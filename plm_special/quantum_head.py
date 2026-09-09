"""Qiskit-backed VQC action head for Quantum-GPT experiments."""

import math

import numpy as np
import qiskit
import qiskit_machine_learning
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch import nn

from utils.bbr import ACTION_LEVELS


ANSATZ_TRAINABLE_RY = "trainable_ry_layers"
ANSATZ_TRAINABLE_RY_RZ = "trainable_ry_rz_layers"
ANSATZ_DATA_REUPLOADING_RY = "data_reuploading_ry"
SUPPORTED_ANSATZES = (ANSATZ_TRAINABLE_RY, ANSATZ_TRAINABLE_RY_RZ, ANSATZ_DATA_REUPLOADING_RY)
ANGLE_SCALE_PI = "pi"
ANGLE_SCALE_HALF_PI = "half_pi"
SUPPORTED_ANGLE_SCALES = (ANGLE_SCALE_PI, ANGLE_SCALE_HALF_PI)

BACKEND_QISKIT = "qiskit"
BACKEND_TORCH = "torch"
SUPPORTED_BACKENDS = (BACKEND_QISKIT, BACKEND_TORCH)


class TorchStatevectorQNN(nn.Module):
    """Exact statevector simulation of the same VQC, in native torch ops.

    Numerically equivalent to the Qiskit `EstimatorQNN` path (which uses
    `StatevectorEstimator` at `default_precision=0.0`, i.e. exact, shot-free
    simulation), but differentiated by autograd instead of Qiskit's
    parameter-shift rule. Parameter-shift costs 2 extra circuit evaluations per
    circuit parameter per sample and Qiskit evaluates samples one at a time, so
    the Qiskit path is ~200ms/sample and single-core-bound; this one is batched
    and runs on whatever device the angles are on.

    The state is carried as separate real/imaginary tensors rather than a
    complex dtype so the head still runs on MPS, where complex support is
    incomplete.

    Mirrors `TorchConnector`'s interface: a `weight` parameter, `(batch,
    n_qubits)` angles in, `(batch, n_qubits)` Pauli-Z expectations out.
    """

    def __init__(self, n_qubits, depth, ansatz, weight_count):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.ansatz = ansatz
        self.weight = nn.Parameter(torch.zeros(weight_count))
        # Qiskit's TorchConnector stores the circuit weights under two keys,
        # `weight` and `_weights`, holding the same tensor. Mirroring that on
        # save, and tolerating it on load, keeps checkpoints interchangeable
        # between the two backends under strict=True -- so a run started on one
        # backend can be resumed on the other without a conversion step.
        self._register_state_dict_hook(self._emit_qiskit_alias)

        dim = 2 ** n_qubits
        index = torch.arange(dim)
        # Qiskit is little-endian: qubit q is bit q of the basis-state index.
        for qubit in range(n_qubits):
            self.register_buffer(
                "z_sign_{}".format(qubit),
                1.0 - 2.0 * ((index >> qubit) & 1).float(),
                persistent=False,
            )
        if n_qubits > 1:
            for control in range(n_qubits):
                target = (control + 1) % n_qubits
                # CNOT permutes basis states: out[i] = in[i with bit `target`
                # flipped when bit `control` is set]. Self-inverse, so the same
                # index serves as gather map.
                self.register_buffer(
                    "cx_perm_{}".format(control),
                    index ^ (((index >> control) & 1) << target),
                    persistent=False,
                )

    @staticmethod
    def _emit_qiskit_alias(module, state_dict, prefix, local_metadata):
        state_dict[prefix + "_weights"] = state_dict[prefix + "weight"]
        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, *args):
        alias = prefix + "_weights"
        if alias in state_dict:
            if prefix + "weight" not in state_dict:
                state_dict[prefix + "weight"] = state_dict[alias]
            state_dict = {key: value for key, value in state_dict.items() if key != alias}
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, *args)

    def _apply_ry(self, real, imag, angles, qubit):
        batch = real.shape[0]
        shape = (batch, 2 ** (self.n_qubits - qubit - 1), 2, 2 ** qubit)
        real, imag = real.reshape(shape), imag.reshape(shape)
        half = angles.reshape(-1, 1, 1) / 2
        cos, sin = torch.cos(half), torch.sin(half)
        r0, r1 = real[:, :, 0, :], real[:, :, 1, :]
        i0, i1 = imag[:, :, 0, :], imag[:, :, 1, :]
        out_real = torch.stack([cos * r0 - sin * r1, sin * r0 + cos * r1], dim=2)
        out_imag = torch.stack([cos * i0 - sin * i1, sin * i0 + cos * i1], dim=2)
        flat = (real.shape[0], 2 ** self.n_qubits)
        return out_real.reshape(flat), out_imag.reshape(flat)

    def _apply_rz(self, real, imag, angles, qubit):
        batch = real.shape[0]
        shape = (batch, 2 ** (self.n_qubits - qubit - 1), 2, 2 ** qubit)
        real, imag = real.reshape(shape), imag.reshape(shape)
        half = angles.reshape(-1, 1, 1) / 2
        cos, sin = torch.cos(half), torch.sin(half)
        r0, r1 = real[:, :, 0, :], real[:, :, 1, :]
        i0, i1 = imag[:, :, 0, :], imag[:, :, 1, :]
        # branch 0 multiplied by exp(-i a/2), branch 1 by exp(+i a/2)
        out_real = torch.stack([r0 * cos + i0 * sin, r1 * cos - i1 * sin], dim=2)
        out_imag = torch.stack([i0 * cos - r0 * sin, i1 * cos + r1 * sin], dim=2)
        flat = (real.shape[0], 2 ** self.n_qubits)
        return out_real.reshape(flat), out_imag.reshape(flat)

    def _apply_cx_ring(self, real, imag):
        for control in range(self.n_qubits):
            perm = getattr(self, "cx_perm_{}".format(control))
            real, imag = real[:, perm], imag[:, perm]
        return real, imag

    def forward(self, angles):
        batch = angles.shape[0]
        real = torch.zeros(batch, 2 ** self.n_qubits, dtype=angles.dtype, device=angles.device)
        real[:, 0] = 1.0
        imag = torch.zeros_like(real)

        reuploading = self.ansatz == ANSATZ_DATA_REUPLOADING_RY
        if not reuploading:
            for qubit in range(self.n_qubits):
                real, imag = self._apply_ry(real, imag, angles[:, qubit], qubit)

        cursor = 0
        for _ in range(self.depth):
            if reuploading:
                for qubit in range(self.n_qubits):
                    real, imag = self._apply_ry(real, imag, angles[:, qubit], qubit)
            for qubit in range(self.n_qubits):
                weight = self.weight[cursor].expand(batch)
                real, imag = self._apply_ry(real, imag, weight, qubit)
                cursor += 1
                if self.ansatz == ANSATZ_TRAINABLE_RY_RZ:
                    weight = self.weight[cursor].expand(batch)
                    real, imag = self._apply_rz(real, imag, weight, qubit)
                    cursor += 1
            if self.n_qubits > 1:
                real, imag = self._apply_cx_ring(real, imag)

        probabilities = real * real + imag * imag
        return torch.stack(
            [
                (probabilities * getattr(self, "z_sign_{}".format(qubit))).sum(dim=1)
                for qubit in range(self.n_qubits)
            ],
            dim=1,
        )


class QuantumActionHead(nn.Module):
    """Hybrid Qiskit/PyTorch head for 11-action BBR prediction.

    The head uses Qiskit Machine Learning for the quantum layer:

    hidden -> Linear(d_model, n_qubits) -> pi*tanh angle bound
    -> Qiskit EstimatorQNN with RY feature encoding, a named trainable ansatz,
    CNOT-ring entanglement, and per-qubit Pauli-Z observables
    -> Linear(n_qubits, 11).

    Qiskit supplies the statevector estimator and PyTorch connector. The output
    remains differentiable through the Qiskit TorchConnector, so LoRA, the input
    projection, quantum weights, and output projection can train end to end.
    """

    def __init__(
        self,
        input_dim,
        action_levels=ACTION_LEVELS,
        n_qubits=4,
        depth=2,
        ansatz=ANSATZ_TRAINABLE_RY,
        input_layernorm=False,
        temperature=1.0,
        angle_scale=ANGLE_SCALE_PI,
        backend=BACKEND_QISKIT,
    ):
        super().__init__()
        if action_levels != ACTION_LEVELS:
            raise ValueError("QuantumActionHead requires {} actions".format(ACTION_LEVELS))
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if depth < 1:
            raise ValueError("depth must be positive")
        if ansatz not in SUPPORTED_ANSATZES:
            raise ValueError(
                "Unknown quantum ansatz {!r}; expected one of {}".format(
                    ansatz, SUPPORTED_ANSATZES
                )
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if angle_scale not in SUPPORTED_ANGLE_SCALES:
            raise ValueError(
                "Unknown angle_scale {!r}; expected one of {}".format(
                    angle_scale, SUPPORTED_ANGLE_SCALES
                )
            )
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                "Unknown backend {!r}; expected one of {}".format(backend, SUPPORTED_BACKENDS)
            )

        self.input_dim = input_dim
        self.action_levels = action_levels
        self.n_qubits = n_qubits
        self.depth = depth
        self.ansatz = ansatz
        self.input_layernorm = bool(input_layernorm)
        self.temperature = float(temperature)
        self.angle_scale_name = angle_scale
        self.angle_scale = math.pi if angle_scale == ANGLE_SCALE_PI else math.pi / 2.0
        self.backend = backend
        self.weight_count = self._weight_count()

        self.input_norm = nn.LayerNorm(input_dim) if self.input_layernorm else nn.Identity()
        self.angle_projection = nn.Linear(input_dim, n_qubits)
        if backend == BACKEND_TORCH:
            self.qnn = None
            self.quantum_layer = TorchStatevectorQNN(
                n_qubits=n_qubits,
                depth=depth,
                ansatz=ansatz,
                weight_count=self.weight_count,
            )
        else:
            self.qnn = self._build_qiskit_qnn()
            self.quantum_layer = TorchConnector(
                self.qnn,
                initial_weights=np.zeros(self.weight_count, dtype=np.float32),
            )
        self.output_projection = nn.Linear(n_qubits, action_levels)
        self.diagnostics_enabled = False
        self.last_diagnostics = None

    def _weight_count(self):
        per_qubit_weights = {
            ANSATZ_TRAINABLE_RY: 1,
            ANSATZ_TRAINABLE_RY_RZ: 2,
            ANSATZ_DATA_REUPLOADING_RY: 1,
        }[self.ansatz]
        return self.depth * self.n_qubits * per_qubit_weights

    def _build_qiskit_qnn(self):
        input_params = ParameterVector("x", self.n_qubits)
        weight_params = ParameterVector("theta", self.weight_count)
        circuit = QuantumCircuit(self.n_qubits)

        reuploading = self.ansatz == ANSATZ_DATA_REUPLOADING_RY
        if not reuploading:
            for qubit in range(self.n_qubits):
                circuit.ry(input_params[qubit], qubit)
        cursor = 0
        for layer in range(self.depth):
            if reuploading:
                # Re-encode the same classical features before every
                # variational layer instead of once up front. This is the
                # standard data-re-uploading construction (Perez-Salinas
                # et al., 2020): repeated encode/vary blocks raise circuit
                # expressivity without adding qubits or trainable weights.
                for qubit in range(self.n_qubits):
                    circuit.ry(input_params[qubit], qubit)
            for qubit in range(self.n_qubits):
                circuit.ry(weight_params[cursor], qubit)
                cursor += 1
                if self.ansatz == ANSATZ_TRAINABLE_RY_RZ:
                    circuit.rz(weight_params[cursor], qubit)
                    cursor += 1
            if self.n_qubits > 1:
                for control in range(self.n_qubits):
                    circuit.cx(control, (control + 1) % self.n_qubits)

        observables = []
        for qubit in range(self.n_qubits):
            pauli = ["I"] * self.n_qubits
            pauli[self.n_qubits - qubit - 1] = "Z"
            observables.append(SparsePauliOp.from_list([("".join(pauli), 1.0)]))

        return EstimatorQNN(
            circuit=circuit,
            estimator=StatevectorEstimator(),
            observables=observables,
            input_params=list(input_params),
            weight_params=list(weight_params),
            input_gradients=True,
            default_precision=0.0,
        )

    def forward(self, hidden_states):
        original_shape = hidden_states.shape[:-1]
        hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
        projected = self.angle_projection(self.input_norm(hidden))
        bounded = torch.tanh(projected / self.temperature)
        angles = self.angle_scale * bounded

        if self.backend == BACKEND_TORCH:
            expectations = self.quantum_layer(angles)
        else:
            # Qiskit's TorchConnector runs on CPU tensors. The device transfers
            # remain in the autograd graph, so gradients still flow back to the LM.
            expectations = self.quantum_layer(angles.to("cpu", dtype=torch.float32))
        expectations = expectations.to(device=hidden.device, dtype=hidden.dtype)
        logits = self.output_projection(expectations)
        if self.diagnostics_enabled:
            self.last_diagnostics = {
                "raw_projection_abs_mean": projected.detach().float().abs().mean().item(),
                "angle_abs_mean": angles.detach().float().abs().mean().item(),
                "bottleneck_std": expectations.detach().float().std(unbiased=False).item(),
                "bottleneck_saturation_ratio": (
                    bounded.detach().float().abs() > 0.99
                ).float().mean().item(),
            }
        return logits.reshape(*original_shape, self.action_levels)

    def enable_diagnostics(self, enabled=True):
        self.diagnostics_enabled = bool(enabled)
        self.last_diagnostics = None

    def manifest_config(self):
        return {
            "type": "qiskit_estimator_qnn_vqc",
            "framework": "qiskit",
            "qiskit_version": qiskit.__version__,
            "qiskit_machine_learning_version": qiskit_machine_learning.__version__,
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "encoding": (
                "bounded_ry_angle_encoding_reuploaded_each_layer"
                if self.ansatz == ANSATZ_DATA_REUPLOADING_RY
                else "bounded_ry_angle_encoding"
            ),
            "input_layernorm": self.input_layernorm,
            "temperature": self.temperature,
            "angle_scale": self.angle_scale_name,
            "ansatz": self.ansatz,
            "trainable_circuit_parameters": self.weight_count,
            "entanglement": "cnot_ring",
            "measurement": "per_qubit_pauli_z_expectation",
            "backend": self.backend,
            "gradient_method": (
                "autograd" if self.backend == BACKEND_TORCH else "qiskit_parameter_shift"
            ),
            "estimator": (
                "torch_exact_statevector"
                if self.backend == BACKEND_TORCH
                else "qiskit.primitives.StatevectorEstimator"
            ),
            "simulator": (
                "torch_statevector"
                if self.backend == BACKEND_TORCH
                else "qiskit_statevector_estimator"
            ),
            "default_precision": 0.0,
            "shots": None,
        }
