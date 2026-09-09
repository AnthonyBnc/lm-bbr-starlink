import unittest

import torch

from plm_special.quantum_head import QuantumActionHead
from utils.bbr import ACTION_LEVELS


class QuantumHeadTests(unittest.TestCase):
    def test_quantum_head_outputs_eleven_logits_and_gradients(self):
        head = QuantumActionHead(input_dim=8, n_qubits=4, depth=2)
        hidden = torch.randn(2, 3, 8, requires_grad=True)
        logits = head(hidden)
        self.assertEqual(tuple(logits.shape), (2, 3, ACTION_LEVELS))
        loss = logits.square().mean()
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(torch.isfinite(head.quantum_layer.weight.grad).all())

    def test_quantum_head_manifest_records_default_vqc(self):
        head = QuantumActionHead(input_dim=8, n_qubits=4, depth=2)
        config = head.manifest_config()
        self.assertEqual(config["n_qubits"], 4)
        self.assertEqual(config["depth"], 2)
        self.assertEqual(config["ansatz"], "trainable_ry_layers")
        self.assertEqual(config["trainable_circuit_parameters"], 8)
        self.assertEqual(config["framework"], "qiskit")
        self.assertEqual(config["measurement"], "per_qubit_pauli_z_expectation")
        self.assertEqual(config["estimator"], "qiskit.primitives.StatevectorEstimator")
        self.assertEqual(config["default_precision"], 0.0)
        self.assertIsNone(config["shots"])

    def test_quantum_head_supports_ry_rz_ansatz(self):
        head = QuantumActionHead(
            input_dim=8,
            n_qubits=4,
            depth=2,
            ansatz="trainable_ry_rz_layers",
        )
        hidden = torch.randn(1, 2, 8, requires_grad=True)
        logits = head(hidden)
        self.assertEqual(tuple(logits.shape), (1, 2, ACTION_LEVELS))
        logits.sum().backward()
        self.assertIsNotNone(hidden.grad)
        config = head.manifest_config()
        self.assertEqual(config["ansatz"], "trainable_ry_rz_layers")
        self.assertEqual(config["trainable_circuit_parameters"], 16)

    def test_quantum_head_records_trainability_diagnostics(self):
        head = QuantumActionHead(input_dim=8, n_qubits=4, depth=2)
        head.enable_diagnostics()
        head(torch.randn(1, 2, 8))
        self.assertIn("bottleneck_std", head.last_diagnostics)
        self.assertIn("bottleneck_saturation_ratio", head.last_diagnostics)
        self.assertGreaterEqual(head.last_diagnostics["bottleneck_saturation_ratio"], 0.0)
        self.assertLessEqual(head.last_diagnostics["bottleneck_saturation_ratio"], 1.0)

    def test_quantum_head_supports_normalized_temperature_scaled_encoding(self):
        head = QuantumActionHead(
            input_dim=8,
            n_qubits=4,
            depth=2,
            input_layernorm=True,
            temperature=4.0,
            angle_scale="half_pi",
        )
        head.enable_diagnostics()
        logits = head(torch.randn(1, 2, 8))
        self.assertEqual(tuple(logits.shape), (1, 2, ACTION_LEVELS))
        self.assertIn("raw_projection_abs_mean", head.last_diagnostics)
        self.assertIn("angle_abs_mean", head.last_diagnostics)
        config = head.manifest_config()
        self.assertTrue(config["input_layernorm"])
        self.assertEqual(config["temperature"], 4.0)
        self.assertEqual(config["angle_scale"], "half_pi")

    def test_torch_backend_matches_qiskit_for_every_ansatz(self):
        """The torch backend must be an exact drop-in: same circuit, same
        expectations. Guards against the two paths silently diverging."""
        for ansatz, depth, n_qubits in (
            ("trainable_ry_layers", 1, 4),
            ("trainable_ry_layers", 3, 5),
            ("trainable_ry_rz_layers", 2, 4),
            ("data_reuploading_ry", 2, 4),
            ("trainable_ry_layers", 2, 1),
        ):
            with self.subTest(ansatz=ansatz, depth=depth, n_qubits=n_qubits):
                shared = dict(input_dim=8, n_qubits=n_qubits, depth=depth, ansatz=ansatz)
                qiskit_head = QuantumActionHead(backend="qiskit", **shared)
                torch_head = QuantumActionHead(backend="torch", **shared)

                weights = torch.randn(qiskit_head.weight_count)
                with torch.no_grad():
                    qiskit_head.quantum_layer.weight.copy_(weights)
                    torch_head.quantum_layer.weight.copy_(weights)

                angles = torch.randn(6, n_qubits)
                expected = qiskit_head.quantum_layer(angles)
                actual = torch_head.quantum_layer(angles)
                self.assertLess((expected - actual).abs().max().item(), 1e-5)

    def test_torch_backend_is_differentiable_and_reports_provenance(self):
        head = QuantumActionHead(input_dim=8, n_qubits=4, depth=2, backend="torch")
        hidden = torch.randn(2, 3, 8, requires_grad=True)
        logits = head(hidden)
        self.assertEqual(tuple(logits.shape), (2, 3, ACTION_LEVELS))
        logits.square().mean().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(torch.isfinite(head.quantum_layer.weight.grad).all())

        config = head.manifest_config()
        self.assertEqual(config["backend"], "torch")
        self.assertEqual(config["gradient_method"], "autograd")
        self.assertEqual(config["estimator"], "torch_exact_statevector")

    def test_qiskit_backend_remains_the_default(self):
        head = QuantumActionHead(input_dim=8, n_qubits=4, depth=2)
        config = head.manifest_config()
        self.assertEqual(config["backend"], "qiskit")
        self.assertEqual(config["gradient_method"], "qiskit_parameter_shift")

    def test_checkpoints_are_interchangeable_between_backends(self):
        """A run started on one backend must be resumable on the other under
        strict=True -- train_modern_lora loads task modules strictly, and
        Qiskit's TorchConnector carries a duplicate `_weights` key that the
        torch backend has to both accept and emit."""
        shared = dict(input_dim=16, n_qubits=4, depth=2)
        sample = torch.randn(3, 2, 16)

        for source_backend, target_backend in (("qiskit", "torch"), ("torch", "qiskit")):
            with self.subTest(source=source_backend, target=target_backend):
                source = QuantumActionHead(backend=source_backend, **shared)
                with torch.no_grad():
                    source.quantum_layer.weight.copy_(torch.randn(source.weight_count))
                    source.angle_projection.weight.normal_()
                    source.output_projection.weight.normal_()

                target = QuantumActionHead(backend=target_backend, **shared)
                target.load_state_dict(source.state_dict(), strict=True)

                with torch.no_grad():
                    difference = (source(sample) - target(sample)).abs().max().item()
                self.assertLess(difference, 1e-5)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            QuantumActionHead(input_dim=8, n_qubits=4, depth=2, backend="pennylane")


if __name__ == "__main__":
    unittest.main()
