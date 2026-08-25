# ADR-0008: Qiskit Quantum Backend

## Status

Accepted for development implementation on 2026-08-12.

## Context

The first Quantum-GPT scaffold used a hand-written PyTorch state-vector
simulator to validate gradients and training integration. The project now needs
the quantum model and documentation to explicitly use Qiskit libraries.

The BBR task, Qwen parent candidate, data split, labels, masks, and metrics must
remain unchanged.

## Decision

Use Qiskit as the declared quantum backend for the Quantum-GPT head.

Implementation choices:

- quantum framework: Qiskit;
- quantum ML layer: `qiskit_machine_learning.neural_networks.EstimatorQNN`;
- PyTorch bridge: `qiskit_machine_learning.connectors.TorchConnector`;
- estimator: `qiskit.primitives.StatevectorEstimator`;
- estimator precision: `default_precision=0.0`;
- circuit: RY feature encoding, trainable RY variational layers, CNOT-ring
  entanglement;
- observables: per-qubit Pauli-Z expectation values;
- shots: `None` for analytic development runs;
- output projection: `Linear(n_qubits, 11)`.

## Consequences

- The quantum model now imports and depends on `qiskit` and
  `qiskit-machine-learning`.
- The quantum circuit is auditable as a Qiskit `QuantumCircuit`.
- Training remains integrated with PyTorch through `TorchConnector`.
- The Qiskit layer runs on CPU while the surrounding LM path may run on MPS; the
  tensor transfers remain part of the autograd path.
- Full training cost may increase relative to the hand-written PyTorch
  state-vector scaffold.
- Tokyo remains held out and must not be used to tune the Qiskit backend.

## Validation

Required checks:

- `QuantumActionHead` returns `[..., 11]` logits.
- Gradients flow through the Qiskit `TorchConnector` weights and the input
  projection.
- Manifests record Qiskit versions, estimator, simulator, qubits, depth,
  encoding, entanglement, measurement, and shots.
- Existing classical and head-ablation paths still use the same data and masks.
