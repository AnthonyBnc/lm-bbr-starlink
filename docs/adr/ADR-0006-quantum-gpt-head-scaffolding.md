# ADR-0006: Quantum-GPT Head Scaffolding

## Status

Proposed for implementation validation on 2026-08-11.

## Context

The architecture contract defines Quantum-GPT as the selected GPT-style parent
with a trainable quantum component added at the action-position hidden state.
Phase 3 selected Qwen3.5-4B-Base as the current leading GPT-style candidate, and
Phase 4 started a multi-seed development confirmation step. The final quantum
parent is not frozen yet.

We still need code that can build, test, and audit the quantum path before any
long training run.

## Decision

Add a small differentiable Qiskit-backed Quantum-GPT action head behind an explicit
`head_type="quantum"` option in `OfflineRLPolicy`.

Initial smoke configuration:

- parent model key: `qwen3_5_4b_base`;
- insertion point: action-position hidden state;
- qubits: `4`;
- depth: `2`;
- quantum framework: Qiskit;
- Qiskit neural network: `EstimatorQNN`;
- PyTorch bridge: `TorchConnector`;
- estimator: `qiskit.primitives.StatevectorEstimator`;
- estimator precision: `default_precision=0.0`;
- encoding: bounded RY angle encoding;
- trainable circuit layers: RY rotations;
- entanglement: CNOT ring;
- measurement: Pauli-Z expectation values;
- simulator: Qiskit local analytic statevector estimator;
- shots: `None`;
- output: exactly 11 BBR action logits.

Also add `head_type="classical_twin"` as the parameter-light classical
bottleneck control path:

```text
hidden -> Linear(d_model, n_qubits) -> tanh -> Linear(n_qubits, 11)
```

## Consequences

- Existing classical runs remain the default and are unchanged unless
  `--head-type` is supplied.
- Quantum and classical-twin heads use the same shared state/return/action
  sequence, labels, masks, metrics, split, and checkpoint path.
- The quantum head depends on `qiskit` and `qiskit-machine-learning`.
- The smoke runner is limited to one train batch and one validation batch by
  default; its output is not a model-quality result.
- Full Quantum-GPT training should wait until Phase 4 confirms or freezes the
  GPT parent and the final comparison policy is approved.
- Tokyo remains held out.

## Commands

Dry-run the quantum smoke command:

```bash
.venv/bin/python run_quantum_smoke.py
```

Execute the tiny quantum smoke check:

```bash
.venv/bin/python -u run_quantum_smoke.py --execute
```
