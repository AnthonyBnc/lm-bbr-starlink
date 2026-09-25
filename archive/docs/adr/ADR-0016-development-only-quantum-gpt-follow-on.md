# ADR-0016: Development-only Quantum-GPT follow-on protocol

## Status

Approved research direction on 2026-09-03. The experiment sequence and Tokyo
boundary are binding; the exact parent checkpoint, Qiskit VQC configuration,
seed set, optimization budget, and checkpoint rule remain pending decisions.

## Context

The four frozen under-400M modern baselines will first be evaluated on the
existing Tokyo pool. The next intended contribution is `Quantum-GPT`, trained
on Ohio, Sao Paulo, London, Mumbai, and Sydney and evaluated on Tokyo only after
its protocol and checkpoints are frozen.

Because the Tokyo pool has already been generated, a future agent could
otherwise select the GPT parent or quantum design after seeing Tokyo results.
That would turn Tokyo into development evidence and invalidate a pristine
held-out generalisation claim.

The codebase already defines the scientifically necessary three-role
comparison: `gpt_classical`, `gpt_classical_twin`, and `gpt_quantum`.
`Qwen3.5-4B-Base` is the current GPT-style parent candidate from development
evidence. Existing diagnostics also identify unresolved `ProbeBW_UP`
representation collapse.

## Decision

- Keep the paper's State Encoder, return/state/action sequence, Low-Rank
  Adaptation (LoRA), networking-head task, phase mask, 11 actions,
  cross-entropy objective, labels, and metrics.
- Use one identical eligible GPT-family parent and revision for
  `gpt_classical`, `gpt_classical_twin`, and `gpt_quantum`.
- Keep the hidden-state extraction point, LoRA policy, data, sample IDs, seed
  set, optimizer budget, stopping/checkpoint rule, and evaluation records
  identical. Only the declared head/integration component may differ.
- Use the existing Qiskit path for `gpt_quantum`: Qiskit Machine Learning
  `EstimatorQNN`, PyTorch `TorchConnector`, and local
  `StatevectorEstimator` for analytic development runs unless a later ADR
  explicitly changes the backend.
- Resolve the `ProbeBW_UP` representation-separability block using the frozen
  development split before a full headline run.
- Freeze the exact parent checkpoint and all three head protocols in a later
  ADR before full training. `Qwen3.5-4B-Base` remains a candidate until then.
- Train on Ohio, Sao Paulo, London, Mumbai, and Sydney; validate only on the
  frozen five-location development holdout; freeze checkpoints and manifests;
  then run inference on the existing Tokyo pool.
- Do not use Tokyo phase counts, labels, predictions, metrics, or model ranking
  to select the parent, quantum circuit, hyperparameters, epoch, seeds, or
  retry policy.

## Naming and reporting

Use the existing roles exactly: `gpt_classical`, `gpt_classical_twin`, and
`gpt_quantum`. Use `Quantum-GPT` only for the quantum variant, not for the
three-model comparison as a whole. Do not invent a replacement method name.

GPT-2, T5, GPT-Neo, and SmolLM2 remain `published_reference` rows. The frozen
under-400M models and later three-role experiment are repository-measured
extensions, not exact reproduction of the paper's four-SLM experiment.

## Consequences

Completing the four-model Tokyo evaluation does not prohibit the predefined
Quantum-GPT follow-on. It does prohibit adapting that follow-on in response to
Tokyo. If any Quantum-GPT design choice uses Tokyo evidence, label the study
post-hoc exploratory and do not claim pristine held-out generalisation.

No full Quantum-GPT training command is approved by this ADR. The user starts
long training only after the remaining experiment variables are frozen and
explicitly authorizes execution.
