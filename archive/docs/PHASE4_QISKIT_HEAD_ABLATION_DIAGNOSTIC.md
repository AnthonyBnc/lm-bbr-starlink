# Phase 4 Qiskit Head-Ablation Diagnostic

This note summarizes the current Phase 4 development evidence. These are
one-seed development results, not final reportable results. Tokyo remains held
out and must not be used for model selection.

## Shared Setup

- Parent model: `Qwen/Qwen3.5-4B-Base`
- Revision: `1001bb4d826a52d1f399e183466143f4da7b741b`
- Seed: `100003`
- Development split: `slm_bbr_w10_eq2_3_dev_split_seed100003`
- Sequence length: `20`
- Sample step: `20`
- Epochs: `1`
- Action space: 11 phase-masked BBR pacing-gain actions
- Held-out location: `Tokyo`

## Result Summary

| Run | Head | Validation accuracy | Macro-phase accuracy | DOWN | CRUISE | UP |
|---|---|---:|---:|---:|---:|---:|
| q4_d2 | classical | 95.08% | 76.27% | 100.00% | 100.00% | 28.80% |
| q4_d2 | classical_twin | 94.26% | 72.33% | 100.00% | 100.00% | 16.99% |
| q4_d2 | quantum | 88.62% | 40.48% | 0.00% | 100.00% | 21.45% |
| q8_d2 | classical | 95.08% | 76.27% | 100.00% | 100.00% | 28.80% |
| q8_d2 | classical_twin | 94.99% | 75.86% | 100.00% | 100.00% | 27.59% |
| q8_d2 | quantum | 94.57% | 73.82% | 100.00% | 100.00% | 21.45% |
| q8_d2 phase-balanced | classical | 95.34% | 77.55% | 100.00% | 100.00% | 32.65% |
| q8_d2 phase-balanced | classical_twin | 94.99% | 75.86% | 100.00% | 100.00% | 27.59% |
| q8_d2 phase-balanced | quantum | 94.57% | 73.82% | 100.00% | 100.00% | 21.45% |
| q8_d3 | classical | 95.08% | 76.27% | 100.00% | 100.00% | 28.80% |
| q8_d3 | classical_twin | 94.99% | 75.86% | 100.00% | 100.00% | 27.59% |
| q8_d3 | quantum | 94.52% | 73.57% | 100.00% | 100.00% | 20.72% |
| q8_d2 RY+RZ | quantum | 94.57% | 73.82% | 100.00% | 100.00% | 21.45% |

All completed usable runs passed checkpoint reload and Tokyo-isolation audits.

## Interpretation

The 4-qubit Qiskit quantum head worked technically but failed on `BW_DOWN`.
It predicted a valid DOWN action, but collapsed to action `3` / gain `0.96`
while the validation DOWN labels were action `0` / gain `0.90`.

Increasing the quantum head from 4 to 8 qubits fixed the DOWN collapse:

```text
Quantum q4_d2 DOWN: 0.00%
Quantum q8_d2 DOWN: 100.00%
```

However, the 8-qubit quantum head still collapses all `BW_UP` validation
predictions to action `8` / gain `1.15`, giving 21.45% UP accuracy. The
phase-balanced-loss ablation did not improve this quantum UP behavior, although
it improved the classical head's UP accuracy from 28.80% to 32.65%.
Increasing the same 8-qubit RY/CNOT-ring circuit from depth 2 to depth 3 also
did not improve UP; quantum UP decreased slightly from 21.45% to 20.72%.
Adding trainable RZ rotations at q8/depth-2 also left quantum UP at 21.45%.

The current evidence therefore supports this conclusion:

```text
More quantum width fixed DOWN, but simply adding another RY/CNOT-ring layer did
not model UP actions as well as the classical controls.
```

## Next Step

Do not intentionally weaken the classical baselines, and do not use Tokyo yet.
Depth 3 and RY+RZ have now answered the immediate circuit-capacity questions.
The next controlled diagnostic trains only q8/depth-2 RY Quantum-GPT on 1,500
UP-focused sampled windows, then evaluates on the unchanged 600-window
validation set. Historical classical controls remain reference rows because
their 2,400-window training distribution differs. The requested 700-window
test waits for an approved independent split or the final Tokyo gate.
