# ADR-0017: Under-400M Quantum-GPT parent, head configuration, and protocol freeze

## Status

Approved by the user on 2026-09-03. This ADR freezes the variables ADR-0016
left pending. It authorizes preparing the classical-twin and quantum training
commands. The user explicitly withheld authorization to execute training in
this session; a human runs the commands below when ready.

## Context

[ADR-0015](ADR-0015-frozen-tokyo-eight-model-evaluation.md) produced a frozen
Tokyo evaluation of four under-400M classical baselines. Their Tokyo numbers
(accuracy 0.9498-0.9606, `granite_4_0_350m` ranked first) were reviewed in
this session only as pre-existing repository state, at the user's request to
continue from "current Tokyo results." [ADR-0016](ADR-0016-development-only-quantum-gpt-follow-on.md)
explicitly prohibits using Tokyo ranking, metrics, or predictions to select
the Quantum-GPT parent, circuit, or hyperparameters, to keep Tokyo a pristine
held-out generalisation test. The user confirmed in this session that parent
selection must use development-only evidence, not Tokyo.

ADR-0016 also named `Qwen3.5-4B-Base` as a *candidate* parent pending freeze.
The user has now scoped this follow-on to a backbone under 400M parameters,
so that candidate no longer applies; this ADR selects the parent from the
four checkpoints already frozen under
[ADR-0013](ADR-0013-under-400m-modern-backbone-extension.md) /
[ADR-0014](ADR-0014-gemma-3-270m-compute-replacement.md).

## Decision

### Parent selection (development evidence only)

Source: epoch-16 **validation** metrics from
`data/processed/lora_training/under400_rank128_epochs16_seed100003_v1/` and
`data/processed/lora_training/gemma3_270m_rank128_epochs16_seed100003_v1/`,
computed on the frozen five-location development holdout (Ohio, Sao Paulo,
London, Mumbai, Sydney train / same-protocol validation split). Tokyo was not
read for this comparison.

| Model key | Val accuracy | Val macro-phase accuracy | Val BW_UP accuracy |
|---|---|---|---|
| `granite_4_0_350m` | 0.9503 | 0.7606 | 0.2819 |
| `pleias_rag_350m` | 0.9505 | 0.7614 | 0.2843 |
| **`lfm2_5_350m`** | **0.9550** | **0.7831** | **0.3494** |
| `gemma_3_270m` | 0.9498 | 0.7582 | 0.2747 |

`lfm2_5_350m` (`LiquidAI/LFM2.5-350M`, revision
`9e6c6ccf47cd318696e137d381a7ded8fe4df09f`) wins on all three development
metrics, including the historically weakest `BW_UP` phase called out in
[ADR-0009](ADR-0009-quantum-up-focused-training-diagnostic.md). It is the
frozen Quantum-GPT parent for this under-400M follow-on. `config.py`
`quantum_defaults.parent_model_key` is updated from `qwen3_5_4b_base` to
`lfm2_5_350m` accordingly.

### Three-role comparison

Reuse the existing frozen classical run as the `gpt_classical` reference
(no retraining): `under400_rank128_epochs16_seed100003_v1/lfm2_5_350m_rank128_epochs16_seed100003_train`
(`model_key=lfm2_5_350m`, rank 128, 16 epochs, seed 100003, Tokyo isolation
PASS, `checkpoint_reload` PASS). Train two new sibling runs with identical
data, split, seed, rank, epochs, and LoRA target modules, varying only the
head:

- `gpt_classical_twin`: `--head-type classical_twin`
- `gpt_quantum`: `--head-type quantum`

### Frozen quantum head configuration

- qubits: `8`
- depth: `2`
- ansatz: `trainable_ry_layers`
- angle scale: `pi`
- input layernorm: `False`
- temperature: `1.0`
- framework/estimator/simulator: unchanged from
  [ADR-0006](ADR-0006-quantum-gpt-head-scaffolding.md) /
  [ADR-0008](ADR-0008-qiskit-quantum-backend.md) (Qiskit `EstimatorQNN`,
  `TorchConnector`, local `StatevectorEstimator`, `default_precision=0.0`,
  `shots=None`).

This reuses the q8/depth-2 RY configuration already exercised in the Qwen
head ablations and the ADR-0009/ADR-0011 diagnostics, rather than inventing
an untested circuit for the new parent.

### Frozen training protocol

Identical across `gpt_classical`, `gpt_classical_twin`, and `gpt_quantum`:

- data: Ohio, Sao Paulo, London, Mumbai, Sydney train pool; frozen
  development validation split; Tokyo excluded.
- split dir: `data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003`
- sequence length 20, sample step 20, batch size 1, gradient accumulation 32
- LoRA rank 128, alpha 32, dropout 0.05, target modules per
  `cfg.modern_lora_registry['lfm2_5_350m']`
  (`q_proj`, `v_proj`, `in_proj`, `out_proj`)
- seed 100003, 16 epochs, learning rate 1e-4, gradient clip norm 0.25
- device/dtype: same as the existing `lfm2_5_350m` classical run (`mps`,
  `float16`), or `--dtype auto` on the execution machine if different
  hardware is used — record whichever is actually used in the run manifest
- new `run_purpose`: `under400m_quantum_gpt_follow_on` (added to
  `train_modern_lora.py`), which maps `--head-type` to the roles
  `gpt_classical` / `gpt_classical_twin` / `gpt_quantum` and marks
  `method_provenance.backbone_substitution = true`

### Checkpoint rule

Select the final completed epoch (epoch 16) for both new runs, matching
[ADR-0015](ADR-0015-frozen-tokyo-eight-model-evaluation.md). Record SHA-256
checksums for the adapter, task modules, training state, and run manifest.
Require `checkpoint_reload: PASS` before any downstream use.

### Pre-headline blocking step (development-only, not a training gate)

Before labelling any `gpt_classical_twin`/`gpt_quantum` comparison a
headline result, rerun the `ProbeBW_UP` representation-separability
diagnostic (`analyze_up_separability.py`) against the new `lfm2_5_350m`
checkpoints instead of the Qwen `trainability_gate200_ln_t4` runs it
currently points at. This only gates the *headline* claim, not whether
training may start.

### Tokyo boundary (unchanged, reaffirmed)

Do not read, evaluate against, or otherwise use Tokyo for any decision in
this ADR or during `gpt_classical_twin`/`gpt_quantum` training, validation,
or checkpoint selection. Tokyo evaluation for these two new roles runs once,
after checkpoints are frozen, through the existing
`evaluate_tokyo_models.py` / ADR-0015 gate, extended to accept `head_type`
and `quantum_config` (already threaded through `evaluate_tokyo_models.py`'s
manifest-driven policy construction — see `analyze_up_separability.py`'s
`build_policy` for the same pattern).

## Commands (prepared, not executed by this ADR)

```bash
.venv/bin/python -u train_modern_lora.py \
  --model-key lfm2_5_350m \
  --head-type classical_twin \
  --rank 128 --alpha 32 --dropout 0.05 \
  --epochs 16 --seed 100003 \
  --sequence-length 20 --sample-step 20 \
  --grad-accum-steps 32 --learning-rate 1e-4 \
  --run-purpose under400m_quantum_gpt_follow_on \
  --output-dir data/processed/lora_training/under400m_quantum_gpt_follow_on_v1/lfm2_5_350m_classical_twin
```

```bash
.venv/bin/python -u train_modern_lora.py \
  --model-key lfm2_5_350m \
  --head-type quantum \
  --n-qubits 8 --quantum-depth 2 --quantum-ansatz trainable_ry_layers \
  --quantum-angle-scale pi \
  --rank 128 --alpha 32 --dropout 0.05 \
  --epochs 16 --seed 100003 \
  --sequence-length 20 --sample-step 20 \
  --grad-accum-steps 32 --learning-rate 1e-4 \
  --run-purpose under400m_quantum_gpt_follow_on \
  --output-dir data/processed/lora_training/under400m_quantum_gpt_follow_on_v1/lfm2_5_350m_quantum
```

Run a one-train-step/one-validation-step preflight for both (`--max-train-steps 1 --max-validation-steps 1`)
before the full 16-epoch run, matching the ADR-0013 preflight practice.

## Consequences

- The Quantum-GPT parent is now `lfm2_5_350m`, not `Qwen3.5-4B-Base`; this
  supersedes that part of ADR-0016 for the under-400M scope.
- Because selection used development evidence only, the eventual Tokyo
  evaluation of `gpt_classical_twin`/`gpt_quantum` can still claim pristine
  held-out generalisation, consistent with ADR-0016.
- No training has run yet. The user runs the two commands above (and their
  preflights) when ready; this session does not execute them.
- Tokyo evaluation for these two new roles remains a one-time action after
  checkpoints, manifests, and the `ProbeBW_UP` re-diagnostic are complete.
