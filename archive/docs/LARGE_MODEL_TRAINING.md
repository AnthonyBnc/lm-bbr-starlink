# Large-Backbone and Quantum-GPT Training Guide

This is the single canonical operations guide for training or resuming the
2.6B-4B modern backbones and for the later `Quantum-GPT` experiment. Do not
duplicate these commands in another document. Result interpretation remains in
`PHASE3_RESULTS_README.md`; current decisions remain in `CONTEXT.md`.

## Scientific boundary

Every run in this guide implements **Small Language Model-based Control for BBR
over Low Earth Orbit Satellite Internet** with a substituted backbone. It keeps
the paper's State Encoder, return/state/action sequence, Low-Rank Adaptation
(LoRA), networking head, cross-entropy objective, 11 pacing-gain actions, and
BBR macro-phase constraint.

The following labels have fixed meanings:

| Display name | Code identifier | Meaning |
|---|---|---|
| Small Language Model (SLM) | legacy `PLM` identifiers | Paper method/model category |
| State Encoder | `EncoderNetwork`, `state_encoder` | Nine-field state encoder |
| Offline RL Policy with Language Model Head | `OfflineRLPolicy` | Return-conditioned policy |
| networking head | `action_head`, `classical` | Eleven-logit task head |
| ProbeBW_DOWN | `BW_DOWN` | Gains 0.90-0.98 |
| ProbeBW_CRUISE | `BW_CRUISE` | Gain 1.00 |
| ProbeBW_UP | `BW_UP` | Gains 1.05-1.25 |
| Quantum-GPT | `gpt_quantum`, `head_type=quantum` | GPT parent plus declared Qiskit VQC head |

Do not invent a new scientific name for any of these components. Machine-safe
run IDs may use the existing code identifiers only.

## Data and evaluation boundary

- Train and development-validation data contain Ohio, Sao Paulo, London,
  Mumbai, and Sydney only.
- Tokyo is held-out evaluation data and must never be used for training,
  validation, normalization, early stopping, architecture selection, or
  hyperparameter selection.
- The generated Tokyo pool already exists, so a future agent must not inspect
  its labels, predictions, or metrics while developing `Quantum-GPT`.
- All model families use the shared preprocessing, labels, returns, sample IDs,
  phase masks, and metrics.

The development split is:

```text
data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003/
  train.pkl
  validation.pkl
  split.manifest.json
```

## Modern large-backbone set

The completed exploratory large-backbone set uses these exact display names
and registry keys:

| Display name | Registry key | Role |
|---|---|---|
| LFM2.5-2.6B | `lfm2_5_2_6b` | modern baseline |
| Llama-3.2-3B | `llama_3_2_3b` | modern baseline |
| Qwen3.5-4B-Base | `qwen3_5_4b_base` | modern baseline and current GPT-style parent candidate |
| Gemma-3-4B-PT | `gemma_3_4b_pt` | modern baseline |

These are a modern-backbone extension. They are not GPT-2, T5, GPT-Neo, or
SmolLM2 and must not be described as the paper's four-SLM reproduction.

The completed Phase 3 runs used rank 8, alpha 32, dropout 0.05, five epochs,
sequence length 20, sample step 20, gradient accumulation 32, learning rate
`1e-4`, weight decay `1e-4`, seed `100003`, original sampling, and unweighted
phase-masked cross-entropy. See `PHASE3_RESULTS_README.md` for the results.

## Local model paths

Never edit `config.py` just to match one machine. Set the model root:

```bash
export LM_BBR_LOCAL_MODEL_ROOT=/path/to/models
```

The directory must contain folders matching `config.py`. On the current Mac,
the downloaded small-model root is `/Users/baoan/models`; agents must not bake
that user-specific path into shared code.

## Plan a large-backbone run

Planning does not start training:

```bash
.venv/bin/python run_slm_bbr_modern_backbones.py \
  --model-set approved-modern \
  --rank 8 \
  --epochs 5 \
  --warmup-steps 0 \
  --output-root data/processed/lora_training/large_backbones_plan
```

Only the user starts a long run. After reviewing the generated plan:

```bash
caffeinate -dimsu \
  .venv/bin/python run_slm_bbr_modern_backbones.py \
  --model-set approved-modern \
  --rank 8 \
  --epochs 5 \
  --warmup-steps 0 \
  --output-root data/processed/lora_training/large_backbones_run \
  --execute \
  --acknowledge-long-run
```

Do not use this template to create a new headline run until epochs, rank, seed
set, and checkpoint rule are approved and recorded. The shown command documents
the completed exploratory policy.

## Resume a completed or interrupted run

`--epochs` means target total epochs when `--resume-from-run` is supplied. A
source at epoch 5 with `--epochs 6` executes epoch 6 only. Resume restores the
LoRA adapter, State Encoder, networking head, AdamW state, scheduler state when
present, metrics history, and completed epoch.

Plan a four-model continuation:

```bash
.venv/bin/python run_slm_bbr_modern_backbones.py \
  --model-set approved-modern \
  --rank 8 \
  --epochs 6 \
  --warmup-steps 0 \
  --resume-from-root data/processed/lora_training/phase3_rank8_epochs5_seed100003 \
  --output-root data/processed/lora_training/modern_backbones_resumed_to_epoch6
```

Run a disposable resume preflight first:

```bash
.venv/bin/python run_slm_bbr_modern_backbones.py \
  --model-set approved-modern \
  --preflight \
  --execute \
  --rank 8 \
  --epochs 6 \
  --warmup-steps 0 \
  --resume-from-root data/processed/lora_training/phase3_rank8_epochs5_seed100003 \
  --output-root data/processed/lora_training/modern_backbones_resume_preflight
```

Resume must reject drift in model revision, dtype, seed, sequence length,
sample step, gradient accumulation, learning rate, weight decay, warmup, head
type, LoRA configuration, split hashes, sampling, or loss weighting. Always use
a new output directory. Never overwrite the source run.

## Cloud execution

Google Colab or an ordinary IBM Cloud GPU VM may run the same command with
`--device cuda`. Persist the repository, split, backbone snapshots, source
checkpoints, optimizer state, metrics, and manifests outside ephemeral storage.
Record the hardware transition because MPS and CUDA are not bit-for-bit
equivalent.

IBM Quantum hardware is not a replacement for the GPU required by the language
model backbone. The declared development backend is Qiskit Machine Learning's
`EstimatorQNN` and `TorchConnector` with local `StatevectorEstimator`. Moving to
IBM Quantum Runtime changes shots, noise, queueing, and estimator behavior and
requires a separately approved ADR.

## Quantum-GPT milestone after the modern baselines

The primary causal comparison is exactly this trio on one frozen GPT-family
parent:

| Model role | Existing ID | Required difference |
|---|---|---|
| GPT classical control | `gpt_classical` | Normal networking head |
| GPT classical twin | `gpt_classical_twin` | Parameter-budget-matched classical bottleneck |
| Quantum-GPT | `gpt_quantum` | Declared Qiskit VQC head |

The parent checkpoint, hidden-state extraction point, State Encoder, LoRA
policy, five-location data, split, sequence sampling, labels, phase mask, seed
set, optimizer budget, checkpoint rule, and metrics must be identical. Only the
declared head/integration component may differ.

`Qwen3.5-4B-Base` is the current GPT-style parent candidate because it was
selected from development evidence and already has classical/twin/quantum
diagnostic support. It is not final until its exact checkpoint and quantum
configuration are frozen in an approved ADR. Do not attach the quantum head to
whichever small model performs best on Tokyo; Granite, Pleias-RAG, LFM2.5, and
Gemma are modern baseline rows, not automatically eligible Quantum-GPT parents.
The binding follow-on boundary is recorded in
`docs/adr/ADR-0016-development-only-quantum-gpt-follow-on.md`.

The current diagnostics found `ProbeBW_UP` action collapse. Full headline
Quantum-GPT training remains blocked until the representation-separability
analysis is reviewed. Do not respond to this block by adding epochs, changing
labels, rebalancing only the quantum model, or inspecting Tokyo.

After the block is resolved, the correct order is:

1. Freeze the exact GPT parent and revision using development evidence only.
2. Freeze the normal networking head, classical twin, and Qiskit VQC designs.
3. Freeze seed set, optimizer budget, LoRA policy, and final-epoch/checkpoint
   selection rule.
4. Run non-Tokyo smoke tests for all three heads.
5. Train all three on the same five-location training pool and validate only on
   the five-location development holdout.
6. Freeze all checkpoints and manifests.
7. Evaluate all three on the already frozen Tokyo pool without tuning.

Because Tokyo preprocessing has already been opened, no future choice may use
its phase counts or any model result. If the parent/head protocol is selected
after inspecting Tokyo model performance, label the quantum study exploratory
and do not claim pristine held-out selection.

## Final audit checklist

Before training:

- exact model ID and revision match `config.py`;
- development split hashes match across models;
- Tokyo files are not opened by the training command;
- output directory is new;
- LoRA targets and expected adapter-module count pass;
- the plan records model role, seed, head type, and checkpoint rule.

After training:

- `completed_epochs` equals the declared budget;
- `checkpoint_reload` and `tokyo_isolation` are `PASS`;
- metrics contain overall, macro-phase, and per-phase results;
- prediction distributions do not hide single-action collapse;
- manifest records git state, software, hardware, dataset, split, and model
  revision;
- failed or incomplete runs remain visible and are not silently excluded.

## Paper-writing boundary

Use the paper's model/method names in prose. Separate evidence into:

1. values published by the SLM-BBR paper;
2. results measured by this repository;
3. exploratory diagnostics;
4. final frozen experiments.

Never merge these categories into one statistical comparison. The paper's
Figure 9 supplies plots rather than exact Tokyo action accuracy/loss values;
missing numbers stay missing unless an explicitly labelled plot-digitization
procedure is used. Every table row must link to a manifest or a paper
table/figure/equation.
