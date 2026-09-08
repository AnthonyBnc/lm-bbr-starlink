# 100-Epoch Backbone Extension — Results Summary

> Generated 2026-09-08 from committed run/evaluation manifests on branch
> `train-4models-quantum-100epochs`. All numbers below are pulled directly
> from `run.manifest.json`, `metrics.json`, and the Tokyo
> `result.manifest.json` / `comparison.manifest.json` files — none are
> estimated.

## Protocol

Identical across all 4 runs: LoRA rank 128, alpha 32, dropout 0.05, AdamW,
learning rate 1e-4, gradient clip 0.25, batch size 1 with gradient
accumulation over 32 sequences, sequence length 20, sample step 20, seed
100003, classical (linear) head, `run_purpose:
slm_bbr_modern_backbone_extension`.

**Checkpoint rule:** `final completed epoch` — fixed before training started,
kept unchanged after seeing results (see "Peak vs. final epoch" below for why).

## Table A — Development-split validation, epoch 100 (final checkpoint)

| Model | Overall Acc. | Macro-Phase Acc. | BW_UP Acc. |
|---|---:|---:|---:|
| Gemma-3-270M | 95.73% | 79.44% | 38.31% |
| Granite-4.0-350M | 95.57% | 78.63% | 35.90% |
| LFM2.5-350M | 95.78% | 79.68% | 39.04% |
| Pleias-RAG-350M | 95.67% | 79.16% | 37.47% |

## Table B — Tokyo held-out evaluation, epoch 100 (final checkpoint) — REAL, FROZEN

| Model | Overall Acc. | Macro-Phase Acc. | BW_CRUISE | BW_DOWN | BW_UP |
|---|---:|---:|---:|---:|---:|
| Granite-4.0-350M | 96.07% | 83.39% | 100% | 100% | 50.16% |
| Gemma-3-270M | 96.02% | 83.17% | 100% | 100% | 49.52% |
| LFM2.5-350M | 95.93% | 82.82% | 100% | 100% | 48.47% |
| Pleias-RAG-350M | 95.88% | 82.61% | 100% | 100% | 47.84% |

All 4 evaluated on identical Tokyo sample IDs (`shared_sample_ids_sha256`
matches across all four in `comparison.manifest.json`). No BW_UP collapse
observed — all four land well above the ~20% majority-class floor for a
5-way phase.

**Note:** the development-split ranking (LFM2.5-350M led on all 3 metrics,
Table A / original Table I) does not hold on Tokyo — Granite and Gemma edge
it out slightly there. The gap is small (≤0.8pp across all four on overall
accuracy) but real, and worth stating as a limitation of development-only
backbone selection rather than smoothing over.

## Table C — Compute cost

| Model | Backbone params | LoRA trainable | Total trainable | Wall-clock (100 ep) | dtype |
|---|---:|---:|---:|---:|---|
| Gemma-3-270M | 273,996,416 | 5,898,240 | 7,588,619 | 17.42 h | bfloat16 |
| Granite-4.0-350M | 364,307,456 | 11,927,552 | 14,629,387 | 17.02 h | bfloat16 |
| LFM2.5-350M | 366,673,664 | 12,189,696 | 14,891,531 | 11.95 h | float16 |
| Pleias-RAG-350M | 365,351,936 | 11,927,552 | 14,629,387 | 15.82 h | bfloat16 |

Hardware: single NVIDIA RTX 4090 (24GB), Windows 11, CUDA 13.0, PyTorch
2.14.0+cu130, Transformers 5.16.1 (from `software_versions` /
`hardware` fields; GPU model confirmed via `nvidia-smi` during setup, not
itself logged in the manifest).

**VRAM usage was not recorded** during these runs — no profiling
instrumentation exists in the training script. The figures below are a
**calculated estimate**, not a live GPU measurement, built from the exact
parameter counts already in each run's manifest plus each model's real
published architecture:

| Model | Weights + Grad + AdamW (MB) | Activations (MB) | +CUDA context | **Estimated peak VRAM** |
|---|---:|---:|---:|---:|
| Gemma-3-270M | 638.4 | 98.8 | 500 | **1.21 GB** |
| Granite-4.0-350M | 918.1 | 307.9 | 500 | **1.69 GB** |
| LFM2.5-350M | 926.6 | 176.0 | 500 | **1.56 GB** |
| Pleias-RAG-350M | 920.1 | 285.9 | 500 | **1.67 GB** |

Method (reproducible):
- Frozen backbone: exact backbone-parameter count × 2 bytes (fp16/bf16).
- Trainable weights + gradients + AdamW (m, v): exact trainable-parameter
  count × 16 bytes (4 bytes × 4, assuming fp32 for LoRA/task modules and
  optimizer state — standard PEFT/AdamW practice for training stability).
- Activations: the Korthikanti et al. 2022 ("Reducing Activation
  Recomputation in Large Transformer Models") per-layer formula
  `s·b·h·(34 + 5ah/s)`, using each model's real hidden size, layer count,
  and attention-head count from its Hugging Face `config.json`
  (`hidden_size`, `num_hidden_layers`, `num_attention_heads`), with
  `s=220` structured embeddings per window and `b=1` micro-batch. Gemma's
  config was read from an unsloth mirror since Google's own repository is
  license-gated; same architecture, unrestricted copy.
- +500MB fixed PyTorch/CUDA context overhead (not model-specific).

**Caveat:** this is a calculated floor, not a measurement. Real `nvidia-smi`
usage typically runs 1.3–2x higher, since PyTorch's caching allocator
reserves memory in pools rather than the exact minimum, and cuDNN/cuBLAS
kernel workspaces aren't included here. All four sit far below the 24GB RTX
4090 either way — VRAM was not the bottleneck in these runs; the wall-clock
differences above come from architecture/kernel efficiency, not memory
pressure. If exact measured figures are needed later, this needs a small
new script (`torch.cuda.max_memory_allocated()` around one training step)
run on the training machine — say the word and I'll write one.

## Peak vs. final epoch (development split)

The checkpoint rule freezes the *final* epoch, not the best one. For
reference, here is how far epoch 100 sits from each model's actual peak
validation accuracy during training:

| Model | Best epoch | Best val. acc. | Epoch 100 val. acc. | Regression |
|---|---:|---:|---:|---:|
| Gemma-3-270M | 41 | 95.83% | 95.73% | −0.09pp |
| Granite-4.0-350M | 11 | 95.99% | 95.57% | −0.42pp |
| LFM2.5-350M | 15 | 96.15% | 95.78% | −0.37pp |
| Pleias-RAG-350M | 13 | 95.85% | 95.67% | −0.18pp |

Decision: kept the pre-registered final-epoch rule rather than switching to
best-epoch after observing this, since changing the rule *after* seeing
validation results would itself be a form of post-hoc tuning.

## Comparison to the original 16-epoch `gpt_classical` baseline

Same model (LFM2.5-350M), same everything except epoch count:

| Metric | 16 epochs (original) | 100 epochs (this run) | Change |
|---|---:|---:|---:|
| Tokyo overall accuracy | 95.75% | 95.93% | +0.18pp |
| Tokyo macro-phase accuracy | 82.05% | 82.82% | +0.77pp |
| Tokyo BW_UP accuracy | 46.15% | 48.47% | +2.32pp |

Modest but real improvement from the additional training, despite the
past-peak overfitting on the development split noted above.

## Data integrity

- `checkpoint_reload: PASS` and `tokyo_isolation: PASS` for all 4 runs.
- All 4 Tokyo evaluations used identical, verified sample IDs.
- Tokyo pool checksum verified against the already-opened reference manifest
  before evaluation (see `reuse_tokyo_gate.py`).
- Committed manifests: `run.manifest.json`, `metrics.json`,
  `progress.manifest.json` (training) and `result.manifest.json`,
  `predictions.csv`, `comparison.manifest.json` (Tokyo) for all 4 models, on
  branch `train-4models-quantum-100epochs`.
