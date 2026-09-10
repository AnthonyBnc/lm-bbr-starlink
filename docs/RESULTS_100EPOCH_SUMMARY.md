# 100-Epoch Backbone Extension — Results Summary

> Generated 2026-09-08, extended 2026-09-10 with the quantum-GPT run. All
> numbers below are pulled directly from `run.manifest.json`,
> `metrics.json`, and the Tokyo `result.manifest.json` /
> `comparison.manifest.json` files — none are estimated, except the VRAM
> table in "Table C", which is explicitly labelled as a calculation.

## Protocol

Identical across all 5 runs: LoRA rank 128, alpha 32, dropout 0.05, AdamW,
learning rate 1e-4, gradient clip 0.25, batch size 1 with gradient
accumulation over 32 sequences, sequence length 20, sample step 20, seed
100003, `run_purpose: slm_bbr_modern_backbone_extension`.

Four runs use a classical (linear) head. The fifth, **gpt_quantum**, keeps
the LFM2.5-350M backbone and replaces only the head with a variational
quantum circuit: 8 qubits, depth 1, `trainable_ry` ansatz, CNOT-ring
entanglement, per-qubit Pauli-Z measurement, bounded RY angle encoding with
LayerNorm and temperature 4.0 — 8 trainable circuit parameters. Because
backbone and every training hyperparameter are shared with the classical
LFM2.5-350M run, the two isolate the head choice.

**Checkpoint rule:** `final completed epoch` — fixed before training started,
kept unchanged after seeing results (see "Peak vs. final epoch" below for why).

## Table A — Development-split validation, epoch 100 (final checkpoint)

| Model | Overall Acc. | Macro-Phase Acc. | BW_UP Acc. |
|---|---:|---:|---:|
| Gemma-3-270M | 95.73% | 79.44% | 38.31% |
| Granite-4.0-350M | 95.57% | 78.63% | 35.90% |
| LFM2.5-350M | 95.78% | 79.68% | 39.04% |
| Pleias-RAG-350M | 95.67% | 79.16% | 37.47% |
| gpt_quantum (LFM2.5-350M + VQC) | 94.52% | 73.57% | 20.72% |

## Table B — Tokyo held-out evaluation, epoch 100 (final checkpoint) — REAL, FROZEN

| Model | Overall Acc. | Macro-Phase Acc. | BW_CRUISE | BW_DOWN | BW_UP | Distinct labels predicted |
|---|---:|---:|---:|---:|---:|---:|
| Granite-4.0-350M | 96.07% | 83.39% | 100% | 100% | 50.16% | 7 |
| Gemma-3-270M | 96.02% | 83.17% | 100% | 100% | 49.52% | 7 |
| LFM2.5-350M | 95.93% | 82.82% | 100% | 100% | 48.47% | 7 |
| Pleias-RAG-350M | 95.88% | 82.61% | 100% | 100% | 47.84% | 7 |
| gpt_quantum (LFM2.5-350M + VQC) | 95.50% | 80.99% | 100% | 100% | 42.98% | **3** |

All 5 evaluated on the identical frozen Tokyo pool
(`tokyo_pool_sha256: 800f1a444fcf1610720b83f6feadf0743fb2acaca81ed8d251f264fad012f27b`),
verified equal across every `result.manifest.json`.

Among the 4 classical backbones there is no BW_UP collapse — all land well
above the ~20% floor for a 5-way phase. **gpt_quantum is a different case
and its 42.98% must not be read as partial success; see "The quantum head
does not learn" below.**

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
| gpt_quantum (LFM2.5-350M + VQC) | 366,673,664 | 12,189,696 | 14,890,611 | 8.41 h (epochs 12–100) | float16 |

**gpt_quantum wall-clock is not comparable to the four above.** Its 8.41 h
covers epochs 12–100 only, and those epochs ran on a different simulation
backend (see below), so the figure measures the torch statevector path
rather than the run as a whole. Epochs 1–11 ran earlier on the Qiskit
parameter-shift path at roughly 2 h/epoch; had the whole run continued that
way, 100 epochs would have taken on the order of 8 days. LoRA trainable
counts are identical to the classical LFM2.5-350M run (12,189,696); the two
differ by 920 in total trainable parameters (14,890,611 vs 14,891,531),
because the VQC head — LayerNorm, `Linear(2048, 8)`, 8 circuit parameters,
`Linear(8, 11)` — is marginally smaller than the classical `Linear(2048,
11)` it replaces. The comparison is therefore not confounded by capacity.

**Backend switch, epochs 11 → 12.** Epochs 1–11 used Qiskit's
`StatevectorEstimator` with parameter-shift gradients; 12–100 used an exact
torch statevector implementation of the same circuit differentiated by
autograd. Both are exact, shot-free simulations of an identical circuit and
agree to 1e-5 in unit tests (all three ansatzes, several depths, and the
single-qubit no-entanglement case). The training curve is continuous across
the boundary: the train-loss step at the switch is 0.0022, against a median
step of 0.0011 and a maximum of 0.0041 over epochs 5–30. The Tokyo
evaluation independently rebuilt the head on the **Qiskit** path and loaded
the torch-trained weights, so the reported held-out numbers are a
cross-backend check.

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
| gpt_quantum | 1 | 94.52% | 94.52% | 0.00pp |

Decision: kept the pre-registered final-epoch rule rather than switching to
best-epoch after observing this, since changing the rule *after* seeing
validation results would itself be a form of post-hoc tuning.

For gpt_quantum the choice is moot: its validation accuracy never changes,
so "best epoch" and "final epoch" select the same value. That is itself the
finding, covered next.

## The quantum head does not learn

Across all 100 epochs the gpt_quantum run produces **exactly one distinct
validation accuracy**, `0.945167`, and **one distinct BW_UP accuracy**,
`0.207229` — identical to six decimal places from epoch 1 to epoch 100.
Training loss does fall (0.183 → 0.113), but it does so by sharpening
confidence on predictions that never change.

The prediction distribution shows the mechanism. On Tokyo:

```
predicted : {0: 634, 10: 947, 5: 10419}          3 distinct labels
actual    : {0: 634, 5: 10419, 10: 407, 6: 136, 7: 140, 8: 96, 9: 168}
```

The head has settled on a constant label per phase:

| Phase | Always predicts | Result |
|---|---|---|
| BW_CRUISE | label 5 | 100% — it is the only label in that phase |
| BW_DOWN | label 0 | 100% — likewise |
| BW_UP | label 10 | correct only when the true label happens to be 10 |

**This is why the 42.98% BW_UP figure is misleading.** Because the head
always emits label 10, its BW_UP score is simply the share of label 10 in
whichever pool it is scored against:

| Pool | Label-10 share | gpt_quantum BW_UP |
|---|---:|---:|
| Tokyo | 407/947 = 42.98% | **42.98%** |
| Development split | 172/830 = 20.72% | **20.72%** |

Both match to the decimal. The same frozen model scores 43% on one pool and
21% on another purely because the pools are skewed differently — the
signature of a constant predictor, not of a model that has learned
anything. On the development split it lands *below* even that split's
majority-label baseline (27.59%, label 6), because label 10 is not the most
common BW_UP label there.

For contrast, classical LFM2.5-350M on the identical Tokyo samples scores
48.47%, beating the majority-label baseline by **+5.49pp**, and spreads its
predictions over all 7 labels. Its development-split accuracy takes 46
distinct values over 100 epochs.

Candidate explanations, not yet tested:

1. Circuit weights initialise to zeros, and with `trainable_ry` an RY(0) is
   the identity — a textbook barren-plateau starting point.
2. `temperature=4.0` squashes the projected angles through `tanh`, so the
   Pauli-Z expectations may vary little between samples; a near-constant
   8-value bottleneck can only produce a near-constant `Linear(8, 11)`.
3. 8 trainable circuit parameters at depth 1 is very low expressivity.

`QuantumActionHead` already records `bottleneck_std` and
`bottleneck_saturation_ratio` under `--trainability-diagnostics`, which
were not enabled for this run. A short diagnostic run is the cheapest next
step — at ~6 min/epoch it costs minutes. Related prior work: ADR-0009
(up-focused training diagnostic), ADR-0011 (bottleneck saturation
ablation).

**This is reported as a negative result on VQC head trainability, not as a
competitive model.** Quoting gpt_quantum's 95.50% overall Tokyo accuracy
without this context would be misleading, since that number comes almost
entirely from BW_CRUISE and BW_DOWN being separable by a constant guess.

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

- `checkpoint_reload: PASS` and `tokyo_isolation: PASS` for all 5 runs,
  gpt_quantum included.
- All 5 Tokyo evaluations ran against the same frozen pool, checksum
  `800f1a444fcf1610720b83f6feadf0743fb2acaca81ed8d251f264fad012f27b`,
  verified equal in every `result.manifest.json`.
- Tokyo pool checksum verified against the already-opened reference manifest
  before evaluation (see `reuse_tokyo_gate.py`).
- gpt_quantum carries `checkpoint_rule: final completed epoch for
  slm_bbr_modern_backbone_extension`, matching the four classical runs, so
  it passes the same `utils/checkpoint_freeze.py` gate.
- Committed manifests: `run.manifest.json`, `metrics.json`,
  `progress.manifest.json` (training) and `result.manifest.json`,
  `predictions.csv` (Tokyo) for all 5 models. Model weights are excluded by
  the repository convention; only manifests, metrics and per-epoch metadata
  are tracked.

**Two provenance caveats worth carrying into the paper.**

1. gpt_quantum's `run.manifest.json` records `resumed_from_epoch: 11` and
   `backend: torch` / `gradient_method: autograd`. Epochs 1–11 were produced
   by the Qiskit parameter-shift path and are not re-derivable from this
   manifest alone; they live in
   `data/processed/lora_training/under400m_quantum_gpt_follow_on_v1/lfm2_5_350m_quantum_100ep`.
2. The `head_config` block inside the Tokyo `result.manifest.json` is copied
   from the *training* manifest, so it reads `backend: torch` even though
   the evaluation itself rebuilt the head on the Qiskit path (the evaluator
   does not pass a backend and so takes the default). The numbers are
   unaffected — that is precisely the cross-backend check — but the field
   describes training, not inference.

## What is not yet regenerated

Two derived artifacts still describe four models and should be rebuilt
before the paper's figures are final:

- `reports/assets/nine_model_comparison.json` — its `gpt_quantum` row still
  holds figures from the earlier 16-epoch run (latency 5.79 ms/action; the
  100-epoch Tokyo run measures 5.74 ms/action).
- `data/processed/evaluation/four_model_100epoch_v1/results/comparison.manifest.json`
  — covers only the four classical models; regenerating it with
  `compare_tokyo_results.py` would make it five.
