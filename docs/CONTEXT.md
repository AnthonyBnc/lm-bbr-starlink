# Project Context

> Living decisions and implementation observations for `lm-bbr-starlink`. Last updated: 2026-09-04.

## Project identity

- Main paper: **Small Language Model-based Control for BBR over Low Earth Orbit Satellite Internet**
- Main task: language-model-based phase-safe BBR pacing control over Starlink traces
- Current phase: train and Tokyo-evaluate `gpt_quantum`, then compare it with the already-frozen `gpt_classical`
- Contribution: does attaching a Qiskit VQC action head to one SLM change its BBR pacing-gain prediction ability, compared with the same SLM's classical head?

## Research direction (simplified, current — 2026-09-03)

The user and tutor agreed to narrow scope to one question, one pair of models,
on one fixed backbone:

- `gpt_classical` — `LFM2.5-350M`, normal linear action head. **Done, frozen, Tokyo-evaluated.**
- `gpt_quantum` — the same `LFM2.5-350M`, Qiskit VQC action head. **Training in progress.**

`classical_twin` (parameter-matched classical bottleneck control) and the
four-way modern-backbone-vs-paper comparison are **out of scope** for this
comparison; they remain background/history in `archive/`.

`LFM2.5-350M` was chosen from four downloadable under-400M checkpoints using
development-only validation metrics (never Tokyo) — see
[ADR-0017](adr/ADR-0017-under400m-quantum-gpt-parent-and-protocol-freeze.md).

## Current snapshot

**`gpt_classical` (frozen):** `LFM2.5-350M`, LoRA rank 128, 16 epochs, seed
`100003`, trained on Ohio/Sao Paulo/London/Mumbai/Sydney, validated on the
frozen development holdout, Tokyo-evaluated (12,000 samples). Checkpoint at
`data/processed/lora_training/under400_rank128_epochs16_seed100003_v1/lfm2_5_350m_rank128_epochs16_seed100003_train/`.
Tokyo result at `data/processed/evaluation/tokyo_8model_comparison_v1/results/lfm2_5_350m/`
(accuracy 0.9575, macro-phase 0.8205, `BW_UP` 0.4615).

**`gpt_quantum` (in progress):** same backbone/data/split/seed/rank/epochs;
only the head differs. Circuit configuration history:

- First attempt: 8 qubits, depth 2, `data_reuploading_ry` ansatz — measured
  ~9.3s/window → ~4.3 day full run. Too slow given time budget; stopped.
- Benchmarked 8 qubits/depth 2 without re-uploading (`trainable_ry_layers`):
  ~9.4s/window — barely faster, not viable either.
- Benchmarked 4 qubits, depth 1, `trainable_ry_layers`: ~0.65-0.7s/window →
  ~8.5-9.5 hour full run. ~13-15x faster than 8q/d2, matching the exponential
  Hilbert-space-size argument (2^4 vs 2^8).
- **Real gate-200 check (200 train steps, real ~4,000-sample validation
  slice, not the tiny 2-window speed benchmarks) exposed a correctness
  problem with 4 qubits**: `BW_DOWN` collapsed to 0% (0/189, every DOWN
  sample predicted action 1 instead of the correct action 0) — a real
  regression, not noise. The same real gate-200 check with **8 qubits,
  depth 1** (same LN/T4, same everything else) recovered `BW_DOWN` to 100%
  (189/189) at ~4.9s/window (~2.1-2.2 day full run, ~7x slower than 4
  qubits but ~2x faster than depth 2). `BW_UP` stayed collapsed
  (majority-class parroting) in every configuration tried, including this
  one — a persistent, unresolved issue unrelated to qubit/depth choice.
- **Final choice (2026-09-04): 8 qubits, depth 1, `trainable_ry_layers`,
  LayerNorm+temperature-4.** Chosen for correctness (`BW_DOWN`) over the
  faster but factually-wrong 4-qubit option, accepting the ~2.1-2.2 day
  runtime. Per-epoch checkpoints (`epoch_checkpoints/epoch_0010/`, etc.)
  let the user inspect or stop at epoch 10 without losing the option to
  continue to epoch 16 later.
- All variants keep `--head-input-layernorm --bottleneck-temperature 4.0`
  (confirmed necessary to avoid `tanh` saturation; adds negligible cost).

Known risk carried over from earlier Qwen-parent diagnostics (see
`archive/docs/PHASE4_QISKIT_HEAD_ABLATION_DIAGNOSTIC.md`): quantum and
classical-twin heads with an 8-dimensional bottleneck have repeatedly
collapsed `BW_UP` predictions to a single majority action rather than
discriminating between the 5 `BW_UP` gains. This has not yet been resolved;
the current simplified scope reports it honestly as an observed limitation
rather than blocking training on it.

## Reporting scope

Final comparison is a 2-row table: `gpt_classical` vs `gpt_quantum`, same
metrics (accuracy, macro-phase accuracy, per-phase accuracy, latency), plus
Tokyo throughput/retransmission box plots. Chart generation is split across
three scripts in `reports/`: `under400m_chart_common.py` (shared config,
styling, data loading — not runnable on its own),
`generate_under400m_line_charts.py` (mean loss/accuracy), and
`generate_under400m_box_charts.py` (Tokyo throughput/retransmission box
plots). Extend `MODELS` in `under400m_chart_common.py` once `gpt_quantum`'s
Tokyo `predictions.csv` exists.

## Tokyo boundary (unchanged)

Tokyo train/validation/checkpoint-selection isolation still applies to
`gpt_quantum` exactly as it did to the four classical baselines: no Tokyo
data, labels, or metrics may influence circuit choice, hyperparameters, or
checkpoint selection. The circuit simplification above used only wall-clock
timing from a training-data benchmark, not Tokyo evidence.

## Historical context

Earlier, broader exploratory work (paper-reproduction fidelity, a four-model
modern-backbone-vs-paper comparison, and an extensive Qwen3.5-4B-Base
quantum-head ablation trail with LayerNorm/temperature/ansatz sweeps) is
preserved under `archive/` for provenance. It explains *why* certain design
choices (e.g. LayerNorm+temperature-4, the `BW_UP` collapse risk) were
carried into this simplified direction, but is not itself the current
research question.
