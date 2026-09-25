# ADR-0018: Simplified research scope and quantum-circuit revision

## Status

Approved by the user (relaying tutor feedback) on 2026-09-03/04.

## Context

The tutor reviewed the direction set by ADR-0016/ADR-0017 — comparing a
backbone-substituted, under-400M modern extension against the original
paper's four SLMs, plus a three-role `gpt_classical` /
`gpt_classical_twin` / `gpt_quantum` ablation on that backbone — and found it
"khá mơ hồ" (fairly vague/unfocused): two research variables (backbone
substitution, and quantum-head addition) were being told as one story,
inflating scope and making the actual claim hard to state and defend. The
tutor's guidance was to keep the study small and do one simple thing well.

Separately, the `gpt_quantum` full 16-epoch training run frozen by ADR-0017
(8 qubits, depth 2) was measured in production to take ~4.3-4.5 days —
impractical given the user's remaining time. A wall-clock benchmark on
training-pool data (never Tokyo) found 4 qubits/depth 1 to be ~13-15x faster
(~0.65-0.7s/window vs ~9.3-9.4s/window), consistent with the exponential
Hilbert-space-size argument ($2^4$ vs $2^8$).

## Decision

### Scope

The primary research question is narrowed to exactly one comparison:

> Does attaching a Qiskit VQC action head to one SLM change its BBR
> pacing-gain prediction ability, compared with the same SLM's classical
> head?

Only two roles are required for this claim:

- `gpt_classical` — `LFM2.5-350M`, classical head. Already frozen and
  Tokyo-evaluated (accuracy 0.9575, macro-phase 0.8205, `BW_UP` 0.4615).
- `gpt_quantum` — same `LFM2.5-350M`, Qiskit VQC head.

`gpt_classical_twin` and the four-model modern-backbone-vs-paper comparison
are **dropped from the primary claim**. They are not deleted — the completed
work (four frozen under-400M Tokyo evaluations, the Qwen3.5-4B-Base
quantum-head ablation trail that motivated the LayerNorm+temperature-4 fix)
is preserved under `archive/` as provenance for why `LFM2.5-350M` and the
LN/T4 fix were chosen. It is background, not part of the current report.

Because there is no `classical_twin` control in this simplified comparison,
any observed difference between `gpt_classical` and `gpt_quantum` must be
reported as-is, without claiming it isolates "quantum-ness" specifically
versus the effect of the 8-dimensional bottleneck. This limitation must be
stated explicitly in any write-up.

### Quantum circuit revision

`gpt_quantum`'s circuit is revised from ADR-0017's `8 qubits, depth 2,
data_reuploading_ry` to:

- qubits: `8`
- depth: `1`
- ansatz: `trainable_ry_layers`
- angle scale: `pi`
- input layernorm: `True`, temperature: `4.0` (unchanged — confirmed
  necessary to avoid `tanh` saturation, and adds negligible wall-clock cost)

A first speed-only benchmark (`--max-train-steps 120`, 2 validation windows)
picked 4 qubits/depth 1 (~13-15x faster than 8 qubits/depth 2). A follow-up
**correctness** check — a real gate-200 diagnostic with a ~4,000-sample
validation slice, not the earlier 2-window speed probe — found 4 qubits
collapses `BW_DOWN` to 0% (0/189), while 8 qubits/depth 1 (same LN/T4)
recovers `BW_DOWN` to 100% (189/189) at roughly 2x the training time of
depth 2 and ~7x that of 4 qubits/depth 1 (~2.1-2.2 days for 16 epochs vs
~8.5-9.5 hours for 4 qubits/depth 1). The user chose 8 qubits/depth 1,
prioritizing correctness over the faster but factually-wrong 4-qubit option.
`BW_UP` remained collapsed (majority-class parroting) in every configuration
tested, including this one; this is treated as a persistent, unresolved
limitation to report honestly, not a blocker.

Every benchmark and gate-200 check used only training-pool/development-split
windows; no Tokyo data, label, or metric was read to make this choice,
preserving Tokyo isolation. Per-epoch checkpoints (saved every epoch during
the full run) let the user inspect intermediate epochs (e.g. epoch 10)
without committing in advance to a shorter epoch budget than `gpt_classical`.

### Documentation reorganization

`docs/`, top-level scripts, and `reports/figures/` were reorganized to match
this scope:

- Moved to `archive/docs/adr/`: ADR-0001–0012, ADR-0016 (paper-reproduction
  fidelity, Qwen quantum-head ablation trail, old three-role follow-on plan).
- Moved to `archive/docs/`: `PAPER_DRAFT.md`, `LARGE_MODEL_TRAINING.md`,
  `RESUME_AND_CLOUD_TRAINING.md`, `NEXT_STEPS_README.md`,
  `PHASE3_RESULTS_README.md`, `PHASE4_QISKIT_HEAD_ABLATION_DIAGNOSTIC.md`.
- Moved to `archive/scripts/`: Qwen-ablation and phase3/phase4 large-backbone
  runners/diagnostics (`run_qwen_*.py`, `diagnose_head_ablation.py`,
  `run_saturation_ablation.py`, `run_trainability_gate200.py`,
  `run_quantum_up_focus.py`, `prepare_quantum_up_focus_data.py`,
  `run_quantum_smoke.py`, `analyze_up_separability.py`,
  `run_slm_bbr_modern_backbones.py`, `run_modern_lora_suite.py`,
  `run_phase4_multiseed.py`, `summarize_phase4_multiseed.py`,
  `summarize_head_ablation.py`).
- Moved to `archive/reports/figures/`: Qwen-ablation, phase3-large-backbone,
  and paper-comparison figures.
- Kept in place (still load-bearing for the current scope): ADR-0013/0014/0015
  (why the four under-400M candidates were evaluated and how Tokyo isolation
  was verified for them — evidence behind picking `LFM2.5-350M`), ADR-0017
  (parent selection), this ADR, `README.md`, `docs/ARCHITECTURE.md`,
  `docs/CONTEXT.md`, `docs/SKILL.md`, `AGENTS.md`,
  `reports/figures/under400m/`, `reports/under400m_chart_common.py`,
  `reports/generate_under400m_line_charts.py`,
  `reports/generate_under400m_box_charts.py` (split from one combined script
  on 2026-09-04 at the user's request, for easier independent editing).
- `README.md`, `docs/ARCHITECTURE.md`, `docs/CONTEXT.md`, `docs/SKILL.md`,
  `AGENTS.md` were rewritten to describe the simplified scope and to stop
  pointing at the moved files as current instructions.

Nothing was deleted; everything moved remains on disk under `archive/` and
git-recoverable where tracked.

## Consequences

- The next required action is training `gpt_quantum` (4 qubits/depth 1) for
  16 epochs (~8.5-9.5 hours, estimated from benchmark, not yet confirmed by a
  full run), then a one-time Tokyo evaluation, then a 2-row comparison table
  against the already-frozen `gpt_classical`.
- Any future report or manuscript for this simplified direction should be
  written fresh, informed by but not built on top of `archive/docs/PAPER_DRAFT.md`.
- If a future direction wants to reintroduce `classical_twin` or the
  modern-backbone-vs-paper comparison, `archive/` has the completed prior
  work and does not need to be redone.
