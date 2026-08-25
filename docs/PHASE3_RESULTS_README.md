# Phase 3 Modern LoRA Results

## Purpose

Phase 3 is an exploratory comparison of modern downloadable language models on
the paper-derived SLM-BBR task. Each model predicts one of 11 phase-constrained
BBR pacing-gain actions from the same Starlink development data.

These results prepare the final classical benchmark and identify a GPT-style
candidate for the later classical-versus-quantum comparison. They are not final
paper results.

## Shared Protocol

All completed models used:

| Setting | Value |
|---|---:|
| Development split | Trace-stratified 80/20 |
| Split seed | `100003` |
| Training traces | 160 |
| Validation traces | 40 |
| LoRA rank | 8 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Epochs | 5 |
| Sequence length | 20 |
| Sample step | 20 |
| Gradient accumulation | 32 |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |

The audit confirmed identical dataset, split, sample-ID hashes, labels, action
masks, optimization budget, and evaluation path. Checkpoint reload and Tokyo
isolation passed for every completed model.

Tokyo was not used for training, validation, checkpoint selection, model
ranking, or hyperparameter decisions.

## Completion Status

| Model | Status | Recorded wall time |
|---|---|---:|
| Qwen3.5-4B-Base | Completed | 25.45 hours* |
| LFM2.5-2.6B | Completed | 6.06 hours |
| Gemma 3-4B PT | Completed | 9.60 hours |
| Llama 3.2-3B | Completed | 7.62 hours |
| OLMo 3-7B | Excluded by compute limit | 13.3 hours partial |

\*Qwen's recorded wall time includes a period when its process was suspended,
so it is not pure active compute time.

## Development Results

| Model | Accuracy | UP accuracy | Macro-phase accuracy | Validation loss |
|---|---:|---:|---:|---:|
| Qwen3.5-4B-Base | **95.74%** | **38.43%** | **79.48%** | **0.09531** |
| LFM2.5-2.6B | 95.63% | 36.87% | 78.96% | 0.10110 |
| Gemma 3-4B PT | 95.44% | 34.10% | 78.03% | 0.10653 |
| Llama 3.2-3B | 95.20% | 30.60% | 76.87% | 0.11112 |

Qwen leads this single-seed development comparison on accuracy, UP accuracy,
macro-phase accuracy, and validation loss. This is evidence that Qwen is the
current leading candidate, not a final superiority claim.

## How To Interpret The Results

Overall accuracy is not sufficient for model selection because the validation
data is strongly dominated by the CRUISE phase. All four models achieved 100%
on DOWN and CRUISE, while their main differences appeared in the UP phase.

The most useful development metrics are:

1. Macro-phase accuracy, which gives each phase equal weight.
2. UP accuracy, which exposes performance on the difficult phase.
3. Validation loss, which measures prediction confidence.
4. Prediction distribution, which reveals collapse to a small action subset.

The result remains exploratory because it uses one split, one seed, and a
training policy that has not yet been approved as final.

## OLMo Exclusion

OLMo reached epoch 1, step 200 of 2400 after approximately 13.3 hours on Apple
MPS. Its observed speed projected approximately 29-33 days for five epochs.

The run was excluded because it exceeded the available local compute budget.
Its partial loss is not included in the comparison, and its protocol was not
shortened because that would create an unfair model-specific experiment.

The decision is recorded in
[`ADR-0004-phase3-olmo-compute-exclusion.md`](adr/ADR-0004-phase3-olmo-compute-exclusion.md).
The Phase 3 result must be described as a four-model exploratory comparison,
not a completed five-model benchmark.

## Artifacts

- Audited comparison: `data/processed/lora_training/phase3_rank8_epochs5_seed100003/comparison.summary.json`
- Suite record: `data/processed/lora_training/phase3_rank8_epochs5_seed100003/suite.plan.json`
- Per-model metrics: `<model-run-directory>/metrics.json`
- Per-model provenance: `<model-run-directory>/run.manifest.json`
- Per-model output: `<model-run-directory>/training.log`

Print the audited summary:

```bash
.venv/bin/python -m json.tool \
  data/processed/lora_training/phase3_rank8_epochs5_seed100003/comparison.summary.json
```

Regenerate and re-audit the summary:

```bash
.venv/bin/python summarize_modern_lora.py \
  --run-root data/processed/lora_training/phase3_rank8_epochs5_seed100003
```

## Phase 4 Recommendation

Before final training or Quantum-GPT implementation, freeze the training and
selection policy with tutor approval. The current proposal is:

Phase 4 has been started as proposed development tooling in
[`ADR-0005`](adr/ADR-0005-phase4-development-multiseed-policy.md).

| Decision | Proposed value |
|---|---|
| LoRA rank | 8 |
| Epochs | 5 |
| Primary ranking metric | Macro-phase accuracy |
| Tie-break metric | Validation loss |
| Final evaluation | Multiple fixed seeds |
| Leading GPT-style candidate | Qwen3.5-4B-Base |

Qwen remains a candidate until the final policy and seed set are approved. The
strongest overall baseline and Quantum-GPT parent are separate roles, although
Qwen is currently eligible for both.

After approval:

1. Record the final training policy in an ADR and configuration.
2. Run the final multi-seed classical benchmark.
3. Freeze the GPT-style quantum parent using development results only.
4. Build the normal classical head, parameter-matched classical twin, and VQC
   head on the same parent checkpoint.
5. Run shared smoke tests, pilots, and final multi-seed training.
6. Evaluate Tokyo once, after every development decision is frozen.

## Research Limitations

- The development split is still pending supervisor approval.
- Only one seed has been evaluated.
- CRUISE represents 87.5% of the full development pool.
- DOWN labels collapse to pacing gain 0.90 under the approved Equation 3 path.
- OLMo did not complete the shared protocol.
- These runs are marked `reportable_result: false`.
- No Tokyo result has been produced.
