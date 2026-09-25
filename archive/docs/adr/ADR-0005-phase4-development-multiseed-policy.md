# ADR-0005: Phase 4 Development Multi-Seed Policy

## Status

Proposed for Phase 4 on 2026-08-11.

## Context

Phase 3 produced an audited four-model exploratory comparison on the shared
non-Tokyo development split. Qwen3.5-4B-Base led the completed runs on
validation accuracy, UP accuracy, macro-phase accuracy, and validation loss, but
the result used one training seed only.

The architecture contract requires mean, standard deviation, and per-seed
results before moving toward the final classical-vs-quantum comparison. Tokyo
must remain held out and cannot be used for ranking, checkpoint selection, or
hyperparameter changes.

## Decision

Start Phase 4 with a development multi-seed confirmation run for
`qwen3_5_4b_base`.

Initial Phase 4 protocol:

- model: `Qwen/Qwen3.5-4B-Base`;
- role: GPT-style modern classical candidate;
- LoRA rank: `8`;
- LoRA alpha: `32`;
- LoRA dropout: `0.05`;
- epochs: `5`;
- sequence length: `20`;
- sample step: `20`;
- state feature dim: `256`;
- optimizer: `AdamW`;
- learning rate: `1e-4`;
- weight decay: `1e-4`;
- gradient accumulation: `32`;
- seeds: `100003`, `100019`, `100043`;
- primary development metric: validation macro-phase accuracy;
- tie-breaker: validation loss;
- held-out location: Tokyo.

These runs remain development runs and keep `reportable_result: false` until the
supervisor approves the final policy and final evaluation plan.

## Consequences

- Phase 4 tests whether the Phase 3 Qwen result is stable across seeds.
- Mixed-seed aggregation is allowed only for the same model, same split, same
  sample IDs, same labels, same masks, and same training protocol.
- The Phase 3 cross-model audit still rejects mixed seeds.
- Tokyo remains untouched.
- Quantum-GPT work must wait until the GPT parent and quantum/classical control
  protocol are frozen.

## Commands

Dry-run and write the Phase 4 plan:

```bash
.venv/bin/python run_phase4_multiseed.py
```

Execute the Phase 4 Qwen multi-seed run:

```bash
.venv/bin/python -u run_phase4_multiseed.py --execute
```

Resume without repeating completed seeds:

```bash
.venv/bin/python -u run_phase4_multiseed.py --execute --resume-completed
```

Regenerate the audited multi-seed summary:

```bash
.venv/bin/python summarize_phase4_multiseed.py \
  --run-root data/processed/lora_training/phase4_qwen_multiseed_rank8_epochs5
```
