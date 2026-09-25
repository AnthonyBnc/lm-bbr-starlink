# ADR-0004: Phase 3 OLMo Compute Exclusion

## Status

Accepted for the exploratory Phase 3 run on 2026-08-10.

## Context

The Phase 3 suite started five modern LoRA baselines with the same development
split and training protocol. LFM2.5, Llama 3.2, Qwen3.5, and Gemma 3 completed
five epochs and produced audited manifests. OLMo 3 7B reached epoch 1, step 200
of 2400 after approximately 13.3 hours on Apple MPS. Its observed throughput
projected a total runtime of roughly 29-33 days.

## Decision

Exclude OLMo from this exploratory comparison because it is not feasible under
the available local compute budget. Preserve its incomplete training log and
record the exclusion in `suite.plan.json`. Do not compare its partial loss with
completed models and do not silently replace it with a shorter or otherwise
different training protocol.

The Phase 3 output is therefore an audited four-model exploratory development
comparison, not a completed five-model benchmark. OLMo may be revisited only as
a separately declared compute ablation or on suitable hardware.

## Consequences

- Four completed models remain directly comparable under the shared protocol.
- Claims must disclose the OLMo compute exclusion.
- The result remains exploratory, one-seed, and non-reportable.
- Tokyo remains isolated and cannot be used to compensate for the missing run.
- At least two modern baselines remain available, satisfying the architecture's
  minimum baseline coverage for later classical-versus-quantum work.
