# ADR-0003: Exploratory modern-model LoRA preparation

## Status

Implemented for compatibility testing on 2026-08-06. The final training rank,
optimization budget, and tuning policy remain pending supervisor approval.

## Context

The SLM-BBR paper uses LoRA but does not report a reproducible LoRA rank,
alpha, dropout, or target-module list. The five modern checkpoints also use
different token-mixing designs. Qwen3.5 and LFM2.5 contain hybrid layers, while
Gemma 3 includes a vision tower that is not part of the structured BBR path.
Applying one suffix list to every model would either miss active layers or tune
unused vision parameters.

## Decision

Use rank 8, alpha 32, and dropout 0.05 only for adapter-construction smoke
tests. This rank is not frozen for training. Target the following modules:

| Model | Exploratory LoRA scope | Adapter modules |
|---|---|---:|
| Qwen3.5-4B-Base | Q/V in 8 full-attention layers; fused QKV input and output in 24 linear-attention layers | 64 |
| Gemma 3 4B PT | Q/V in 34 text layers; exclude the vision tower | 68 |
| Llama 3.2 3B | Q/V in 28 attention layers | 56 |
| LFM2.5-2.6B | Q/V attention plus hybrid convolution input/output projections | 68 |
| OLMo 3 7B | Q/V in 32 attention layers | 64 |

Every target count is checked against the runtime `AutoModel` module tree.
Construction manifests pin the local model revision and saved adapter hashes.

## Consequences

All five models can construct and save PEFT adapters. This establishes wiring
compatibility only. It does not establish that rank 8 is fair or effective, and
it is not evidence of model quality. Full LoRA training must use a declared,
shared optimization budget, preserve the same data and metrics, and record any
family-specific target differences. Tokyo remains unavailable for these
decisions.
