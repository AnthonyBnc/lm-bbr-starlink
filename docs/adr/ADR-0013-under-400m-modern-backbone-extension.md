# ADR-0013: Under-400M Modern-Backbone Extension

## Status

Accepted by user direction on 2026-09-02.

## Context

The user selected four downloadable checkpoints near or below 400 million
parameters for a compute-bounded extension of the SLM-BBR comparison. The
checkpoints are already stored locally and expose Hugging Face model metadata.
They are not the GPT-2, T5, GPT-Neo, and SmolLM2 set evaluated by the paper.

The requested 16-epoch budget is also shorter than the paper's reported
150-epoch budget. Consequently, this experiment may preserve the paper-derived
BBR task and training path, but it cannot be described as an exact paper
reproduction or compared as though the optimization budgets were identical.

## Decision

Add a separately named `under-400m` model set containing these pinned revisions:

| Model key | Hugging Face ID | Revision |
|---|---|---|
| `granite_4_0_350m` | `ibm-granite/granite-4.0-350m` | `bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2` |
| `pleias_rag_350m` | `PleIAs/Pleias-RAG-350M` | `db001b29a33532583d14979c3c38ef04b6d59352` |
| `lfm2_5_350m` | `LiquidAI/LFM2.5-350M` | `9e6c6ccf47cd318696e137d381a7ded8fe4df09f` |
| `granite_4_0_h_350m` | `ibm-granite/granite-4.0-h-350m` | `3b17b717b8f2f5d305b0a92c1491e239aeda19c8` |

The existing approved-modern model set remains the runner default. Selecting
`under-400m` changes only the declared backbone set. Every model must use the
same development split, State Encoder, return/action sequences, 11-action
phase mask, training sampling, loss, metrics, seed, and optimization budget.

Run a one-train-step/one-validation-step preflight for all four checkpoints
before the 16-epoch suite. The preflight and training validation data come only
from Ohio, Sao Paulo, London, Mumbai, and Sydney. Tokyo remains excluded from
training, validation, checkpoint selection, thresholds, and tuning. A future
Tokyo evaluation must be evaluation-only, use frozen checkpoints and protocol,
and be run once after development decisions are complete.

The user owns execution of both preflight and training commands.

## Consequences

- Results are an exploratory 16-epoch under-400M modern-backbone extension.
- Results must not be labelled as exact reproduction of the paper.
- A failed model preflight must be surfaced and fixed; the runner must not
  silently substitute another model or skip it in the comparison.
- Forward/backward and checkpoint-reload evidence is required before full
  training, even though local model loading and LoRA construction already pass.
