# ADR-0014: Gemma 3 270M Compute Replacement

## Status

Accepted by user direction on 2026-09-03.

## Context

Granite 4.0 H 350M used the naive Mamba implementation on Apple MPS because
the optimized selective-state and causal-convolution kernels were unavailable.
Its incomplete run is excluded from model-quality comparisons.

The user selected the pretrained `google/gemma-3-270m` checkpoint as a
compute-feasible replacement. This is Gemma **3** 270M; names that omit the
generation number or append an invented `xs` suffix are incorrect.

## Decision

Register the checkpoint as `gemma_3_270m`, with local directory
`gemma-3-270m` and pinned revision
`9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`. Expose a separate
`gemma-3-270m-replacement` model set so the three completed under-400M runs do
not need to be repeated and the original `under-400m` plan remains immutable.

Gemma 3 270M must use the same development split, labels, State Encoder,
11-action phase mask, unweighted phase-masked cross-entropy, seed, 16-epoch
budget, and final-epoch checkpoint rule as the three completed models. Tokyo
remains excluded until the final frozen evaluation.

## Consequences

- This remains a modern-backbone extension, not an exact paper reproduction.
- The expected LoRA scope is Q/V projections across 18 text-transformer
  layers, for 36 adapter modules; construction and pipeline preflights must
  verify that count before training.
- The incomplete Granite 4.0 H 350M run remains a documented compute exclusion
  and must not enter model-quality rankings.
