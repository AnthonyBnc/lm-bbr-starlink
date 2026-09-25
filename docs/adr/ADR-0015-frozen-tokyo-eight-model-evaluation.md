# ADR-0015: Frozen Tokyo eight-model evaluation

## Status

Approved by the user on 2026-09-03; the four new-model checkpoints are frozen
and the four old models use paper-published results as contextual references.

## Context

The four completed 16-epoch modern-backbone runs are Granite 4.0 350M,
Pleias-RAG 350M, LFM2.5 350M, and Gemma 3 270M. The final objective is to
compare them with GPT-2, T5, GPT-Neo, and SmolLM2 from the paper. The user chose
not to retrain the four paper models, so their published values cannot be
treated as same-pipeline or same-sample measurements.

The released paper/code does not unambiguously specify the numeric location
flag for the unseen Tokyo location. Development currently occupies flags 0-4.

## Decision

- Select the final completed epoch 16 for all four modern models before Tokyo.
- Record SHA-256 checksums for the adapter, task modules, training state, and
  run manifest. A checksum mismatch blocks evaluation.
- Keep Tokyo behind an explicit acknowledgement gate and a dedicated
  `held_out_test` preprocessing entry point.
- Select only the primary `bbr_` flow from all ten runs in each of the four
  stream groups, matching development trace discovery (40 traces total).
- Use sequence length 20, sample step 20, the same return scaling, phase mask,
  labels, and metric implementation used for development evaluation.
- Apply the phase mask before both cross-entropy and argmax.
- Run with `model.eval()`, `torch.no_grad()`, and every parameter frozen.
- Record per-sample predictions, overall/loss/macro/per-phase metrics, paper
  Equations 21-27 surrogate outcomes, latency, and all relevant hashes.
- Require identical Tokyo sample-ID hashes among the four newly evaluated
  models.
- Import old-model values with source/figure/metric provenance and mark exact
  numeric values unavailable when the paper only provides a plot.
- Present paper-published values in a separate contextual-reference section;
  do not calculate same-sample significance or claim a controlled head-to-head.

## Frozen Tokyo location encoding

The user approved Tokyo location flag `5` on 2026-09-03. It is an explicit
unseen value because it does not alias development flags 0-4. The paper does
not uniquely specify this encoding, so it remains a documented reproduction
assumption. It must not be changed in response to results.

The Tokyo gate may open after the four new-model checkpoints, preprocessing,
metrics, and location flag are frozen. The old models require no checkpoint
because the user selected published-reference comparison.

## Consequences

Opening Tokyo is a one-time final evaluation action for this frozen suite.
Tokyo outcomes must not influence later training, checkpoint selection,
hyperparameter tuning, model replacement, or protocol changes. A separately
pre-registered follow-on study may proceed using development evidence only; it
must not use Tokyo to choose what to train. A genuine evaluator defect
invalidates the whole affected suite and requires a documented all-model
rerun.
