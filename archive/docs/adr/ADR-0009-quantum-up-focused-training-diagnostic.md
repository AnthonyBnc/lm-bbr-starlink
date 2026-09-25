# ADR-0009: Quantum UP-Focused Training Diagnostic

## Status

Exploratory, pending supervisor approval for any headline use.

## Context

The q8/depth-2 Qiskit quantum head learns DOWN and CRUISE but collapses UP
predictions to action 8. Phase-balanced loss, depth 3, and an RY+RZ ansatz did
not improve UP. The full training-window distribution contains only about 7%
UP positions and about 88% CRUISE positions.

## Decision

Run one training-only diagnostic for the q8/depth-2 RY Quantum-GPT variant:

- sample exactly 1,500 sequence windows with replacement;
- use deterministic seed 100003;
- use windows whose UP positions belong to one UP action, balance quotas across
  actions 6-10, weight by UP-density power 3, and penalise DOWN positions;
- keep sequence length and sample step at 20;
- keep cross-entropy, labels, rewards, actions, masks, model parent, Qiskit
  backend, and quantum architecture unchanged;
- evaluate on all 600 original validation windows without resampling;
- do not access Tokyo.

The runner records source and selected phase/action distributions, repeated and
unique window counts, and the selected-index hash in the run manifest.
The selected indices are also materialised in a deterministic index-only data
design. Training validates the source train pool, validation pool, and split
manifest checksums before using that design. No source pool is rewritten.

## Comparison Limitation

The historical classical and classical-twin q8/depth-2 runs used all 2,400
original training windows. They may be displayed as validation references
because the validation records are identical, but the new result is not a fair
headline training-protocol comparison. A fair claim would require retraining
all three heads under the same sampling protocol.

## Test Set

The requested 700-window test is not created in this ADR. The current frozen
development split provides 2,400 train and 600 validation windows. Creating an
additional independent test would change split membership; using Tokyo now
would leak the final held-out location into model selection. That evaluation
waits for supervisor approval or the final Tokyo gate.
