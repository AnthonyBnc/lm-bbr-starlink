# ADR-0010: CRUISE-80 Balanced-UP Fair Head Ablation

## Status

Completed as a one-seed exploratory development protocol on 2026-08-25.

## Context

The original 2,400-window training distribution is dominated by CRUISE and the
q8/depth-2 quantum head collapses its UP predictions. A proposed 1,500-window
quantum-only diagnostic would reduce the training budget and would not be
comparable with previously trained controls. The tutor advised retaining the
window budget and rerunning every compared model when sampling changes.

## Decision

Use one deterministic 2,400-window index design for Qwen classical,
classical-twin, and q8/depth-2 RY quantum heads:

- retain 800 high-UP-context base windows;
- draw 1,600 UP-focused replacement windows;
- keep exactly 480 windows from each of the five development locations;
- balance final UP action positions across actions 6-10;
- keep sequence length and sample step at 20;
- keep the original 600-window validation set unmodified;
- do not access Tokyo.

The realised training distribution is 79.365% CRUISE, 14.246% UP, and 6.390%
DOWN. UP action counts are 1,364, 1,367, 1,382, 1,361, and 1,364. The design
contains 835 unique and 1,565 repeated windows.

## Consequences

All three heads must be retrained from scratch with the same sampling design.
Historical results remain useful references but are not directly comparable as
headline rows. The strong replacement rate may increase overfitting risk, so
validation remains naturally distributed and prediction diversity must be
reported alongside accuracy.

## Outcome

All three heads passed checkpoint reload and Tokyo-isolation checks. Validation
UP accuracy was 25.783% for classical and 16.988% for both classical-twin and
q8/depth-2 RY quantum. Classical-twin and quantum each predicted action 7 for
all 830 validation UP positions. Relative to the original-distribution q8/d2
pilots, UP accuracy decreased for all three heads. The replacement-heavy design
therefore did not solve UP generalisation and is not the next tuning direction.

This outcome is development evidence from seed 100003 only. It is not a Tokyo
result and does not support a final model-ranking or quantum-advantage claim.
