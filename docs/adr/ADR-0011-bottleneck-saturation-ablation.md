# ADR-0011: Bottleneck Saturation Ablation

## Status

Completed as a limited one-seed development smoke evaluation on 2026-08-25.

## Context

The CRUISE-80 pilot left classical-twin and quantum UP predictions collapsed.
Short diagnostics found high bounded-encoding saturation in both heads and
100% validation angle saturation in the quantum head. Quantum circuit gradients
were finite and non-zero, so a controlled encoding diagnostic has stronger
current evidence than increasing circuit depth or changing training data.

## Decision

Keep the existing head behavior as the default control and expose three named
experiment variables:

- optional LayerNorm immediately before the bottleneck input projection;
- a positive fixed temperature dividing projection values before `tanh`;
- quantum angle scale of pi or pi/2 after `tanh`.

Run only 25 training and 50 validation steps for the initial matrix. Use the
same Qwen revision, seed 100003, q8/depth-2 RY circuit, CRUISE-80 sampling plan,
labels, masks, loss, and development validation set. Do not access Tokyo.

## Consequences

These runs diagnose representation and gradient behavior; they are not model
quality results. The manifest records LayerNorm, temperature, and angle scale.
The next full run may be selected only after saturation decreases without
destroying gradients or UP prediction diversity. Any selected architecture
must later be compared fairly against an equivalent classical control and
confirmed across predefined seeds.

## Outcome

All four limited runs passed checkpoint reload and Tokyo isolation. LayerNorm
with temperature 4 reduced measured validation saturation to 0% in both the
classical-twin and quantum paths. Quantum circuit gradients remained finite
and increased relative to the saturated control. Nevertheless, both quantum
temperature-4 variants still predicted one UP action across the 37 UP positions
in the limited validation slice. Reducing the angle range to pi/2 did not
improve early loss, UP accuracy, prediction diversity, or circuit-gradient
strength over pi.

The decision is therefore partially supported: normalization and temperature
correct the measured saturation condition, but saturation was not the sole
cause of UP collapse. The pi/2 variant is not advanced. Quantum LN/T4/pi and
its classical-twin LN/T4 control may proceed only to a 200-step development
gate before any full training decision.
