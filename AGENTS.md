# Repository Instructions for AI Agents

This repository implements and extends **Small Language Model-based Control for
BBR over Low Earth Orbit Satellite Internet**. Do not infer the research method
from conversation history or filenames alone.

## Mandatory reading order

Before understanding, editing, reviewing, testing, documenting, configuring,
or running an experiment, read completely:

1. `README.md`
2. `docs/ARCHITECTURE.md` - binding research contract
3. `docs/CONTEXT.md` - current state and next milestone
4. `docs/SKILL.md` - mandatory workflow and naming rules
5. `docs/adr/ADR-0017-under400m-quantum-gpt-parent-and-protocol-freeze.md` - the current frozen parent/protocol decision
6. affected source files/tests and the current git diff

`archive/` (docs, ADRs, scripts, figures) holds an earlier, broader-scope
direction (paper-vs-modern-backbone comparison, an extensive Qwen3.5-4B-Base
quantum-head ablation trail). Read it only for historical context/provenance
— it is not the current research question and must not be treated as current
instructions.

## Non-negotiable boundaries

- Preserve the paper's nine state fields, return/state/action sequence, State
  Encoder, LoRA method, networking head, cross-entropy objective, exactly 11
  pacing-gain actions, and phase mask.
- Train/development locations are Ohio, Sao Paulo, London, Mumbai, and Sydney.
  Tokyo is evaluation-only and must not affect any later development choice.
- Tokyo location flag is frozen as 5. Do not inspect Tokyo labels/results
  while selecting or tuning `gpt_quantum`.
- **Current simplified scope (since 2026-09-03):** exactly two model roles,
  same fixed backbone `LFM2.5-350M` (chosen from four under-400M candidates
  using development-only evidence, ADR-0017):
  - `gpt_classical` — classical head. Frozen, Tokyo-evaluated.
  - `gpt_quantum` — Qiskit VQC head, 4 qubits/depth 1 (chosen by wall-clock
    benchmark, not Tokyo). In progress.
  `gpt_classical_twin`, other modern-LM backbones, and the paper's
  GPT-2/T5/GPT-Neo/SmolLM2 rows are out of scope for this comparison; do not
  reintroduce them into the primary claim without the user's explicit
  instruction.
- The user runs long training. Do not start or resume a long/multi-hour
  training run unless the current request explicitly authorizes it — ask
  first, and prefer a small timed benchmark over guessing at duration.

## Naming and evidence

Use names from the main paper or existing upstream/current code. Search those
sources and approved ADRs before adding a name. Do not invent scientific
acronyms, model roles, method names, head labels, metrics, or dataset names.

Keep these evidence classes separate in code, figures, tables, and prose:

- `published_reference`
- `measured_exploratory`
- `measured_final`
- `interpretation`

Never claim "exact paper reproduction" for a modern-backbone substitution,
turn missing paper values into numbers, or combine published and locally
measured rows as a controlled same-sample comparison.

## Required handoff

After changes, report files changed, purpose, tests, experiment/config impact,
risks, and:

```text
Scope check: PASS | BLOCKED
Task class: IMPLEMENTATION_ONLY | EXPERIMENT_VARIABLE | RESEARCH_INVARIANT | OUT_OF_SCOPE
Data/split changed: yes/no
Labels/reward/actions changed: yes/no
Classical-vs-quantum comparability preserved: yes/no/not-applicable
Tokyo isolation verified: yes/no/not-applicable
```
