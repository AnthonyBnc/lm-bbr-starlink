---
name: lm-bbr-starlink
description: Repository workflow instructions for AI agents working on lm-bbr-starlink. Use before understanding, editing, reviewing, testing, documenting, configuring, or running experiments in this repo, especially work involving BBR actions, Starlink traces, SLM-BBR reproduction, GPT/LM baselines, LoRA, quantum/VQC heads, data splits, evaluation, or experiment results.
---

# LM-BBR Starlink Agent Skill

## Required First Read

Before changing code or reporting experiment conclusions, read these files from the current checkout:

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/CONTEXT.md`
4. `docs/adr/ADR-0017-under400m-quantum-gpt-parent-and-protocol-freeze.md` for the current frozen parent/protocol
5. The applicable decision record under `docs/adr/`
6. The files and tests directly affected by the user request
7. The current git diff/status

`archive/` holds an earlier, broader-scope direction (paper-vs-modern-backbone
comparison, Qwen3.5-4B-Base quantum-head ablation) for provenance only; it is
not the current research question.

For manuscript work, additionally read the source paper and every
run/evaluation manifest supporting the proposed text (`archive/docs/PAPER_DRAFT.md`
is a draft view of the earlier, broader-scope evidence, not current). Do not
update a claim from conversation memory alone.

Treat `docs/ARCHITECTURE.md` as the research contract. Treat `docs/CONTEXT.md` as the living decision log and known-risk list.

## Canonical Naming Boundary

Use the main paper's terms in all new scientific prose, manifests, plots, and
experiment descriptions: `Small Language Model (SLM)`, `State Encoder`,
`Low-Rank Adaptation (LoRA)`, `Offline RL Policy with Language Model Head`,
`networking head`, `ProbeBW_UP`, `ProbeBW_DOWN`, and `ProbeBW_CRUISE`. Preserve
the displayed model names `GPT-2`, `T5`, `GPT-Neo`, and `SmolLM2`.

Existing implementation aliases such as `PLM`, `OfflineRLPolicy`,
`action_head`, `BW_UP`, `BW_DOWN`, and `BW_CRUISE` may remain where required for
backward compatibility. Do not spontaneously rename those identifiers, invent
a replacement method name, or use a new label that obscures which paper concept
it implements. A new experiment ID may add a neutral qualifier, but the report
must retain the canonical paper term.

Name authority is: paper display name, existing upstream/current public code
identifier, then an approved ADR. Search all three before introducing a name.
Do not coin a method acronym, model role, head name, metric name, dataset name,
or result label merely to make prose sound new. Allowed run suffixes are neutral
metadata already used by the repo, such as rank, epoch, seed, `preflight`,
`smoke`, and `pilot`.

Never call a run `exact`, `same as the paper`, or `paper reproduction` until the
full reproduction gate in `docs/ARCHITECTURE.md` is satisfied. The exact
four-SLM comparison is `GPT-2`, `T5`, `GPT-Neo`, and `SmolLM2`; four locally
available modern models are an extension, not a substitute for that set.

The user selected the four old SLMs as `published_reference` rows rather than
local reruns. Never attach local action accuracy, sample hashes, or hardware to
those rows. Figure-only values remain figure-only unless a documented
digitization explicitly marks them as estimates.

## Scope Check

Before editing, complete this sentence and make sure it is true:

> This change supports shared BBR pipeline, GPT classical control, GPT classical twin, Quantum-GPT, modern baseline, or evaluation, and does not alter the 11-action task, phase mask, shared data split, shared labels, or held-out Tokyo protocol.

Classify the task:

- `IMPLEMENTATION_ONLY`: refactor, bug fix, logging, tests, portability, or performance work that preserves outputs and methodology.
- `EXPERIMENT_VARIABLE`: model checkpoint, LoRA target, learning rate, qubit count, VQC depth, seed set, or another declared experiment parameter.
- `RESEARCH_INVARIANT`: action mapping, phase mask, split membership, label/reward construction, primary metrics, or Tokyo isolation.
- `OUT_OF_SCOPE`: generic chatbot, channel allocation, A2C/DDPG replacement, unrelated quantum task, or anything that abandons SLM-BBR BBR pacing control.

Proceed only for `IMPLEMENTATION_ONLY`, or for `EXPERIMENT_VARIABLE` when the choice is configurable, named, logged, and not tuned on Tokyo. Stop and ask for explicit approval for `RESEARCH_INVARIANT` or `OUT_OF_SCOPE` changes.

## Locked Research Invariants

Preserve these unless the user explicitly says there is tutor/supervisor approval and the decision is recorded durably:

- Task: offline return-conditioned BBR pacing-gain action prediction from structured Starlink telemetry sequences.
- State fields: location flag, stream flag, time, throughput, retransmissions, congestion window, receiver window, RTT, RTT variance.
- Action count: exactly 11.
- Action gains: `0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25`.
- Phase mask:
  - `BW_DOWN`: `0.90` through `0.98`
  - `BW_CRUISE`: `1.00`
  - `BW_UP`: `1.05` through `1.25`
- Invalid logits must be masked before action selection.
- Training/development locations: Ohio, Sao Paulo, London, Mumbai, Sydney.
- Held-out generalisation test: Tokyo.
- Tokyo must not affect training, validation, fitted preprocessing statistics, checkpoint selection, hyperparameter choice, or model ranking.
- Labels, reward construction, preprocessing, sample IDs, masks, metrics, and evaluation records must be shared across headline models.

Never silently introduce a 12th action, continuous action output, unconstrained classifier, text-generation task, different labels for one model, or a private model-specific data pipeline.

## Model Comparison Rules

For primary quantum claims, the classical GPT parent and Quantum-GPT must use the same:

- GPT-family checkpoint and revision
- hidden-state extraction point
- input pipeline and sample IDs
- split and checkpoint-selection rule
- LoRA/tuning policy, except for the declared head/integration difference
- evaluation metrics and records

Quantum-GPT is an added GPT-family variant, not automatically applied to whichever modern baseline performs best if that model is not an eligible GPT-style parent. The Qiskit VQC is the declared implementation path; its exact qubit count, depth, ansatz, and other experiment settings are not facts to assume before the GPT parent and quantum design are frozen.

The current declared Quantum-GPT backend is Qiskit: implement the quantum head with
Qiskit Machine Learning `EstimatorQNN` and PyTorch `TorchConnector`, using
Qiskit's local `StatevectorEstimator` for analytic development runs unless a new
ADR explicitly approves another backend. Do not silently replace Qiskit with a
private hand-written simulator in headline quantum experiments.

Modern baselines must be downloadable/open-weight, pinned by revision, compatible with local training, and routed through the same task head/masks/data/evaluation path.

## Repository Boundaries

Respect these ownership boundaries:

- `plm_special/`: model adapters, classical heads, quantum heads, training integration, model tests.
- `utils/`: model-independent parsing, BBR masks, splits, metrics, seeding, manifests.
- `config.py`: resolved paths, model IDs/revisions, split rules, seeds, hyperparameters, quantum settings.
- `run_plm.py`: thin single-run entry point, not the home for research logic.
- `run_all.py`: thin fair-comparison orchestrator, not a separate training/evaluation implementation.
- `tcp-cc-starlink-main/`: upstream data/code provenance; do not silently modify it.

Keep shared preprocessing, labels, masks, loss, and metrics in one authoritative path. Prefer explicit errors over silent fallback when data, checkpoints, shapes, quantum execution, or masks are unavailable.

## Known Risks To Check First

When relevant, inspect these before building on the current code:

- `plm_special/utils/constants.py` may define `ACTION_LEVELS = 12`, conflicting with the 11-action contract.
- Phase-safe masking may be missing or unclear in the policy path.
- `run_plm.py` may hard-code training and Tokyo experience-pool paths.
- Checkpoint selection may compare against `min_loss` without updating it.
- `state_encoder.py` may map the ninth feature through `fc8` instead of `fc9`.
- `rl_policy.py` may truncate sequence length using `plm_embed_size`.
- Model loading and LoRA targets are family-specific and need smoke tests for each candidate.
- `config.py` may contain non-portable local paths and model lists that do not match newer metadata.

Do not fix these by changing methodology. Restore or verify the documented SLM-BBR behaviour with tests where possible.

## Stop Conditions

Stop and ask one focused question when:

- The requested change conflicts with `docs/ARCHITECTURE.md` or a committed ADR.
- The task requires changing action count/order, phases, reward, labels, split membership, or primary metrics.
- The exact GPT checkpoint, model revision, tensor interface, or quantum insertion point cannot be verified.
- Completing the task would require mutating raw data or upstream reference code.
- A dependency or compute limit would require silently weakening the scientific protocol.
- The change would make classical and quantum GPT controls non-equivalent.
- A development or quantum-design decision would depend on Tokyo phase counts,
  labels, predictions, metrics, or model ranking.
- A request for exact paper reproduction lacks any exact model revision,
  released preprocessing/Experience Pool, or training detail required by the
  reproduction gate.
- A result table mixes paper-published references with local measurements
  without separate provenance labels.

## Current Milestone and Required Direction

The current frozen under-400M modern baselines are IBM Granite 4.0 350M,
Pleias-RAG 350M, LiquidAI LFM2.5-350M, and Gemma 3 270M. Their epoch-16
checkpoints are frozen. Tokyo location flag 5 and the held-out pool are also
frozen; model inference is the next operation. GPT-2, T5, GPT-Neo, and SmolLM2
are paper `published_reference` rows, not local reruns.

After the small-model Tokyo report, development continues with the existing
`Quantum-GPT` direction. Do not select its parent from Tokyo results.
`Qwen3.5-4B-Base` is the current development-selected GPT-style candidate and
the only currently documented parent path. Before full training, resolve the
`ProbeBW_UP` separability block and freeze the exact parent/head protocol in an
ADR.

The primary quantum experiment is the three-role comparison already named in
the codebase:

- `gpt_classical`: same GPT parent with normal networking head;
- `gpt_classical_twin`: same parent with parameter-budget-matched classical bottleneck;
- `gpt_quantum`: same parent with the declared Qiskit VQC head.

All three train on the identical Ohio, Sao Paulo, London, Mumbai, and Sydney
development data and use the identical validation split. Only after their
checkpoints are frozen may they run inference on the existing Tokyo pool.

Do not call all three “Quantum-GPT”, do not attach quantum to an arbitrary
Tokyo winner, and do not give the quantum variant private sampling, labels,
loss, metrics, or extra optimization budget.

## Paper-Writing Workflow

Before editing the manuscript:

1. Identify each statement as `published_reference`, `measured_exploratory`,
   `measured_final`, or `interpretation`.
2. Open the source paper table/figure/equation or the exact local manifest.
3. Preserve paper display names and codebase role IDs.
4. Keep modern-backbone extension results separate from paper reproduction and
   published-reference results.
5. Report negative, incomplete, and failed results; never select only favorable
   models/seeds.
6. Leave unavailable values blank. Never turn plot-only data into an exact
   number without a documented digitization method.
7. Update `docs/CONTEXT.md` when a decision/status changes and add an ADR for a
   frozen experiment choice. Any new manuscript draft should only be written
   after evidence is audited; `archive/docs/PAPER_DRAFT.md` is the earlier,
   broader-scope draft and should not be extended in place for the current
   simplified scope.

An agent must be able to map every result sentence to a paper locator or local
manifest path. If it cannot, it must not write the claim.

## Training Execution Ownership

The user runs training. Agents may generate or validate a plan and may audit
completed logs, manifests, metrics, and checkpoints, but must not execute or
resume training unless the user explicitly reauthorises it in the current
request. Read-only result checking does not require renewed authorisation.

## Validation

Prefer small, targeted tests and smoke checks that verify:

- action logits are shaped `[..., 11]`;
- phase masks only allow valid gains;
- Tokyo is absent from training/validation inputs;
- all headline models receive identical sample IDs and labels;
- forward/backward/save/reload works for each model candidate;
- manifests include model revision, seed, dataset/split version, resolved config, and quantum settings when applicable.

Smoke-test outputs are exploratory only. Do not present them as final research results.

## Required Handoff After Code Changes

After changing code, report:

- files changed;
- research purpose supported;
- invariants checked;
- commands/tests run and outcomes;
- experiment/config impact;
- unresolved assumptions or risks.

Include this checklist:

```text
Scope check: PASS | BLOCKED
Task class: IMPLEMENTATION_ONLY | EXPERIMENT_VARIABLE | RESEARCH_INVARIANT | OUT_OF_SCOPE
Data/split changed: yes/no
Labels/reward/actions changed: yes/no
Classical-vs-quantum comparability preserved: yes/no/not-applicable
Tokyo isolation verified: yes/no/not-applicable
```
