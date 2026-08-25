# Project Context

> Living decisions and implementation observations for `lm-bbr-starlink`. Last updated: 2026-08-07.

## Project identity

- Main repository: `https://github.com/AnthonyBnc/lm-bbr-starlink`
- Default branch: `main`
- Main paper: **Small Language Model-based Control for BBR over Low Earth Orbit Satellite Internet**
- Main task: language-model-based phase-safe BBR pacing control over Starlink traces
- Current phase: verify the paper implementation and select 4-5 modern compatible models
- Proposed contribution: one additional quantum-enhanced model based on a modern GPT-style backbone

## User-defined research direction

The intended study is:

1. use the main SLM-BBR paper as the methodological source of truth;
2. keep its dataset, state/action formulation, training method, and evaluation;
3. replace the paper's older model comparison with 4-5 newer downloadable models;
4. benchmark those models under one shared protocol;
5. freeze the best eligible modern GPT-style model as the quantum parent;
6. add a suitable quantum component to that GPT backbone;
7. compare quantum-GPT with the unchanged classical GPT parent and all other modern models.

This corrects two earlier misunderstandings:

- quantum is not automatically added to an arbitrary non-GPT winner;
- VQC is not a predetermined requirement.

## Baseline from the main paper

The main paper evaluates:

- GPT-2;
- T5;
- GPT-Neo;
- SmolLM2;
- LLaMA 3.2 as a larger comparison model.

Its core method uses:

- nine structured state values;
- a return/state/action temporal sequence;
- LoRA-based adaptation;
- a task-specific linear networking head;
- cross-entropy action learning;
- 11 discrete phase-constrained pacing gains;
- Ohio, Sao Paulo, London, Mumbai, and Sydney for training/development;
- Tokyo for unseen evaluation;
- surrogate throughput and retransmission models;
- inference latency, trainable parameters, and VRAM measurements.

These items must be verified in code before reportable experiments.

## Modern-model decision still required

The exact 4-5 modern models are not yet frozen. Each must be checked for:

- downloadable and pin-able weights;
- release recency relative to the legacy baseline;
- local PyTorch/Hugging Face compatibility;
- `inputs_embeds` or equivalent structured-input support;
- hidden-state access;
- LoRA/PEFT compatibility;
- 11-action-head compatibility;
- compute and licence feasibility.

At least one candidate must be GPT-style and eligible to become the quantum parent. Exact model names must not be added to headline experiments until their revisions and compatibility tests are recorded.

## Quantum decisions still required

The backbone family is fixed as GPT-style, but the exact checkpoint depends on the modern benchmark.

The following remain open until the GPT parent is frozen:

- exact GPT model and checkpoint;
- quantum method;
- insertion point;
- dimensionality reduction and data encoding;
- qubit count, depth, ansatz, and measurement if applicable;
- quantum framework;
- simulator versus hardware;
- training/gradient method;
- parameter-matched classical ablation;
- compute budget.

VQC may be evaluated as an option, but agents must not treat it as already selected.

## Dataset context

The expected raw source contains sequential and competitive logs for downlink and uplink across London, Mumbai, Ohio, Sao Paulo, Sydney, and Tokyo.

Rules:

- raw downloads remain immutable;
- competitive/sequential and uplink/downlink streams are not silently merged;
- preprocessing writes versioned derived data;
- data source, date/version, checksums, schema, and split manifests are recorded;
- Tokyo remains unseen until final evaluation;
- all compared models receive identical eligible sample IDs.

Do not assume raw logs already contain the paper's final expert labels. Trace reward, phase, and action generation from the main paper and released code.

## Current repository observations

The following were observed on `main` on 2026-08-05 and must be resolved or verified before reportable benchmarking:

1. `run_all.py` still launches the legacy set: T5, GPT-2, SmolLM2, GPT-Neo, and LLaMA 3.
2. `config.py` contains metadata for newer families such as Qwen and Gemma, but `cfg.plm_types` does not currently admit all of them.
3. `config.py` hard-codes a Windows checkpoint path (`d:\\downloaded_plms`), which is not portable.
4. `plm_special/utils/constants.py` defines `ACTION_LEVELS = 12`, while the main paper defines 11 pacing-gain actions. This is a blocking paper/code mismatch.
5. The inspected policy path exposes a linear action head, but phase-safe masking is not obvious in that path and must be traced before claiming paper equivalence.
6. `run_plm.py` hard-codes training and Tokyo experience-pool paths instead of consistently using the command-line/config value.
7. The best-checkpoint loop compares against `min_loss` but does not visibly update `min_loss`; checkpoint-selection behavior must be tested.
8. `state_encoder.py` maps the ninth feature through `fc8` rather than `fc9`; verify whether this is a defect.
9. `rl_policy.py` truncates sequence length using `plm_embed_size`; verify this against each model's actual context limit and the paper's intended sequence behavior.
10. Model loading and LoRA targets contain family-specific assumptions that require a smoke test for every new candidate.

These are observations, not permission to change methodology. Fixes must preserve the main paper's task and pass shared tests.

## Required validation split

The paper clearly reserves Tokyo for final testing. The project also needs a validation procedure for model and checkpoint selection that uses only the five development/training locations.

The exact strategy remains to be frozen before benchmarking, for example a location-aware inner holdout or a reproducible within-development split. An agent must not invent it during a run or tune it after seeing Tokyo.

## Final comparison matrix

| Model row | Required role |
|---|---|
| Modern classical 1-4/5 | Updated baselines |
| Classical GPT parent | Direct quantum control and one member of the classical group |
| Quantum-enhanced GPT | Proposed contribution |

Legacy paper models may be reported as reproduction/reference rows, but they must be visually separated from the headline modern comparison.

## Immediate next decisions

1. Verify and reconcile the 11-action contract in paper and code.
2. Freeze the validation design without using Tokyo.
3. Define the candidate-screening date and compute limit.
4. Select and pin 4-5 modern model revisions.
5. Run forward/backward/save/reload compatibility tests.
6. Freeze the ranking rule for `best_overall_classical` and `quantum_parent`.
7. Only after the classical benchmark, choose the quantum design.

## Phase 1 implementation status

Started on `QGPT` on 2026-08-06:

- `utils/bbr.py` is now the authoritative 11-action pacing-gain and phase-mask module.
- The legacy `plm_special.utils.constants` import resolves to the shared 11-action contract.
- Policy construction rejects action heads that do not have exactly 11 outputs.
- Batch processing rejects non-integer or out-of-range action labels.
- Dependency-free tests lock the action order and all three phase masks.

The shared experience-pool schema now requires a paper-defined BBR macro-phase for every sample.
Dataset construction validates phase/action compatibility, and `OfflineRLPolicy.sample` masks invalid
logits before choosing an action. Legacy pools without phase metadata fail explicitly rather than
running an unconstrained policy.

Phase detection itself is not implemented yet. The checked-out upstream data contains raw telemetry
but no reusable released phase-construction code. The next data-generation step must reproduce and
test the paper's rolling-throughput/local-extrema procedure before generating replacement pools;
guessing phase from an existing state field would change sample meaning.

Paper preprocessing reproduction was approved on 2026-08-06 with `w = 10` used as the half-window
for all centered rolling calculations. `utils/starlink_preprocessing.py` now implements an
exploratory shared parser, phase detector, Equations 16 and 22-27 label/reward path, deterministic
sample IDs, and a hard Tokyo exclusion. The exact assumptions are recorded in
`docs/adr/ADR-0001-paper-preprocessing-reproduction.md`.

The first Ohio/downlink-sequential smoke pool produced 301 valid samples with phase counts of 261
CRUISE, 20 UP, and 20 DOWN. Literal Equation 16 optimization using Equations 22-27 collapsed labels
to actions 0.98, 1.00, and 1.05. On 2026-08-06 the user approved using Equations 2-3 as the
authoritative expert-label construction: compute the utilization-based continuous gain, choose the
nearest phase-safe discrete action, and then use Equation 16 to assign that selected action's reward.
The user approved full non-Tokyo pool generation on 2026-08-06. The exploratory pool contains all
200 discovered development traces and 60,084 samples: 52,576 CRUISE (87.50%), 3,292 DOWN (5.48%),
and 4,216 UP (7.02%). Tokyo is absent, all sample IDs are unique, all states contain nine fields,
all rewards are finite, and all labels pass the phase mask. The five UP gains all occur, but every
DOWN sample maps to 0.90. This is the direct result of Equation 3: observed DOWN targets range from
0.50 to 0.820765, below the 0.91 nearest-action boundary between 0.90 and 0.92. Model training is
paused until this full-pool imbalance is reviewed and the validation split is frozen.

One Sydney trace places an iperf diagnostic line before otherwise valid JSON. The shared parser now
preserves and records diagnostics on both sides of the JSON object instead of dropping that trace.
The experience-pool pickle loader also restores episode-end flags, covered by a round-trip test.

On 2026-08-06 the user approved implementing an exploratory 80/20 development split. ADR-0002
records a deterministic SHA-256-ranked holdout of two traces per location by stream stratum using
seed 100003. The generated split has 160 training traces (48,065 samples) and 40 validation traces
(12,019 samples), with disjoint sample IDs and no Tokyo data. `ExperienceDataset` now builds windows
inside episode boundaries, preventing sequences from crossing between raw traces. The split remains
pending supervisor approval before headline experiments.

Model-screening support started on `QGPT` on 2026-08-06:

- `config.py` now records the downloaded modern shortlist under a local-model registry rooted at
  `~/models/lm-bbr-starlink` by default, with explicit Hugging Face IDs and release dates.
- The repo now admits `qwen3` and `gemma3` as first-class `plm_type` values for config-level
  selection instead of silently keeping them outside `cfg.plm_types`.
- `screen_modern_models.py` provides a dependency-light metadata screen for the downloaded modern
  models, including whether the current repo loader has a plausible mapping and whether config
  hidden-size/layer metadata matches the downloaded checkpoint metadata.

The local environment now contains `torch`, `transformers`, and `peft`. An exploratory generic
`AutoModel` smoke test passed for the pinned local LFM2.5-2.6B revision: one phase-masked BBR
forward/backward optimizer step produced `[1, 4, 11]` logits and a non-zero gradient, followed by a
successful task-checkpoint reload. The backbone was frozen for this compatibility smoke, so this is
not a training result or evidence for model quality. Full model training remains blocked on final
split approval and a declared tuning policy.

On 2026-08-06 the same frozen-backbone BBR smoke protocol was completed for all five pinned local
models: Qwen3.5-4B-Base, Gemma 3 4B PT, Llama 3.2 3B, LFM2.5-2.6B, and OLMo 3 7B. Every model
produced `[1, 1, 11]` logits for the same development-only BW_UP sample, completed backward and one
task-head optimizer step with a finite gradient, and passed checkpoint reload. Qwen produced a NaN
gradient under FP16 but passed under BF16; its manifest records BF16 as the required smoke dtype.
No Tokyo data was accessed.

ADR-0003 records exploratory LoRA preparation. Runtime module-tree checks now validate family-specific
targets, including Qwen hybrid linear-attention layers, LFM hybrid convolution projections, and text-
only Gemma targets that exclude its vision tower. Rank 8, alpha 32, and dropout 0.05 were used only to
construct and save test adapters for all five models. These are not frozen training hyperparameters.

Phase 2 training-runner preparation started on 2026-08-07. `train_modern_lora.py` now provides one
shared modern-model LoRA path with phase-masked cross-entropy, deterministic train/validation loaders,
per-phase and per-action metrics, provenance manifests, checkpoint save/reload verification, and hard
Tokyo isolation checks. A limited real-data LFM2.5 rank-8 check passed with sequence length 20, one
train batch, and one validation batch; its finite train loss was 0.273499 and checkpoint reload passed.
The run is marked `exploratory_pipeline_check` and is not a model-quality result. At that point, full
one-epoch LoRA training had not started. Loading model shards directly onto MPS exposed a native concurrent Metal
copy/cast crash on this macOS/PyTorch runtime, so the shared loader now loads MPS weights on CPU first
and moves the completed model to MPS sequentially; model weights and numerical dtype are unchanged.

The full Phase 2 LFM2.5 rank-8 pilot completed on 2026-08-07 in 4,659.84 seconds. It processed 2,400
train windows (48,000 positions) and 600 validation windows (12,000 positions), produced finite train
loss 0.125753 and validation loss 0.115458, and passed checkpoint reload and Tokyo-isolation audits.
Validation overall accuracy was 94.60%, but this is dominated by the phase distribution: CRUISE and
DOWN accuracy were both 100%, while UP accuracy was 21.93%; macro-phase accuracy was 73.98%. The UP
prediction distribution also concentrated on action 10 (gain 1.25), with 802 of 830 UP predictions.
This is an exploratory one-seed development result, not evidence of model superiority and not a Tokyo
result. Its manifest keeps `reportable_result: false`.

Phase 3 ran with rank 8, alpha 32, dropout 0.05, five epochs, sequence/sample step 20, gradient
accumulation 32, and seed 100003 on the same development split. LFM2.5, Llama 3.2, Qwen3.5, and
Gemma 3 completed with audited manifests, checkpoint reload, and Tokyo isolation. Qwen3.5 currently
leads this one-seed development comparison with 95.74% overall accuracy, 38.43% UP accuracy, 79.48%
macro-phase accuracy, and validation loss 0.09531. This is exploratory and cannot be ranked by
overall accuracy alone because CRUISE dominates the data.

OLMo 3 7B was excluded after reaching only epoch 1 step 200 in approximately 13.3 hours on Apple
MPS, projecting 29-33 days for the full run. ADR-0004 records this compute-limit decision. Its partial
loss is not compared with completed models. The output is an audited four-model exploratory
comparison, not a completed five-model benchmark. Tokyo remains untouched.

Phase 4 started on 2026-08-11 as a development multi-seed confirmation step for Qwen3.5-4B-Base,
the leading GPT-style candidate from Phase 3. ADR-0005 proposes keeping the Phase 3 rank-8,
five-epoch protocol and running fixed seeds 100003, 100019, and 100043 on the same development split.
The new `run_phase4_multiseed.py` runner plans or executes those runs, and
`summarize_phase4_multiseed.py` audits same-model multi-seed results by allowing the seed to vary
while requiring all data, split, sample-ID, label, mask, and training-protocol fields to match.
This remains non-reportable development work until the final policy and quantum parent are approved.

Quantum-GPT scaffolding started on 2026-08-11. `plm_special/quantum_head.py` implements the proposed
4-qubit, depth-2 VQC head with bounded RY angle encoding, trainable RY layers, CNOT-ring
entanglement, per-qubit Pauli-Z expectation measurement, and an 11-logit output projection.
On 2026-08-12, ADR-0008 changed the declared quantum backend to Qiskit: the head now uses
Qiskit Machine Learning `EstimatorQNN`, `TorchConnector`, and Qiskit's local `StatevectorEstimator`
instead of a private hand-written PyTorch state-vector simulator. `OfflineRLPolicy` accepts explicit
`head_type` values: `classical`, `classical_twin`, and `quantum`. The default remains `classical`,
so existing Phase 2-4 classical runs are unchanged. `run_quantum_smoke.py` plans or executes a
one-train-batch/one-validation-batch Quantum-GPT smoke check on the non-Tokyo development split.
ADR-0006 records that this is implementation scaffolding only; full Quantum-GPT training still waits
for Phase 4 parent/policy approval.
Any quantum smoke artifacts generated before ADR-0008 are historical PyTorch-simulator smoke records,
not Qiskit-backend evidence, and should be rerun before being cited.
The fresh Qiskit smoke `quantum_smoke_qwen_qiskit_seq1_r8_q4_d2_v2` passed on 2026-08-12 with
`qiskit==2.5.1`, `qiskit-machine-learning==0.9.0`, `StatevectorEstimator`, `default_precision=0.0`,
one train step, one validation step, checkpoint reload PASS, and Tokyo isolation PASS. A full
sequence-length-20 Qiskit smoke was interrupted because the first Qiskit training step was too slow
for an interactive check; Qiskit pilot/full training should be treated as a compute-risk item.

On 2026-08-12, `run_qwen_head_ablation.py` and `summarize_head_ablation.py` were added to plan,
execute, and audit same-parent Qwen head ablations across `classical`, `classical_twin`, and
`quantum` heads. The default runner mode is a limited smoke check with one train batch and one
validation batch per head. Unlimited pilot summaries require all three heads, the same Qwen revision,
the same seed, and identical shared data/protocol fields. ADR-0007 records this as development
validation scaffolding, not a final quantum comparison.

On 2026-08-13, the one-epoch Qwen head-ablation pilot completed with the Qiskit backend and the
same non-Tokyo development split. All three heads passed checkpoint reload and Tokyo-isolation audits.
The classical head reached 95.075% validation accuracy and 76.265% macro-phase accuracy; the
classical bottleneck twin reached 94.258% and 72.329%; the Qiskit quantum head reached 88.625% and
40.482%. The quantum head failed on `BW_DOWN` with 0.0% accuracy and collapsed its DOWN predictions
to action 3 (`0.96`) while all DOWN labels in this split are action 0 (`0.90`). It also collapsed
all UP predictions to action 8 (`1.15`). This is a development diagnostic result only, not evidence
for final model ranking. The next step is controlled quantum-head diagnosis/capacity ablation before
larger training or Tokyo evaluation.

On 2026-08-14, the 8-qubit/depth-2 Qiskit head-ablation pilot completed. Increasing quantum capacity
from 4 to 8 qubits fixed the DOWN collapse: the quantum head reached 100% DOWN accuracy, 100% CRUISE
accuracy, 21.45% UP accuracy, 94.567% overall validation accuracy, and 73.815% macro-phase accuracy.
The classical head still led with 95.075% overall and 76.265% macro-phase accuracy. The remaining
problem is UP action collapse: the 8-qubit quantum head predicted action 8 (`1.15`) for all UP
validation samples. This result supports a controlled imbalance experiment rather than increasing
qubits again.

`train_modern_lora.py` now supports `--loss-weighting none|phase_balanced`. The weighting affects
training loss only; validation loss/accuracy remain unweighted for comparison with prior runs. The
option is routed through `run_qwen_head_ablation.py` so classical, classical-twin, and quantum heads
can be trained under the same balanced-loss protocol. A one-batch Qwen classical smoke with
`--loss-weighting phase_balanced` passed checkpoint reload and Tokyo isolation on 2026-08-14.

On 2026-08-16, the fair 8-qubit/depth-2 phase-balanced-loss head ablation completed after rerunning
the interrupted quantum head into a separate `_rerun` output folder. All usable heads passed
checkpoint reload and Tokyo-isolation audits. The classical head improved to 95.342% overall,
77.550% macro-phase accuracy, and 32.65% UP accuracy. The classical bottleneck twin reached 94.992%
overall, 75.863% macro-phase accuracy, and 27.59% UP accuracy. The 8-qubit Qiskit quantum head
remained at 94.567% overall, 73.815% macro-phase accuracy, and 21.45% UP accuracy, with all UP
predictions still collapsed to action 8 (`1.15`). Balanced loss therefore helped the classical
control but did not solve the quantum UP collapse. The next controlled step is a small quantum
circuit-depth or ansatz ablation, starting with an 8-qubit/depth-3 smoke test, not Tokyo evaluation
or intentionally weakening the classical baselines.

On 2026-08-18, the 8-qubit/depth-3 Qiskit path passed a sequence-length-5 smoke check and then
completed the one-epoch fair Qwen head-ablation pilot. All three heads passed checkpoint reload and
Tokyo-isolation audits. The classical head matched the earlier unweighted q8 pilots at 95.075%
overall, 76.265% macro-phase accuracy, and 28.80% UP accuracy. The classical bottleneck twin reached
94.992% overall, 75.863% macro-phase accuracy, and 27.59% UP accuracy. The 8-qubit/depth-3 Qiskit
quantum head reached 94.517% overall, 73.574% macro-phase accuracy, and 20.72% UP accuracy. Compared
with the 8-qubit/depth-2 quantum head, depth 3 slightly reduced UP accuracy from 21.45% to 20.72%
and did not improve validation loss. This suggests that simply adding another RY/CNOT-ring layer is
not the right next lever for the UP-state failure. The next controlled development step should focus
on the UP label/action imbalance directly or on a richer Qiskit ansatz, while preserving the same
Qwen parent, split, labels, masks, metrics, and Tokyo isolation.

The next quantum-design ablation is implemented as an explicit `--quantum-ansatz` experiment
variable. The existing circuit remains `trainable_ry_layers`: bounded RY input encoding, trainable
RY rotations, CNOT-ring entanglement, and per-qubit Pauli-Z measurements. The new comparison circuit
is `trainable_ry_rz_layers`: it keeps the same encoding, entanglement, measurement, Qiskit
`StatevectorEstimator`, qubit count, depth, and output head, but adds a trainable RZ rotation after
each trainable RY rotation. For q8/depth-2 this doubles trainable circuit parameters from 16 to 32.
This is a named ansatz ablation, not a change to labels, actions, split, metrics, parent model, or
Tokyo protocol. The next gate is a q8/depth-2 RY+RZ smoke run before launching the long fair
head-ablation pilot.
The smoke run `quantum_smoke_qwen_qiskit_seq5_r8_q8_d2_ry_rz` passed on 2026-08-18 with sequence
length 5, checkpoint reload PASS, Tokyo isolation PASS, and `trainable_circuit_parameters=32`.

The full q8/depth-2 RY+RZ one-epoch head ablation subsequently completed. Its quantum head reached
94.567% overall validation accuracy, 73.815% macro-phase accuracy, 100% DOWN, 100% CRUISE, and
21.446% UP, effectively matching the earlier RY-only q8/depth-2 result and retaining the action-8 UP
collapse. Circuit depth and the added RZ parameters therefore did not solve the current bottleneck.

On 2026-08-24, ADR-0009 introduced a training-only UP-focused sampling diagnostic. The new runner
trains only the q8/depth-2 RY Quantum-GPT head path using 1,500 deterministically sampled sequence
windows with replacement. The sampler uses pure-UP windows, balances its quotas across actions 6-10,
prioritises windows with higher UP density, and penalises DOWN positions. With seed 100003 and UP
density power 3, the planned training distribution changes from 7.05% UP / 5.37% DOWN / 87.57%
CRUISE positions to 16.00% UP / 3.57% DOWN / 80.43% CRUISE. Its five UP action counts are
951, 950, 946, 974, and 979. Labels, rewards, actions, phase masks, source pools, and the full
600-window validation distribution remain unchanged. Historical classical and classical-twin q8/d2
results are shown only as validation references because they trained on all 2,400 original windows;
this is not a fair headline training-protocol comparison. A requested 700-window test is deferred
until an independent development-test split or the final Tokyo gate is approved.

On 2026-08-24, tutor feedback superseded the 1,500-window quantum-only diagnostic: the training
budget must remain 2,400 windows, and any sampling change must be applied to every compared head.
ADR-0010 therefore defines one shared CRUISE-80 design for Qwen classical, classical-twin, and
q8/depth-2 RY quantum runs. It selects 2,400 windows from the unchanged development pool using 800
retained base windows and 1,600 UP-focused replacement draws. The realised position distribution is
79.365% CRUISE, 14.246% UP, and 6.390% DOWN. UP action counts are 1,364, 1,367, 1,382, 1,361, and
1,364 for actions 6-10, while every development location contributes exactly 480 windows. The
design has 835 unique and 1,565 repeated windows. Validation remains the original 600 windows,
Tokyo remains unused, and the source pool/split checksums are unchanged.

The CRUISE-80 fair head ablation completed on 2026-08-25 for all three Qwen heads with seed 100003.
All runs passed checkpoint reload and Tokyo-isolation checks. On the unchanged 600-window
development validation set, the classical head reached 94.867% overall accuracy, 75.261%
macro-phase accuracy, and 25.783% UP accuracy. The classical bottleneck twin reached 94.258%
overall, 72.329% macro-phase, and 16.988% UP. The q8/depth-2 RY Qiskit head also reached 94.258%
overall, 72.329% macro-phase, and 16.988% UP. All three reached 100% validation DOWN and CRUISE
accuracy.

The identical twin and quantum UP scores hide an important failure mode: both predicted action 7
(`1.10`) for every one of the 830 validation UP positions. The classical head produced predictions
across actions 6-10 and therefore retained more UP discrimination, although its 25.783% accuracy
remains weak. Relative to the original-distribution q8/depth-2 pilots, CRUISE-80 reduced UP accuracy
from 28.795% to 25.783% for classical, 27.590% to 16.988% for classical-twin, and 21.446% to
16.988% for quantum. The balanced replacement design therefore did not improve natural-validation
UP generalisation. Its 835 unique windows and 1,565 repeated selections are a plausible overfitting
risk, but this one-seed result does not by itself prove the cause.

This result remains non-reportable development evidence and must not be presented as a Tokyo or
final quantum comparison. The next controlled diagnostic is quantum trainability rather than more
sampling, depth, or ansatz expansion: preserve zero initialization as a control, add a deterministic
small non-zero circuit-weight initialization option, and log gradients separately for the angle
projection, Qiskit circuit parameters, and output projection. Any promising initialization must be
confirmed across predefined seeds without using Tokyo.

On 2026-08-25, 25-step trainability diagnostics showed a shared saturation
problem in the two bottleneck heads. Classical-twin had 83.7% mean training
`tanh` saturation and approximately 88.0% validation saturation. Quantum had
92.5% mean training angle saturation and 100% validation saturation. Qiskit
circuit gradients remained finite and non-zero, so the evidence points to
information loss at the bounded input encoding rather than a disconnected
quantum layer. The earlier quantum diagnostic field named
`projected_abs_mean` measured the encoded angle magnitude, not the raw Linear
projection; the implementation now records `raw_projection_abs_mean` and
`angle_abs_mean` separately.

ADR-0011 defines the next limited saturation ablation. Default behavior remains
unchanged: no head-input LayerNorm, temperature 1, and angle scale pi. Named
experiment variables can now add LayerNorm, divide projection outputs by a
positive fixed temperature before `tanh`, and select pi or pi/2 quantum angle
range. A four-configuration, 25-step runner compares LayerNorm/temperature
effects on the classical twin and quantum q8/depth-2 head using the same frozen
CRUISE-80 design. Tokyo remains unused. No configuration proceeds to full
training until gradients, saturation, and UP prediction diversity are reviewed.

The four ADR-0011 saturation smokes completed on 2026-08-25. Every run passed
checkpoint reload and Tokyo isolation. On the common first 50 validation
windows, the prior quantum control had 100% angle saturation and mean quantum
weight-gradient norm 0.00476. Quantum LayerNorm with temperature 1 reduced
validation saturation to 72.99% and increased the mean circuit-gradient norm to
0.02839, but all 37 UP positions still mapped to action 9. Quantum LayerNorm
with temperature 4 reduced saturation to 0% for both pi and pi/2 angle scales.
The corresponding mean circuit-gradient norms were 0.03228 and 0.02365, but
both variants still mapped every UP position to action 6 and reached only
10.81% UP accuracy on this small slice. The classical-twin LayerNorm/T4 control
also reached 0% saturation and produced two UP actions rather than one, with
16.22% UP accuracy.

These results confirm that the original bounded encoding was saturated and
that normalization/temperature improves gradient conditions. They also show
that removing saturation alone does not immediately solve UP action collapse.
The pi/2 scale offers no early benefit over pi. Because this is a 25-step smoke
on 37 validation UP positions, it must not be used as a model-quality ranking.
The next evidence gate is a 200-step quantum LN/T4/pi diagnostic and the
classical-twin LN/T4 control on the same frozen development protocol. A full
2,400-step run remains premature. If UP diversity does not emerge by that gate,
the next diagnosis must measure UP-class separability before and after the
bottleneck rather than adding depth, epochs, or new sampling.
