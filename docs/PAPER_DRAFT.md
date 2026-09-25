# Quantum-Enhanced Language-Model-Based Control for BBR over Low-Earth-Orbit Satellite Internet

> **Manuscript status: development draft, simplified scope (current since 2026-09-03).** This draft replaces the earlier three-role, multi-backbone direction preserved at `archive/docs/PAPER_DRAFT.md`. The methodology, dataset, and `gpt_classical` results below are complete and frozen. `gpt_quantum` is training; its results, discussion, and conclusion are left as explicit placeholders (marked **[PENDING]**) rather than estimated or invented. Do not fill those sections until `gpt_quantum` has a frozen checkpoint and a one-time Tokyo evaluation.

**Author:** Chung Bao An Nguyen

## Abstract

Low Earth Orbit (LEO) satellite Internet exhibits rapid capacity variation, intermittent path changes, and non-terrestrial delay dynamics that complicate transport-layer congestion control. Bottleneck Bandwidth and Round-trip propagation time (BBR) achieves high throughput in this environment, but its bandwidth-probing behaviour can increase retransmissions and destabilise pacing. Small Language Model-based Control for BBR (SLM-BBR) casts pacing-gain selection as offline, return-conditioned sequence modelling over structured Starlink telemetry. This paper asks a single, narrow question: **when the data, backbone, tuning policy, and evaluation protocol are held fixed, does replacing a language model's classical action head with a trainable Qiskit variational quantum circuit (VQC) change its BBR pacing-gain prediction ability?** We select `LFM2.5-350M` as the shared backbone from four downloadable under-400M candidates using development-only validation metrics (Tokyo is never read for this choice), then attach two task heads to it: a normal linear head (`gpt_classical`) and an eight-qubit Qiskit `EstimatorQNN` head (`gpt_quantum`). Both variants share identical preprocessing, expert labels, the eleven-action phase-safe pacing-gain space, LoRA configuration, training budget, and held-out Tokyo evaluation. `gpt_classical` is trained and Tokyo-evaluated: 95.75% overall action accuracy and 82.05% macro-phase accuracy. `gpt_quantum` is still training at the time of writing; its results are reported as pending rather than estimated. Because this simplified comparison has no parameter-matched classical bottleneck control, any observed `gpt_classical`-versus-`gpt_quantum` difference is reported as-is and is not attributed to "quantum-ness" specifically. No claim of quantum advantage or unseen-location generalisation is made until `gpt_quantum` completes training and its one-time Tokyo evaluation is recorded.

**Index Terms:** BBR, congestion control, Low Earth Orbit satellite Internet, Starlink, language models, Low-Rank Adaptation, quantum machine learning, variational quantum circuits.

## I. Introduction

Low Earth Orbit (LEO) satellite constellations, exemplified by Starlink, provide broadband Internet with lower propagation delay than geostationary systems. Their moving topology, handovers, and load- and weather-dependent channel conditions nonetheless create a difficult environment for transport-layer congestion control. BBR differs from loss-based algorithms by explicitly estimating bottleneck bandwidth and round-trip propagation time and using these estimates to regulate pacing and in-flight data. Measurements over Starlink show that this model-based behaviour preserves high throughput, although bandwidth probing can also increase retransmissions and queue pressure [1], [2].

SLM-BBR reformulates BBR pacing-gain selection as offline, return-conditioned sequence modelling over real Starlink telemetry [1]. It combines a structured state encoder, a pretrained language-model backbone, Low-Rank Adaptation (LoRA) for parameter-efficient tuning, and a constrained task head that emits one of eleven phase-safe pacing gains per decision step. The direct head avoids free-form text generation and enforces safety through a phase-dependent action mask.

This project extends SLM-BBR with a single, causal research question:

> Does attaching a trainable Qiskit variational quantum circuit to the action head of one SLM change its BBR pacing-gain prediction ability, compared with the same SLM's classical head?

The scope was deliberately narrowed on 2026-09-03, after review found an earlier direction — a four-backbone modern-model comparison combined with a three-role classical/classical-bottleneck/quantum ablation on one 4B-parameter backbone — too broad to state and defend as one claim. The current design keeps exactly one research variable (the task head) fixed against exactly one backbone, so that any measured difference can be attributed to the head alone rather than to a mixture of backbone substitution and quantum integration. The earlier, broader work is preserved under `archive/` and is used only as background and as justification for design choices carried forward (Section II, Section III-F).

This paper makes four contributions:

1. It reproduces the SLM-BBR experience-pool construction deterministically and auditably: phase detection, eleven phase-safe pacing gains, trace-level development splitting, and strict isolation of Tokyo as a held-out evaluation location.
2. It selects a single under-400M-parameter backbone (`LFM2.5-350M`) from four downloadable candidates using development-only validation metrics, without reading Tokyo.
3. It implements a Qiskit-backed quantum action head and attaches it to the selected backbone alongside a normal classical head, holding every other pipeline element fixed.
4. It reports the classical head's Tokyo result as a frozen baseline and documents the quantum head's status, configuration, and remaining evaluation steps rather than reporting numbers that do not yet exist.

## II. Related Work

### A. BBR over Starlink

BBR regulates sending behaviour using an explicit model of bottleneck bandwidth and round-trip propagation time. Experimental studies over geographically distributed Starlink paths show that BBR can outperform Cubic, Vegas, and Hybla in throughput, while its active probing produces a trade-off involving retransmissions and stability [1], [2]. These observations motivate finer control over the ProbeBW pacing cycle rather than wholesale replacement of BBR.

### B. Small Language Model-Based BBR Control

The methodological foundation of this study is SLM-BBR [1], which represents Starlink telemetry as return-state-action sequences and applies LoRA-tuned language models to predict phase-constrained pacing gains. Its head directly predicts one of eleven discrete BBR pacing gains, constrained by the detected macro-phase (`BW_DOWN`, `BW_CRUISE`, `BW_UP`). This paper preserves the original state fields, sequence formulation, action space, phase mask, and held-out-location protocol unchanged, and extends only the task head.

### C. Language Models for Network Control

Related work has adapted language models to other network-control tasks, including active queue management through structured encoders, task-specific action heads, and parameter-efficient tuning [4]. These studies support the general pattern of a structured encoder plus a constrained head for network control, but do not define the BBR phase-safe pacing task used here.

### D. Variational Quantum Models

Variational quantum circuits (VQCs) encode classical features into parameterised quantum gates, apply trainable rotations and entangling operations, and expose measured expectation values to a classical optimiser. Hybrid quantum reinforcement-learning systems have been proposed for satellite beam and power allocation [3], using simulated quantum backends. That task differs from BBR pacing control in its state representation, action space, and objective; it is used here only as architectural motivation for combining a language-model backbone with a quantum action head, not as a source for the networking task itself.

### E. Research Gap and Scope of This Study

SLM-BBR establishes that language models can learn phase-safe BBR pacing decisions from Starlink telemetry. Separately, quantum reinforcement learning has been shown to be trainable for other satellite-control problems. These two directions have not previously been combined and evaluated under one controlled BBR experiment. This study addresses that gap with the narrowest design that isolates the head: the same backbone, data, labels, phase masks, LoRA policy, and evaluation protocol are used for both the classical and the quantum head, and only the head differs.

A parameter-budget-matched classical bottleneck control (`gpt_classical_twin`) was used in earlier exploratory work on a larger backbone (`archive/`) to separate the effect of low-dimensional compression from the effect of quantum processing specifically. It is not part of the present comparison. Its absence is a stated limitation (Section VI): an observed `gpt_classical`-versus-`gpt_quantum` difference in this study cannot be attributed to quantum processing specifically as opposed to the effect of routing the hidden state through an eight-dimensional bottleneck.

## III. Methodology

### A. Research Design

This study compares exactly two task-head variants on exactly one backbone:

- `gpt_classical` — `LFM2.5-350M` with a normal linear action head.
- `gpt_quantum` — the same `LFM2.5-350M` with a Qiskit-based variational quantum action head.

Both variants share the same checkpoint and revision, hidden-state extraction point, input pipeline, expert labels, phase masks, sequence windows, LoRA configuration, optimisation budget, checkpoint-selection rule, random seed, and evaluation implementation; only the task head differs. `LFM2.5-350M` itself was selected from four downloadable under-400M candidates using development-only validation metrics (Section III-F); Tokyo was not read for that choice and is reserved for a single, final evaluation of the frozen `gpt_classical` and `gpt_quantum` checkpoints.

### B. Dataset and Data Collection

This study uses the Starlink measurement traces associated with the SLM-BBR framework [1], collected using iperf3 across six geographically distributed locations: Ohio, São Paulo, London, Mumbai, Sydney, and Tokyo. Traces cover sequential and competitive traffic under downlink and uplink conditions. Each interval contributes sender-side throughput, retransmissions, congestion window, receiver window, round-trip time, and RTT variance, together with a location and stream-category flag. At decision step \(t\), the state is

\[
s_t = [l_t, u_t, \tau_t, y_t, r_t, c_t, w_t, d_t, v_t],
\]

a nine-dimensional vector combining the location flag \(l_t\), stream-category flag \(u_t\), observation time \(\tau_t\), throughput \(y_t\), retransmissions \(r_t\), congestion window \(c_t\), receiver window \(w_t\), RTT \(d_t\), and RTT variance \(v_t\).

Raw captures are treated as immutable source data; every derived label, split, and sequence sample is deterministic and stored separately, with source-file checksums and stable sample identifiers.

### C. BBR Phase Detection and Action Construction

The controller operates over three BBR macro-phases: `BW_DOWN`, `BW_CRUISE`, and `BW_UP`. Let \(\bar y_t\) be the centred rolling mean of throughput with half-window \(w=10\), and define the deviation \(\delta_t = y_t - \bar y_t\) with population standard deviation \(\sigma_\delta\). A local maximum with \(\delta_t > 0.7\,\sigma_\delta\) marks a `BW_UP` event; the first subsequent local minimum with \(\delta_t < -0.7\,\sigma_\delta\) marks `BW_DOWN`. Remaining positions are `BW_CRUISE`. Detection runs independently within each raw trace.

The global action set contains exactly eleven discrete pacing gains:

\[
\mathcal{A} = \{0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25\}.
\]

The feasible subset depends on the detected phase:

\[
\mathcal{A}_{\phi_t} =
\begin{cases}
\{0.90, 0.92, 0.94, 0.96, 0.98\}, & \phi_t = \mathrm{BW\_DOWN}, \\
\{1.00\}, & \phi_t = \mathrm{BW\_CRUISE}, \\
\{1.05, 1.10, 1.15, 1.20, 1.25\}, & \phi_t = \mathrm{BW\_UP}.
\end{cases}
\]

Using the trace-maximum utilisation proxy \(\bar B_t = y_t / \max_j y_j\), the continuous target gain follows the source formulation \(G_t^{\uparrow} = 3/(\bar B_t + 2)\) for `BW_UP` and \(G_t^{\downarrow} = (\bar B_t + 1)/2\) for `BW_DOWN`; `BW_CRUISE` is fixed to 1.00. The expert action is the nearest feasible discrete gain, with ties resolved by the lower action index. Reward assignment follows the source hybrid rate/delay utilisation model, unchanged from SLM-BBR [1] and not modified for either task head in this comparison.

The resulting non-Tokyo experience pool contains 60,084 samples from 200 traces: 52,576 `BW_CRUISE` (87.50%), 3,292 `BW_DOWN` (5.48%), and 4,216 `BW_UP` (7.02%). All five `BW_UP` actions occur in the pool. Every observed `BW_DOWN` target maps to gain 0.90 under this rule; this label degeneracy is retained rather than adjusted, and both task heads see it identically.

### D. Development Split and Sequence Sampling

The five development locations are split at the trace level using a deterministic SHA-256 ranking with seed 100003: within every location-by-stream stratum, two of ten traces are held out for validation. This produces 160 training traces (48,065 samples) and 40 validation traces (12,019 samples), with disjoint sample-identifier sets. Non-overlapping windows of length 20 are sampled with step 20 within trace boundaries, yielding 2,400 training windows and 600 validation windows (12,000 positions: 10,457 `BW_CRUISE`, 713 `BW_DOWN`, 830 `BW_UP`). Tokyo is excluded from this split entirely and is not read for preprocessing statistics, training, validation, checkpoint selection, architecture choice, hyperparameter selection, or model ranking.

### E. Structured Sequence and Backbone Adaptation

Each scalar state feature is independently embedded and aligned to the backbone hidden size; return and previous-action values are linearly embedded, and a learned timestep embedding is added. At step \(t\) the ordered token block is \([R_t, s_t^{(1)}, \ldots, s_t^{(9)}, a_t]\); a length-20 window therefore produces 220 structured embeddings, supplied to the backbone through its `inputs_embeds` interface rather than as text tokens. The hidden representation at the final state-feature position, immediately before the action token, is used for action prediction: \(h_t \in \mathbb{R}^d\).

Low-Rank Adaptation is applied to architecture-specific attention projections while the remaining backbone parameters are frozen:

\[
W = W_0 + \frac{\alpha}{r} BA.
\]

The shared configuration for both task heads is LoRA rank \(r=128\), scaling factor \(\alpha=32\), dropout 0.05, AdamW with learning rate \(10^{-4}\) and gradient clipping at 0.25, batch size 1 with gradient accumulation over 32 sequences, 16 epochs, sequence length 20, sample step 20, and seed 100003. Target modules are declared per backbone in `config.py`'s `modern_lora_registry`.

### F. Backbone Selection (Development-Only)

Four downloadable checkpoints under 400M parameters — `granite_4_0_350m`, `pleias_rag_350m`, `lfm2_5_350m`, and `gemma_3_270m` — were compared under the identical shared pipeline (Sections III-B–III-E) with a normal classical head, using only the epoch-16 development validation split. Tokyo was not read for this comparison.

**Table I. Development-only backbone selection (epoch-16 validation, seed 100003).**

| Backbone | Val. accuracy | Val. macro-phase accuracy | Val. `BW_UP` accuracy |
|---|---:|---:|---:|
| `granite_4_0_350m` | 0.9503 | 0.7606 | 0.2819 |
| `pleias_rag_350m` | 0.9505 | 0.7614 | 0.2843 |
| **`lfm2_5_350m`** | **0.9550** | **0.7831** | **0.3494** |
| `gemma_3_270m` | 0.9498 | 0.7582 | 0.2747 |

`LFM2.5-350M` (`LiquidAI/LFM2.5-350M`) leads on all three development metrics, including the historically weakest `BW_UP` phase, and is used as the fixed backbone for both `gpt_classical` and `gpt_quantum`. This selection is development-only evidence over a single seed; it supports backbone choice for the controlled head comparison but is not a general claim of superiority over the other three candidates.

### G. Classical Action Head

The classical head maps the backbone hidden state directly to the eleven-action logit space:

\[
o_t^{\text{classical}} = W_c h_t + b_c, \qquad o_t^{\text{classical}} \in \mathbb{R}^{11}.
\]

`gpt_classical` uses this head with no intermediate bottleneck.

### H. Quantum-Enhanced Action Head

The quantum head is a hybrid quantum-classical module positioned between the backbone hidden state and the action logits; it does not replace the state encoder, backbone, LoRA adapters, labels, or training objective. The hidden state is first projected to \(n_q\) scalar features and bounded into rotation angles, optionally through head-input layer normalisation and a fixed temperature \(T\):

\[
\theta_t = s_\theta \tanh\!\left(\frac{W_q\,\mathrm{LayerNorm}(h_t) + b_q}{T}\right),
\]

where \(s_\theta\) is the angle scale. Each angle \(\theta_{t,i}\) is encoded on qubit \(i\) with an \(R_Y(\theta_{t,i})\) gate. A depth-\(D\) variational circuit applies one trainable \(R_Y(\omega_{d,i})\) rotation per qubit per layer, followed by ring-structured CNOT entanglement. Per-qubit Pauli-Z expectation values \(q_t \in [-1,1]^{n_q}\) are mapped to eleven logits by a final linear layer:

\[
o_t^{\text{quantum}} = W_{out}\, q_t + b_{out}.
\]

The circuit is implemented using Qiskit Machine Learning's `EstimatorQNN` through `TorchConnector`, backed by Qiskit's local `StatevectorEstimator` with analytic expectation values (`default_precision=0.0`, `shots=None`, no quantum hardware).

**Frozen configuration for `gpt_quantum` (current):** 8 qubits, depth 1, one trainable RY layer per qubit with ring CNOT entanglement (no data re-uploading), angle scale \(\pi\), head-input LayerNorm enabled, temperature \(T=4\).

This configuration was chosen from two development-only checks, neither of which read Tokyo. First, a wall-clock benchmark on training-pool windows found that reducing circuit width and depth from an earlier 8-qubit/depth-2 configuration substantially reduced per-window training time, at up to a 13-15x speedup for a 4-qubit/depth-1 variant relative to 8-qubit/depth-2. Second, a longer correctness check (200 real training steps against a several-thousand-sample validation slice, not the short speed benchmark) showed that the 4-qubit/depth-1 variant collapsed `BW_DOWN` accuracy to 0%, while an 8-qubit/depth-1 variant recovered `BW_DOWN` accuracy to 100% at roughly one third the runtime of the original depth-2 configuration. 8 qubits/depth 1 was therefore chosen for correctness, at an estimated 16-epoch training time of approximately 2.1-2.2 days. LayerNorm and \(T=4\) are retained from earlier diagnostics (`archive/`) that found them necessary to avoid saturating the bounded angle encoding.

### I. Phase-Safe Prediction and Training Objective

Each head produces an unmasked logit vector \(o_t \in \mathbb{R}^{11}\). Given the feasible action subset \(\mathcal{A}_{\phi_t}\) for the detected phase, the phase-safe logits are

\[
\tilde o_{t,j} =
\begin{cases}
o_{t,j}, & j \in \mathcal{A}_{\phi_t}, \\
-\infty, & j \notin \mathcal{A}_{\phi_t}.
\end{cases}
\]

Invalid actions are removed before probability normalisation, loss computation, and action selection; the same mask is shared by both heads. The training objective is phase-masked cross-entropy against the expert action \(a_t^{*}\):

\[
\mathcal{L} = -\frac{1}{L}\sum_{t=1}^{L} \log \frac{\exp(\tilde o_{t,a_t^{*}})}{\sum_{j \in \mathcal{A}_{\phi_t}} \exp(\tilde o_{t,j})}.
\]

### J. Metrics and Reproducibility

Both variants are evaluated with the same model-independent implementation, reporting phase-masked cross-entropy loss, overall action accuracy, per-phase accuracy (`BW_DOWN`, `BW_CRUISE`, `BW_UP`), macro-phase accuracy (the unweighted mean of the three per-phase accuracies), and predicted-action distributions. Because `BW_CRUISE` dominates the data and permits only one valid action, macro-phase accuracy and per-phase accuracy are treated as primary evidence; overall accuracy is secondary.

Each run records a manifest with the git commit, dataset and split versions, sample-identifier hashes, model identifier and revision, task-head type and quantum configuration where applicable, seed, resolved training configuration, checkpoint path and checksum, hardware and software versions, and Tokyo-isolation status. A checkpoint is used downstream only after it passes a checkpoint-reload verification.

## IV. Results

### A. Backbone Selection

Reported in Table I (Section III-F). `LFM2.5-350M` is the frozen shared backbone for both `gpt_classical` and `gpt_quantum`.

### B. `gpt_classical` — Frozen, Tokyo-Evaluated

`gpt_classical` completed 16 epochs of training on the five development locations and validation on the frozen development holdout, then a single Tokyo evaluation (12,000 samples).

**Table II. `gpt_classical` results (`LFM2.5-350M`, classical head, seed 100003, epoch 16).**

| Split | Overall accuracy | Macro-phase accuracy | `BW_UP` accuracy |
|---|---:|---:|---:|
| Tokyo (held out) | 0.9575 | 0.8205 | 0.4615 |

This is a frozen, single-seed, single-checkpoint result. Full per-phase and per-action breakdowns are recorded in the corresponding run and evaluation manifests rather than restated in full here.

### C. `gpt_quantum` — In Progress **[PENDING]**

`gpt_quantum` (`LFM2.5-350M`, 8-qubit/depth-1 quantum head as configured in Section III-H) is training on the same five development locations, validation split, seed, LoRA configuration, and epoch budget as `gpt_classical`. Training had not completed at the time of writing.

**Table III. `gpt_classical` vs. `gpt_quantum` — PLACEHOLDER, to be completed after `gpt_quantum` training and Tokyo evaluation.**

| Model | Overall accuracy | Macro-phase accuracy | `BW_DOWN` accuracy | `BW_CRUISE` accuracy | `BW_UP` accuracy | Latency |
|---|---:|---:|---:|---:|---:|---:|
| `gpt_classical` | 0.9575 | 0.8205 | — | — | 0.4615 | — |
| `gpt_quantum` | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |

No `gpt_quantum` number in this table may be filled in from estimation, an earlier backbone's ablation trail, or a partial/step-limited diagnostic. Every cell must come from the frozen epoch-16 checkpoint's Tokyo evaluation manifest once it exists. Tokyo throughput and retransmission box plots (extending `reports/generate_under400m_box_charts.py`) are likewise deferred until then.

## V. Discussion **[PENDING — quantitative discussion requires Table III]**

A comparative discussion of accuracy, macro-phase behaviour, and any `BW_UP` action-collapse pattern cannot be written honestly before `gpt_quantum` has a frozen, Tokyo-evaluated checkpoint. This section is deliberately left as a placeholder rather than populated with expected or provisional claims.

One risk is already known and should be checked explicitly once `gpt_quantum` results exist, rather than assumed: earlier exploratory diagnostics on a different (4B-parameter) backbone repeatedly found that quantum and classical-bottleneck heads with an eight-dimensional bottleneck collapsed `BW_UP` predictions to a single majority action, even after saturation correction (`archive/docs/PHASE4_QISKIT_HEAD_ABLATION_DIAGNOSTIC.md`). Whether this pattern recurs, is absent, or differs for `LFM2.5-350M`'s eight-qubit/depth-one head is an open, testable question for this section once data is available — not a conclusion to state in advance.

Because this comparison has no classical-bottleneck twin, any difference between `gpt_classical` and `gpt_quantum` — in either direction — must be reported as an observed difference between "classical head" and "quantum head as implemented here," not as evidence isolating quantum computation from the effect of routing the hidden state through an eight-dimensional bottleneck before the final projection.

## VI. Limitations

1. **`gpt_quantum` is incomplete.** Its training, checkpoint freeze, and Tokyo evaluation are not yet done; Table III, Section V, and Section VIII are placeholders pending that work.
2. **No classical-bottleneck control.** This simplified comparison does not include a parameter-matched classical bottleneck (`gpt_classical_twin`), so an observed head difference cannot be attributed to quantum processing specifically. Earlier work with this control on a different backbone is preserved in `archive/` but is not part of this comparison.
3. **Single seed.** Both `gpt_classical` and `gpt_quantum` use seed 100003 only; no variance estimate is available.
4. **Backbone selection is development-only.** `LFM2.5-350M` was chosen from four candidates using development-split metrics alone (Table I); this is a valid development decision but not a final ranking claim, and it was not re-verified against Tokyo.
5. **Simulator-only quantum execution.** All quantum results use an analytic local statevector estimator with no shot noise or hardware error; they establish trainability and behaviour in simulation, not feasibility or performance on physical quantum hardware.
6. **Label degeneracy in `BW_DOWN`.** Every observed `BW_DOWN` expert label maps to pacing gain 0.90 under the reproduced reward rule (Section III-C); `BW_DOWN` accuracy therefore reflects correct phase/gain identification rather than five-way discrimination within that phase.

## VII. Remaining Work

1. Complete `gpt_quantum`'s 16-epoch training run (Section III-H configuration); per-epoch checkpoints allow inspection at intermediate epochs without committing in advance to a shorter budget than `gpt_classical`.
2. Verify checkpoint reload and Tokyo-isolation status for the frozen `gpt_quantum` checkpoint.
3. Run the one-time Tokyo evaluation for `gpt_quantum`, using the same sample IDs and metrics as `gpt_classical`.
4. Fill in Table III and the accompanying Tokyo throughput/retransmission box plots from the resulting manifest.
5. Write Section V (Discussion) and Section VIII (Conclusion) from the completed comparison, explicitly checking for the `BW_UP` collapse pattern described in Section V and reporting whatever is found, including a negative result.

## VIII. Conclusion **[PENDING]**

This section will summarise what the completed `gpt_classical`-versus-`gpt_quantum` comparison shows once Table III is filled in from a frozen, Tokyo-evaluated `gpt_quantum` checkpoint. At present, the paper establishes the shared pipeline, the development-only backbone selection (`LFM2.5-350M`), the frozen `gpt_classical` baseline (Tokyo accuracy 0.9575, macro-phase 0.8205), and the exact configuration under training for `gpt_quantum` (8 qubits, depth 1, `trainable_ry_layers`, LayerNorm, temperature 4). No claim about the effect of the quantum head, in either direction, is made until that comparison exists.

## References

[1] R. De Silva, S. R. Pokhrel, and J. Kua, "Small Language Model-based Control for BBR over Low Earth Orbit Satellite Internet," arXiv:2607.07142v1, 2026.

[2] R. De Silva, S. R. Pokhrel, and J. Kua, "Unveiling TCP BBR Dominance in Starlink Internet: Experimental Insights and Analysis," arXiv:2607.07133v1, 2026.

[3] Q. T. Ngo, Y. He, B. Jayawickrama, E. Dutkiewicz, and S. R. Pokhrel, "Quantum Reinforcement Learning With Classical Policy Deployment for Resource Allocation in Multibeam GEO-LEO Satellite Networks," *IEEE Internet of Things Journal*, vol. 13, no. 9, 2026, doi:10.1109/JIOT.2026.3663363.

[4] S. R. Pokhrel, D. Satish, J. Kua, and A. Walid, "Distilling Large Language Models for Network Active Queue Management," *IEEE Transactions on Networking*, vol. 34, 2026, doi:10.1109/TON.2026.3690076.

[5] N. Cardwell, I. Swett, and J. Beshay, "BBR Congestion Control," IETF Congestion Control Working Group, Internet-Draft draft-ietf-ccwg-bbr-03, 2024.

[6] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," in *Proc. ICLR*, 2022.

## Appendix A. Evidence-to-Claim Boundary

| Claim | Current status | Evidence required for final wording |
|---|---|---|
| `LFM2.5-350M` is the selected development backbone | Supported, one seed, development-only | Confirmed; not revisited unless a new ADR reopens backbone selection |
| `gpt_classical` Tokyo result (0.9575 / 0.8205 / 0.4615) | Supported — frozen, Tokyo-evaluated | None; treat as final for this seed |
| `gpt_quantum` trains without error under the frozen 8-qubit/depth-1 configuration | Supported by development-only wall-clock and gate-200 correctness checks | Full 16-epoch run completing without error |
| `gpt_quantum` Tokyo result | **Not available** | Completed training + one-time Tokyo evaluation |
| Quantum head changes `BW_UP` discrimination relative to classical | **Not evaluated for this backbone** | Table III populated from frozen checkpoints |
| Quantum advantage | **Not supported** | Would additionally require a classical-bottleneck control and multi-seed confirmation, neither in current scope |
| Unseen-location generalisation of `gpt_quantum` | **Unknown** | Frozen Tokyo evaluation |

## Appendix B. Reproducibility Snapshot

| Item | Current value |
|---|---|
| Development locations | Ohio, São Paulo, London, Mumbai, Sydney |
| Held-out location | Tokyo (location flag 5, frozen) |
| Seed | `100003` |
| Sequence length / step | 20 / 20 |
| LoRA rank / alpha / dropout | 128 / 32 / 0.05 |
| Epochs | 16 |
| Optimizer / learning rate / grad clip | AdamW / `1e-4` / 0.25 |
| Backbone | `LiquidAI/LFM2.5-350M` (selected via Table I) |
| `gpt_classical` checkpoint | `data/processed/lora_training/under400_rank128_epochs16_seed100003_v1/lfm2_5_350m_rank128_epochs16_seed100003_train/` |
| `gpt_classical` Tokyo result | `data/processed/evaluation/tokyo_8model_comparison_v1/results/lfm2_5_350m/` |
| Quantum software | Qiskit; Qiskit Machine Learning `EstimatorQNN` + `TorchConnector` |
| Quantum backend | Local analytic `StatevectorEstimator`, `default_precision=0.0`, no shots |
| `gpt_quantum` circuit (current) | 8 qubits, depth 1, `trainable_ry_layers`, ring CNOT, angle scale π, head-input LayerNorm, temperature 4 |
| `gpt_quantum` status | Training (16 epochs); checkpoint and Tokyo result **pending** |
