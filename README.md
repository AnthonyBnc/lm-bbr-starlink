# Quantum-Enhanced GPT for BBR over Starlink

Research code for integrating a Qiskit-backed trainable variational quantum circuit (VQC) into a downloadable GPT-family model for BBR pacing-gain control over Starlink traces.

The proposed **Quantum-GPT** is evaluated against:

- the same GPT backbone with a classical head;
- a parameter-budget-matched classical bottleneck;
- modern downloadable language-model baselines; and
- reproducible models from the original SLM-BBR study where practical.

The goal is to test whether quantum integration changes performance, efficiency, and unseen-location generalisation. The project does not assume that the quantum model will outperform classical models.

## Research scope

The primary question is:

> Can a quantum-enhanced GPT controller improve BBR pacing-gain prediction and unseen-location performance compared with the equivalent classical GPT and modern language-model baselines?

The original networking task remains fixed:

- exactly 11 phase-constrained pacing-gain actions;
- shared preprocessing, labels, loss, masks, and metrics;
- Ohio, Sao Paulo, London, Mumbai, and Sydney for training/validation;
- Tokyo reserved for held-out generalisation testing;
- no separate data or evaluation path for Quantum-GPT.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full research contract and mandatory coding-agent guardrails.

## Repository map

```text
lm-bbr-starlink/
|-- plm_special/             # GPT adapters, classical controls, VQC integration
|-- tcp-cc-starlink-main/    # upstream Starlink code/data reference
|-- utils/                   # shared data, masks, metrics, manifests, seeding
|-- config.py                # central experiment configuration
|-- run_plm.py               # one model/run entry point
|-- run_all.py               # fair multi-model orchestration
|-- README.md                # project entry point
|-- docs/                    # architecture, context, ADRs, plans, and result notes
|-- docs/ARCHITECTURE.md     # authoritative research and architecture contract
|-- docs/CONTEXT.md          # current decisions, assumptions, and open questions
`-- docs/SKILL.md            # workflow instructions for coding agents
```

This map describes intended ownership. Agents must inspect the actual files before modifying or importing them.

## Model design

The default proposed path is:

```text
BBR state/return sequence
  -> downloadable GPT backbone
  -> action-position hidden state
  -> Linear(d_model, n_qubits)
  -> bounded angle encoding and Qiskit VQC
  -> Pauli-Z measurements
  -> Linear(n_qubits, 11)
  -> BBR phase mask
  -> pacing-gain action
```

The initial experimental VQC uses Qiskit Machine Learning's `EstimatorQNN`
through `TorchConnector`, backed by Qiskit's local `StatevectorEstimator`. It
uses 4 qubits, 2 variational layers, analytic expectation values with
`default_precision=0.0`, and no finite shots or quantum hardware. These are
configurable experiment defaults, not final conclusions.

## Dataset handling

Preserve downloaded traces as immutable raw data. Derived samples must be deterministic and written separately. Do not commit large raw data, processed data, model weights, caches, secrets, or local environments.

Expected source groups include downlink/uplink and competitive/sequential logs across London, Mumbai, Ohio, Sao Paulo, Sydney, and Tokyo. Do not combine groups or infer training labels until the released schema and preprocessing code have been verified.

## Before running experiments

1. Read `ARCHITECTURE.md`, `CONTEXT.md`, and `SKILL.md`.
2. Confirm the raw dataset path and schema.
3. Pin the GPT checkpoint and every comparison-model revision.
4. Validate the 11-action mapping, phase mask, and Tokyo isolation tests.
5. Use a committed configuration and fixed seed set.
6. Mark smoke-test results as exploratory.

Exact commands and dependencies should be added only after the current codebase and environment files are inspected. Do not invent setup commands in documentation.

## Minimum fair comparison

| Model role | Purpose |
|---|---|
| `gpt_classical` | Same selected GPT with its classical task head |
| `gpt_classical_twin` | Same GPT with a small classical bottleneck |
| `gpt_quantum` | Same GPT with the proposed VQC component |
| `modern_<name>` | Modern downloadable LM using the shared task pipeline |

All headline variants must use the same eligible samples, labels, masks, metrics, seed policy, and checkpoint-selection protocol.

## Reproducibility

Each reportable run must record the git commit, dataset/split version, model ID and revision, model role, seed, resolved training configuration, LoRA configuration, quantum configuration, checkpoint rule, hardware, and software versions.

Results without a complete manifest are exploratory and must not enter the final comparison table.

## Current status

The architecture and guardrails are defined. Qwen3.5-4B-Base is the current exploratory GPT-style parent candidate, and the Qiskit-backed head path has passed smoke and one-seed development pilot checks. Increasing the quantum head from 4 to 8 qubits fixed the original `BW_DOWN` collapse, but phase-balanced loss, depth 3, and the richer RY+RZ ansatz did not improve quantum `BW_UP`.

The tutor-approved CRUISE-80 diagnostic is complete for the classical, parameter-matched classical-twin, and q8/depth-2 RY quantum heads. All three used the same 2,400-window training design and unchanged 600-window validation set. Classical reached 25.78% validation UP accuracy; classical-twin and quantum each reached 16.99% and predicted action 7 for every validation UP position. The replacement-heavy design therefore did not solve UP generalisation.

Follow-up trainability and saturation smokes found finite, non-zero Qiskit gradients but severe bounded-encoding saturation. Adding head-input LayerNorm and temperature 4 reduced measured saturation to 0% for both bottleneck paths and increased the quantum circuit-gradient signal, but did not yet prevent UP collapse within 25 training steps. Reducing the angle range from pi to pi/2 added no early benefit. Saturation is therefore a confirmed optimization problem, but not a sufficient explanation for UP failure. These remain one-seed development diagnostics, not Tokyo or reportable final results. Tokyo remains unused. See `docs/CONTEXT.md` and `docs/NEXT_STEPS_README.md`.
