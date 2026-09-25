# Quantum-Enhanced GPT for BBR over Starlink

Research code testing whether adding a Qiskit-backed trainable variational
quantum circuit (VQC) to the action head of a Small Language Model (SLM)
changes its ability to predict BBR pacing-gain actions, compared with the
same SLM using a normal classical head.

## Research scope (simplified, current)

The primary question is:

> Does attaching a quantum action head to one SLM change its BBR pacing-gain
> prediction ability compared with the same SLM's classical head?

Only two model roles are compared, on the same fixed backbone:

- `gpt_classical` — `LFM2.5-350M` with a normal linear action head;
- `gpt_quantum` — the same `LFM2.5-350M` with a Qiskit VQC action head.

`LFM2.5-350M` was selected from four downloadable under-400M checkpoints
using development-only validation metrics (never Tokyo) — see
[ADR-0017](docs/adr/ADR-0017-under400m-quantum-gpt-parent-and-protocol-freeze.md).
Everything else (data, State Encoder, LoRA, loss, phase mask, split, seed) is
identical between the two roles; only the head differs.

The original networking task remains fixed:

- exactly 11 phase-constrained pacing-gain actions;
- shared preprocessing, labels, loss, masks, and metrics;
- Ohio, Sao Paulo, London, Mumbai, and Sydney for training/validation;
- Tokyo reserved for held-out generalisation testing.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full research
contract and mandatory coding-agent guardrails.

## Start here for a new AI agent

1. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — binding research contract.
2. [`docs/CONTEXT.md`](docs/CONTEXT.md) — current state, completed work, and next milestone.
3. [`docs/SKILL.md`](docs/SKILL.md) — mandatory agent workflow and naming rules.
4. [`docs/adr/ADR-0017-under400m-quantum-gpt-parent-and-protocol-freeze.md`](docs/adr/ADR-0017-under400m-quantum-gpt-parent-and-protocol-freeze.md) — the current frozen parent/protocol decision, and the current git diff.

Conversation history is not an experiment specification. If a request
conflicts with the checked-in contract or an ADR, stop and ask the user which
durable source should be changed.

## `archive/`

Earlier exploratory directions (comparing four modern backbones against the
original SLM-BBR paper's GPT-2/T5/GPT-Neo/SmolLM2 results, and an extensive
Qwen3.5-4B-Base quantum-head ablation trail with a 3-role
classical/classical-twin/quantum comparison) are preserved under `archive/`
(`archive/docs/`, `archive/docs/adr/`, `archive/scripts/`,
`archive/reports/figures/`) for provenance — including why `LFM2.5-350M` was
selected as the current backbone, and prior quantum-circuit findings (e.g. the
persistent `BW_UP` collapse). They are historical context, not the current
research direction. Do not build new work on top of them without checking
they still apply.

## Repository map

```text
lm-bbr-starlink/
|-- AGENTS.md                # automatic entrypoint for coding-agent sessions
|-- plm_special/             # GPT adapters, classical control, VQC integration
|-- tcp-cc-starlink-main/    # upstream Starlink code/data reference
|-- utils/                   # shared data, masks, metrics, manifests, seeding
|-- config.py                # central experiment configuration
|-- train_modern_lora.py     # training/evaluation entry point (classical + quantum heads)
|-- README.md                # project entry point
|-- docs/                    # architecture, context, ADRs, current guides
|-- docs/ARCHITECTURE.md     # authoritative research and architecture contract
|-- docs/CONTEXT.md          # current decisions, assumptions, and open questions
|-- docs/SKILL.md            # workflow instructions for coding agents
|-- reports/figures/under400m/ # current charts (loss/accuracy, Tokyo box plots)
`-- archive/                 # superseded exploratory directions (see above)
```

Agents must inspect the actual files before modifying or importing them.

## Model design

```text
BBR state/return sequence
  -> LFM2.5-350M backbone (LoRA)
  -> action-position hidden state
  -> Linear(d_model, n_qubits)
  -> bounded angle encoding and Qiskit VQC
  -> Pauli-Z measurements
  -> Linear(n_qubits, 11)
  -> BBR phase mask
  -> pacing-gain action
```

The VQC uses Qiskit Machine Learning's `EstimatorQNN` through `TorchConnector`,
backed by Qiskit's local `StatevectorEstimator` (analytic, `default_precision=0.0`,
no finite shots or quantum hardware). Qubit count and circuit depth are
configurable via `--n-qubits` / `--quantum-depth` on `train_modern_lora.py`;
see `config.py`'s `quantum_defaults` for the current default and
[ADR-0017](docs/adr/ADR-0017-under400m-quantum-gpt-parent-and-protocol-freeze.md)
for the frozen protocol.

## Dataset handling

Preserve downloaded traces as immutable raw data. Derived samples must be
deterministic and written separately. Do not commit large raw data, processed
data, model weights, caches, secrets, or local environments.

## Reproducibility

Each reportable run must record the git commit, dataset/split version, model
ID and revision, model role, seed, resolved training configuration, LoRA
configuration, quantum configuration (when applicable), checkpoint rule,
hardware, and software versions. Results without a complete manifest are
exploratory and must not enter the final comparison table.

## Current status

`gpt_classical` (`LFM2.5-350M`, classical head) is trained and frozen at
epoch 16, Tokyo-evaluated. `gpt_quantum` (same backbone, Qiskit VQC head) is
in progress — see `docs/CONTEXT.md` for the latest circuit configuration and
training status.
