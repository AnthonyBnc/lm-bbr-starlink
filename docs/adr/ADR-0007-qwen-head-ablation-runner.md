# ADR-0007: Qwen Head-Ablation Runner

## Status

Proposed for development validation on 2026-08-12.

## Context

Quantum-GPT must be compared with controls that use the same GPT-style parent,
same hidden-state extraction point, same LoRA policy, same data split, same
labels, same masks, and same metrics. The current parent candidate is
Qwen3.5-4B-Base, pending Phase 4 confirmation.

## Decision

Add a Qwen head-ablation runner with three explicit heads:

- `classical`: normal Qwen classical action head;
- `classical_twin`: Qwen with a small classical bottleneck;
- `quantum`: Qwen with the 4-qubit, depth-2 Qiskit `EstimatorQNN` VQC head.

The runner is dry-run by default and writes a plan before execution. The default
execution mode is a limited smoke check with one train batch and one validation
batch per head. Full summaries are only generated for unlimited runs, because
limited smoke checks are not model-quality comparisons.

## Consequences

- The three heads are routed through the same `train_modern_lora.py` path.
- Manifests record `model_role`, `head_type`, `head_config`, and
  `quantum_config`.
- The head-ablation audit requires one model/revision, one seed, one shared
  protocol, and all three head types.
- Tokyo remains held out.

## Commands

Dry-run the smoke plan:

```bash
.venv/bin/python run_qwen_head_ablation.py
```

Execute the smoke plan:

```bash
.venv/bin/python -u run_qwen_head_ablation.py --execute
```

Resume without repeating completed head runs:

```bash
.venv/bin/python -u run_qwen_head_ablation.py --execute --resume-completed
```

Plan an unlimited one-epoch pilot:

```bash
.venv/bin/python run_qwen_head_ablation.py \
  --output-root data/processed/lora_training/qwen_head_ablation_pilot_epoch1 \
  --max-train-steps 0 \
  --max-validation-steps 0
```
