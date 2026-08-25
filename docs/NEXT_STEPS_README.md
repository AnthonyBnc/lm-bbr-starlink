# Next Steps

The previous 1,500-window quantum-only plan is superseded by tutor feedback.
Do not run `run_quantum_up_focus.py` for the next comparison.

The fair 2,400-window CRUISE-80 experiment has now completed. Do not rerun it
unless reproducing the recorded result or adding an approved seed.

## Shared Data Design

The completed experiment kept the original training budget of 2,400 windows
and used one index-only design for all three Qwen heads:

| Item | Value |
|---|---:|
| Training windows | 2,400 |
| Unique / repeated windows | 835 / 1,565 |
| CRUISE positions | 79.365% |
| UP positions | 14.246% |
| DOWN positions | 6.390% |
| Validation windows | 600, unchanged |

UP actions 6-10 contain 1,364, 1,367, 1,382, 1,361, and 1,364 positions.
London, Mumbai, Ohio, Sao Paulo, and Sydney each contribute 480 training
windows. Source pools, labels, rewards, actions, phase masks, split membership,
and Tokyo isolation remain unchanged.

The frozen design is written to:

```text
data/processed/lora_training/qwen_head_ablation_cruise80_up_balanced_q8_d2/training_sampling.design.json
```

## Completed Result

All three runs passed checkpoint reload and Tokyo-isolation checks:

| Head | Validation accuracy | Macro-phase accuracy | DOWN | CRUISE | UP |
|---|---:|---:|---:|---:|---:|
| Classical | 94.87% | 75.26% | 100% | 100% | 25.78% |
| Classical twin | 94.26% | 72.33% | 100% | 100% | 16.99% |
| Quantum q8/d2 RY | 94.26% | 72.33% | 100% | 100% | 16.99% |

Classical-twin and quantum predicted action 7 (`1.10`) for all 830 validation
UP positions. This is an UP-action collapse, despite the near-balanced UP
action counts in training. Classical produced more diverse UP predictions but
still reached only 25.78% UP accuracy.

Compared with the original-distribution q8/d2 pilot, validation UP accuracy
changed from 28.80% to 25.78% for classical, 27.59% to 16.99% for
classical-twin, and 21.45% to 16.99% for quantum. The design repeated 1,565 of
2,400 selected windows and did not improve validation generalisation.

The audited summary is:

```text
data/processed/lora_training/qwen_head_ablation_cruise80_up_balanced_q8_d2/head_ablation.summary.json
```

## Next Controlled Diagnostic

Do not increase depth, add another sampling scheme, or access Tokyo next.
First test whether the bottleneck and Qiskit heads are trainable from their
current initialization. The runner now supports `--trainability-diagnostics`,
which records component gradient norms, bottleneck variation, and `tanh`/angle
saturation in `metrics.json`.

1. run the 25-step classical-twin diagnostic below;
2. run the equivalent q8/depth-2 quantum diagnostic;
3. compare component gradients, bottleneck standard deviation, saturation, and
   UP prediction distributions;
4. only if the quantum circuit is demonstrably receiving useful gradients, add
   a seed-controlled small non-zero initialization as a separate ablation.

Both commands must use the frozen CRUISE-80 design. Their output directories
must be new and must not replace completed pilot artifacts.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u train_modern_lora.py \
  --model-key qwen3_5_4b_base --device mps --dtype bfloat16 \
  --output-dir data/processed/lora_training/trainability_diag_cruise80_twin_q8_steps25 \
  --head-type classical_twin --n-qubits 8 --quantum-depth 2 \
  --rank 8 --alpha 32 --dropout 0.05 --epochs 1 \
  --sequence-length 20 --sample-step 20 --grad-accum-steps 1 \
  --seed 100003 --learning-rate 1e-4 --weight-decay 1e-4 \
  --max-train-steps 25 --max-validation-steps 50 \
  --training-sampling-strategy cruise80_location_up_action_balanced_replacement \
  --target-train-windows 2400 --up-density-power 12 \
  --down-window-penalty 10 --replacement-train-windows 1600 \
  --training-sampling-plan data/processed/lora_training/qwen_head_ablation_cruise80_up_balanced_q8_d2/training_sampling.design.json \
  --trainability-diagnostics
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u train_modern_lora.py \
  --model-key qwen3_5_4b_base --device mps --dtype bfloat16 \
  --output-dir data/processed/lora_training/trainability_diag_cruise80_quantum_q8_d2_steps25 \
  --head-type quantum --n-qubits 8 --quantum-depth 2 \
  --quantum-ansatz trainable_ry_layers \
  --rank 8 --alpha 32 --dropout 0.05 --epochs 1 \
  --sequence-length 20 --sample-step 20 --grad-accum-steps 1 \
  --seed 100003 --learning-rate 1e-4 --weight-decay 1e-4 \
  --max-train-steps 25 --max-validation-steps 50 \
  --training-sampling-strategy cruise80_location_up_action_balanced_replacement \
  --target-train-windows 2400 --up-density-power 12 \
  --down-window-penalty 10 --replacement-train-windows 1600 \
  --training-sampling-plan data/processed/lora_training/qwen_head_ablation_cruise80_up_balanced_q8_d2/training_sampling.design.json \
  --trainability-diagnostics
```

This is an architecture/optimization diagnostic, not permission to give the
quantum model different labels, validation data, or Tokyo information. If one
initialization appears better, confirm it over the predefined seed set before
making a quantum-advantage claim.

## Saturation Ablation

The next short experiment is implemented by `run_saturation_ablation.py`. It
preserves the old encoding as the control and adds named options for head-input
LayerNorm, fixed pre-`tanh` temperature, and quantum angle scale. It runs four
limited configurations on the same frozen CRUISE-80 plan:

| Name | Head | LayerNorm | Temperature | Angle scale |
|---|---|---:|---:|---:|
| `twin_ln_t4` | Classical twin | yes | 4 | n/a |
| `quantum_ln_t1_pi` | Quantum | yes | 1 | pi |
| `quantum_ln_t4_pi` | Quantum | yes | 4 | pi |
| `quantum_ln_t4_half_pi` | Quantum | yes | 4 | pi/2 |

Run the dry plan first, then execute:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u run_saturation_ablation.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u run_saturation_ablation.py \
  --execute 2>&1 | tee data/processed/lora_training/saturation_ablation_steps25_terminal.log
```

Use `--resume-completed --execute` after an interruption. These are 25-step
diagnostics, not full training or reportable model comparisons. Select no full
run until the saturation, gradients, and UP prediction diversity are reviewed.

### Completed outcome

All four configurations completed with checkpoint reload and Tokyo isolation
PASS. The 50-window validation slice contained 37 UP positions.

| Configuration | Validation saturation | Quantum gradient mean | UP accuracy | UP prediction behavior |
|---|---:|---:|---:|---|
| Previous quantum control | 100% | 0.00476 | 16.22% | one UP action |
| Quantum LN, T1, pi | 72.99% | 0.02839 | 16.22% | one UP action |
| Quantum LN, T4, pi | 0% | 0.03228 | 10.81% | one UP action |
| Quantum LN, T4, pi/2 | 0% | 0.02365 | 10.81% | one UP action |
| Classical twin LN, T4 | 0% | n/a | 16.22% | two UP actions |

LayerNorm plus temperature 4 successfully removes measured saturation, and
the quantum circuit still receives finite non-zero gradients. However, the
25-step quantum predictions remain collapsed. Saturation is therefore a real
trainability issue but is not, by itself, a complete explanation for UP
failure. The pi/2 angle range has no early advantage over pi and should not be
advanced.

The next gate is a medium-length development diagnostic using only the most
informative normalized configuration, quantum LN/T4/pi, together with the
classical-twin LN/T4 control. Do not launch a full 2,400-step run yet. First
confirm whether 200 steps produce UP prediction diversity while saturation
remains low. If collapse remains, diagnose UP-class separability at the head
input/bottleneck instead of adding more epochs or circuit depth.
