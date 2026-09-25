"""Plan or run the q8/d2 Quantum-GPT UP-focused development diagnostic."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from prepare_quantum_up_focus_data import build_data_design, write_data_design
from utils.training_sampling import TRAINING_SAMPLING_UP_FOCUS


DEFAULT_REFERENCE_ROOT = Path(
    "data/processed/lora_training/qwen_head_ablation_pilot_epoch1_qiskit_q8_d2"
)


def build_command(args, output_dir, sampling_plan=None):
    command = [
        sys.executable,
        "-u",
        "train_modern_lora.py",
        "--model-key",
        "qwen3_5_4b_base",
        "--split-dir",
        str(args.split_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--dtype",
        "bfloat16",
        "--rank",
        "8",
        "--alpha",
        "32",
        "--dropout",
        "0.05",
        "--epochs",
        str(args.epochs),
        "--sequence-length",
        "20",
        "--sample-step",
        "20",
        "--state-feature-dim",
        "256",
        "--learning-rate",
        "0.0001",
        "--weight-decay",
        "0.0001",
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--seed",
        str(args.seed),
        "--loss-weighting",
        "none",
        "--head-type",
        "quantum",
        "--n-qubits",
        "8",
        "--quantum-depth",
        "2",
        "--quantum-ansatz",
        "trainable_ry_layers",
        "--training-sampling-strategy",
        "up_action_balanced_replacement",
        "--target-train-windows",
        str(args.train_windows),
        "--up-density-power",
        str(args.up_density_power),
        "--down-window-penalty",
        str(args.down_penalty),
        "--run-purpose",
        "quantum_up_focus",
    ]
    if sampling_plan is not None:
        command.extend(["--training-sampling-plan", str(sampling_plan)])
    return command


def _last_validation(metrics_path):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return metrics["epochs"][-1]["validation"]


def write_comparison(output_root, reference_root, quantum_run_dir):
    reference = json.loads(
        (reference_root / "head_ablation.summary.json").read_text(encoding="utf-8")
    )
    quantum_manifest = json.loads(
        (quantum_run_dir / "run.manifest.json").read_text(encoding="utf-8")
    )
    quantum_validation = _last_validation(quantum_run_dir / "metrics.json")
    reference_heads = [
        head for head in reference["heads"]
        if head["head_type"] in ("classical", "classical_twin")
    ]
    shared_validation = all(
        quantum_manifest[field] == reference["shared_protocol"][field]
        for field in (
            "validation_pool_sha256",
            "validation_sample_ids_sha256",
            "sequence_length",
            "sample_step",
        )
    )
    phase_values = [
        value for value in quantum_validation["per_phase_accuracy"].values()
        if value is not None
    ]
    result = {
        "status": "exploratory_quantum_up_focus_comparison",
        "reportable_result": False,
        "held_out_location": "Tokyo",
        "tokyo_used": False,
        "shared_validation_comparable": shared_validation,
        "fair_training_protocol_comparison": False,
        "comparison_note": (
            "All rows use the same 600-window validation set, but the new quantum "
            "run uses 1,500 UP-focused sampled training windows while the historical "
            "classical controls used all 2,400 original training windows."
        ),
        "historical_controls": reference_heads,
        "quantum_up_focus": {
            "head_type": "quantum",
            "validation_accuracy": quantum_validation["accuracy"],
            "validation_macro_phase_accuracy": sum(phase_values) / len(phase_values),
            "validation_per_phase_accuracy": quantum_validation["per_phase_accuracy"],
            "validation_loss": quantum_validation["loss"],
            "validation_label_distribution": quantum_validation["label_distribution"],
            "validation_prediction_distribution": quantum_validation["prediction_distribution"],
            "training_sampling": quantum_manifest["training_sampling"],
            "checkpoint_reload": quantum_manifest["checkpoint_reload"],
            "tokyo_isolation": quantum_manifest["tokyo_isolation"],
        },
    }
    path = output_root / "quantum_up_focus.comparison.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/lora_training/quantum_q8_d2_up_focus_1500"),
    )
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument("--train-windows", type=int, default=1500)
    parser.add_argument("--validation-windows", type=int, default=600)
    parser.add_argument("--requested-test-windows", type=int, default=700)
    parser.add_argument("--up-density-power", type=float, default=3.0)
    parser.add_argument("--down-penalty", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.validation_windows != 600:
        parser.error("The frozen validation dataset contains exactly 600 sequence windows")
    if args.requested_test_windows != 700:
        parser.error("This diagnostic records the requested 700-window future test gate")
    run_dir = args.output_root / "qwen_gpt_quantum_rank8_q8_d2_up_focus"
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error("Refusing to overwrite non-empty run directory: {}".format(run_dir))
    args.output_root.mkdir(parents=True, exist_ok=True)
    sampling_plan_path = args.output_root / "training_sampling.design.json"
    sampling_design = build_data_design(
        args.split_dir,
        train_windows=args.train_windows,
        seed=args.seed,
        sequence_length=20,
        sample_step=20,
        up_density_power=args.up_density_power,
        down_penalty=args.down_penalty,
        replacement_windows=0,
        sampling_strategy=TRAINING_SAMPLING_UP_FOCUS,
    )
    write_data_design(sampling_plan_path, sampling_design)
    command = build_command(args, run_dir, sampling_plan=sampling_plan_path)
    plan = {
        "status": "running" if args.execute else "planned",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "quantum_only_up_focused_development_diagnostic",
        "reportable_result": False,
        "quantum": {"n_qubits": 8, "depth": 2, "ansatz": "trainable_ry_layers"},
        "training_windows": args.train_windows,
        "training_sampling": {
            "strategy": "up_action_balanced_replacement",
            "up_density_power": args.up_density_power,
            "down_penalty": args.down_penalty,
            "design": str(sampling_plan_path),
            "selected_distribution": sampling_design["selected_distribution"],
        },
        "validation_windows": args.validation_windows,
        "requested_test_windows": args.requested_test_windows,
        "test_status": "deferred_until_an_independent_split_or_final_Tokyo_gate_is_approved",
        "historical_control_reference": str(args.reference_root),
        "comparison_limit": "training distributions differ; validation-only diagnostic comparison",
        "command": command,
    }
    plan_path = args.output_root / "quantum_up_focus.plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Plan: {}".format(plan_path))
    print("Command: {}".format(" ".join(command)))
    if not args.execute:
        return

    completed = subprocess.run(command).returncode
    if completed:
        raise SystemExit(completed)
    comparison_path = write_comparison(args.output_root, args.reference_root, run_dir)
    plan["status"] = "completed"
    plan["comparison"] = str(comparison_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Comparison: {}".format(comparison_path))


if __name__ == "__main__":
    main()
