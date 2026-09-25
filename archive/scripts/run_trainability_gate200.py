"""Run the 200-step normalized twin/quantum trainability gate sequentially."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


GATE_CONFIGS = (
    ("twin_ln_t4", "classical_twin"),
    ("quantum_ln_t4_pi", "quantum"),
)


def build_command(args, name, head_type):
    output_dir = args.output_root / name
    command = [
        sys.executable,
        "-u",
        "train_modern_lora.py",
        "--model-key", "qwen3_5_4b_base",
        "--split-dir", str(args.split_dir),
        "--output-dir", str(output_dir),
        "--device", args.device,
        "--dtype", "bfloat16",
        "--head-type", head_type,
        "--n-qubits", "8",
        "--quantum-depth", "2",
        "--quantum-ansatz", "trainable_ry_layers",
        "--head-input-layernorm",
        "--bottleneck-temperature", "4.0",
        "--quantum-angle-scale", "pi",
        "--rank", "8",
        "--alpha", "32",
        "--dropout", "0.05",
        "--epochs", "1",
        "--sequence-length", "20",
        "--sample-step", "20",
        "--grad-accum-steps", "1",
        "--seed", str(args.seed),
        "--learning-rate", "1e-4",
        "--weight-decay", "1e-4",
        "--max-train-steps", "200",
        "--max-validation-steps", "200",
        "--training-sampling-strategy",
        "cruise80_location_up_action_balanced_replacement",
        "--target-train-windows", "2400",
        "--up-density-power", "12",
        "--down-window-penalty", "10",
        "--replacement-train-windows", "1600",
        "--training-sampling-plan", str(args.sampling_plan),
        "--trainability-diagnostics",
        "--run-purpose", "head_ablation_smoke",
    ]
    return output_dir, command


def completed_run(output_dir):
    manifest_path = output_dir / "run.manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("checkpoint_reload") == "PASS"
        and manifest.get("tokyo_isolation") == "PASS"
        and manifest.get("max_train_steps") == 200
        and manifest.get("max_validation_steps") == 200
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/lora_training/trainability_gate200_ln_t4"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003"),
    )
    parser.add_argument(
        "--sampling-plan",
        type=Path,
        default=Path(
            "data/processed/lora_training/"
            "qwen_head_ablation_cruise80_up_balanced_q8_d2/"
            "training_sampling.design.json"
        ),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()
    if not args.sampling_plan.is_file():
        parser.error("Sampling plan not found: {}".format(args.sampling_plan))

    args.output_root.mkdir(parents=True, exist_ok=True)
    planned = []
    for config in GATE_CONFIGS:
        output_dir, command = build_command(args, *config)
        planned.append({"name": config[0], "output_dir": str(output_dir), "command": command})
    plan_path = args.output_root / "trainability_gate200.plan.json"
    plan_path.write_text(json.dumps(planned, indent=2) + "\n", encoding="utf-8")

    for item in planned:
        output_dir = Path(item["output_dir"])
        if args.resume_completed and completed_run(output_dir):
            print("Skipping completed {}".format(item["name"]))
            continue
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError("Refusing to overwrite non-empty run directory: {}".format(output_dir))
        print("Command: {}".format(" ".join(item["command"])), flush=True)
        if args.execute:
            subprocess.run(item["command"], check=True)

    if not args.execute:
        print("Dry run only. Add --execute to run twin and then quantum.")
    print("Plan: {}".format(plan_path))


if __name__ == "__main__":
    main()
