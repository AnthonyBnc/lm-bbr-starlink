"""Plan or execute short bottleneck-saturation diagnostics on frozen data."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


CONFIGS = (
    ("twin_ln_t4", "classical_twin", True, "4.0", "pi"),
    ("quantum_ln_t1_pi", "quantum", True, "1.0", "pi"),
    ("quantum_ln_t4_pi", "quantum", True, "4.0", "pi"),
    ("quantum_ln_t4_half_pi", "quantum", True, "4.0", "half_pi"),
)


def build_command(args, name, head_type, layernorm, temperature, angle_scale):
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
        "--quantum-angle-scale", angle_scale,
        "--bottleneck-temperature", temperature,
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
        "--max-train-steps", str(args.train_steps),
        "--max-validation-steps", str(args.validation_steps),
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
    if layernorm:
        command.append("--head-input-layernorm")
    return output_dir, command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/lora_training/saturation_ablation_steps25"),
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
    parser.add_argument("--train-steps", type=int, default=25)
    parser.add_argument("--validation-steps", type=int, default=50)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()
    if args.train_steps < 1 or args.validation_steps < 1:
        parser.error("train-steps and validation-steps must be positive")
    if not args.sampling_plan.is_file():
        parser.error("Sampling plan not found: {}".format(args.sampling_plan))

    args.output_root.mkdir(parents=True, exist_ok=True)
    commands = []
    for config in CONFIGS:
        output_dir, command = build_command(args, *config)
        commands.append({"name": config[0], "output_dir": str(output_dir), "command": command})
    plan_path = args.output_root / "saturation_ablation.plan.json"
    plan_path.write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")

    for item in commands:
        output_dir = Path(item["output_dir"])
        manifest = output_dir / "run.manifest.json"
        if args.resume_completed and manifest.is_file():
            print("Skipping completed {}".format(item["name"]))
            continue
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError("Refusing to overwrite non-empty run directory: {}".format(output_dir))
        print("Command: {}".format(" ".join(item["command"])))
        if args.execute:
            subprocess.run(item["command"], check=True)

    if not args.execute:
        print("Dry run only. Add --execute to run the four short diagnostics.")
    print("Plan: {}".format(plan_path))


if __name__ == "__main__":
    main()
