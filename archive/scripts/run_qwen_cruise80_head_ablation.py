"""Prepare and run the fair 2,400-window CRUISE-80 Qwen head ablation."""

import argparse
from pathlib import Path
import subprocess
import sys

from prepare_quantum_up_focus_data import build_data_design, write_data_design
from utils.training_sampling import TRAINING_SAMPLING_CRUISE80


def build_command(args, design_path):
    command = [
        sys.executable,
        "-u",
        "run_qwen_head_ablation.py",
        "--output-root",
        str(args.output_root),
        "--split-dir",
        str(args.split_dir),
        "--device",
        args.device,
        "--rank",
        "8",
        "--alpha",
        "32",
        "--dropout",
        "0.05",
        "--epochs",
        "1",
        "--sequence-length",
        "20",
        "--sample-step",
        "20",
        "--grad-accum-steps",
        "1",
        "--seed",
        str(args.seed),
        "--n-qubits",
        "8",
        "--quantum-depth",
        "2",
        "--quantum-ansatz",
        "trainable_ry_layers",
        "--max-train-steps",
        "0",
        "--max-validation-steps",
        "0",
        "--loss-weighting",
        "none",
        "--training-sampling-strategy",
        TRAINING_SAMPLING_CRUISE80,
        "--target-train-windows",
        "2400",
        "--up-density-power",
        "12.0",
        "--down-window-penalty",
        "10.0",
        "--replacement-train-windows",
        "1600",
        "--training-sampling-plan",
        str(design_path),
    ]
    if args.execute:
        command.append("--execute")
    if args.resume_completed:
        command.append("--resume-completed")
    return command


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
        default=Path(
            "data/processed/lora_training/qwen_head_ablation_cruise80_up_balanced_q8_d2"
        ),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    design_path = args.output_root / "training_sampling.design.json"
    design = build_data_design(
        args.split_dir,
        train_windows=2400,
        seed=args.seed,
        sequence_length=20,
        sample_step=20,
        up_density_power=12.0,
        down_penalty=10.0,
        replacement_windows=1600,
        sampling_strategy=TRAINING_SAMPLING_CRUISE80,
    )
    write_data_design(design_path, design)
    command = build_command(args, design_path)
    print("Data design: {}".format(design_path))
    print("Command: {}".format(" ".join(command)))
    if not args.execute:
        print("Dry run only. Add --execute to train classical, classical-twin, then quantum.")
        return
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
