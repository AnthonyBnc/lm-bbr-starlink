"""Plan or execute Qwen classical/classical-twin/quantum head ablations."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from config import cfg


HEAD_ORDER = ("classical", "classical_twin", "quantum")
HEAD_OUTPUT_NAMES = {
    "classical": "qwen_gpt_classical",
    "classical_twin": "qwen_gpt_classical_twin",
    "quantum": "qwen_gpt_quantum",
}


def output_name(head_type, rank, epochs, seed, limited):
    suffix = "smoke" if limited else "pilot"
    return "{}_rank{}_epochs{}_seed{}_{}".format(
        HEAD_OUTPUT_NAMES[head_type], rank, epochs, seed, suffix
    )


def build_command(args, head_type, output_dir):
    model_info = cfg.modern_model_registry[args.model_key]
    command = [
        sys.executable,
        "-u",
        "train_modern_lora.py",
        "--model-key",
        args.model_key,
        "--split-dir",
        str(args.split_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--dtype",
        model_info["preferred_dtype"],
        "--rank",
        str(args.rank),
        "--alpha",
        str(args.alpha),
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.epochs),
        "--sequence-length",
        str(args.sequence_length),
        "--sample-step",
        str(args.sample_step),
        "--state-feature-dim",
        str(args.state_feature_dim),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--seed",
        str(args.seed),
        "--loss-weighting",
        args.loss_weighting,
        "--training-sampling-strategy",
        args.training_sampling_strategy,
        "--target-train-windows",
        str(args.target_train_windows),
        "--up-density-power",
        str(args.up_density_power),
        "--down-window-penalty",
        str(args.down_window_penalty),
        "--replacement-train-windows",
        str(args.replacement_train_windows),
        "--head-type",
        head_type,
        "--n-qubits",
        str(args.n_qubits),
        "--quantum-depth",
        str(args.quantum_depth),
        "--quantum-ansatz",
        args.quantum_ansatz,
        "--run-purpose",
        "head_ablation_smoke" if args.limited else "head_ablation_pilot",
    ]
    if args.max_train_steps:
        command.extend(["--max-train-steps", str(args.max_train_steps)])
    if args.max_validation_steps:
        command.extend(["--max-validation-steps", str(args.max_validation_steps)])
    if args.training_sampling_plan is not None:
        command.extend(["--training-sampling-plan", str(args.training_sampling_plan)])
    return command


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_command(command, log_path):
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
            log_stream.flush()
        return process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default=cfg.quantum_defaults["parent_model_key"])
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/lora_training/qwen_head_ablation_smoke"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--sample-step", type=int, default=20)
    parser.add_argument("--state-feature-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument("--n-qubits", type=int, default=cfg.quantum_defaults["n_qubits"])
    parser.add_argument("--quantum-depth", type=int, default=cfg.quantum_defaults["depth"])
    parser.add_argument(
        "--quantum-ansatz",
        choices=("trainable_ry_layers", "trainable_ry_rz_layers"),
        default=cfg.quantum_defaults.get("ansatz", "trainable_ry_layers"),
    )
    parser.add_argument("--max-train-steps", type=int, default=1)
    parser.add_argument("--max-validation-steps", type=int, default=1)
    parser.add_argument(
        "--loss-weighting",
        choices=("none", "phase_balanced"),
        default="none",
    )
    parser.add_argument(
        "--training-sampling-strategy",
        choices=(
            "original",
            "up_action_balanced_replacement",
            "cruise80_location_up_action_balanced_replacement",
        ),
        default="original",
    )
    parser.add_argument("--target-train-windows", type=int, default=0)
    parser.add_argument("--up-density-power", type=float, default=3.0)
    parser.add_argument("--down-window-penalty", type=float, default=1.0)
    parser.add_argument("--replacement-train-windows", type=int, default=0)
    parser.add_argument("--training-sampling-plan", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()

    if args.model_key not in cfg.modern_model_registry:
        parser.error("Unknown model key: {}".format(args.model_key))
    args.limited = bool(args.max_train_steps or args.max_validation_steps)

    args.output_root.mkdir(parents=True, exist_ok=True)
    model_info = cfg.modern_model_registry[args.model_key]
    entries = []
    for head_type in HEAD_ORDER:
        run_dir = args.output_root / output_name(
            head_type, args.rank, args.epochs, args.seed, args.limited
        )
        entries.append(
            {
                "head_type": head_type,
                "model_key": args.model_key,
                "model_id": model_info["hf_id"],
                "revision": model_info["revision"],
                "dtype": model_info["preferred_dtype"],
                "output_dir": str(run_dir),
                "command": build_command(args, head_type, run_dir),
            }
        )

    plan = {
        "status": "planned" if not args.execute else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "qwen_head_ablation",
        "reportable_result": False,
        "held_out_location": "Tokyo",
        "limited_smoke": args.limited,
        "selection_metric": "validation_macro_phase_accuracy",
        "tie_breaker": "validation_loss",
        "training_loss_weighting": args.loss_weighting,
        "training_sampling_strategy": args.training_sampling_strategy,
        "training_sampling_plan": (
            str(args.training_sampling_plan)
            if args.training_sampling_plan is not None
            else None
        ),
        "quantum_ansatz": args.quantum_ansatz,
        "heads": entries,
    }
    plan_path = args.output_root / "head_ablation.plan.json"
    write_json(plan_path, plan)

    if not args.execute:
        for entry in entries:
            print(" ".join(entry["command"]))
        print("Dry run only. Add --execute to start sequential head runs.")
        print("Plan: {}".format(plan_path))
        return

    for entry in entries:
        run_dir = Path(entry["output_dir"])
        manifest_path = run_dir / "run.manifest.json"
        if manifest_path.is_file() and args.resume_completed:
            entry["status"] = "skipped_completed"
            write_json(plan_path, plan)
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(
                "Refusing to overwrite non-empty run directory: {}".format(run_dir)
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        entry["status"] = "running"
        write_json(plan_path, plan)
        return_code = run_command(entry["command"], run_dir / "training.log")
        if return_code != 0:
            entry["status"] = "failed"
            entry["return_code"] = return_code
            plan["status"] = "failed"
            write_json(plan_path, plan)
            raise SystemExit(return_code)
        if not manifest_path.is_file():
            raise RuntimeError("Training succeeded without a run manifest")
        entry["status"] = "completed"
        write_json(plan_path, plan)

    plan["status"] = "completed"
    plan["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(plan_path, plan)
    if not args.limited:
        subprocess.run(
            [
                sys.executable,
                "summarize_head_ablation.py",
                "--run-root",
                str(args.output_root),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
