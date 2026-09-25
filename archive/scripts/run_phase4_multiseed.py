"""Plan or execute the Phase 4 multi-seed development run for one model."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from config import cfg


DEFAULT_SEEDS = (100003, 100019, 100043)


def output_name(model_key, rank, epochs, seed):
    return "{}_rank{}_epochs{}_seed{}".format(model_key, rank, epochs, seed)


def build_command(args, seed, output_dir):
    model_info = cfg.modern_model_registry[args.model_key]
    return [
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
        str(seed),
        "--run-purpose",
        "phase4_dev_multiseed",
    ]


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
    parser.add_argument("--model-key", default="qwen3_5_4b_base")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/lora_training/phase4_qwen_multiseed_rank8_epochs5"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--sample-step", type=int, default=20)
    parser.add_argument("--state-feature-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()

    if args.model_key not in cfg.modern_model_registry:
        parser.error("Unknown model key: {}".format(args.model_key))
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be unique")

    args.output_root.mkdir(parents=True, exist_ok=True)
    model_info = cfg.modern_model_registry[args.model_key]
    entries = []
    for seed in args.seeds:
        run_dir = args.output_root / output_name(
            args.model_key, args.rank, args.epochs, seed
        )
        entries.append(
            {
                "seed": seed,
                "model_key": args.model_key,
                "model_id": model_info["hf_id"],
                "revision": model_info["revision"],
                "dtype": model_info["preferred_dtype"],
                "output_dir": str(run_dir),
                "command": build_command(args, seed, run_dir),
            }
        )

    plan = {
        "status": "planned" if not args.execute else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "phase4_development_multiseed",
        "reportable_result": False,
        "held_out_location": "Tokyo",
        "selection_metric": "validation_macro_phase_accuracy",
        "tie_breaker": "validation_loss",
        "models": entries,
    }
    plan_path = args.output_root / "phase4.plan.json"
    write_json(plan_path, plan)

    if not args.execute:
        for entry in entries:
            print(" ".join(entry["command"]))
        print("Dry run only. Add --execute to start sequential training.")
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
    summary_command = [
        sys.executable,
        "summarize_phase4_multiseed.py",
        "--run-root",
        str(args.output_root),
    ]
    subprocess.run(summary_command, check=True)


if __name__ == "__main__":
    main()
