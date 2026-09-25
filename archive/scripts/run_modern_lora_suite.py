"""Plan or execute the shared Phase 3 modern LoRA suite sequentially."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from config import cfg


MODEL_ORDER = (
    "lfm2_5_2_6b",
    "llama_3_2_3b",
    "qwen3_5_4b_base",
    "gemma_3_4b_pt",
    "olmo_3_1025_7b",
)


def output_name(model_key, rank, epochs, seed):
    return "{}_rank{}_epochs{}_seed{}".format(model_key, rank, epochs, seed)


def build_command(args, model_key, output_dir):
    return [
        sys.executable,
        "-u",
        "train_modern_lora.py",
        "--model-key", model_key,
        "--split-dir", str(args.split_dir),
        "--output-dir", str(output_dir),
        "--device", args.device,
        "--dtype", cfg.modern_model_registry[model_key]["preferred_dtype"],
        "--rank", str(args.rank),
        "--alpha", str(args.alpha),
        "--dropout", str(args.dropout),
        "--epochs", str(args.epochs),
        "--sequence-length", str(args.sequence_length),
        "--sample-step", str(args.sample_step),
        "--state-feature-dim", str(args.state_feature_dim),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--grad-accum-steps", str(args.grad_accum_steps),
        "--seed", str(args.seed),
        "--run-purpose", "phase3_exploratory",
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
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for model_key in MODEL_ORDER:
        run_dir = args.output_root / output_name(
            model_key, args.rank, args.epochs, args.seed
        )
        entries.append(
            {
                "model_key": model_key,
                "model_id": cfg.modern_model_registry[model_key]["hf_id"],
                "revision": cfg.modern_model_registry[model_key]["revision"],
                "dtype": cfg.modern_model_registry[model_key]["preferred_dtype"],
                "output_dir": str(run_dir),
                "command": build_command(args, model_key, run_dir),
            }
        )
    plan = {
        "status": "planned" if not args.execute else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "phase3_exploratory_modern_lora",
        "reportable_result": False,
        "held_out_location": "Tokyo",
        "models": entries,
    }
    plan_path = args.output_root / "suite.plan.json"
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
    write_json(plan_path, plan)
    summary_command = [
        sys.executable,
        "summarize_modern_lora.py",
        "--run-root",
        str(args.output_root),
    ]
    subprocess.run(summary_command, check=True)


if __name__ == "__main__":
    main()
