"""Plan, preflight, or run SLM-BBR with a declared local-backbone set."""

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
)
UNDER_400M_MODEL_ORDER = (
    "granite_4_0_350m",
    "pleias_rag_350m",
    "lfm2_5_350m",
    "granite_4_0_h_350m",
)
GEMMA_3_270M_REPLACEMENT_ORDER = (
    "gemma_3_270m",
)
MODEL_SETS = {
    "approved-modern": MODEL_ORDER,
    "under-400m": UNDER_400M_MODEL_ORDER,
    "gemma-3-270m-replacement": GEMMA_3_270M_REPLACEMENT_ORDER,
}
OFFICIAL_SOURCE_COMMIT = "c0afba6521e62c09d4f558095fb83e577a1f7c80"


def output_name(model_key, rank, epochs, seed, preflight=False):
    suffix = "preflight" if preflight else "train"
    return "{}_rank{}_epochs{}_seed{}_{}".format(
        model_key, rank, epochs, seed, suffix
    )


def build_command(args, model_key, output_dir, resume_run=None):
    executed_epochs = args.epochs if resume_run is not None else (
        1 if args.preflight else args.epochs
    )
    command = [
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
        "--epochs", str(executed_epochs),
        "--sequence-length", str(args.sequence_length),
        "--sample-step", str(args.sample_step),
        "--state-feature-dim", str(args.state_feature_dim),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--warmup-steps", str(args.warmup_steps),
        "--grad-accum-steps", str(args.grad_accum_steps),
        "--seed", str(args.seed),
        "--training-sampling-strategy", "original",
        "--loss-weighting", "none",
        "--head-type", "classical",
        "--run-purpose", "slm_bbr_modern_backbone_extension",
    ]
    if resume_run is not None:
        command.extend(["--resume-from-run", str(resume_run)])
    if args.early_stopping_patience:
        command.extend(
            [
                "--early-stopping-patience",
                str(args.early_stopping_patience),
                "--early-stopping-min-delta",
                str(args.early_stopping_min_delta),
            ]
        )
    for option, value in (
        ("--success-overall-accuracy", args.success_overall_accuracy),
        ("--success-up-accuracy", args.success_up_accuracy),
        ("--success-macro-phase-accuracy", args.success_macro_phase_accuracy),
        ("--max-up-prediction-share", args.max_up_prediction_share),
    ):
        if value is not None:
            command.extend([option, str(value)])
    if args.preflight:
        command.extend(["--max-train-steps", "1", "--max-validation-steps", "1"])
    return command


def find_resume_run(root, model_key):
    matches = []
    for manifest_path in sorted(Path(root).glob("*/run.manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_key") == model_key:
            matches.append(manifest_path.parent)
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one resume run for {} under {}, found {}".format(
                model_key, root, len(matches)
            )
        )
    return matches[0]


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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/lora_training/slm_bbr_modern_backbones"),
    )
    parser.add_argument(
        "--model-set",
        choices=tuple(MODEL_SETS),
        default="approved-modern",
        help="Named, pinned backbone set to run sequentially.",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--sample-step", type=int, default=20)
    parser.add_argument("--state-feature-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument(
        "--resume-from-root",
        type=Path,
        help="Root containing one completed run manifest for each approved model.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--success-overall-accuracy", type=float, default=0.96)
    parser.add_argument(
        "--success-up-accuracy", type=float, default=0.42168674698795183
    )
    parser.add_argument(
        "--success-macro-phase-accuracy", type=float, default=0.8072289156626506
    )
    parser.add_argument("--max-up-prediction-share", type=float, default=0.90)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument(
        "--acknowledge-long-run",
        action="store_true",
        help="Required for any non-preflight training suite.",
    )
    args = parser.parse_args()

    if args.execute and not args.preflight and not args.acknowledge_long_run:
        parser.error("non-preflight execution requires --acknowledge-long-run")

    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_model_order = MODEL_SETS[args.model_set]
    entries = []
    for model_key in selected_model_order:
        run_dir = args.output_root / output_name(
            model_key, args.rank, args.epochs, args.seed, args.preflight
        )
        model_info = cfg.modern_model_registry[model_key]
        resume_run = (
            find_resume_run(args.resume_from_root, model_key)
            if args.resume_from_root
            else None
        )
        entries.append(
            {
                "model_key": model_key,
                "model_id": model_info["hf_id"],
                "revision": model_info["revision"],
                "dtype": model_info["preferred_dtype"],
                "resume_from_run": str(resume_run) if resume_run else None,
                "output_dir": str(run_dir),
                "command": build_command(args, model_key, run_dir, resume_run),
            }
        )

    plan = {
        "status": "planned" if not args.execute else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canonical_method": "Small Language Model-based Control for BBR over Low Earth Orbit Satellite Internet",
        "experiment_role": "modern_backbone_extension",
        "model_set": args.model_set,
        "model_keys": list(selected_model_order),
        "backbone_substitution_approved_by_user": True,
        "paper_reproduction_claim": False,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "held_out_location": "Tokyo",
        "reportable_result": False,
        "preflight": args.preflight,
        "protocol": {
            "epochs": args.epochs,
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "sequence_length": args.sequence_length,
            "sample_step": args.sample_step,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "gradient_accumulation_steps": args.grad_accum_steps,
            "gradient_clip_norm": 0.25,
            "training_sampling": "original",
            "loss": "phase-masked cross-entropy",
            "checkpoint_rule": "final epoch",
            "resume_from_root": (
                str(args.resume_from_root) if args.resume_from_root else None
            ),
            "epochs_semantics": "target_total_epochs",
            "validation_success_thresholds": {
                "overall_accuracy": args.success_overall_accuracy,
                "up_accuracy": args.success_up_accuracy,
                "macro_phase_accuracy": args.success_macro_phase_accuracy,
                "max_up_prediction_share": args.max_up_prediction_share,
            },
            "early_stopping": {
                "patience": args.early_stopping_patience,
                "min_delta": args.early_stopping_min_delta,
            },
        },
        "known_source_discrepancy": (
            "The paper reports mini-batches of 20 with gradient accumulation, "
            "while upstream Trainer uses batch_size=1 and run_all.py selects "
            "gradient accumulation 32. This adaptation follows the executable "
            "upstream batch/accumulation behavior and records the discrepancy."
        ),
        "models": entries,
    }
    plan_path = args.output_root / (
        "preflight.plan.json" if args.preflight else "training.plan.json"
    )
    write_json(plan_path, plan)

    if not args.execute:
        for entry in entries:
            print(" ".join(entry["command"]))
        print("Plan only. Add --execute to start.")
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


if __name__ == "__main__":
    main()
