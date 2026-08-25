"""Plan or execute a tiny Quantum-GPT smoke run on the development split."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from config import cfg


def build_command(args):
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
        str(args.output_dir),
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
        "1",
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
        "1",
        "--seed",
        str(args.seed),
        "--max-train-steps",
        str(args.max_train_steps),
        "--max-validation-steps",
        str(args.max_validation_steps),
        "--head-type",
        "quantum",
        "--n-qubits",
        str(args.n_qubits),
        "--quantum-depth",
        str(args.quantum_depth),
        "--quantum-ansatz",
        args.quantum_ansatz,
        "--run-purpose",
        "quantum_smoke",
    ]


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
        "--output-dir",
        type=Path,
        default=Path("data/processed/lora_training/quantum_smoke_qwen_r8_q4_d2"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--sample-step", type=int, default=20)
    parser.add_argument("--state-feature-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.model_key not in cfg.modern_model_registry:
        parser.error("Unknown model key: {}".format(args.model_key))
    command = build_command(args)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "planned" if not args.execute else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "quantum_gpt_smoke",
        "reportable_result": False,
        "held_out_location": "Tokyo",
        "command": command,
        "quantum_config": {
            "framework": cfg.quantum_defaults["framework"],
            "n_qubits": args.n_qubits,
            "depth": args.quantum_depth,
            "ansatz": args.quantum_ansatz,
            "estimator": cfg.quantum_defaults["estimator"],
            "simulator": cfg.quantum_defaults["simulator"],
            "default_precision": cfg.quantum_defaults["default_precision"],
            "shots": cfg.quantum_defaults["shots"],
        },
    }
    plan_path = args.output_dir.parent / "quantum_smoke.plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.execute:
        print(" ".join(command))
        print("Dry run only. Add --execute to run the smoke check.")
        print("Plan: {}".format(plan_path))
        return

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(
            "Refusing to overwrite non-empty run directory: {}".format(args.output_dir)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_code = run_command(command, args.output_dir / "training.log")
    if return_code != 0:
        plan["status"] = "failed"
        plan["return_code"] = return_code
        plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise SystemExit(return_code)
    plan["status"] = "completed"
    plan["completed_at"] = datetime.now(timezone.utc).isoformat()
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
