"""Rebuild the frozen Tokyo evaluation pool from the raw Starlink traces.

Why: the original tokyo.pkl (sha256 800f1a44...) was produced on the lab PC and
is not in git (data/processed/ is ignored). Its bytes cannot be reproduced
exactly because its metadata holds the gate-opening timestamp, so this script
rebuilds the pool with the same code and frozen location flag and then PROVES
the content is identical: every one of the 12,000 samples evaluated for seed
100003 must match in sample_id, pool index, phase, expert action, stream flag,
observed throughput and observed retransmissions.

This does not open or re-open the Tokyo gate and uses no model: it only
reconstructs the already-frozen held-out pool. No labels are inspected for any
tuning or selection decision.

Run (needs the raw traces in tcp-cc-starlink-main/):
    python reports/rebuild_tokyo_pool.py
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.starlink_preprocessing import (  # noqa: E402
    HELD_OUT_LOCATION,
    STREAM_FLAGS,
    build_held_out_experience_pool,
    write_experience_pool,
)

ORIGINAL_POOL_SHA256 = "800f1a444fcf1610720b83f6feadf0743fb2acaca81ed8d251f264fad012f27b"
REFERENCE_PREDICTIONS = ROOT / (
    "data/processed/evaluation/four_model_100epoch_v1/results/lfm2_5_350m_quantum/predictions.csv"
)


def discover_tokyo_bbr_traces(raw_root):
    """Use the exact function from prepare_tokyo_evaluation.py (single source of
    truth) without importing that module, which needs PyTorch."""
    source = (ROOT / "prepare_tokyo_evaluation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "discover_tokyo_bbr_traces")
    namespace = {"Path": Path, "STREAM_FLAGS": STREAM_FLAGS, "HELD_OUT_LOCATION": HELD_OUT_LOCATION}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "prepare_tokyo_evaluation.py", "exec"), namespace)
    return namespace["discover_tokyo_bbr_traces"](raw_root)


def verify_against_reference(pool, predictions_csv):
    with open(predictions_csv, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 12000:
        raise SystemExit("Reference predictions should have 12000 rows, found {}".format(len(rows)))
    mismatches = []
    for row in rows:
        index = int(row["pool_index"])
        state = pool.states[index]
        checks = {
            "sample_id": pool.sample_ids[index] == row["sample_id"],
            "phase": pool.phases[index] == row["phase"],
            "target_action": int(pool.actions[index]) == int(row["target_action"]),
            "stream_flag": int(state[1]) == int(row["stream_flag"]),
            "observed_throughput": float(state[3]) == float(row["observed_throughput"]),
            "observed_retransmissions": float(state[4]) == float(row["observed_retransmissions"]),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            mismatches.append((index, failed))
    if mismatches:
        raise SystemExit("Rebuilt pool differs from the evaluated Tokyo pool at {} samples, e.g. {}".format(
            len(mismatches), mismatches[:5]))
    evaluated_ids = [row["sample_id"] for row in sorted(rows, key=lambda r: int(r["pool_index"]))]
    return len(rows), sha256("\n".join(evaluated_ids).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ROOT / "tcp-cc-starlink-main")
    parser.add_argument("--freeze-manifest", type=Path, default=ROOT / (
        "data/processed/evaluation/tokyo_8model_comparison_v1/frozen_checkpoints.manifest.json"))
    parser.add_argument("--output", type=Path, default=ROOT / (
        "data/processed/evaluation/tokyo_8model_comparison_v1/tokyo.pkl"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite {}".format(args.output))

    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if freeze.get("tokyo_opened") is not True or freeze.get("tokyo_pool_sha256") != ORIGINAL_POOL_SHA256:
        raise SystemExit("Freeze manifest does not describe the original opened Tokyo pool")
    flag = int(freeze["tokyo_location_flag"])

    traces = discover_tokyo_bbr_traces(args.raw_root)
    if len(traces) != 40:
        raise SystemExit("Expected 40 primary Tokyo BBR traces, found {}".format(len(traces)))
    pool = build_held_out_experience_pool(traces, args.raw_root, tokyo_location_flag=flag)

    evaluated, evaluated_ids_sha = verify_against_reference(pool, REFERENCE_PREDICTIONS)
    pool.metadata.update({
        "evaluation_protocol": "tokyo_8model_comparison_v1",
        "sequence_length": 20,
        "sample_step": 20,
        "selection_rule": "primary bbr_ flow only; all 10 runs; four stream groups",
        "frozen_checkpoint_manifest": "data/processed/evaluation/tokyo_8model_comparison_v1/"
                                      "frozen_checkpoints.manifest.json",
        "opened_at": freeze.get("tokyo_opened_at"),
        "rebuild": {
            "reason": "original tokyo.pkl not available on this machine; rebuilt from raw traces",
            "rebuilt_at": datetime.now(timezone.utc).isoformat(),
            "original_pool_sha256": ORIGINAL_POOL_SHA256,
            "verified_against": REFERENCE_PREDICTIONS.relative_to(ROOT).as_posix(),
            "verified_samples": evaluated,
            "verified_fields": ["sample_id", "pool_index", "phase", "target_action", "stream_flag",
                                "observed_throughput", "observed_retransmissions"],
            "evaluated_sample_ids_sha256": evaluated_ids_sha,
        },
    })
    manifest_path = write_experience_pool(pool, args.output)
    digest = sha256(args.output.read_bytes()).hexdigest()
    print("Rebuilt Tokyo pool: {} ({} samples)".format(args.output, len(pool)))
    print("Matches the seed-100003 evaluation on all {} evaluated samples.".format(evaluated))
    print("sha256: {}".format(digest))
    print("manifest: {}".format(manifest_path))


if __name__ == "__main__":
    main()
