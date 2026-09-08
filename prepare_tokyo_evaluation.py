"""Prepare the one-time Tokyo held-out Experience Pool after protocol freeze."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from plm_special.data.dataset import ExperienceDataset
from utils.checkpoint_freeze import file_sha256, verify_frozen_entry, write_json
from utils.starlink_preprocessing import (
    HELD_OUT_LOCATION,
    STREAM_FLAGS,
    build_held_out_experience_pool,
    write_experience_pool,
)


def discover_tokyo_bbr_traces(raw_root, stream_groups=tuple(STREAM_FLAGS)):
    """Select the primary BBR flow only, matching development-pool discovery."""
    paths = []
    for stream_group in stream_groups:
        paths.extend(
            Path(raw_root).glob(
                "{}/{}/**/bbr_{}__*.json".format(
                    stream_group, HELD_OUT_LOCATION, HELD_OUT_LOCATION
                )
            )
        )
    return sorted(paths)


def validate_frozen_suite(manifest):
    if manifest.get("status") != "modern_checkpoints_frozen":
        raise ValueError("Checkpoints are not frozen")
    if manifest.get("tokyo_opened") is not False:
        raise ValueError("Frozen manifest does not represent an unopened Tokyo gate")
    if len(manifest.get("modern_models", ())) != 4:
        raise ValueError("Exactly four frozen modern models are required")
    for entry in manifest["modern_models"]:
        verify_frozen_entry(entry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("tcp-cc-starlink-main"))
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path(
            "data/processed/evaluation/tokyo_8model_comparison_v1/"
            "frozen_checkpoints.manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/evaluation/tokyo_8model_comparison_v1/tokyo.pkl"
        ),
    )
    parser.add_argument(
        "--tokyo-location-flag",
        type=int,
        required=True,
        help="Explicit frozen unseen-location code; it must not alias development flags 0-4.",
    )
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--sample-step", type=int, default=20)
    parser.add_argument(
        "--acknowledge-final-held-out-evaluation",
        action="store_true",
        help="Required because this command opens the one-time Tokyo gate.",
    )
    args = parser.parse_args()
    if not args.acknowledge_final_held_out_evaluation:
        parser.error("Tokyo preparation requires --acknowledge-final-held-out-evaluation")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("Refusing to overwrite an existing Tokyo evaluation pool")

    freeze_manifest = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    validate_frozen_suite(freeze_manifest)
    if args.tokyo_location_flag != freeze_manifest.get("tokyo_location_flag"):
        raise ValueError(
            "Tokyo location flag must match the frozen value {}".format(
                freeze_manifest.get("tokyo_location_flag")
            )
        )
    invalid_paper_models = [
        item["model_key"]
        for item in freeze_manifest.get("paper_models", ())
        if item.get("status") not in ("checkpoint_frozen", "published_reference")
    ]
    if invalid_paper_models:
        raise ValueError(
            "Paper-model comparison mode is unresolved for: {}".format(
                ", ".join(invalid_paper_models)
            )
        )
    trace_paths = discover_tokyo_bbr_traces(args.raw_root)
    if len(trace_paths) != 40:
        raise ValueError("Expected 40 primary Tokyo BBR traces, found {}".format(len(trace_paths)))
    pool = build_held_out_experience_pool(
        trace_paths,
        args.raw_root,
        tokyo_location_flag=args.tokyo_location_flag,
    )
    dataset = ExperienceDataset(
        pool,
        gamma=1.0,
        scale=1000,
        max_length=args.sequence_length,
        sample_step=args.sample_step,
    )
    pool.metadata.update(
        {
            "evaluation_protocol": "tokyo_8model_comparison_v1",
            "sequence_length": args.sequence_length,
            "sample_step": args.sample_step,
            "selection_rule": "primary bbr_ flow only; all 10 runs; four stream groups",
            "frozen_checkpoint_manifest": args.freeze_manifest.as_posix(),
            "frozen_checkpoint_manifest_sha256_before_gate": file_sha256(
                args.freeze_manifest
            ),
            "windows": len(dataset),
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    pool_manifest_path = write_experience_pool(pool, args.output)
    freeze_manifest["tokyo_opened"] = True
    freeze_manifest["tokyo_opened_at"] = pool.metadata["opened_at"]
    freeze_manifest["tokyo_pool"] = args.output.as_posix()
    freeze_manifest["tokyo_pool_sha256"] = file_sha256(args.output)
    freeze_manifest["tokyo_pool_manifest"] = pool_manifest_path.as_posix()
    write_json(args.freeze_manifest, freeze_manifest)

    print("Tokyo pool: {}".format(args.output))
    print("Tokyo manifest: {}".format(pool_manifest_path))
    print("Traces: 40; windows: {}; samples: {}".format(len(dataset), len(pool)))
    print("Phases: {}".format(dict(sorted(Counter(pool.phases).items()))))


if __name__ == "__main__":
    main()
