"""Freeze the four completed under-400M checkpoints before opening Tokyo."""

import argparse
from pathlib import Path

from utils.checkpoint_freeze import build_freeze_manifest, write_json


MODEL_KEYS = (
    "granite_4_0_350m",
    "pleias_rag_350m",
    "lfm2_5_350m",
    "gemma_3_270m",
)
DEFAULT_RUN_DIRS = (
    Path("data/processed/lora_training/under400_rank128_epochs16_seed100003_v1/granite_4_0_350m_rank128_epochs16_seed100003_train"),
    Path("data/processed/lora_training/under400_rank128_epochs16_seed100003_v1/pleias_rag_350m_rank128_epochs16_seed100003_train"),
    Path("data/processed/lora_training/under400_rank128_epochs16_seed100003_v1/lfm2_5_350m_rank128_epochs16_seed100003_train"),
    Path("data/processed/lora_training/gemma3_270m_rank128_epochs16_seed100003_v1/gemma_3_270m_rank128_epochs16_seed100003_train"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, dest="run_dirs")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/evaluation/tokyo_8model_comparison_v1/"
            "frozen_checkpoints.manifest.json"
        ),
    )
    parser.add_argument(
        "--expected-epochs",
        type=int,
        default=16,
        help="Epoch count each run must match before it can be frozen.",
    )
    args = parser.parse_args()
    manifest = build_freeze_manifest(
        args.run_dirs or DEFAULT_RUN_DIRS, MODEL_KEYS, expected_epochs=args.expected_epochs
    )
    write_json(args.output, manifest)
    print("Frozen checkpoint manifest: {}".format(args.output))
    for entry in manifest["modern_models"]:
        print("{}: epoch {} PASS".format(entry["model_key"], entry["epochs"]))


if __name__ == "__main__":
    main()
