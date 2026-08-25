"""Create an index-only UP-focused training design without mutating data pools."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle

from plm_special.data.dataset import ExperienceDataset
from utils.training_sampling import (
    TRAINING_SAMPLING_CRUISE80,
    TRAINING_SAMPLING_UP_FOCUS,
    build_training_window_indices,
)


DEFAULT_SPLIT_DIR = Path("data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003")


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def sample_ids_sha256(pool):
    return sha256("\n".join(pool.sample_ids).encode("utf-8")).hexdigest()


def verify_pool(pool, role):
    if pool.metadata.get("split_role") != role:
        raise ValueError("Expected {} development pool".format(role))
    if pool.metadata.get("held_out_location") != "Tokyo":
        raise ValueError("Development pool does not declare Tokyo as held out")
    if any(
        "tokyo" in source["path"].lower()
        for source in pool.metadata.get("source_files", ())
    ):
        raise ValueError("Tokyo source found in development pool")
    if len(pool.sample_ids) != len(set(pool.sample_ids)):
        raise ValueError("Development pool contains duplicate sample IDs")


def build_data_design(
    split_dir,
    train_windows=2400,
    seed=100003,
    sequence_length=20,
    sample_step=20,
    up_density_power=12.0,
    down_penalty=10.0,
    replacement_windows=1600,
    sampling_strategy=TRAINING_SAMPLING_CRUISE80,
):
    split_dir = Path(split_dir)
    train_path = split_dir / "train.pkl"
    validation_path = split_dir / "validation.pkl"
    split_manifest_path = split_dir / "split.manifest.json"
    with train_path.open("rb") as stream:
        train_pool = pickle.load(stream)
    with validation_path.open("rb") as stream:
        validation_pool = pickle.load(stream)
    verify_pool(train_pool, "train")
    verify_pool(validation_pool, "validation")
    if set(train_pool.sample_ids) & set(validation_pool.sample_ids):
        raise ValueError("Train and validation sample IDs overlap")

    train_dataset = ExperienceDataset(
        train_pool,
        gamma=1.0,
        scale=1000,
        max_length=sequence_length,
        sample_step=sample_step,
    )
    validation_dataset = ExperienceDataset(
        validation_pool,
        gamma=1.0,
        scale=1000,
        max_length=sequence_length,
        sample_step=sample_step,
    )
    selected_indices, record = build_training_window_indices(
        train_dataset,
        strategy=sampling_strategy,
        target_windows=train_windows,
        seed=seed,
        up_density_power=up_density_power,
        down_penalty=down_penalty,
        replacement_windows=replacement_windows,
    )
    return {
        "status": "exploratory_index_only_training_design",
        "reportable_result": False,
        "development_pool_mutated": False,
        "split_membership_changed": False,
        "labels_rewards_actions_changed": False,
        "held_out_location": "Tokyo",
        "tokyo_isolation": "PASS",
        "split_version": train_pool.metadata["split_version"],
        "sequence_length": sequence_length,
        "sample_step": sample_step,
        "source_train": {
            "path": train_path.as_posix(),
            "sha256": file_sha256(train_path),
            "sample_ids_sha256": sample_ids_sha256(train_pool),
            "pool_samples": len(train_pool),
            "windows": len(train_dataset),
        },
        "source_validation": {
            "path": validation_path.as_posix(),
            "sha256": file_sha256(validation_path),
            "sample_ids_sha256": sample_ids_sha256(validation_pool),
            "pool_samples": len(validation_pool),
            "windows": len(validation_dataset),
            "sampling": "original_unmodified_distribution",
        },
        "split_manifest": {
            "path": split_manifest_path.as_posix(),
            "sha256": file_sha256(split_manifest_path),
        },
        "training_sampling": {
            key: record[key]
            for key in (
                "strategy",
                "seed",
                "target_windows",
                "with_replacement",
                "up_density_power",
                "down_penalty",
                "up_action_window_quotas",
                "replacement_windows",
                "location_action_window_quotas",
            )
        },
        "source_distribution": record["source_distribution"],
        "selected_distribution": record["selected_distribution"],
        "source_location_windows": record["source_location_windows"],
        "selected_location_windows": record["selected_location_windows"],
        "selected_dataset_indices": selected_indices,
        "test_design": {
            "requested_windows": 700,
            "status": "deferred_no_independent_pool_without_split_change_or_Tokyo_use",
        },
    }


def write_data_design(path, design):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-windows", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--sample-step", type=int, default=20)
    parser.add_argument("--up-density-power", type=float, default=12.0)
    parser.add_argument("--down-penalty", type=float, default=10.0)
    parser.add_argument("--replacement-windows", type=int, default=1600)
    args = parser.parse_args()
    design = build_data_design(
        args.split_dir,
        train_windows=args.train_windows,
        seed=args.seed,
        sequence_length=args.sequence_length,
        sample_step=args.sample_step,
        up_density_power=args.up_density_power,
        down_penalty=args.down_penalty,
        replacement_windows=args.replacement_windows,
        sampling_strategy=TRAINING_SAMPLING_CRUISE80,
    )
    path = write_data_design(args.output, design)
    print("Data design: {}".format(path))
    print(json.dumps({
        "development_pool_mutated": design["development_pool_mutated"],
        "train_source_windows": design["source_train"]["windows"],
        "selected_train_windows": design["training_sampling"]["target_windows"],
        "validation_windows": design["source_validation"]["windows"],
        "selected_phase_positions": design["selected_distribution"]["phase_positions"],
        "selected_up_actions": {
            action: design["selected_distribution"]["action_positions"][action]
            for action in ("6", "7", "8", "9", "10")
        },
        "tokyo_isolation": design["tokyo_isolation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
