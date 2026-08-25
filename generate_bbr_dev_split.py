"""Generate an exploratory deterministic train/validation split by trace."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import pickle

from utils.splits import stratified_trace_split
from utils.starlink_preprocessing import write_experience_pool


def _summary(pool):
    return {
        "samples": len(pool),
        "traces": len(pool.metadata["source_files"]),
        "phases": dict(sorted(Counter(pool.phases).items())),
        "actions": {str(key): value for key, value in sorted(Counter(pool.actions).items())},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument("--validation-traces-per-stratum", type=int, default=2)
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("rb") as stream:
        pool = pickle.load(stream)
    train_pool, validation_pool, split_manifest = stratified_trace_split(
        pool,
        validation_traces_per_stratum=args.validation_traces_per_stratum,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    train_path = output_dir / "train.pkl"
    validation_path = output_dir / "validation.pkl"
    write_experience_pool(train_pool, train_path)
    write_experience_pool(validation_pool, validation_path)

    split_manifest.update(
        {
            "parent_pool": input_path.as_posix(),
            "parent_pool_sha256": sha256(input_path.read_bytes()).hexdigest(),
            "train": _summary(train_pool),
            "validation": _summary(validation_pool),
            "train_pool_sha256": sha256(train_path.read_bytes()).hexdigest(),
            "validation_pool_sha256": sha256(validation_path.read_bytes()).hexdigest(),
        }
    )
    manifest_path = output_dir / "split.manifest.json"
    manifest_path.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Split manifest: {}".format(manifest_path))
    print("Train: {}".format(_summary(train_pool)))
    print("Validation: {}".format(_summary(validation_pool)))


if __name__ == "__main__":
    main()
