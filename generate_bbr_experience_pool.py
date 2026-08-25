"""Generate an exploratory paper-compatible BBR experience pool."""

import argparse
from collections import Counter
from pathlib import Path

from utils.bbr import action_index_to_gain
from utils.starlink_preprocessing import (
    DEVELOPMENT_LOCATIONS,
    STREAM_FLAGS,
    build_experience_pool,
    write_experience_pool,
)


def discover_traces(raw_root, locations, stream_groups, max_files=None):
    paths = []
    for stream_group in stream_groups:
        for location in locations:
            paths.extend(
                Path(raw_root).glob(
                    "{}/{}/**/bbr_{}__*.json".format(stream_group, location, location)
                )
            )
    paths = sorted(paths)
    return paths[:max_files] if max_files is not None else paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="tcp-cc-starlink-main")
    parser.add_argument("--output", required=True)
    parser.add_argument("--locations", nargs="+", choices=DEVELOPMENT_LOCATIONS, default=["Ohio"])
    parser.add_argument(
        "--stream-groups",
        nargs="+",
        choices=tuple(STREAM_FLAGS),
        default=["downlink-sequential-logs"],
    )
    parser.add_argument("--max-files", type=int, default=1)
    args = parser.parse_args()

    trace_paths = discover_traces(
        args.raw_root, args.locations, args.stream_groups, args.max_files
    )
    if not trace_paths:
        raise SystemExit("No matching BBR traces found")
    pool = build_experience_pool(trace_paths, args.raw_root)
    manifest_path = write_experience_pool(pool, args.output)
    phase_counts = Counter(pool.phases)
    action_counts = Counter(pool.actions)
    print("Exploratory experience pool: {}".format(args.output))
    print("Manifest: {}".format(manifest_path))
    print("Samples: {}".format(len(pool)))
    print("Phases: {}".format(dict(sorted(phase_counts.items()))))
    print(
        "Actions: {}".format(
            {
                "{} ({:.2f})".format(index, action_index_to_gain(index)): count
                for index, count in sorted(action_counts.items())
            }
        )
    )


if __name__ == "__main__":
    main()
