"""Create an audited exploratory comparison from modern LoRA run artifacts."""

import argparse
import json
from pathlib import Path

from utils.experiment_audit import build_comparison_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_paths = sorted(args.run_root.glob("*/run.manifest.json"))
    summary = build_comparison_summary(manifest_paths)
    output = args.output or args.run_root / "comparison.summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Summary: {}".format(output))


if __name__ == "__main__":
    main()
