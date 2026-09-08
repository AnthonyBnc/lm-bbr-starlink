"""Attach an already-opened Tokyo pool's gate stamp to a newly frozen manifest.

Tokyo may only be opened once (prepare_tokyo_evaluation.py, requires raw traces).
This script does not reopen it: it verifies the existing frozen pool file is
byte-identical to the one already opened, then copies the opened-gate fields
onto a fresh freeze manifest (built by freeze_tokyo_comparison.py) for a new
round of checkpoints, so the same pool can be reused for later models without
regenerating it or touching the one-time-open rule.
"""

import argparse
import json
from pathlib import Path

from utils.checkpoint_freeze import file_sha256, write_json

OPENED_FIELDS = (
    "tokyo_opened",
    "tokyo_opened_at",
    "tokyo_pool",
    "tokyo_pool_manifest",
    "tokyo_pool_sha256",
    "tokyo_location_flag",
    "tokyo_location_flag_status",
    "held_out_location",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--opened-reference-manifest", type=Path, required=True)
    parser.add_argument("--tokyo-pool", type=Path, required=True)
    args = parser.parse_args()

    new_manifest = json.loads(args.new_manifest.read_text(encoding="utf-8"))
    if new_manifest.get("tokyo_opened") is not False:
        raise ValueError("--new-manifest must be a freshly frozen, unopened manifest")

    reference = json.loads(args.opened_reference_manifest.read_text(encoding="utf-8"))
    if reference.get("tokyo_opened") is not True:
        raise ValueError("--opened-reference-manifest must already have Tokyo opened")

    actual_pool_sha256 = file_sha256(args.tokyo_pool)
    if actual_pool_sha256 != reference["tokyo_pool_sha256"]:
        raise ValueError(
            "Tokyo pool file does not match the already-opened reference checksum; "
            "refusing to attach the gate stamp"
        )

    for field in OPENED_FIELDS:
        new_manifest[field] = reference[field]
    new_manifest["status"] = "modern_checkpoints_frozen"

    write_json(args.new_manifest, new_manifest)
    print("Attached opened-Tokyo gate to {}".format(args.new_manifest))
    print("tokyo_pool_sha256 verified: {}".format(actual_pool_sha256))


if __name__ == "__main__":
    main()
