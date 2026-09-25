"""Cryptographic freeze records for final held-out evaluation checkpoints."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


REQUIRED_CHECKPOINT_FILES = (
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "task_modules.pt",
    "training_state.json",
)


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def freeze_completed_run(run_dir, expected_epochs=16):
    """Validate and checksum a completed final-epoch LoRA run."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run.manifest.json"
    progress_path = run_dir / "progress.manifest.json"
    if not manifest_path.is_file() or not progress_path.is_file():
        raise ValueError("Run is missing run/progress manifest: {}".format(run_dir))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if manifest.get("checkpoint_reload") != "PASS":
        raise ValueError("Checkpoint reload did not pass: {}".format(run_dir))
    if manifest.get("tokyo_isolation") != "PASS":
        raise ValueError("Tokyo isolation did not pass: {}".format(run_dir))
    if manifest.get("epochs") != expected_epochs:
        raise ValueError("Run is not a {}-epoch run: {}".format(expected_epochs, run_dir))
    if progress.get("completed_epochs") != expected_epochs:
        raise ValueError("Progress epoch count disagrees: {}".format(run_dir))
    if manifest.get("checkpoint_rule") != (
        "final completed epoch for slm_bbr_modern_backbone_extension"
    ):
        raise ValueError("Run does not use the frozen final-epoch rule: {}".format(run_dir))

    checkpoint_dir = run_dir / manifest["checkpoint"]
    hashes = {}
    for relative_name in REQUIRED_CHECKPOINT_FILES:
        path = checkpoint_dir / relative_name
        if not path.is_file():
            raise ValueError("Frozen checkpoint file is missing: {}".format(path))
        hashes[relative_name] = file_sha256(path)

    progress.update(
        {
            "status": "training_completed",
            "completed_at": manifest["run_completed_at"],
            "checkpoint": manifest["checkpoint"],
            "checkpoint_reload": "PASS",
            "frozen_for_held_out_evaluation": True,
            "frozen_checkpoint_sha256": hashes,
        }
    )
    write_json(progress_path, progress)
    return {
        "model_key": manifest["model_key"],
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "run_dir": run_dir.as_posix(),
        "run_manifest": manifest_path.as_posix(),
        "run_manifest_sha256": file_sha256(manifest_path),
        "checkpoint": checkpoint_dir.as_posix(),
        "checkpoint_rule": "final_epoch",
        "epochs": expected_epochs,
        "seed": manifest["seed"],
        "checkpoint_reload": "PASS",
        "checkpoint_files_sha256": hashes,
    }


def verify_frozen_entry(entry):
    """Fail closed if any frozen checkpoint or run manifest has changed."""
    if file_sha256(entry["run_manifest"]) != entry["run_manifest_sha256"]:
        raise ValueError("Frozen run manifest changed for {}".format(entry["model_key"]))
    checkpoint_dir = Path(entry["checkpoint"])
    for relative_name, expected_hash in entry["checkpoint_files_sha256"].items():
        path = checkpoint_dir / relative_name
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(
                "Frozen checkpoint changed for {}: {}".format(
                    entry["model_key"], relative_name
                )
            )
    return True


def build_freeze_manifest(run_dirs, expected_model_keys, expected_epochs=16):
    entries = [freeze_completed_run(path, expected_epochs) for path in run_dirs]
    observed_keys = [entry["model_key"] for entry in entries]
    if set(observed_keys) != set(expected_model_keys) or len(observed_keys) != len(
        expected_model_keys
    ):
        raise ValueError(
            "Expected exactly {}, got {}".format(expected_model_keys, observed_keys)
        )
    shared_fields = ("seed",)
    for field in shared_fields:
        values = {entry[field] for entry in entries}
        if len(values) != 1:
            raise ValueError("Frozen runs disagree on {}: {}".format(field, values))
    run_manifests = [
        json.loads(Path(entry["run_manifest"]).read_text(encoding="utf-8"))
        for entry in entries
    ]
    for field in (
        "train_pool_sha256",
        "validation_pool_sha256",
        "split_manifest_sha256",
        "train_sample_ids_sha256",
        "validation_sample_ids_sha256",
        "sequence_length",
        "sample_step",
        "training_loss_weighting",
        "validation_loss_weighting",
    ):
        values = {json.dumps(item.get(field), sort_keys=True) for item in run_manifests}
        if len(values) != 1:
            raise ValueError("Frozen runs disagree on {}".format(field))
    entries.sort(key=lambda item: expected_model_keys.index(item["model_key"]))
    return {
        "schema_version": 1,
        "status": "modern_checkpoints_frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "held_out_location": "Tokyo",
        "tokyo_location_flag": 5,
        "tokyo_location_flag_status": "user_approved_2026-09-03",
        "tokyo_opened": False,
        "checkpoint_selection_rule": "final_epoch_16_fixed_before_tokyo",
        "modern_models": entries,
        "paper_models": [
            {
                "model_key": key,
                "status": "published_reference",
                "source": "project_sources/2607.07142v1 copy.pdf",
            }
            for key in ("gpt2", "t5", "gpt_neo", "smollm2")
        ],
        "comparison_claim": (
            "The four new models are evaluated directly on the frozen Tokyo pool. "
            "The four paper models use published aggregate/figure values and are a "
            "contextual reference, not a same-sample or same-runtime rerun."
        ),
    }
