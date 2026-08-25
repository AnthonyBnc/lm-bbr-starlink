"""Audit and summarize comparable modern LoRA development runs."""

from hashlib import sha256
import json
from pathlib import Path

from utils.bbr import BBR_PHASES


SHARED_COMPARISON_FIELDS = (
    "dataset_version",
    "split_version",
    "train_pool_sha256",
    "validation_pool_sha256",
    "train_sample_ids_sha256",
    "validation_sample_ids_sha256",
    "split_manifest_sha256",
    "seed",
    "sequence_length",
    "sample_step",
    "batch_size",
    "gradient_accumulation_steps",
    "effective_sequences_per_optimizer_step",
    "epochs",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "training_loss_weighting",
    "training_sampling",
)

SHARED_MULTI_SEED_FIELDS = tuple(
    field for field in SHARED_COMPARISON_FIELDS if field != "seed"
)

REQUIRED_HEAD_ABLATION_HEADS = ("classical", "classical_twin", "quantum")


def _sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _mean(values):
    return sum(values) / len(values)


def _stddev(values):
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def load_completed_run(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("tokyo_isolation") != "PASS":
        raise ValueError("Tokyo isolation did not pass: {}".format(manifest_path))
    if manifest.get("checkpoint_reload") != "PASS":
        raise ValueError("Checkpoint reload did not pass: {}".format(manifest_path))
    if manifest.get("max_train_steps") is not None:
        raise ValueError("Limited train run cannot enter a model comparison")
    if manifest.get("max_validation_steps") is not None:
        raise ValueError("Limited validation run cannot enter a model comparison")

    metrics_path = Path(manifest["metrics"])
    if not metrics_path.is_absolute():
        candidate = manifest_path.parent / metrics_path.name
        metrics_path = candidate if candidate.is_file() else metrics_path
    if _sha256(metrics_path) != manifest["metrics_sha256"]:
        raise ValueError("Metrics hash mismatch: {}".format(metrics_path))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if len(metrics.get("epochs", ())) != manifest["epochs"]:
        raise ValueError("Metrics epoch count does not match manifest")
    return manifest, metrics


def audit_comparable_runs(manifest_paths):
    runs = [load_completed_run(path) for path in manifest_paths]
    if not runs:
        raise ValueError("No completed run manifests were supplied")
    reference = runs[0][0]
    for manifest, _ in runs[1:]:
        mismatches = {
            field: (reference.get(field), manifest.get(field))
            for field in SHARED_COMPARISON_FIELDS
            if reference.get(field) != manifest.get(field)
        }
        if mismatches:
            raise ValueError(
                "Run {} is not comparable: {}".format(
                    manifest.get("model_key"), mismatches
                )
            )
    return runs


def summarize_run(manifest, metrics):
    validation = metrics["epochs"][-1]["validation"]
    phase_values = [
        validation["per_phase_accuracy"][phase]
        for phase in BBR_PHASES
        if validation["per_phase_accuracy"][phase] is not None
    ]
    return {
        "model_key": manifest["model_key"],
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "model_role": manifest.get("model_role"),
        "head_type": manifest.get("head_type", "classical"),
        "head_config": manifest.get("head_config"),
        "quantum_config": manifest.get("quantum_config"),
        "seed": manifest["seed"],
        "dtype": manifest["dtype"],
        "epochs": manifest["epochs"],
        "validation_loss": validation["loss"],
        "validation_accuracy": validation["accuracy"],
        "validation_macro_phase_accuracy": sum(phase_values) / len(phase_values),
        "validation_per_phase_accuracy": validation["per_phase_accuracy"],
        "validation_label_distribution": validation["label_distribution"],
        "validation_prediction_distribution": validation["prediction_distribution"],
        "lora_rank": manifest["lora_config"]["rank"],
        "lora_trainable_parameters": manifest["lora_trainable_parameters"],
        "total_trainable_parameters": manifest["total_trainable_parameters"],
        "wall_clock_seconds": manifest["wall_clock_seconds"],
        "tokyo_isolation": manifest["tokyo_isolation"],
        "checkpoint_reload": manifest["checkpoint_reload"],
        "reportable_result": False,
    }


def build_comparison_summary(manifest_paths):
    runs = audit_comparable_runs(manifest_paths)
    reference = runs[0][0]
    return {
        "status": "exploratory_development_comparison",
        "reportable_result": False,
        "held_out_location": "Tokyo",
        "shared_protocol": {
            field: reference.get(field) for field in SHARED_COMPARISON_FIELDS
        },
        "models": [summarize_run(manifest, metrics) for manifest, metrics in runs],
    }


def audit_multiseed_runs(manifest_paths):
    runs = [load_completed_run(path) for path in manifest_paths]
    if not runs:
        raise ValueError("No completed run manifests were supplied")
    reference = runs[0][0]
    for manifest, _ in runs[1:]:
        if manifest.get("model_key") != reference.get("model_key"):
            raise ValueError("Multi-seed summary requires one model_key")
        mismatches = {
            field: (reference.get(field), manifest.get(field))
            for field in SHARED_MULTI_SEED_FIELDS
            if reference.get(field) != manifest.get(field)
        }
        if mismatches:
            raise ValueError(
                "Run seed {} is not comparable: {}".format(
                    manifest.get("seed"), mismatches
                )
            )
    seeds = [manifest["seed"] for manifest, _ in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seed in multi-seed summary")
    return runs


def build_multiseed_summary(manifest_paths):
    runs = audit_multiseed_runs(manifest_paths)
    reference = runs[0][0]
    per_seed = [summarize_run(manifest, metrics) for manifest, metrics in runs]
    metrics = {
        "validation_loss": [item["validation_loss"] for item in per_seed],
        "validation_accuracy": [item["validation_accuracy"] for item in per_seed],
        "validation_macro_phase_accuracy": [
            item["validation_macro_phase_accuracy"] for item in per_seed
        ],
        "wall_clock_seconds": [item["wall_clock_seconds"] for item in per_seed],
    }
    return {
        "status": "phase4_development_multiseed_summary",
        "reportable_result": False,
        "held_out_location": "Tokyo",
        "model_key": reference["model_key"],
        "model_id": reference["model_id"],
        "model_revision": reference["model_revision"],
        "seeds": sorted(item["seed"] for item, _ in runs),
        "shared_protocol": {
            field: reference.get(field) for field in SHARED_MULTI_SEED_FIELDS
        },
        "aggregate": {
            key: {
                "mean": _mean(values),
                "stddev": _stddev(values),
                "min": min(values),
                "max": max(values),
            }
            for key, values in metrics.items()
        },
        "per_seed": sorted(per_seed, key=lambda item: item["seed"]),
    }


def audit_head_ablation_runs(manifest_paths):
    runs = [load_completed_run(path) for path in manifest_paths]
    if not runs:
        raise ValueError("No completed run manifests were supplied")
    reference = runs[0][0]
    model_fields = ("model_key", "model_id", "model_revision")
    for manifest, _ in runs[1:]:
        mismatches = {
            field: (reference.get(field), manifest.get(field))
            for field in model_fields + SHARED_COMPARISON_FIELDS
            if reference.get(field) != manifest.get(field)
        }
        if mismatches:
            raise ValueError(
                "Head run {} is not comparable: {}".format(
                    manifest.get("head_type"), mismatches
                )
            )
    heads = [manifest.get("head_type") for manifest, _ in runs]
    if len(heads) != len(set(heads)):
        raise ValueError("Duplicate head_type in head ablation summary")
    missing = [
        head for head in REQUIRED_HEAD_ABLATION_HEADS if head not in set(heads)
    ]
    if missing:
        raise ValueError("Missing required head_type values: {}".format(missing))
    return runs


def build_head_ablation_summary(manifest_paths):
    runs = audit_head_ablation_runs(manifest_paths)
    reference = runs[0][0]
    return {
        "status": "qwen_head_ablation_development_summary",
        "reportable_result": False,
        "held_out_location": "Tokyo",
        "model_key": reference["model_key"],
        "model_id": reference["model_id"],
        "model_revision": reference["model_revision"],
        "seed": reference["seed"],
        "shared_protocol": {
            field: reference.get(field) for field in SHARED_COMPARISON_FIELDS
        },
        "heads": sorted(
            [summarize_run(manifest, metrics) for manifest, metrics in runs],
            key=lambda item: REQUIRED_HEAD_ABLATION_HEADS.index(item["head_type"]),
        ),
    }
