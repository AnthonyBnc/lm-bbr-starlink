import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from utils.experiment_audit import (
    build_comparison_summary,
    build_head_ablation_summary,
    build_multiseed_summary,
)


def write_run(root, model_key, seed=100003, head_type="classical"):
    run_dir = root / "{}_seed{}_{}".format(model_key, seed, head_type)
    run_dir.mkdir()
    metrics = {
        "epochs": [
            {
                "epoch": 1,
                "validation": {
                    "loss": 0.2,
                    "accuracy": 0.8,
                    "per_phase_accuracy": {
                        "BW_DOWN": 0.7,
                        "BW_CRUISE": 1.0,
                        "BW_UP": 0.4,
                    },
                    "label_distribution": {str(i): 0 for i in range(11)},
                    "prediction_distribution": {str(i): 0 for i in range(11)},
                },
            }
        ]
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    manifest = {
        "model_key": model_key,
        "model_id": model_key,
        "model_revision": "revision",
        "model_role": {
            "classical": "gpt_classical",
            "classical_twin": "gpt_classical_twin",
            "quantum": "gpt_quantum",
        }.get(head_type, "modern_baseline"),
        "head_type": head_type,
        "head_config": {"type": head_type},
        "quantum_config": {"type": "vqc"} if head_type == "quantum" else None,
        "dtype": "float16",
        "epochs": 1,
        "tokyo_isolation": "PASS",
        "checkpoint_reload": "PASS",
        "max_train_steps": None,
        "max_validation_steps": None,
        "metrics": str(metrics_path),
        "metrics_sha256": digest,
        "dataset_version": "dataset",
        "split_version": "split",
        "train_pool_sha256": "train",
        "validation_pool_sha256": "validation",
        "train_sample_ids_sha256": "train-ids",
        "validation_sample_ids_sha256": "validation-ids",
        "split_manifest_sha256": "split-manifest",
        "seed": seed,
        "sequence_length": 20,
        "sample_step": 20,
        "batch_size": 1,
        "gradient_accumulation_steps": 32,
        "effective_sequences_per_optimizer_step": 32,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 0.25,
        "lora_config": {"rank": 8},
        "lora_trainable_parameters": 10,
        "total_trainable_parameters": 20,
        "wall_clock_seconds": 30,
    }
    manifest_path = run_dir / "run.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class ExperimentAuditTests(unittest.TestCase):
    def test_comparable_runs_produce_macro_phase_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = [write_run(root, "one"), write_run(root, "two")]
            summary = build_comparison_summary(manifests)
            self.assertEqual(len(summary["models"]), 2)
            self.assertAlmostEqual(
                summary["models"][0]["validation_macro_phase_accuracy"], 0.7
            )

    def test_different_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = [write_run(root, "one"), write_run(root, "two", seed=7)]
            with self.assertRaisesRegex(ValueError, "not comparable"):
                build_comparison_summary(manifests)

    def test_multiseed_summary_accepts_same_model_different_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = [
                write_run(root, "qwen", seed=100003),
                write_run(root, "qwen", seed=100019),
            ]
            summary = build_multiseed_summary(manifests)
            self.assertEqual(summary["model_key"], "qwen")
            self.assertEqual(summary["seeds"], [100003, 100019])
            self.assertAlmostEqual(
                summary["aggregate"]["validation_macro_phase_accuracy"]["mean"], 0.7
            )

    def test_multiseed_summary_rejects_multiple_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = [
                write_run(root, "qwen", seed=100003),
                write_run(root, "lfm", seed=100019),
            ]
            with self.assertRaisesRegex(ValueError, "one model_key"):
                build_multiseed_summary(manifests)

    def test_head_ablation_accepts_three_heads_same_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = [
                write_run(root, "qwen", head_type="classical"),
                write_run(root, "qwen", head_type="classical_twin"),
                write_run(root, "qwen", head_type="quantum"),
            ]
            summary = build_head_ablation_summary(manifests)
            self.assertEqual(summary["model_key"], "qwen")
            self.assertEqual(
                [item["head_type"] for item in summary["heads"]],
                ["classical", "classical_twin", "quantum"],
            )

    def test_head_ablation_rejects_missing_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = [
                write_run(root, "qwen", head_type="classical"),
                write_run(root, "qwen", head_type="quantum"),
            ]
            with self.assertRaisesRegex(ValueError, "Missing required head_type"):
                build_head_ablation_summary(manifests)


if __name__ == "__main__":
    unittest.main()
