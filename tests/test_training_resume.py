import json
from pathlib import Path
import tempfile
import unittest

import torch

from train_modern_lora import (
    load_resume_record,
    validate_resume_compatibility,
    validate_resume_lora,
)


class TrainingResumeTests(unittest.TestCase):
    def test_resume_record_accepts_portable_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint = run_dir / "checkpoint"
            adapter = checkpoint / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            torch.save({}, checkpoint / "task_modules.pt")
            torch.save({}, checkpoint / "optimizer.pt")
            metrics = {"epochs": [{"epoch": 1}]}
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            manifest = {
                "completed_epochs": 1,
                "checkpoint": "checkpoint",
                "metrics": "metrics.json",
            }
            (run_dir / "progress.manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            result = load_resume_record(run_dir)

            self.assertEqual(result["completed_epochs"], 1)
            self.assertEqual(result["checkpoint_dir"], checkpoint)

    def test_resume_compatibility_rejects_changed_method_setting(self):
        with self.assertRaisesRegex(ValueError, "sequence_length"):
            validate_resume_compatibility(
                {"sequence_length": 20}, {"sequence_length": 10}
            )

    def test_resume_lora_rejects_changed_rank(self):
        with self.assertRaisesRegex(ValueError, "rank"):
            validate_resume_lora(
                {"lora_config": {"rank": 8, "alpha": 32, "dropout": 0.05}},
                rank=4,
                alpha=32,
                dropout=0.05,
            )


if __name__ == "__main__":
    unittest.main()
