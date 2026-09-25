import json
from pathlib import Path
import tempfile
import unittest

from utils.checkpoint_freeze import freeze_completed_run, verify_frozen_entry


class CheckpointFreezeTests(unittest.TestCase):
    def _run(self, root):
        run = Path(root) / "run"
        checkpoint = run / "checkpoint"
        (checkpoint / "adapter").mkdir(parents=True)
        for name in (
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "task_modules.pt",
            "training_state.json",
        ):
            (checkpoint / name).write_bytes(name.encode("utf-8"))
        manifest = {
            "checkpoint_reload": "PASS",
            "tokyo_isolation": "PASS",
            "epochs": 16,
            "checkpoint_rule": "final completed epoch for slm_bbr_modern_backbone_extension",
            "checkpoint": "checkpoint",
            "run_completed_at": "2026-09-03T00:00:00+00:00",
            "model_key": "model",
            "model_id": "owner/model",
            "model_revision": "revision",
            "seed": 100003,
        }
        (run / "run.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run / "progress.manifest.json").write_text(
            json.dumps({"status": "training_in_progress", "completed_epochs": 16}),
            encoding="utf-8",
        )
        return run

    def test_freeze_finalizes_progress_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(directory)
            entry = freeze_completed_run(run)
            progress = json.loads((run / "progress.manifest.json").read_text())
            self.assertEqual(progress["status"], "training_completed")
            self.assertTrue(progress["frozen_for_held_out_evaluation"])
            self.assertTrue(verify_frozen_entry(entry))
            (run / "checkpoint" / "task_modules.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_frozen_entry(entry)


if __name__ == "__main__":
    unittest.main()
