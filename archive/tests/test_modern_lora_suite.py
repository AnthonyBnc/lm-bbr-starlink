from pathlib import Path
from types import SimpleNamespace
import unittest

from run_modern_lora_suite import MODEL_ORDER, build_command, output_name


class ModernLoraSuiteTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            split_dir=Path("split"),
            device="mps",
            rank=8,
            alpha=32,
            dropout=0.05,
            epochs=5,
            sequence_length=20,
            sample_step=20,
            state_feature_dim=256,
            learning_rate=1e-4,
            weight_decay=1e-4,
            grad_accum_steps=32,
            seed=100003,
        )

    def test_phase3_order_contains_each_registered_model_once(self):
        self.assertEqual(len(MODEL_ORDER), len(set(MODEL_ORDER)))
        self.assertEqual(MODEL_ORDER[0], "lfm2_5_2_6b")
        self.assertEqual(MODEL_ORDER[-1], "olmo_3_1025_7b")

    def test_command_is_full_unlimited_phase3_run(self):
        command = build_command(self.args, "lfm2_5_2_6b", Path("output"))
        self.assertIn("phase3_exploratory", command)
        self.assertIn("float16", command)
        self.assertNotIn("--max-train-steps", command)
        self.assertNotIn("--max-validation-steps", command)
        self.assertEqual(output_name("model", 8, 5, 100003), "model_rank8_epochs5_seed100003")


if __name__ == "__main__":
    unittest.main()
