from pathlib import Path
from types import SimpleNamespace
import unittest

from run_phase4_multiseed import DEFAULT_SEEDS, build_command, output_name


class Phase4MultiseedTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            model_key="qwen3_5_4b_base",
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
        )

    def test_default_seeds_are_unique(self):
        self.assertEqual(len(DEFAULT_SEEDS), len(set(DEFAULT_SEEDS)))
        self.assertIn(100003, DEFAULT_SEEDS)

    def test_command_is_phase4_qwen_run(self):
        command = build_command(self.args, 100019, Path("output"))
        self.assertIn("phase4_dev_multiseed", command)
        self.assertIn("qwen3_5_4b_base", command)
        self.assertIn("bfloat16", command)
        self.assertEqual(
            output_name("qwen3_5_4b_base", 8, 5, 100019),
            "qwen3_5_4b_base_rank8_epochs5_seed100019",
        )


if __name__ == "__main__":
    unittest.main()
