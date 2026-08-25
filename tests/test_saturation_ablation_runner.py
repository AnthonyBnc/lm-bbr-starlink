from pathlib import Path
from types import SimpleNamespace
import unittest

from run_saturation_ablation import CONFIGS, build_command


class SaturationAblationRunnerTests(unittest.TestCase):
    def test_matrix_contains_controlled_normalization_and_scale_tests(self):
        self.assertEqual(len(CONFIGS), 4)
        self.assertIn(("quantum_ln_t4_half_pi", "quantum", True, "4.0", "half_pi"), CONFIGS)

    def test_command_keeps_frozen_data_and_limits_steps(self):
        args = SimpleNamespace(
            output_root=Path("output"),
            split_dir=Path("split"),
            sampling_plan=Path("design.json"),
            device="mps",
            seed=100003,
            train_steps=25,
            validation_steps=50,
        )
        _, command = build_command(args, *CONFIGS[-1])
        self.assertIn("--head-input-layernorm", command)
        self.assertEqual(command[command.index("--bottleneck-temperature") + 1], "4.0")
        self.assertEqual(command[command.index("--quantum-angle-scale") + 1], "half_pi")
        self.assertEqual(command[command.index("--max-train-steps") + 1], "25")
        self.assertEqual(command[command.index("--training-sampling-plan") + 1], "design.json")


if __name__ == "__main__":
    unittest.main()
