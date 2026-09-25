from pathlib import Path
from types import SimpleNamespace
import unittest

from run_qwen_cruise80_head_ablation import build_command


class Cruise80RunnerTests(unittest.TestCase):
    def test_command_routes_one_design_to_fair_three_head_runner(self):
        args = SimpleNamespace(
            output_root=Path("output"),
            split_dir=Path("split"),
            device="mps",
            seed=100003,
            execute=True,
            resume_completed=False,
        )
        command = build_command(args, Path("design.json"))
        joined = " ".join(command)
        self.assertIn("run_qwen_head_ablation.py", joined)
        self.assertIn("--target-train-windows 2400", joined)
        self.assertIn("--replacement-train-windows 1600", joined)
        self.assertIn("cruise80_location_up_action_balanced_replacement", joined)
        self.assertIn("--training-sampling-plan design.json", joined)
        self.assertIn("--execute", command)


if __name__ == "__main__":
    unittest.main()
