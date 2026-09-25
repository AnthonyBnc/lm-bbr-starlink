from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

from run_trainability_gate200 import GATE_CONFIGS, build_command, completed_run


class TrainabilityGate200RunnerTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            output_root=Path("output"),
            split_dir=Path("split"),
            sampling_plan=Path("design.json"),
            device="mps",
            seed=100003,
        )

    def test_gate_runs_twin_then_quantum(self):
        self.assertEqual(
            GATE_CONFIGS,
            (("twin_ln_t4", "classical_twin"), ("quantum_ln_t4_pi", "quantum")),
        )

    def test_command_uses_normalized_200_step_frozen_protocol(self):
        _, command = build_command(self.args, *GATE_CONFIGS[-1])
        self.assertIn("--head-input-layernorm", command)
        self.assertEqual(command[command.index("--bottleneck-temperature") + 1], "4.0")
        self.assertEqual(command[command.index("--max-train-steps") + 1], "200")
        self.assertEqual(command[command.index("--max-validation-steps") + 1], "200")
        self.assertEqual(command[command.index("--training-sampling-plan") + 1], "design.json")

    def test_completed_run_requires_audits_and_exact_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = {
                "checkpoint_reload": "PASS",
                "tokyo_isolation": "PASS",
                "max_train_steps": 200,
                "max_validation_steps": 200,
            }
            (output / "run.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(completed_run(output))


if __name__ == "__main__":
    unittest.main()
