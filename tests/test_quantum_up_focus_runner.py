from pathlib import Path
from types import SimpleNamespace
import unittest

from run_quantum_up_focus import build_command


class QuantumUpFocusRunnerTests(unittest.TestCase):
    def test_command_trains_only_q8_d2_quantum_with_up_sampling(self):
        args = SimpleNamespace(
            split_dir=Path("split"),
            device="mps",
            epochs=1,
            grad_accum_steps=1,
            seed=100003,
            train_windows=1500,
            up_density_power=3.0,
            down_penalty=1.0,
        )
        command = build_command(args, Path("output"))
        joined = " ".join(command)
        self.assertIn("--head-type quantum", joined)
        self.assertIn("--n-qubits 8", joined)
        self.assertIn("--quantum-depth 2", joined)
        self.assertIn("--target-train-windows 1500", joined)
        self.assertIn("--training-sampling-strategy up_action_balanced_replacement", joined)
        self.assertNotIn("classical_twin", joined)


if __name__ == "__main__":
    unittest.main()
