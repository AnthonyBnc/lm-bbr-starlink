from pathlib import Path
from types import SimpleNamespace
import unittest

from run_quantum_smoke import build_command


class QuantumSmokeRunnerTests(unittest.TestCase):
    def test_command_uses_quantum_head_and_limited_steps(self):
        args = SimpleNamespace(
            model_key="qwen3_5_4b_base",
            split_dir=Path("split"),
            output_dir=Path("output"),
            device="mps",
            rank=8,
            alpha=32,
            dropout=0.05,
            sequence_length=20,
            sample_step=20,
            state_feature_dim=256,
            learning_rate=1e-4,
            weight_decay=1e-4,
            seed=100003,
            max_train_steps=1,
            max_validation_steps=1,
            n_qubits=4,
            quantum_depth=2,
            quantum_ansatz="trainable_ry_rz_layers",
        )
        command = build_command(args)
        self.assertIn("--head-type", command)
        self.assertIn("quantum", command)
        self.assertIn("--max-train-steps", command)
        self.assertIn("--max-validation-steps", command)
        self.assertIn("quantum_smoke", command)
        self.assertIn("bfloat16", command)
        self.assertIn("--quantum-ansatz", command)
        self.assertIn("trainable_ry_rz_layers", command)


if __name__ == "__main__":
    unittest.main()
