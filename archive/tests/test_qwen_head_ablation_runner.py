from pathlib import Path
from types import SimpleNamespace
import unittest

from run_qwen_head_ablation import HEAD_ORDER, build_command, output_name


class QwenHeadAblationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            model_key="qwen3_5_4b_base",
            split_dir=Path("split"),
            device="mps",
            rank=8,
            alpha=32,
            dropout=0.05,
            epochs=1,
            sequence_length=20,
            sample_step=20,
            state_feature_dim=256,
            learning_rate=1e-4,
            weight_decay=1e-4,
            grad_accum_steps=1,
            seed=100003,
            n_qubits=4,
            quantum_depth=2,
            quantum_ansatz="trainable_ry_rz_layers",
            max_train_steps=1,
            max_validation_steps=1,
            loss_weighting="phase_balanced",
            training_sampling_strategy="original",
            target_train_windows=0,
            up_density_power=3.0,
            down_window_penalty=1.0,
            replacement_train_windows=0,
            training_sampling_plan=None,
            limited=True,
        )

    def test_head_order_contains_required_ablation_heads(self):
        self.assertEqual(HEAD_ORDER, ("classical", "classical_twin", "quantum"))

    def test_quantum_command_uses_same_qwen_protocol_and_quantum_head(self):
        command = build_command(self.args, "quantum", Path("output"))
        self.assertIn("qwen3_5_4b_base", command)
        self.assertIn("bfloat16", command)
        self.assertIn("--head-type", command)
        self.assertIn("quantum", command)
        self.assertIn("head_ablation_smoke", command)
        self.assertIn("--loss-weighting", command)
        self.assertIn("phase_balanced", command)
        self.assertIn("--max-train-steps", command)
        self.assertIn("--quantum-ansatz", command)
        self.assertIn("trainable_ry_rz_layers", command)

    def test_output_name_marks_limited_smoke(self):
        self.assertEqual(
            output_name("classical_twin", 8, 1, 100003, True),
            "qwen_gpt_classical_twin_rank8_epochs1_seed100003_smoke",
        )


if __name__ == "__main__":
    unittest.main()
