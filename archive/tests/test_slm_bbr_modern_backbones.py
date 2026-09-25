from pathlib import Path
from types import SimpleNamespace
import unittest

from run_slm_bbr_modern_backbones import (
    GEMMA_3_270M_REPLACEMENT_ORDER,
    MODEL_ORDER,
    MODEL_SETS,
    UNDER_400M_MODEL_ORDER,
    build_command,
    output_name,
)


class SlmBbrModernBackbonesTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            split_dir=Path("split"),
            device="mps",
            rank=128,
            alpha=32,
            dropout=0.05,
            epochs=150,
            sequence_length=20,
            sample_step=20,
            state_feature_dim=256,
            learning_rate=1e-4,
            weight_decay=1e-4,
            warmup_steps=2000,
            grad_accum_steps=32,
            seed=100003,
            preflight=False,
            early_stopping_patience=0,
            early_stopping_min_delta=0.0,
            success_overall_accuracy=0.96,
            success_up_accuracy=0.42168674698795183,
            success_macro_phase_accuracy=0.8072289156626506,
            max_up_prediction_share=0.90,
        )

    def test_approved_order_has_exactly_four_local_models(self):
        self.assertEqual(
            MODEL_ORDER,
            (
                "lfm2_5_2_6b",
                "llama_3_2_3b",
                "qwen3_5_4b_base",
                "gemma_3_4b_pt",
            ),
        )

    def test_under_400m_order_has_requested_models(self):
        self.assertEqual(
            UNDER_400M_MODEL_ORDER,
            (
                "granite_4_0_350m",
                "pleias_rag_350m",
                "lfm2_5_350m",
                "granite_4_0_h_350m",
            ),
        )
        self.assertEqual(MODEL_SETS["under-400m"], UNDER_400M_MODEL_ORDER)
        self.assertTrue(set(UNDER_400M_MODEL_ORDER).isdisjoint(MODEL_ORDER))

    def test_under_400m_command_uses_registered_model_dtype(self):
        command = build_command(
            self.args,
            "lfm2_5_350m",
            Path("output"),
        )
        self.assertEqual(command[command.index("--model-key") + 1], "lfm2_5_350m")
        self.assertEqual(command[command.index("--dtype") + 1], "float16")

    def test_command_preserves_slm_bbr_protocol_settings(self):
        command = build_command(self.args, MODEL_ORDER[0], Path("output"))
        self.assertEqual(command[command.index("--rank") + 1], "128")
        self.assertEqual(command[command.index("--epochs") + 1], "150")
        self.assertEqual(command[command.index("--warmup-steps") + 1], "2000")
        self.assertEqual(command[command.index("--grad-accum-steps") + 1], "32")
        self.assertIn("slm_bbr_modern_backbone_extension", command)
        self.assertIn("original", command)
        self.assertIn("none", command)
        self.assertEqual(
            command[command.index("--success-overall-accuracy") + 1], "0.96"
        )

    def test_resume_command_uses_target_total_epoch_and_source_run(self):
        self.args.epochs = 6
        command = build_command(
            self.args,
            MODEL_ORDER[0],
            Path("output"),
            resume_run=Path("phase3/model"),
        )
        self.assertEqual(command[command.index("--epochs") + 1], "6")
        self.assertEqual(
            command[command.index("--resume-from-run") + 1], "phase3/model"
        )

    def test_preflight_limits_both_train_and_validation(self):
        self.args.preflight = True
        command = build_command(self.args, MODEL_ORDER[0], Path("output"))
        self.assertEqual(command[command.index("--epochs") + 1], "1")
        self.assertEqual(command[command.index("--max-train-steps") + 1], "1")
        self.assertEqual(command[command.index("--max-validation-steps") + 1], "1")
        self.assertTrue(output_name("model", 128, 150, 100003, True).endswith("preflight"))

    def test_resume_preflight_keeps_target_epoch_and_limits_steps(self):
        self.args.preflight = True
        self.args.epochs = 6
        command = build_command(
            self.args,
            MODEL_ORDER[0],
            Path("output"),
            resume_run=Path("phase3/model"),
        )
        self.assertEqual(command[command.index("--epochs") + 1], "6")
        self.assertEqual(command[command.index("--max-train-steps") + 1], "1")
        self.assertEqual(command[command.index("--max-validation-steps") + 1], "1")
    
    def test_gemma_replacement_set_contains_only_gemma_3_270m(self):
        self.assertEqual(
            GEMMA_3_270M_REPLACEMENT_ORDER,
            ("gemma_3_270m",),
        )
        self.assertEqual(
            MODEL_SETS["gemma-3-270m-replacement"],
            GEMMA_3_270M_REPLACEMENT_ORDER,
        )

if __name__ == "__main__":
    unittest.main()
