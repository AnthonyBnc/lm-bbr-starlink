import json
from pathlib import Path
import tempfile
import unittest

from diagnose_head_ablation import build_diagnostic, build_markdown, load_summary


def make_head(head_type, accuracy, macro, loss, phase_accuracy, predictions):
    labels = {str(index): 0 for index in range(11)}
    labels.update({"0": 713, "5": 10457, "6": 229, "7": 141, "8": 178, "9": 110, "10": 172})
    return {
        "checkpoint_reload": "PASS",
        "head_type": head_type,
        "model_role": "gpt_{}".format(head_type),
        "validation_accuracy": accuracy,
        "validation_label_distribution": labels,
        "validation_loss": loss,
        "validation_macro_phase_accuracy": macro,
        "validation_per_phase_accuracy": phase_accuracy,
        "validation_prediction_distribution": predictions,
        "wall_clock_seconds": 3600,
        "tokyo_isolation": "PASS",
    }


class HeadAblationDiagnosticTests(unittest.TestCase):
    def test_quantum_down_collapse_is_explained(self):
        summary = {
            "held_out_location": "Tokyo",
            "model_id": "Qwen/Qwen3.5-4B-Base",
            "model_key": "qwen3_5_4b_base",
            "model_revision": "revision",
            "reportable_result": False,
            "seed": 100003,
            "heads": [
                make_head(
                    "classical",
                    0.95,
                    0.76,
                    0.10,
                    {"BW_DOWN": 1.0, "BW_CRUISE": 1.0, "BW_UP": 0.28},
                    {str(index): 0 for index in range(11)} | {"0": 713, "5": 10457, "10": 830},
                ),
                make_head(
                    "classical_twin",
                    0.94,
                    0.72,
                    0.12,
                    {"BW_DOWN": 1.0, "BW_CRUISE": 1.0, "BW_UP": 0.16},
                    {str(index): 0 for index in range(11)} | {"0": 713, "5": 10457, "7": 830},
                ),
                make_head(
                    "quantum",
                    0.88,
                    0.40,
                    0.17,
                    {"BW_DOWN": 0.0, "BW_CRUISE": 1.0, "BW_UP": 0.21},
                    {str(index): 0 for index in range(11)} | {"3": 713, "5": 10457, "8": 830},
                ),
            ],
        }
        diagnostic = build_diagnostic(summary)
        self.assertEqual(diagnostic["status"], "phase4_qiskit_head_ablation_diagnostic")
        self.assertIn("collapses to action 3", " ".join(diagnostic["findings"]))
        self.assertIn("8 qubits", " ".join(diagnostic["recommendations"]))
        markdown = build_markdown(diagnostic)
        self.assertIn("Phase 4 Qiskit Head-Ablation Diagnostic", markdown)
        self.assertIn("`quantum`", markdown)

    def test_load_summary_requires_heads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps({"status": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "heads"):
                load_summary(path)


if __name__ == "__main__":
    unittest.main()
