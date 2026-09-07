import unittest

import torch

from utils.bbr import BW_CRUISE, BW_DOWN, BW_UP, mask_sequence_logits
from utils.training_metrics import (
    BBRMetricAccumulator,
    LOSS_WEIGHTING_PHASE_BALANCED,
    evaluate_validation_criteria,
    masked_cross_entropy,
    phase_balanced_weights,
)


class TrainingMetricsTests(unittest.TestCase):
    def test_sequence_mask_rejects_invalid_actions_at_each_position(self):
        logits = torch.arange(33, dtype=torch.float32).reshape(1, 3, 11)
        masked = mask_sequence_logits(logits, (BW_DOWN, BW_CRUISE, BW_UP))
        self.assertEqual(masked[0, 0].argmax().item(), 4)
        self.assertEqual(masked[0, 1].argmax().item(), 5)
        self.assertEqual(masked[0, 2].argmax().item(), 10)
        self.assertTrue(torch.isneginf(masked[0, 0, 5]))
        self.assertTrue(torch.isneginf(masked[0, 2, 5]))

    def test_accumulator_reports_phase_and_action_distributions(self):
        logits = torch.full((1, 3, 11), -5.0)
        logits[0, 0, 0] = 5.0
        logits[0, 1, 5] = 5.0
        logits[0, 2, 10] = 5.0
        labels = torch.tensor([[0, 5, 9]])
        metrics = BBRMetricAccumulator()
        _, loss = metrics.update(logits, labels, (BW_DOWN, BW_CRUISE, BW_UP))
        result = metrics.compute()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(result["samples"], 3)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["per_phase_accuracy"][BW_CRUISE], 1.0)
        self.assertAlmostEqual(result["macro_phase_accuracy"], 2 / 3)
        self.assertEqual(result["prediction_distribution"]["10"], 1)
        self.assertEqual(result["label_distribution"]["9"], 1)
        self.assertEqual(
            result["per_phase_prediction_distribution"][BW_UP]["10"], 1
        )

    def test_validation_criteria_reports_accuracy_and_up_collapse(self):
        metrics = {
            "accuracy": 0.96,
            "macro_phase_accuracy": 0.81,
            "per_phase_accuracy": {BW_DOWN: 1.0, BW_CRUISE: 1.0, BW_UP: 0.43},
            "per_phase_prediction_distribution": {
                BW_UP: {"6": 8, "7": 1, "8": 1}
            },
        }
        result = evaluate_validation_criteria(
            metrics,
            overall_accuracy=0.96,
            up_accuracy=0.421,
            macro_phase_accuracy=0.807,
            max_up_prediction_share=0.9,
        )
        self.assertTrue(result["all_declared_criteria_pass"])
        self.assertAlmostEqual(result["observed"]["max_up_prediction_share"], 0.8)

    def test_phase_balanced_weights_give_equal_phase_total(self):
        weights = phase_balanced_weights(
            (BW_CRUISE, BW_CRUISE, BW_UP, BW_DOWN), batch_size=1, device="cpu"
        )
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]))
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[3]))

    def test_phase_balanced_loss_changes_training_objective(self):
        logits = torch.full((1, 4, 11), -5.0)
        logits[0, 0, 5] = 5.0
        logits[0, 1, 5] = 5.0
        logits[0, 2, 10] = 5.0
        logits[0, 3, 4] = 5.0
        labels = torch.tensor([[5, 5, 6, 0]])
        phases = (BW_CRUISE, BW_CRUISE, BW_UP, BW_DOWN)

        _, unweighted = masked_cross_entropy(logits, labels, phases)
        _, balanced = masked_cross_entropy(
            logits, labels, phases, loss_weighting=LOSS_WEIGHTING_PHASE_BALANCED
        )

        self.assertGreater(float(balanced), float(unweighted))


if __name__ == "__main__":
    unittest.main()
