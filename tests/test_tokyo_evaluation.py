import unittest

import numpy as np
import torch

from plm_special.data.dataset import ExperienceDataset
from utils.bbr import BW_CRUISE, BW_DOWN, BW_UP
from utils.exp_pool import ExperiencePool
from utils.tokyo_evaluation import evaluate_frozen_policy, paper_surrogate_episode


class DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.ones(()))

    def forward(self, states, actions, returns, timesteps):
        logits = torch.zeros((1, states.shape[1], 11), device=states.device)
        logits[..., 10] = 10.0
        logits[..., 5] = 9.0
        logits[..., 0] = 8.0
        return logits


def _pool():
    pool = ExperiencePool()
    phases = [BW_DOWN, BW_CRUISE, BW_UP, BW_CRUISE]
    actions = [0, 5, 10, 5]
    for index, (phase, action) in enumerate(zip(phases, actions)):
        pool.add(
            state=np.asarray([5, 0, index + 1, 100.0, 2.0, 1, 1, 50, 5]),
            action=action,
            reward=float(index + 1),
            done=index == 3,
            phase=phase,
            sample_id="sample-{}".format(index),
        )
    pool.metadata = {"split_role": "held_out_test", "held_out_location": "Tokyo"}
    return pool


class TokyoEvaluationTests(unittest.TestCase):
    def test_surrogate_equations_return_finite_outputs(self):
        pool = _pool()
        result = paper_surrogate_episode(pool.states, pool.actions)
        self.assertEqual(len(result["predicted_throughput"]), 4)
        self.assertTrue(np.isfinite(result["predicted_retransmissions"]).all())

    def test_evaluator_masks_logits_and_never_builds_gradients(self):
        pool = _pool()
        dataset = ExperienceDataset(pool, gamma=1.0, scale=1000, max_length=2, sample_step=2)
        policy = DummyPolicy()
        metrics, records = evaluate_frozen_policy(
            policy, dataset, pool, "cpu", latency_warmup_batches=0
        )
        self.assertEqual([item["predicted_action"] for item in records], [0, 5, 10, 5])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertTrue(all(not parameter.requires_grad for parameter in policy.parameters()))


if __name__ == "__main__":
    unittest.main()
