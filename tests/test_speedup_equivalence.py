"""The GPU speed-up changes must give exactly the same numbers as the old code.

The pre-change versions of the edited files are kept in
reports/speedup_baseline/ and loaded here as reference implementations.
Runs on CPU; the CUDA variants run too when a GPU is present.

Run: python -m pytest tests/test_speedup_equivalence.py -q
"""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASELINE = ROOT / "reports" / "speedup_baseline"

from plm_special.utils import utils as new_utils  # noqa: E402
from utils import training_metrics as new_metrics  # noqa: E402
from utils.bbr import (  # noqa: E402
    BBR_PHASES,
    BW_CRUISE,
    BW_DOWN,
    BW_UP,
    PHASE_ACTION_INDICES,
    mask_action_logits,
    mask_sequence_logits,
)


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, BASELINE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old_bbr = _load("old_bbr", "utils/bbr.py")
old_utils = _load("old_utils", "plm_special/utils/utils.py")
old_metrics = _load("old_training_metrics", "utils/training_metrics.py")

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def random_phases(n, rng):
    return tuple(rng.choice(BBR_PHASES) for _ in range(n))


def old_stack(embeddings):
    """Verbatim copy of the old rl_policy.forward stacking loop."""
    r, s1, s2, s3, s4, s5, s6, s7, s8, s9, a = embeddings
    stacked_inputs, positions = [], []
    for i in range(r.shape[1]):
        stacked_inputs.append(torch.cat((
            r[0, i:i + 1], s1[0, i:i + 1], s2[0, i:i + 1], s3[0, i:i + 1],
            s4[0, i:i + 1], s5[0, i:i + 1], s6[0, i:i + 1], s7[0, i:i + 1],
            s8[0, i:i + 1], s9[0, i:i + 1], a[0, i:i + 1]), dim=0))
        positions.append((i + 1) * (2 + 9))
    return torch.cat(stacked_inputs, dim=0).unsqueeze(0), positions


def new_stack(embeddings):
    """Same code as the new rl_policy.forward."""
    seq_len, block = embeddings[0].shape[1], 2 + 9
    return torch.stack(embeddings, dim=2).reshape(1, seq_len * block, -1), seq_len, block


class SpeedupEquivalenceTests(unittest.TestCase):
    def test_sequence_mask_values_and_gradients_identical(self):
        rng = random.Random(0)
        for device in DEVICES:
            for batch in (1, 3):
                phases = random_phases(20, rng)
                base = torch.randn(batch, 20, 11, device=device)
                x_old = base.clone().requires_grad_(True)
                x_new = base.clone().requires_grad_(True)
                out_old = old_bbr.mask_sequence_logits(x_old, phases)
                out_new = mask_sequence_logits(x_new, phases)
                self.assertTrue(torch.equal(out_old, out_new))
                weight = torch.randn_like(base)
                (torch.where(torch.isinf(out_old), 0, out_old) * weight).sum().backward()
                (torch.where(torch.isinf(out_new), 0, out_new) * weight).sum().backward()
                self.assertTrue(torch.equal(x_old.grad, x_new.grad))

    def test_single_phase_mask_identical(self):
        for device in DEVICES:
            logits = torch.randn(4, 11, device=device)
            for phase in BBR_PHASES:
                self.assertTrue(torch.equal(
                    old_bbr.mask_action_logits(logits, phase), mask_action_logits(logits, phase)))
            with self.assertRaises(Exception):
                mask_action_logits(logits, "NOT_A_PHASE")

    def test_stacking_and_action_positions_identical(self):
        for device in DEVICES:
            base = [torch.randn(1, 20, 32, device=device) for _ in range(11)]
            old_inputs = [t.clone().requires_grad_(True) for t in base]
            new_inputs = [t.clone().requires_grad_(True) for t in base]
            stacked_old, positions = old_stack(old_inputs)
            stacked_new, seq_len, block = new_stack(new_inputs)
            self.assertTrue(torch.equal(stacked_old, stacked_new))
            # stand-in for the backbone output, then the action-position selection
            hidden_old = torch.tanh(stacked_old * 1.5)
            hidden_new = torch.tanh(stacked_new * 1.5)
            index = torch.as_tensor(positions, dtype=torch.long, device=device)
            used_old = hidden_old[:, index - 2]
            used_new = hidden_new[:, block - 2::block].contiguous()
            self.assertEqual(used_new.shape[1], seq_len)
            self.assertTrue(torch.equal(used_old, used_new))
            weight = torch.randn_like(used_old)
            (used_old * weight).sum().backward()
            (used_new * weight).sum().backward()
            for a, b in zip(old_inputs, new_inputs):
                self.assertTrue(torch.equal(a.grad, b.grad))

    def _fake_batch(self, rng, phases):
        states = [torch.randn(1, 9, 1) for _ in range(20)]
        actions = [torch.tensor(float(rng.choice(sorted(PHASE_ACTION_INDICES[p])))) for p in phases]
        returns = [torch.tensor(rng.random()) for _ in range(20)]
        timesteps = [torch.tensor(i) for i in range(20)]
        return states, actions, returns, timesteps, [(p,) for p in phases]

    def test_batch_processing_identical(self):
        rng = random.Random(1)
        for device in DEVICES:
            phases = random_phases(20, rng)
            batch = self._fake_batch(rng, phases)
            old = old_utils.process_bbr_batch(batch, device=device)
            new = new_utils.process_bbr_batch(batch, device=device)
            self.assertEqual(old[-1], new[-1])
            for a, b in zip(old[:-1], new[:-1]):
                self.assertEqual(a.dtype, b.dtype)
                self.assertEqual(a.device.type, b.device.type)
                self.assertTrue(torch.equal(a.cpu(), b.cpu()))
            bad = list(batch)
            bad[1] = [torch.tensor(11.0)] * 20
            with self.assertRaises(ValueError):
                new_utils.process_bbr_batch(tuple(bad), device=device)

    def test_metric_accumulator_identical(self):
        rng = random.Random(2)
        for device in DEVICES:
            old_acc, new_acc = old_metrics.BBRMetricAccumulator(), new_metrics.BBRMetricAccumulator()
            for _ in range(50):
                phases = random_phases(20, rng)
                labels = torch.tensor(
                    [[rng.choice(sorted(PHASE_ACTION_INDICES[p])) for p in phases]], device=device)
                logits = torch.randn(1, 20, 11, device=device)
                _, loss_old = old_acc.update(logits, labels, phases)
                _, loss_new = new_acc.update(logits, labels, phases)
                self.assertTrue(torch.equal(loss_old, loss_new))
            self.assertEqual(old_acc.compute(), new_acc.compute())


if __name__ == "__main__":
    unittest.main()
