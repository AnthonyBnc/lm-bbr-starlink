"""Shared metrics for phase-masked BBR action prediction."""

from collections import Counter

import torch
import torch.nn.functional as F

from utils.bbr import ACTION_LEVELS, BBR_PHASES, mask_sequence_logits, validate_phase_action

LOSS_WEIGHTING_NONE = "none"
LOSS_WEIGHTING_PHASE_BALANCED = "phase_balanced"
LOSS_WEIGHTING_OPTIONS = (LOSS_WEIGHTING_NONE, LOSS_WEIGHTING_PHASE_BALANCED)


def phase_balanced_weights(phases, batch_size, device):
    """Return equal-total phase weights for a flattened batch/sequence."""
    repeated_phases = tuple(phases) * batch_size
    counts = Counter(repeated_phases)
    present_phases = [phase for phase in BBR_PHASES if counts[phase]]
    if not present_phases:
        raise ValueError("Cannot build phase-balanced weights without phases")
    total = len(repeated_phases)
    weights = [
        total / (len(present_phases) * counts[phase])
        for phase in repeated_phases
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def masked_cross_entropy(logits, labels, phases, loss_weighting=LOSS_WEIGHTING_NONE):
    """Compute phase-masked CE with an optional training-only weighting scheme."""
    if loss_weighting not in LOSS_WEIGHTING_OPTIONS:
        raise ValueError("Unknown loss_weighting: {}".format(loss_weighting))
    masked_logits = mask_sequence_logits(logits, phases)
    flat_logits = masked_logits.reshape(-1, ACTION_LEVELS).float()
    flat_labels = labels.reshape(-1).long()
    if flat_labels.numel() != len(phases) * logits.shape[0]:
        raise ValueError("Label count does not match the masked logits")
    per_sample_loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    if loss_weighting == LOSS_WEIGHTING_NONE:
        return masked_logits, per_sample_loss.mean()
    weights = phase_balanced_weights(phases, logits.shape[0], flat_logits.device)
    return masked_logits, (per_sample_loss * weights).sum() / weights.sum()


class BBRMetricAccumulator:
    def __init__(self):
        self.loss_sum = 0.0
        self.samples = 0
        self.correct = 0
        self.phase_samples = Counter()
        self.phase_correct = Counter()
        self.label_distribution = Counter()
        self.prediction_distribution = Counter()

    def update(self, logits, labels, phases, loss_weighting=LOSS_WEIGHTING_NONE):
        masked_logits, loss = masked_cross_entropy(
            logits, labels, phases, loss_weighting=loss_weighting
        )
        flat_logits = masked_logits.reshape(-1, ACTION_LEVELS).float()
        flat_labels = labels.reshape(-1).long()
        loss_sum = F.cross_entropy(flat_logits, flat_labels, reduction="sum")
        predictions = flat_logits.argmax(dim=-1)
        repeated_phases = tuple(phases) * logits.shape[0]
        for phase, label, prediction in zip(
            repeated_phases,
            flat_labels.detach().cpu().tolist(),
            predictions.detach().cpu().tolist(),
        ):
            validate_phase_action(phase, label)
            validate_phase_action(phase, prediction)
            self.samples += 1
            self.loss_sum += float(loss_sum.detach().cpu()) / flat_labels.numel()
            self.correct += int(label == prediction)
            self.phase_samples[phase] += 1
            self.phase_correct[phase] += int(label == prediction)
            self.label_distribution[label] += 1
            self.prediction_distribution[prediction] += 1
        return masked_logits, loss

    def compute(self):
        if not self.samples:
            raise ValueError("Cannot compute BBR metrics without samples")
        return {
            "loss": self.loss_sum / self.samples,
            "accuracy": self.correct / self.samples,
            "samples": self.samples,
            "per_phase_accuracy": {
                phase: (
                    self.phase_correct[phase] / self.phase_samples[phase]
                    if self.phase_samples[phase]
                    else None
                )
                for phase in BBR_PHASES
            },
            "per_phase_samples": {
                phase: self.phase_samples[phase] for phase in BBR_PHASES
            },
            "label_distribution": {
                str(action): self.label_distribution[action]
                for action in range(ACTION_LEVELS)
            },
            "prediction_distribution": {
                str(action): self.prediction_distribution[action]
                for action in range(ACTION_LEVELS)
            },
        }
