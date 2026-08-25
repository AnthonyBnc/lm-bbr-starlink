"""Shared BBR action-space and phase-mask definitions.

This module is the authoritative, model-independent implementation of the
paper's discrete pacing-gain contract. Model code should consume these values
rather than defining its own action mapping.
"""

from numbers import Integral


PACING_GAINS = (
    0.90,
    0.92,
    0.94,
    0.96,
    0.98,
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.25,
)
ACTION_LEVELS = len(PACING_GAINS)

BW_DOWN = "BW_DOWN"
BW_CRUISE = "BW_CRUISE"
BW_UP = "BW_UP"
BBR_PHASES = (BW_DOWN, BW_CRUISE, BW_UP)

PHASE_ACTION_INDICES = {
    BW_DOWN: (0, 1, 2, 3, 4),
    BW_CRUISE: (5,),
    BW_UP: (6, 7, 8, 9, 10),
}


def validate_phase(phase):
    """Validate and return one BBR macro-phase name."""
    if phase not in PHASE_ACTION_INDICES:
        expected = ", ".join(BBR_PHASES)
        raise ValueError("Unknown BBR phase {!r}; expected one of: {}".format(phase, expected))
    return phase


def validate_action_index(action_index):
    """Validate and return one integer action index."""
    if isinstance(action_index, bool) or not isinstance(action_index, Integral):
        raise TypeError("Action index must be an integer, got {!r}".format(action_index))
    if not 0 <= action_index < ACTION_LEVELS:
        raise ValueError(
            "Action index {} is outside the valid range [0, {}]".format(
                action_index, ACTION_LEVELS - 1
            )
        )
    return int(action_index)


def action_index_to_gain(action_index):
    """Map a validated action index to its fixed pacing gain."""
    return PACING_GAINS[validate_action_index(action_index)]


def valid_action_indices(phase):
    """Return the action indices permitted in a BBR macro-phase."""
    return PHASE_ACTION_INDICES[validate_phase(phase)]


def validate_phase_action(phase, action_index):
    """Validate that an action index is feasible for the supplied phase."""
    phase = validate_phase(phase)
    action_index = validate_action_index(action_index)
    if action_index not in PHASE_ACTION_INDICES[phase]:
        raise ValueError(
            "Action index {} (gain {}) is not valid for phase {}".format(
                action_index, PACING_GAINS[action_index], phase
            )
        )
    return action_index


def valid_pacing_gains(phase):
    """Return the pacing gains permitted in a BBR macro-phase."""
    return tuple(PACING_GAINS[index] for index in valid_action_indices(phase))


def phase_action_mask(phase):
    """Return an immutable 11-element boolean mask for one BBR phase."""
    valid = set(valid_action_indices(phase))
    return tuple(index in valid for index in range(ACTION_LEVELS))


def mask_action_values(values, phase, invalid_value=float("-inf")):
    """Mask a one-dimensional action-value sequence without mutating it."""
    if len(values) != ACTION_LEVELS:
        raise ValueError(
            "Expected {} action values, got {}".format(ACTION_LEVELS, len(values))
        )
    mask = phase_action_mask(phase)
    return tuple(value if is_valid else invalid_value for value, is_valid in zip(values, mask))


def mask_action_logits(logits, phase, invalid_value=float("-inf")):
    """Return PyTorch logits with invalid actions masked on the final axis.

    ``phase`` is one phase name applied across all leading dimensions. PyTorch
    is imported lazily so the action contract remains usable by data tooling
    and tests that do not install model dependencies.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("mask_action_logits requires PyTorch") from exc

    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim == 0 or logits.shape[-1] != ACTION_LEVELS:
        raise ValueError(
            "Expected logits with final dimension {}, got {}".format(
                ACTION_LEVELS, tuple(logits.shape)
            )
        )

    mask = torch.tensor(phase_action_mask(phase), dtype=torch.bool, device=logits.device)
    return logits.masked_fill(~mask, invalid_value)


def mask_sequence_logits(logits, phases, invalid_value=float("-inf")):
    """Mask one BBR phase per sequence position for a batch of logits."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("mask_sequence_logits requires PyTorch") from exc

    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim != 3 or logits.shape[-1] != ACTION_LEVELS:
        raise ValueError(
            "Expected [batch, sequence, {}] logits, got {}".format(
                ACTION_LEVELS, tuple(logits.shape)
            )
        )
    if len(phases) != logits.shape[1]:
        raise ValueError(
            "Phase count {} does not match sequence length {}".format(
                len(phases), logits.shape[1]
            )
        )
    return torch.stack(
        [
            mask_action_logits(logits[:, position, :], phase, invalid_value)
            for position, phase in enumerate(phases)
        ],
        dim=1,
    )
