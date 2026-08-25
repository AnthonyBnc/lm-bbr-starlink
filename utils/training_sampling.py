"""Deterministic training-only window sampling for declared BBR ablations."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from pathlib import PurePosixPath
import random

from utils.bbr import ACTION_LEVELS, BBR_PHASES, BW_CRUISE, BW_DOWN, BW_UP


TRAINING_SAMPLING_ORIGINAL = "original"
TRAINING_SAMPLING_UP_FOCUS = "up_action_balanced_replacement"
TRAINING_SAMPLING_CRUISE80 = "cruise80_location_up_action_balanced_replacement"
TRAINING_SAMPLING_OPTIONS = (
    TRAINING_SAMPLING_ORIGINAL,
    TRAINING_SAMPLING_UP_FOCUS,
    TRAINING_SAMPLING_CRUISE80,
)


def _window_values(dataset, index):
    _, actions, _, _, phases = dataset[index]
    return tuple(int(action) for action in actions), tuple(phases)


def summarize_window_indices(dataset, indices):
    phase_counts = Counter()
    action_counts = Counter()
    for index in indices:
        actions, phases = _window_values(dataset, index)
        phase_counts.update(phases)
        action_counts.update(actions)
    return {
        "windows": len(indices),
        "unique_windows": len(set(indices)),
        "repeated_windows": len(indices) - len(set(indices)),
        "phase_positions": {
            phase: phase_counts[phase] for phase in BBR_PHASES
        },
        "action_positions": {
            str(action): action_counts[action] for action in range(ACTION_LEVELS)
        },
        "dataset_indices_sha256": sha256(
            "\n".join(str(index) for index in indices).encode("utf-8")
        ).hexdigest(),
    }


def window_locations(dataset):
    """Resolve one development location for every sequence window."""
    supplied = getattr(dataset, "window_locations", None)
    if supplied is not None:
        if len(supplied) != len(dataset):
            raise ValueError("window_locations length does not match the dataset")
        return tuple(supplied)

    source_files = dataset.exp_pool.metadata.get("source_files", ())
    if not source_files:
        raise ValueError("Cannot resolve window locations without source_files metadata")
    boundaries = []
    end = 0
    for source in source_files:
        end += int(source["intervals"])
        parts = PurePosixPath(source["path"]).parts
        if len(parts) < 2:
            raise ValueError("Unexpected source path: {}".format(source["path"]))
        boundaries.append((end, parts[1]))

    locations = []
    for start in dataset.dataset_indices:
        for end, location in boundaries:
            if start < end:
                locations.append(location)
                break
        else:
            raise ValueError("Window start is outside source-file boundaries")
    return tuple(locations)


def summarize_window_locations(dataset, indices):
    locations = window_locations(dataset)
    counts = Counter(locations[index] for index in indices)
    return {location: counts[location] for location in sorted(counts)}


def _pure_up_buckets(dataset, indices, up_density_power, down_penalty):
    buckets = {}
    weights = {}
    expected_positions = {}
    for action in range(6, ACTION_LEVELS):
        buckets[action] = []
        weights[action] = []
        target_counts = []
        for index in indices:
            actions, phases = _window_values(dataset, index)
            up_actions = Counter(value for value in actions if value >= 6)
            if set(up_actions) != {action}:
                continue
            phase_counts = Counter(phases)
            target_count = up_actions[action]
            weight = (target_count ** up_density_power) / (
                1.0 + down_penalty * phase_counts[BW_DOWN]
            )
            buckets[action].append(index)
            weights[action].append(weight)
            target_counts.append(target_count)
        if not buckets[action]:
            raise ValueError("No pure-UP windows found for action {}".format(action))
        expected_positions[action] = sum(
            weight * count
            for weight, count in zip(weights[action], target_counts)
        ) / sum(weights[action])
    return buckets, weights, expected_positions


def _balanced_action_quotas(base_action_counts, expected_positions, total_windows):
    lower = max(base_action_counts[action] for action in range(6, ACTION_LEVELS))
    upper = lower + total_windows * max(expected_positions.values())
    for _ in range(80):
        target = (lower + upper) / 2.0
        quota_sum = sum(
            max(0.0, (target - base_action_counts[action]) / expected_positions[action])
            for action in range(6, ACTION_LEVELS)
        )
        if quota_sum < total_windows:
            lower = target
        else:
            upper = target
    raw = {
        action: max(
            0.0,
            (upper - base_action_counts[action]) / expected_positions[action],
        )
        for action in range(6, ACTION_LEVELS)
    }
    quotas = {action: int(raw[action]) for action in raw}
    remaining = total_windows - sum(quotas.values())
    order = sorted(
        raw,
        key=lambda action: (raw[action] - quotas[action], -action),
        reverse=True,
    )
    for action in order[:remaining]:
        quotas[action] += 1
    return quotas


def _cruise80_indices(
    dataset,
    target_windows,
    replacement_windows,
    seed,
    up_density_power,
    down_penalty,
):
    if target_windows != len(dataset):
        raise ValueError("CRUISE-80 sampling must preserve the original window budget")
    locations = window_locations(dataset)
    location_indices = {}
    for index, location in enumerate(locations):
        location_indices.setdefault(location, []).append(index)
    location_sizes = {len(indices) for indices in location_indices.values()}
    if len(location_sizes) != 1:
        raise ValueError("CRUISE-80 sampling requires equal source windows per location")
    if replacement_windows < 1 or replacement_windows >= target_windows:
        raise ValueError("replacement_windows must be between 1 and target_windows - 1")
    if replacement_windows % len(location_indices):
        raise ValueError("replacement_windows must divide evenly across locations")

    replacements_per_location = replacement_windows // len(location_indices)
    selected = []
    location_quotas = {}
    for location, indices in sorted(location_indices.items()):
        base_windows = len(indices) - replacements_per_location
        ranked = sorted(
            indices,
            key=lambda index: (
                Counter(_window_values(dataset, index)[1])[BW_UP],
                -Counter(_window_values(dataset, index)[1])[BW_CRUISE],
                -Counter(_window_values(dataset, index)[1])[BW_DOWN],
                index,
            ),
        )
        base_indices = ranked[-base_windows:]
        base_actions = Counter(
            action
            for index in base_indices
            for action in _window_values(dataset, index)[0]
        )
        buckets, weights, expected = _pure_up_buckets(
            dataset,
            indices,
            up_density_power,
            down_penalty,
        )
        quotas = _balanced_action_quotas(
            base_actions,
            expected,
            replacements_per_location,
        )
        location_seed = int(
            sha256("{}:{}".format(seed, location).encode("utf-8")).hexdigest()[:16],
            16,
        )
        generator = random.Random(location_seed)
        extras = []
        for action in range(6, ACTION_LEVELS):
            extras.extend(
                generator.choices(
                    buckets[action],
                    weights=weights[action],
                    k=quotas[action],
                )
            )
        generator.shuffle(extras)
        selected.extend(base_indices + extras)
        location_quotas[location] = {
            str(action): quotas[action] for action in range(6, ACTION_LEVELS)
        }
    random.Random(seed).shuffle(selected)
    return selected, location_quotas


def build_training_window_indices(
    dataset,
    strategy=TRAINING_SAMPLING_ORIGINAL,
    target_windows=0,
    seed=100003,
    up_density_power=3.0,
    down_penalty=1.0,
    replacement_windows=0,
):
    """Select training windows and return an auditable sampling record.

    UP-focused sampling repeats complete sequence windows. It does not alter
    samples, labels, phase masks, rewards, or the validation distribution.
    """
    if strategy not in TRAINING_SAMPLING_OPTIONS:
        raise ValueError("Unknown training sampling strategy: {}".format(strategy))
    if target_windows < 0:
        raise ValueError("target_windows cannot be negative")
    if not len(dataset):
        raise ValueError("Cannot sample an empty training dataset")

    all_indices = list(range(len(dataset)))
    if strategy == TRAINING_SAMPLING_ORIGINAL:
        if target_windows not in (0, len(dataset)):
            raise ValueError(
                "The original strategy requires all {} windows; use the UP-focused "
                "strategy for a declared reduced-window ablation".format(len(dataset))
            )
        selected = all_indices
    elif strategy == TRAINING_SAMPLING_UP_FOCUS:
        if target_windows < 1:
            raise ValueError("UP-focused sampling requires target_windows >= 1")
        if up_density_power <= 0:
            raise ValueError("up_density_power must be positive")
        if down_penalty < 0:
            raise ValueError("down_penalty cannot be negative")

        buckets, bucket_weights, expected_up_positions = _pure_up_buckets(
            dataset, all_indices, up_density_power, down_penalty
        )
        action_quotas = _balanced_action_quotas(
            Counter(), expected_up_positions, target_windows
        )

        generator = random.Random(seed)
        selected = []
        for action in range(6, ACTION_LEVELS):
            selected.extend(
                generator.choices(
                    buckets[action],
                    weights=bucket_weights[action],
                    k=action_quotas[action],
                )
            )
        generator.shuffle(selected)
        location_quotas = None
    else:
        if up_density_power <= 0:
            raise ValueError("up_density_power must be positive")
        if down_penalty < 0:
            raise ValueError("down_penalty cannot be negative")
        selected, location_quotas = _cruise80_indices(
            dataset,
            target_windows,
            replacement_windows,
            seed,
            up_density_power,
            down_penalty,
        )
        action_quotas = None

    record = {
        "strategy": strategy,
        "seed": seed,
        "target_windows": len(selected),
        "with_replacement": strategy != TRAINING_SAMPLING_ORIGINAL,
        "up_density_power": (
            float(up_density_power)
            if strategy != TRAINING_SAMPLING_ORIGINAL
            else None
        ),
        "down_penalty": (
            float(down_penalty)
            if strategy != TRAINING_SAMPLING_ORIGINAL
            else None
        ),
        "up_action_window_quotas": (
            {str(action): action_quotas[action] for action in range(6, ACTION_LEVELS)}
            if strategy == TRAINING_SAMPLING_UP_FOCUS
            else None
        ),
        "replacement_windows": (
            replacement_windows
            if strategy == TRAINING_SAMPLING_CRUISE80
            else None
        ),
        "location_action_window_quotas": (
            location_quotas
            if strategy == TRAINING_SAMPLING_CRUISE80
            else None
        ),
        "source_distribution": summarize_window_indices(dataset, all_indices),
        "selected_distribution": summarize_window_indices(dataset, selected),
        "source_location_windows": summarize_window_locations(dataset, all_indices),
        "selected_location_windows": summarize_window_locations(dataset, selected),
    }
    return selected, record


def load_training_window_plan(
    plan_path,
    dataset,
    strategy,
    target_windows,
    seed,
    up_density_power,
    down_penalty,
    replacement_windows=0,
):
    """Load and verify a precomputed index-only training design."""
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected = {
        "strategy": strategy,
        "target_windows": target_windows,
        "seed": seed,
        "up_density_power": float(up_density_power),
        "down_penalty": float(down_penalty),
    }
    if strategy == TRAINING_SAMPLING_CRUISE80:
        expected["replacement_windows"] = replacement_windows
    actual = {
        "strategy": plan.get("training_sampling", {}).get("strategy"),
        "target_windows": plan.get("training_sampling", {}).get("target_windows"),
        "seed": plan.get("training_sampling", {}).get("seed"),
        "up_density_power": plan.get("training_sampling", {}).get("up_density_power"),
        "down_penalty": plan.get("training_sampling", {}).get("down_penalty"),
    }
    if strategy == TRAINING_SAMPLING_CRUISE80:
        actual["replacement_windows"] = plan.get("training_sampling", {}).get(
            "replacement_windows"
        )
    if actual != expected:
        raise ValueError(
            "Training sampling plan configuration mismatch: expected {}, got {}".format(
                expected, actual
            )
        )

    indices = plan.get("selected_dataset_indices")
    if not isinstance(indices, list) or len(indices) != target_windows:
        raise ValueError("Training sampling plan has an invalid selected-index list")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(dataset)
        for index in indices
    ):
        raise ValueError("Training sampling plan contains an out-of-range dataset index")

    source_distribution = summarize_window_indices(dataset, range(len(dataset)))
    selected_distribution = summarize_window_indices(dataset, indices)
    if source_distribution != plan.get("source_distribution"):
        raise ValueError("Training sampling plan source distribution does not match the pool")
    if selected_distribution != plan.get("selected_distribution"):
        raise ValueError("Training sampling plan selected distribution does not match its indices")
    source_locations = summarize_window_locations(dataset, range(len(dataset)))
    selected_locations = summarize_window_locations(dataset, indices)
    if source_locations != plan.get("source_location_windows"):
        raise ValueError("Training sampling plan source locations do not match the pool")
    if selected_locations != plan.get("selected_location_windows"):
        raise ValueError("Training sampling plan selected locations do not match its indices")

    record = dict(plan["training_sampling"])
    record.update(
        {
            "source_distribution": source_distribution,
            "selected_distribution": selected_distribution,
            "source_location_windows": source_locations,
            "selected_location_windows": selected_locations,
            "plan_path": plan_path.as_posix(),
            "plan_sha256": sha256(plan_path.read_bytes()).hexdigest(),
        }
    )
    return indices, record
