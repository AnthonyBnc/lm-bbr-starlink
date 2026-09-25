"""Shared held-out metrics and paper surrogate calculations for Tokyo."""

from collections import defaultdict
from hashlib import sha256
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from plm_special.utils.utils import process_bbr_batch
from utils.bbr import action_index_to_gain
from utils.starlink_preprocessing import PaperPreprocessingConfig
from utils.training_metrics import BBRMetricAccumulator


def sample_ids_sha256(sample_ids):
    return sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def _window(values, index, half_window):
    start = max(0, index - half_window)
    end = min(len(values), index + half_window + 1)
    return values[start:end]


def _rolling(values, half_window, operation):
    return np.asarray(
        [operation(_window(values, i, half_window)) for i in range(len(values))],
        dtype=np.float64,
    )


def paper_surrogate_episode(states, action_indices, config=PaperPreprocessingConfig()):
    """Apply paper Equations 21-27 to one raw trace episode."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 9:
        raise ValueError("Expected episode states with shape [samples, 9]")
    if len(states) != len(action_indices):
        raise ValueError("Surrogate actions must cover the complete episode")
    throughput = states[:, 3]
    retransmits = states[:, 4]
    rtt = states[:, 7]
    half_window = config.half_window
    b_cap = _rolling(throughput, half_window, lambda values: np.percentile(values, 95))
    rtt_min = _rolling(rtt, half_window, np.min)
    queue_delay = np.maximum(rtt - rtt_min, 0.0)
    queue_ref = np.minimum(
        _rolling(queue_delay, half_window, lambda values: np.percentile(values, 95)),
        0.15 * rtt_min,
    )
    rate_utilization = np.divide(
        throughput, b_cap, out=np.zeros_like(throughput), where=b_cap > 0
    )
    delay_utilization = np.divide(
        queue_delay,
        queue_ref,
        out=np.zeros_like(queue_delay),
        where=queue_ref > 0,
    )
    utilization = np.maximum(
        np.minimum(rate_utilization, 1.0), np.minimum(delay_utilization, 1.0)
    )
    tau_min = float(np.min(retransmits))
    tau_max = _rolling(retransmits, half_window, np.max)
    gains = np.asarray([action_index_to_gain(action) for action in action_indices])

    max_gain = 1.25
    softplus_zero = float(np.logaddexp(0.0, 0.0))
    maximum_strength = max(
        float(np.logaddexp(0.0, config.probe_sharpness * (max_gain - 1.0)))
        - softplus_zero,
        0.0,
    )
    strength = np.maximum(
        np.logaddexp(0.0, config.probe_sharpness * (gains - 1.0))
        - softplus_zero,
        0.0,
    )
    phi = np.power(strength / maximum_strength, config.probe_growth)
    loss_factor = utilization * (
        config.loss_floor + (1.0 - config.loss_floor) * phi
    )
    below_one = gains < 1.0
    loss_factor[below_one] *= 1.0 - config.probe_down_reduction * (
        1.0 - gains[below_one]
    )
    loss_factor = np.clip(loss_factor, 0.0, 1.0)
    return {
        "predicted_throughput": b_cap * np.minimum(gains, 1.0),
        "predicted_retransmissions": tau_min + (tau_max - tau_min) * loss_factor,
        "bottleneck_bandwidth": b_cap,
        "utilization": utilization,
    }


def _synchronize(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif str(device) == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def evaluate_frozen_policy(policy, dataset, pool, device, latency_warmup_batches=5):
    """Evaluate one frozen policy on fixed windows with no parameter updates."""
    if pool.metadata.get("split_role") != "held_out_test":
        raise ValueError("Tokyo evaluator requires a held_out_test pool")
    if pool.metadata.get("held_out_location") != "Tokyo":
        raise ValueError("Tokyo evaluator received the wrong held-out location")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    metrics = BBRMetricAccumulator()
    records = []
    batch_latencies = []
    predicted_by_pool_index = {}

    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            states, actions, returns, timesteps, labels, phases = process_bbr_batch(
                batch, device=device
            )
            _synchronize(device)
            started = time.perf_counter()
            logits = policy(states, actions, returns, timesteps)
            _synchronize(device)
            elapsed = time.perf_counter() - started
            masked_logits, _ = metrics.update(logits, labels, phases)
            predictions = masked_logits.argmax(dim=-1).reshape(-1).cpu().tolist()
            targets = labels.reshape(-1).cpu().tolist()
            pool_start = dataset.dataset_indices[batch_index]
            if batch_index >= latency_warmup_batches:
                batch_latencies.append(elapsed)
            for offset, (phase, target, prediction) in enumerate(
                zip(phases, targets, predictions)
            ):
                pool_index = pool_start + offset
                if pool_index in predicted_by_pool_index:
                    raise ValueError("Evaluation windows overlap at pool index {}".format(pool_index))
                predicted_by_pool_index[pool_index] = prediction
                state = pool.states[pool_index]
                records.append(
                    {
                        "sample_id": pool.sample_ids[pool_index],
                        "pool_index": pool_index,
                        "phase": phase,
                        "target_action": target,
                        "target_gain": action_index_to_gain(target),
                        "predicted_action": prediction,
                        "predicted_gain": action_index_to_gain(prediction),
                        "correct": int(target == prediction),
                        "stream_flag": int(state[1]),
                        "observed_throughput": float(state[3]),
                        "observed_retransmissions": float(state[4]),
                    }
                )

    # Surrogate references must use each complete raw episode, even when the
    # final partial sequence is excluded from classification metrics.
    episode_start = 0
    surrogate_by_index = {}
    for episode_end, done in enumerate(pool.dones):
        if not done:
            continue
        stop = episode_end + 1
        episode_actions = [
            predicted_by_pool_index.get(index, pool.actions[index])
            for index in range(episode_start, stop)
        ]
        surrogate = paper_surrogate_episode(
            pool.states[episode_start:stop], episode_actions
        )
        for offset in range(stop - episode_start):
            pool_index = episode_start + offset
            surrogate_by_index[pool_index] = {
                "surrogate_throughput": float(surrogate["predicted_throughput"][offset]),
                "surrogate_retransmissions": float(
                    surrogate["predicted_retransmissions"][offset]
                ),
            }
        episode_start = stop
    if episode_start != len(pool):
        raise ValueError("Tokyo pool ends with an incomplete episode")
    for record in records:
        record.update(surrogate_by_index[record["pool_index"]])

    result = metrics.compute()
    evaluated_sample_ids = [record["sample_id"] for record in records]
    result["sample_ids_sha256"] = sample_ids_sha256(evaluated_sample_ids)
    result["latency"] = {
        "warmup_batches_excluded": min(latency_warmup_batches, len(dataset)),
        "timed_batches": len(batch_latencies),
        "mean_milliseconds_per_action": (
            1000.0 * sum(batch_latencies) / (len(batch_latencies) * dataset.max_length)
            if batch_latencies
            else None
        ),
    }
    stream_names = {
        int(value): key for key, value in pool.metadata.get("stream_flags", {}).items()
    }
    grouped = defaultdict(lambda: {"samples": 0, "throughput": [], "retransmissions": []})
    for record in records:
        group_name = stream_names.get(
            record["stream_flag"], "stream_flag_{}".format(record["stream_flag"])
        )
        record["stream_group"] = group_name
        group = grouped[group_name]
        group["samples"] += 1
        group["throughput"].append(record["surrogate_throughput"])
        group["retransmissions"].append(record["surrogate_retransmissions"])
    result["paper_surrogate_by_stream_group"] = {
        key: {
            "samples": value["samples"],
            "mean_throughput": float(np.mean(value["throughput"])),
            "mean_retransmissions": float(np.mean(value["retransmissions"])),
        }
        for key, value in sorted(grouped.items())
    }
    return result, records
