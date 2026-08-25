"""Deterministic preprocessing for the paper's Starlink BBR experience pool."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import pickle

import numpy as np

from utils.bbr import (
    BW_CRUISE,
    BW_DOWN,
    BW_UP,
    PACING_GAINS,
    action_index_to_gain,
    valid_action_indices,
)
from utils.exp_pool import ExperiencePool


DEVELOPMENT_LOCATIONS = ("Ohio", "SaoPaulo", "London", "Mumbai", "Sydney")
HELD_OUT_LOCATION = "Tokyo"
LOCATION_FLAGS = {location: index for index, location in enumerate(DEVELOPMENT_LOCATIONS)}
STREAM_FLAGS = {
    "downlink-sequential-logs": 0,
    "uplink-sequential-logs": 1,
    "downlink-competitive-logs": 2,
    "uplink-competitive-logs": 3,
}
REQUIRED_TELEMETRY = (
    "end",
    "bits_per_second",
    "retransmits",
    "snd_cwnd",
    "snd_wnd",
    "rtt",
    "rttvar",
)


@dataclass(frozen=True)
class PaperPreprocessingConfig:
    half_window: int = 10
    phase_threshold_scale: float = 0.7
    cruise_samples_after_down: int = 6
    retransmission_weight: float = 0.5
    aggressive_probe_weight: float = 0.1
    probe_growth: float = 1.5
    probe_sharpness: float = 5.0
    loss_floor: float = 1e-3
    probe_down_reduction: float = 0.5

    def __post_init__(self):
        if self.half_window < 1:
            raise ValueError("half_window must be at least 1")


def read_iperf3_json(path):
    """Read the iperf JSON object while preserving surrounding diagnostics."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="strict")
    decoder = json.JSONDecoder()
    last_error = None
    for start, character in enumerate(raw):
        if character != "{":
            continue
        try:
            document, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(document, dict) and "intervals" in document:
            return document, raw[:start], raw[end:]
    detail = last_error or "no JSON object with an intervals field"
    raise ValueError("Invalid iperf3 JSON in {}: {}".format(path, detail))


def parse_iperf3_intervals(path):
    document, leading_text, trailing_text = read_iperf3_json(path)
    rows = []
    for interval_index, interval in enumerate(document.get("intervals", [])):
        streams = interval.get("streams") or []
        sender_streams = [stream for stream in streams if stream.get("sender") is True]
        if len(sender_streams) != 1:
            raise ValueError(
                "Expected one sender stream at interval {} in {}, got {}".format(
                    interval_index, path, len(sender_streams)
                )
            )
        stream = sender_streams[0]
        missing = [field for field in REQUIRED_TELEMETRY if field not in stream]
        if missing:
            raise ValueError(
                "Missing telemetry fields at interval {} in {}: {}".format(
                    interval_index, path, ", ".join(missing)
                )
            )
        row = {field: float(stream[field]) for field in REQUIRED_TELEMETRY}
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError("Non-finite telemetry at interval {} in {}".format(interval_index, path))
        rows.append(row)
    if not rows:
        raise ValueError("No iperf3 intervals found in {}".format(path))
    return rows, leading_text, trailing_text


def _window(values, index, half_window):
    start = max(0, index - half_window)
    end = min(len(values), index + half_window + 1)
    return values[start:end]


def _rolling_mean(values, half_window):
    return np.asarray(
        [np.mean(_window(values, index, half_window)) for index in range(len(values))],
        dtype=np.float64,
    )


def _rolling_percentile(values, half_window, percentile=95):
    return np.asarray(
        [
            np.percentile(_window(values, index, half_window), percentile)
            for index in range(len(values))
        ],
        dtype=np.float64,
    )


def _rolling_min(values, half_window):
    return np.asarray(
        [np.min(_window(values, index, half_window)) for index in range(len(values))],
        dtype=np.float64,
    )


def _rolling_max(values, half_window):
    return np.asarray(
        [np.max(_window(values, index, half_window)) for index in range(len(values))],
        dtype=np.float64,
    )


def detect_bbr_phases(throughput, config=PaperPreprocessingConfig()):
    """Reproduce Algorithm 1's rolling-deviation macro-phase detector."""
    throughput = np.asarray(throughput, dtype=np.float64)
    if throughput.ndim != 1 or len(throughput) < 3:
        raise ValueError("Phase detection requires at least three throughput samples")

    deviation = throughput - _rolling_mean(throughput, config.half_window)
    sigma = float(np.std(deviation))
    upper = config.phase_threshold_scale * sigma
    lower = -upper
    local_maxima = {
        index
        for index in range(1, len(deviation) - 1)
        if deviation[index] > upper
        and deviation[index] > deviation[index - 1]
        and deviation[index] >= deviation[index + 1]
    }
    local_minima = {
        index
        for index in range(1, len(deviation) - 1)
        if deviation[index] < lower
        and deviation[index] < deviation[index - 1]
        and deviation[index] <= deviation[index + 1]
    }

    phases = [BW_CRUISE] * len(throughput)
    cursor = 1
    while cursor < len(throughput) - 1:
        if cursor not in local_maxima:
            cursor += 1
            continue
        phases[cursor] = BW_UP
        down_index = next((index for index in sorted(local_minima) if index > cursor), None)
        if down_index is None:
            cursor += 1
            continue
        phases[down_index] = BW_DOWN
        cruise_end = min(
            len(phases), down_index + 1 + config.cruise_samples_after_down
        )
        phases[down_index + 1 : cruise_end] = [BW_CRUISE] * (
            cruise_end - down_index - 1
        )
        cursor = cruise_end
    return phases


def _softplus(value):
    return float(np.logaddexp(0.0, value))


def _nearest_action_index(target_gain, phase):
    """Select the nearest phase-safe discrete action for Equations 2-3."""
    permitted_actions = valid_action_indices(phase)
    return min(
        permitted_actions,
        key=lambda action_index: (
            abs(action_index_to_gain(action_index) - target_gain),
            action_index,
        ),
    )


def _equation_2_3_action(phase, utilization_proxy):
    if phase == BW_UP:
        target_gain = 3.0 / (utilization_proxy + 2.0)
    elif phase == BW_DOWN:
        target_gain = (utilization_proxy + 1.0) / 2.0
    elif phase == BW_CRUISE:
        target_gain = 1.0
    else:
        raise ValueError("Unknown BBR phase: {}".format(phase))
    return _nearest_action_index(target_gain, phase)


def _candidate_reward(index, action_index, utilization, retransmit_ref, retransmit_min, retransmit_max, config):
    gain = PACING_GAINS[action_index]
    max_probe_strength = _softplus(
        config.probe_sharpness * (max(PACING_GAINS) - 1.0)
    ) - _softplus(0.0)
    probe_strength = max(
        _softplus(config.probe_sharpness * (gain - 1.0)) - _softplus(0.0),
        0.0,
    )
    probe_component = (
        (probe_strength / max_probe_strength) ** config.probe_growth
        if max_probe_strength > 0
        else 0.0
    )
    loss_factor = utilization[index] * (
        config.loss_floor + (1.0 - config.loss_floor) * probe_component
    )
    if gain < 1.0:
        loss_factor *= 1.0 - config.probe_down_reduction * (1.0 - gain)
    loss_factor = min(max(loss_factor, 0.0), 1.0)
    predicted_retransmits = retransmit_min + (
        retransmit_max[index] - retransmit_min
    ) * loss_factor
    predicted_throughput_ratio = min(gain, 1.0)
    return (
        predicted_throughput_ratio
        - config.retransmission_weight
        * predicted_retransmits
        / retransmit_ref[index]
        - config.aggressive_probe_weight
        * utilization[index]
        * max(gain - 1.0, 0.0)
    )


def build_paper_labels(rows, config=PaperPreprocessingConfig()):
    throughput = np.asarray([row["bits_per_second"] for row in rows], dtype=np.float64)
    retransmits = np.asarray([row["retransmits"] for row in rows], dtype=np.float64)
    rtt = np.asarray([row["rtt"] for row in rows], dtype=np.float64)
    phases = detect_bbr_phases(throughput, config)

    throughput_ref = _rolling_percentile(throughput, config.half_window)
    retransmit_ref = _rolling_percentile(retransmits, config.half_window) + 1.0
    rtt_min = _rolling_min(rtt, config.half_window)
    queue_delay = np.maximum(rtt - rtt_min, 0.0)
    queue_ref = np.minimum(
        _rolling_percentile(queue_delay, config.half_window), 0.15 * rtt_min
    )
    rate_utilization = np.divide(
        throughput,
        throughput_ref,
        out=np.zeros_like(throughput),
        where=throughput_ref > 0,
    )
    rate_utilization = np.minimum(rate_utilization, 1.0)
    delay_utilization = np.divide(
        queue_delay,
        queue_ref,
        out=np.zeros_like(queue_delay),
        where=queue_ref > 0,
    )
    delay_utilization = np.minimum(delay_utilization, 1.0)
    utilization = np.maximum(rate_utilization, delay_utilization)
    retransmit_min = float(np.min(retransmits))
    retransmit_max = _rolling_max(retransmits, config.half_window)
    max_throughput = float(np.max(throughput))
    utilization_proxy = np.divide(
        throughput,
        max_throughput,
        out=np.zeros_like(throughput),
        where=max_throughput > 0,
    )

    labels = []
    rewards = []
    for index, phase in enumerate(phases):
        action_index = _equation_2_3_action(phase, utilization_proxy[index])
        reward = _candidate_reward(
            index,
            action_index,
            utilization,
            retransmit_ref,
            retransmit_min,
            retransmit_max,
            config,
        )
        labels.append(action_index)
        rewards.append(float(reward))
    return phases, labels, rewards


def infer_trace_identity(path, raw_root):
    path = Path(path).resolve()
    raw_root = Path(raw_root).resolve()
    try:
        relative_path = path.relative_to(raw_root)
    except ValueError as exc:
        raise ValueError("Trace is outside raw data root: {}".format(path)) from exc
    parts = relative_path.parts
    if len(parts) < 3:
        raise ValueError("Unexpected Starlink trace path: {}".format(relative_path))
    stream_name, location = parts[0], parts[1]
    if location == HELD_OUT_LOCATION:
        raise ValueError("Tokyo is held out and cannot enter development preprocessing")
    if location not in LOCATION_FLAGS:
        raise ValueError("Unknown development location: {}".format(location))
    if stream_name not in STREAM_FLAGS:
        raise ValueError("Unknown stream group: {}".format(stream_name))
    return relative_path, location, stream_name


def build_experience_pool(trace_paths, raw_root, config=PaperPreprocessingConfig()):
    trace_paths = tuple(trace_paths)
    if not trace_paths:
        raise ValueError("At least one development trace is required")
    pool = ExperiencePool()
    source_files = []
    for trace_path in sorted(Path(path) for path in trace_paths):
        relative_path, location, stream_name = infer_trace_identity(trace_path, raw_root)
        rows, leading_text, trailing_text = parse_iperf3_intervals(trace_path)
        phases, labels, rewards = build_paper_labels(rows, config)
        source_files.append(
            {
                "path": relative_path.as_posix(),
                "sha256": sha256(Path(trace_path).read_bytes()).hexdigest(),
                "intervals": len(rows),
                "leading_diagnostic_bytes": len(leading_text.encode("utf-8")),
                "trailing_diagnostic_bytes": len(trailing_text.encode("utf-8")),
            }
        )
        for interval_index, (row, phase, action, reward) in enumerate(
            zip(rows, phases, labels, rewards)
        ):
            state = np.asarray(
                [
                    LOCATION_FLAGS[location],
                    STREAM_FLAGS[stream_name],
                    row["end"],
                    row["bits_per_second"],
                    row["retransmits"],
                    row["snd_cwnd"],
                    row["snd_wnd"],
                    row["rtt"],
                    row["rttvar"],
                ],
                dtype=np.float32,
            )
            sample_id = sha256(
                "{}:{}".format(relative_path.as_posix(), interval_index).encode("utf-8")
            ).hexdigest()
            pool.add(
                state=state,
                action=action,
                reward=reward,
                done=interval_index == len(rows) - 1,
                phase=phase,
                sample_id=sample_id,
            )
    if len(set(pool.sample_ids)) != len(pool.sample_ids):
        raise ValueError("Generated sample IDs are not unique")
    pool.metadata = {
        "dataset_version": "slm-bbr-paper-w10-eq2-3-labels-v1-exploratory",
        "paper": "arXiv:2607.07142v1",
        "label_rule": "Equations 2-3 nearest phase-safe gain; Equation 16 reward for selected action",
        "config": asdict(config),
        "location_flags": LOCATION_FLAGS,
        "stream_flags": STREAM_FLAGS,
        "held_out_location": HELD_OUT_LOCATION,
        "source_files": source_files,
    }
    return pool


def write_experience_pool(pool, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(pool, stream)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = dict(pool.metadata)
    manifest["samples"] = len(pool)
    manifest["experience_pool_sha256"] = sha256(output_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
