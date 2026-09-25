"""Boxplot of real macro-phase time-share (BW_UP / BW_CRUISE / BW_DOWN)
across all 6 Starlink testbed locations, computed directly from the raw BBR
traces using the project's own phase detector.

Why this exists: the frozen-checkpoint model comparison
(generate_model_comparison_boxplots.py, gpt_classical vs gpt_quantum) can
NOT be broken down by location, because the models were only ever run on
the Tokyo held-out set (see evaluate_tokyo_models.py: "held_out_location":
"Tokyo", and utils/starlink_preprocessing.py: HELD_OUT_LOCATION = "Tokyo",
DEVELOPMENT_LOCATIONS = Ohio/SaoPaulo/London/Mumbai/Sydney). Ohio, SaoPaulo,
London, Mumbai, and Sydney were used only to build the *training* experience
pool -- there is no predictions.csv for them. What DOES exist for all 6
locations is the raw BBR telemetry itself, so this figure characterizes
that instead: how much of each real trace is spent in each macro-phase, per
location -- using exactly the phase detector
(utils.starlink_preprocessing.detect_bbr_phases, the paper's Algorithm 1
rolling-deviation detector, half_window=10, phase_threshold_scale=0.7,
cruise_samples_after_down=6) and exactly the raw files
(bbr_<Location>__REV_run<N>.json) that
generate_bbr_experience_pool.py itself uses to build the training pool.

Each box = the per-run phase time-share (fraction of the 300 one-second
samples in that run spent in that phase) across the 10 runs available for
that location, for stream_group="downlink-sequential-logs" (dedicated
downlink, matching the rest of this project's reports).

Run: python3 reports/generate_macrophase_location_boxplots.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.starlink_preprocessing import detect_bbr_phases  # noqa: E402
from utils.bbr import BW_UP, BW_CRUISE, BW_DOWN  # noqa: E402

RAW_ROOT = ROOT / "tcp-cc-starlink-main" / "downlink-sequential-logs"
FIG_DIR = ROOT / "reports" / "figures" / "cc_algorithms"

LOCATIONS = ["Tokyo", "SaoPaulo", "Ohio", "London", "Mumbai", "Sydney"]
N_RUNS = 10

PHASES = [BW_UP, BW_CRUISE, BW_DOWN]
PHASE_LABELS = {BW_UP: "Bw Up", BW_CRUISE: "Cruise", BW_DOWN: "Bw Down"}
PHASE_COLORS = {BW_UP: "#D55E00", BW_CRUISE: "#0072B2", BW_DOWN: "#009E73"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Times", "serif"],
    "font.size": 16,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

_decoder = json.JSONDecoder()


def load_throughput(location, run):
    path = RAW_ROOT / location / "iperf3-downlink-sequential-logs" / f"bbr_{location}__REV_run{run}.json"
    with open(path) as f:
        text = f.read()
    data, _ = _decoder.raw_decode(text)
    values = []
    for interval in data.get("intervals", []):
        streams = interval.get("streams", [])
        if streams:
            values.append(streams[0]["bits_per_second"])
    return values


def phase_shares(location):
    """Return {phase: [share_per_run, ...]} across the N_RUNS runs."""
    shares = {phase: [] for phase in PHASES}
    for run in range(1, N_RUNS + 1):
        try:
            throughput = load_throughput(location, run)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        phases = detect_bbr_phases(throughput)
        counts = Counter(phases)
        total = len(phases)
        for phase in PHASES:
            shares[phase].append(counts.get(phase, 0) / total)
    return shares


def main():
    fig, ax = plt.subplots(figsize=(10, 4.6))

    n_phases = len(PHASES)
    width = 0.8 / n_phases
    positions = np.arange(len(LOCATIONS))

    per_location = {loc: phase_shares(loc) for loc in LOCATIONS}

    for i, phase in enumerate(PHASES):
        offset = (i - (n_phases - 1) / 2) * width
        color = PHASE_COLORS[phase]
        data = [per_location[loc][phase] for loc in LOCATIONS]
        bp = ax.boxplot(
            data,
            positions=positions + offset,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, markerfacecolor=color,
                             markeredgecolor="none", alpha=0.6),
            medianprops=dict(color="black", linewidth=1.2),
            boxprops=dict(facecolor=color, alpha=0.7, edgecolor=color),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
        )
        bp["boxes"][0].set_label(PHASE_LABELS[phase])

    ax.set_xticks(positions)
    ax.set_xticklabels(LOCATIONS)
    ax.set_ylabel("Share of trace duration")
    ax.set_xlabel("Location")
    ax.set_ylim(0, 1.05)
    ax.set_title("Macro-phase time-share of real BBR traces by location", pad=45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)

    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(FIG_DIR / f"macrophase_location_boxplots.{ext}", bbox_inches="tight", **kwargs)
    print("written:", FIG_DIR / "macrophase_location_boxplots.pdf")


if __name__ == "__main__":
    main()
