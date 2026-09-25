"""Time-series figure: pacing-gain predictions over a real Tokyo trace window,
epoch-100 frozen checkpoints, gpt_classical vs gpt_quantum vs the expert
(ground-truth) target gain. Data is read live from
data/processed/evaluation/four_model_100epoch_v1/results -- no hardcoded
numbers.

Single-panel report figure: a zoomed-in slice of one contiguous 300-step
Tokyo trace segment (stream_group="downlink-sequential-logs", pool_index
3014-3313), wide format so decision steps aren't cramped together. Font
family and text sizes match the other kept charts (Times New Roman, larger
legend/label/tick sizes) -- see the plt.rcParams.update(...) block below,
the one place to edit either.

Run: python3 reports/generate_pacing_gain_timeseries.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "data/processed/evaluation/four_model_100epoch_v1/results"
FIG_DIR = ROOT / "reports" / "figures" / "100epoch"

CLASSICAL_COLOR = "#0072B2"
QUANTUM_COLOR = "#D55E00"
TARGET_COLOR = "#000000"

STREAM_GROUP = "downlink-sequential-logs"
POOL_START, POOL_END = 3014, 3313
ZOOM_END = 100  # steps 0..99 of the window, relative index -- raise/lower to show more/fewer steps

# Figure size in inches: (width, height). This is the knob for "make the
# x-axis longer" -- raise the first number to stretch the plot horizontally
# without changing font size.
FIGSIZE = (10, 4.2)

# Same font family + size convention as reports/under400m_chart_common.py
# (the shared config for the other 4 kept charts) -- edit here to change
# font or text size for this figure; nothing else in this file needs to
# change for a font/size-only tweak. The fallback list matters: this Linux
# box has no real "Times New Roman" font installed, so if the list is ever
# trimmed down to just that one name, matplotlib silently falls back to its
# default sans-serif font instead. Keep "Liberation Serif" (metrically
# identical to Times New Roman) in the list as the working fallback.
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


def load_window(model_key):
    path = RESULTS_ROOT / model_key / "predictions.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    window = [
        r for r in rows
        if r["stream_group"] == STREAM_GROUP and POOL_START <= int(r["pool_index"]) <= POOL_END
    ]
    window.sort(key=lambda r: int(r["pool_index"]))
    return window


def get_series():
    classical = load_window("lfm2_5_350m")
    quantum = load_window("lfm2_5_350m_quantum")
    assert len(classical) == len(quantum)
    assert all(a["sample_id"] == b["sample_id"] for a, b in zip(classical, quantum)), (
        "classical/quantum rows are not aligned to the same Tokyo samples"
    )
    n = len(classical)
    t = list(range(n))
    target = [float(r["target_gain"]) for r in classical]
    pred_classical = [float(r["predicted_gain"]) for r in classical]
    pred_quantum = [float(r["predicted_gain"]) for r in quantum]
    return t, target, pred_classical, pred_quantum


def main():
    t, target, pred_classical, pred_quantum = get_series()
    tz = t[:ZOOM_END]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(tz, target[:ZOOM_END], color=TARGET_COLOR, linewidth=1.4, linestyle="--", marker="s", markersize=3.5,
            label="Expert target", zorder=3)
    ax.plot(tz, pred_classical[:ZOOM_END], color=CLASSICAL_COLOR, linewidth=1.7, marker="o", markersize=4,
            label="GPT Classical", zorder=4, alpha=0.9)
    ax.plot(tz, pred_quantum[:ZOOM_END], color=QUANTUM_COLOR, linewidth=1.7, marker="^", markersize=4,
            label="GPT Quantum", zorder=5, alpha=0.8)

    ax.set_xlim(0, ZOOM_END - 1)
    ax.set_ylim(0.80, 1.42)
    ax.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Pacing gain")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=True, framealpha=0.9,
              edgecolor="#cccccc", handletextpad=0.4, columnspacing=0.9)

    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(FIG_DIR / f"pacing_gain_timeseries.{ext}", bbox_inches="tight", **kwargs)
    print("written:", FIG_DIR / "pacing_gain_timeseries.pdf")


if __name__ == "__main__":
    main()
