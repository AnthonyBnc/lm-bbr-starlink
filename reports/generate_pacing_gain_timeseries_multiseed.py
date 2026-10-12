"""Single-panel pacing-gain time series on one Tokyo trace
(downlink-sequential-logs, trace 10, pool_index 3014-3313), seeds 300003,
400003 and 500003 of multiseed_head_comparison_v1 (same evaluation run).

The expert gain is drawn as a line over time. Outside BW_UP every model
matches the expert in this window (checked below), so only the BW_UP
choices are drawn as markers, side by side around each BW_UP time:
classical seeds left (blue), quantum seeds right (orange), one marker
shape per seed. The x-axis is the iperf3 interval end time t_i (s),
read from the Tokyo pool.

Run: python3 reports/generate_pacing_gain_timeseries_multiseed.py
"""
import csv
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS_ROOT = ROOT / "data/processed/evaluation/multiseed_head_comparison_v1"
POOL_PATH = ROOT / "data/processed/evaluation/tokyo_8model_comparison_v1/tokyo.pkl"
FIG_DIR = ROOT / "reports" / "figures" / "paper"

SEEDS = [300003, 400003, 500003]
MARKERS = {300003: "o", 400003: "s", 500003: "D"}
STREAM_GROUP = "downlink-sequential-logs"
POOL_START, POOL_END = 3014, 3313
T_MAX = 60.0          # seconds shown
OFFSET = 0.85         # seconds between side-by-side markers

CLASSICAL_COLOR = "#0072B2"
QUANTUM_COLOR = "#D55E00"
TARGET_COLOR = "#000000"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Times", "serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def load_window(seed, model_key):
    path = RESULTS_ROOT / f"seed{seed}" / model_key / "predictions.csv"
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r["stream_group"] == STREAM_GROUP
                and POOL_START <= int(r["pool_index"]) <= POOL_END]
    rows.sort(key=lambda r: int(r["pool_index"]))
    return rows


def main():
    with open(POOL_PATH, "rb") as f:
        pool = pickle.load(f)

    preds = {}
    for seed in SEEDS:
        for head in ("classical", "quantum"):
            preds[(seed, head)] = load_window(seed, f"lfm2_5_350m_{head}")
    ref = preds[(SEEDS[0], "classical")]
    for rows in preds.values():
        assert [r["sample_id"] for r in rows] == [r["sample_id"] for r in ref]
    for r in ref:
        assert pool.sample_ids[int(r["pool_index"])] == r["sample_id"]

    times = [float(pool.states[int(r["pool_index"])][2]) for r in ref]
    keep = [i for i, t in enumerate(times) if t <= T_MAX]
    t = [times[i] for i in keep]
    target = [float(ref[i]["target_gain"]) for i in keep]
    up = [i for i in keep if ref[i]["phase"] == "BW_UP"]

    # every model matches the expert outside BW_UP in the shown window
    for rows in preds.values():
        assert all(rows[i]["correct"] == "1" for i in keep if rows[i]["phase"] != "BW_UP")

    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    ax.plot(t, target, color=TARGET_COLOR, lw=0.9, ls="--", zorder=2)

    offsets = {("classical", s): -OFFSET * (3 - k) for k, s in enumerate(SEEDS)}
    offsets.update({("quantum", s): OFFSET * (k + 1) for k, s in enumerate(SEEDS)})
    for i in up:
        ax.plot([times[i] - 3.5 * OFFSET, times[i] + 3.5 * OFFSET],
                [float(ref[i]["target_gain"])] * 2, color=TARGET_COLOR, lw=1.4, zorder=3)
        for seed in SEEDS:
            for head, color in (("classical", CLASSICAL_COLOR), ("quantum", QUANTUM_COLOR)):
                g = float(preds[(seed, head)][i]["predicted_gain"])
                ax.plot(times[i] + offsets[(head, seed)], g, ls="none",
                        marker=MARKERS[seed], ms=3.6, mfc=color, mec="white",
                        mew=0.4, color=color, zorder=4)
        print(f"t={times[i]:.0f}s expert={ref[i]['target_gain']}",
              {f"{h[0]}{s}": preds[(s, h)][i]["predicted_gain"]
               for h in ("classical", "quantum") for s in SEEDS})

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0.85, 1.30)
    ax.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pacing gain")

    # legend: colour = head, marker shape = seed
    handles = [
        Line2D([], [], color=TARGET_COLOR, ls="--", lw=0.9, label="Expert"),
        Line2D([], [], ls="none", marker="o", ms=4, mfc=CLASSICAL_COLOR,
               mec=CLASSICAL_COLOR, label="Classical head"),
        Line2D([], [], ls="none", marker="o", ms=4, mfc=QUANTUM_COLOR,
               mec=QUANTUM_COLOR, label="Quantum head"),
    ]
    for seed in SEEDS:
        handles.append(Line2D([], [], ls="none", marker=MARKERS[seed], ms=3.6,
                              mfc="#777777", mec="#777777", label=f"Seed {seed}"))
    # ncol=3 fills column-wise: pair rows so row 1 = expert/heads, row 2 = seeds
    order = [handles[0], handles[3], handles[1], handles[4], handles[2], handles[5]]
    ax.legend(handles=order, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.2,
              borderaxespad=0.0)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        kw = {"dpi": 300} if ext == "png" else {}
        fig.savefig(FIG_DIR / f"pacing_gain_timeseries_multiseed.{ext}",
                    bbox_inches="tight", **kw)
    print("written:", FIG_DIR / "pacing_gain_timeseries_multiseed.pdf")


if __name__ == "__main__":
    main()
