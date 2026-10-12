"""Pacing-gain time series for one seed (classical vs quantum head vs expert)
on one Tokyo trace (downlink-sequential-logs, trace 10, pool_index
3014-3313), multiseed_head_comparison_v1. x-axis is the iperf3 interval end
time t_i (s) from the Tokyo pool. One column-width figure per seed.

Run: python3 reports/generate_pacing_gain_timeseries_seed.py 400003 500003
"""
import csv
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS_ROOT = ROOT / "data/processed/evaluation/multiseed_head_comparison_v1"
POOL_PATH = ROOT / "data/processed/evaluation/tokyo_8model_comparison_v1/tokyo.pkl"
FIG_DIR = ROOT / "reports" / "figures" / "paper"

STREAM_GROUP = "downlink-sequential-logs"
POOL_START, POOL_END = 3014, 3313
T_MAX = 100.0

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
    "legend.fontsize": 7.5,
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


def plot_seed(seed, pool):
    c = load_window(seed, "lfm2_5_350m_classical")
    q = load_window(seed, "lfm2_5_350m_quantum")
    assert [r["sample_id"] for r in c] == [r["sample_id"] for r in q]
    assert all(pool.sample_ids[int(r["pool_index"])] == r["sample_id"] for r in c)
    t_all = [float(pool.states[int(r["pool_index"])][2]) for r in c]
    keep = [i for i, t in enumerate(t_all) if t <= T_MAX]
    t = [t_all[i] for i in keep]
    tgt = [float(c[i]["target_gain"]) for i in keep]
    pc = [float(c[i]["predicted_gain"]) for i in keep]
    pq = [float(q[i]["predicted_gain"]) for i in keep]

    fig, ax = plt.subplots(figsize=(3.5, 1.95))
    ax.plot(t, tgt, color=TARGET_COLOR, lw=0.9, ls="--", marker="s", ms=1.8,
            label="Expert", zorder=3)
    ax.plot(t, pc, color=CLASSICAL_COLOR, lw=0.9, marker="o", ms=2.2,
            label="Classical head", zorder=4, alpha=0.9)
    ax.plot(t, pq, color=QUANTUM_COLOR, lw=0.9, marker="^", ms=2.4,
            label="Quantum head", zorder=5, alpha=0.85)
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0.85, 1.30)
    ax.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pacing gain")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, handletextpad=0.4, columnspacing=1.0,
              borderaxespad=0.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kw = {"dpi": 300} if ext == "png" else {}
        fig.savefig(FIG_DIR / f"pacing_gain_timeseries_seed{seed}.{ext}",
                    bbox_inches="tight", **kw)
    plt.close(fig)

    up = [i for i in keep if c[i]["phase"] == "BW_UP"]
    nonup_ok = all(c[i]["correct"] == "1" and q[i]["correct"] == "1"
                   for i in keep if c[i]["phase"] != "BW_UP")
    print(seed, "non-UP all correct:", nonup_ok, "| UP times", len(up),
          "| classical matches", sum(c[i]["correct"] == "1" for i in up),
          "| quantum matches", sum(q[i]["correct"] == "1" for i in up))
    for i in up:
        print(f"  t={t_all[i]:.0f}s expert={c[i]['target_gain']} "
              f"classical={c[i]['predicted_gain']} quantum={q[i]['predicted_gain']}")


def main():
    with open(POOL_PATH, "rb") as f:
        pool = pickle.load(f)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in sys.argv[1:]] or [400003, 500003]:
        plot_seed(seed, pool)


if __name__ == "__main__":
    main()
