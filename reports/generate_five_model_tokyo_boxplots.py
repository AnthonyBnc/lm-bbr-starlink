"""Paper Fig. 9-style 2x4 grid of box plots comparing our 5 models' Tokyo
surrogate predictions (the 4 classical backbones + gpt_quantum, all from the
100-epoch run) against the real observed Tokyo BBR flows, across the same 4
stream conditions used throughout this project (downlink/uplink x
individual/competing).

Layout mirrors Fig. 9 of the source paper (project_sources/2607.07142v1)
exactly, panel-for-panel:
  (a) Downlink individual throughput      (b) Downlink individual retransmissions
  (c) Uplink individual throughput        (d) Uplink individual retransmissions
  (e) Downlink competing throughput       (f) Downlink competing retransmissions
  (g) Uplink competing throughput         (h) Uplink competing retransmissions
-- i.e. throughput on top, retransmissions on the bottom, same column order
(downlink individual, uplink individual, downlink competing, uplink
competing) as STREAM_GROUPS in under400m_chart_common.py.

Data source (real, no fabricated numbers): each model's own
predictions.csv under
data/processed/evaluation/tokyo_8model_comparison_v1/results/<model>/ --
the SAME frozen Tokyo held-out evaluation already used by the
under400m line/box charts. "REAL" is the observed_* column (identical
across all 5 models' files, since they all evaluate the same Tokyo
samples); each model's box is its own surrogate_* column. gpt_quantum's
results directory is "lfm2_5_350m_quantum" (see PRED_CSV in
under400m_chart_common.py) -- it reuses the LFM2.5-350M backbone with a
Qiskit VQC head instead of a classical head.

Drawn with seaborn's boxplot() (on top of matplotlib axes we still control
for the layout/log-scale/zoom logic below) rather than a hand-rolled
ax.boxplot() call, for the same look-and-feel seaborn gives everywhere else
box/whisker plots are wanted -- less custom style code, standard black
box outlines instead of colour-on-colour.

Box plots hide individual outlier dots (showfliers=False) for readability,
matching the convention already used in generate_cc_algorithm_boxplots.py --
median/IQR/whiskers are still computed from the complete per-sample
distribution (3000 pooled samples per model per stream condition). The
retransmissions row uses a symmetric-log y-scale for the same reason as
generate_cc_algorithm_boxplots.py's Retransmissions panel: retransmission
counts are extremely skewed (median near 0, rare spikes into the
thousands), and symlog is the only linear-or-log choice that keeps both the
near-zero boxes and the real spiky tail visible on one axis. The throughput
row uses a linear axis, percentile-zoomed (99.5th + 10% headroom) the same
way, so a rare extreme sample doesn't squash the readable bulk of the
distribution.

Run: python3 reports/generate_five_model_tokyo_boxplots.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from under400m_chart_common import (
    COLORS,
    FIG_DIR,
    MODELS_WITH_QUANTUM,
    STREAM_GROUPS,
    load_tokyo_series,
)

SERIES = ["REAL"] + MODELS_WITH_QUANTUM

# Short, single-word/line labels for the x-axis -- DISPLAY's full names
# (e.g. "gpt_quantum (LFM2.5-350M+VQC)") are too long to sit under a box
# without heavy overlap; matches the brevity of the reference paper's own
# x-axis labels (REAL, GPT2, T5, SmolLM2, GPT-Neo, LLaMA3).
SHORT_LABELS = {
    "REAL": "REAL",
    "granite_4_0_350m": "Granite",
    "pleias_rag_350m": "Pleias",
    "lfm2_5_350m": "LFM2.5",
    "gemma_3_270m": "Gemma3",
    "gpt_quantum": "GPT Quantum",
}

# Same font family + size convention as every other kept chart under reports/
# (under400m_chart_common.py, generate_pacing_gain_timeseries.py,
# generate_cc_algorithm_boxplots.py) -- edit here to change font/size for
# this figure only; edit under400m_chart_common.py to change it everywhere.
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

# (letter, field, transform, ylabel-on-this-row, y-scale) per row
ROWS = [
    ("throughput", "Throughput (Mbps)", "linear", list("aceg")),
    ("retransmissions", "Retransmissions (count)", "symlog", list("bdfh")),
]


def _series_boxplot(ax, values_by_series):
    df = pd.DataFrame({
        "series": [s for s in SERIES for _ in values_by_series[s]],
        "value": [v for s in SERIES for v in values_by_series[s]],
    })
    sns.boxplot(
        data=df,
        x="series",
        y="value",
        order=SERIES,
        hue="series",
        hue_order=SERIES,
        palette=COLORS,
        dodge=False,
        legend=False,
        width=0.6,
        linewidth=1.1,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.3},
        ax=ax,
    )
    ax.set_xticks(range(len(SERIES)))
    ax.set_xticklabels([SHORT_LABELS[s] for s in SERIES], rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("")


def main():
    data_by_group = load_tokyo_series()

    # Wider + taller than the initial cut, and titles wrap onto 2 lines --
    # both needed once the project-wide font bump (see under400m_chart_common.py)
    # made "(b) Downlink individual retransmissions" wider than a 4-column
    # panel at the old figsize, which collided into the next panel's title.
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))

    for row_index, (field, ylabel, scale, letters) in enumerate(ROWS):
        for col_index, (group_key, group_label) in enumerate(STREAM_GROUPS):
            ax = axes[row_index, col_index]
            values_by_series = {s: data_by_group[group_key][s][field] for s in SERIES}
            _series_boxplot(ax, values_by_series)

            ax.set_title(f"({letters[col_index]}) {group_label}\n{field.capitalize()}")
            if col_index == 0:
                ax.set_ylabel(ylabel)

            all_vals = [v for s in SERIES for v in values_by_series[s]]
            if not all_vals:
                continue
            if scale == "symlog":
                # linthresh=1: values below 1 (i.e. ~0) render linearly near the
                # origin, values above render log-scale -- lets a box sitting at
                # 0 and a whisker reaching into the thousands share one axis.
                ax.set_yscale("symlog", linthresh=1)
                ax.set_ylim(0, max(all_vals) * 1.3)
            else:
                # View-only zoom to the 99.5th percentile (+10% headroom); no
                # data is removed, boxes/whiskers are still computed from the
                # complete pooled distribution -- same convention as
                # generate_cc_algorithm_boxplots.py.
                cap = np.percentile(all_vals, 99.5) * 1.1
                lo = min(0, min(all_vals))
                ax.set_ylim(lo, max(cap, 1e-6))

    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(FIG_DIR / f"five_model_tokyo_boxplots.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print("written:", FIG_DIR / "five_model_tokyo_boxplots.pdf")


if __name__ == "__main__":
    main()
