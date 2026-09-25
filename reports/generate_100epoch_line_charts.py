"""Generate the single-axes "combined" comparison figures overlaying our 5
models (4 classical backbones + gpt_quantum, from four_model_100epoch_v1)
against the source paper's 4 SLMs (GPT-2/T5/GPT-Neo/SmolLM2, digitized from
Fig.7), clipped to a shared 0-100 epoch window: one figure for mean loss,
one for mean accuracy.

gpt_quantum's trace is flat by epoch 2 and stays flat: its validation
accuracy takes a single value across all 100 epochs. That is the finding,
not a plotting artifact -- see "The quantum head does not learn" in
docs/RESULTS_100EPOCH_SUMMARY.md.

IMPORTANT -- the paper series plotted here is a DIGITIZED ESTIMATE, not
exact data: see reports/digitize_paper_fig7.py's docstring for the full
caveat on reports/assets/digitized_paper_fig7.json. Do not treat any value
from that series as an exact published number.

(The separate side-by-side "ours vs. paper" panel figures were dropped in
the 2026-09 cleanup as unused; this script now only produces the two
combined-overlay figures actually referenced downstream.)

Run: .venv/bin/python reports/generate_100epoch_line_charts.py
"""

import json

import matplotlib.pyplot as plt

from under400m_chart_common import DISPLAY, LINE_STYLE, ROOT

DIGITIZED_PAPER_DATA = ROOT / "reports" / "assets" / "digitized_paper_fig7.json"
FIG_DIR_100EPOCH = ROOT / "reports" / "figures" / "100epoch"

CLASSICAL_MODELS = ["granite_4_0_350m", "pleias_rag_350m", "lfm2_5_350m", "gemma_3_270m"]
# gpt_quantum shares the LFM2.5-350M backbone and every training hyperparameter
# with lfm2_5_350m; only the head differs, so the pair isolates the head choice.
MODELS_100EPOCH = CLASSICAL_MODELS + ["gpt_quantum"]

_RUN_ROOT = ROOT / "data/processed/lora_training/four_model_100epoch_v1"
TRAIN_METRICS_100EPOCH = {
    model: _RUN_ROOT / f"{model}_classical/metrics.json" for model in CLASSICAL_MODELS
}
TRAIN_METRICS_100EPOCH["gpt_quantum"] = _RUN_ROOT / "lfm2_5_350m_quantum/metrics.json"

# Paper Fig.7's own 4 lines: solid blue circle, dashed green square,
# dash-dot red diamond, dotted cyan triangle -- matches the source image.

# Single-axes variant: the two panels above deliberately reuse the same 4
# colors/markers, so overlaying them needs a second, non-colliding style set.
# Ours stay solid with filled markers; the paper's are dashed with hollow
# markers, so a digitized line is never mistaken for a measured one.
COMBINED_OURS_STYLE = {
    "granite_4_0_350m": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "pleias_rag_350m": {"color": "#d62728", "marker": "s", "linestyle": "-"},
    "lfm2_5_350m": {"color": "#2ca02c", "marker": "D", "linestyle": "-"},
    "gemma_3_270m": {"color": "#17becf", "marker": "^", "linestyle": "-"},
    # Thicker so the flat quantum trace stays visible where it runs under the
    # classical curves -- it is the line the reader is meant to notice.
    "gpt_quantum": {"color": "#9467bd", "marker": "x", "linestyle": "-", "linewidth": 2.6},
}
COMBINED_PAPER_STYLE = {
    "GPT2": {"color": "#ff7f0e", "marker": "v", "linestyle": "--"},
    "T5": {"color": "#9467bd", "marker": "X", "linestyle": "--"},
    "SMOLLM2": {"color": "#8c564b", "marker": "P", "linestyle": "--"},
    "GPT_NEO": {"color": "#e377c2", "marker": "*", "linestyle": "--"},
}
COMBINED_MAX_EPOCH = 100
PAPER_DISPLAY = {"GPT2": "GPT2", "T5": "T5", "SMOLLM2": "SMOLLM2", "GPT_NEO": "GPT_NEO"}




def _plot_combined(ax, metric_key, ylabel, legend_loc):
    """All 8 models on one axes: our 4 at 100 epochs + the paper's 4, clipped
    to the same 0-100 window so both sets cover an identical epoch range."""
    for model in MODELS_100EPOCH:
        record = json.loads(TRAIN_METRICS_100EPOCH[model].read_text())
        epochs = [entry["epoch"] for entry in record["epochs"]]
        values = [entry["train"][metric_key] for entry in record["epochs"]]
        ax.plot(
            epochs,
            values,
            markersize=5,
            markevery=10,
            label=DISPLAY[model],
            **COMBINED_OURS_STYLE[model],
        )

    data = json.loads(DIGITIZED_PAPER_DATA.read_text())
    paper_epochs = [e for e in data["epochs"] if e <= COMBINED_MAX_EPOCH]
    for name in ("GPT2", "T5", "SMOLLM2", "GPT_NEO"):
        series = data[metric_key][name]
        cutoff = series["reliable_until_epoch"]
        values = series["value"][: len(paper_epochs)]
        style = COMBINED_PAPER_STYLE[name]

        reliable_epochs = [e for e in paper_epochs if e <= cutoff]
        reliable_values = values[: len(reliable_epochs)]
        ax.plot(
            reliable_epochs,
            reliable_values,
            markersize=5,
            markevery=10,
            markerfacecolor="none",
            label=f"{PAPER_DISPLAY[name]} (paper)",
            **style,
        )

        if cutoff < paper_epochs[-1]:
            tail_epochs = [e for e in paper_epochs if e >= cutoff]
            tail_values = values[len(reliable_epochs) - 1:]
            ax.plot(tail_epochs, tail_values, linestyle=(0, (2, 2)), linewidth=1, color="#aaaaaa", alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc, frameon=True, fontsize=12, ncol=2)


def generate_combined_mean_loss():
    fig, ax = plt.subplots(figsize=(8, 5.2))
    _plot_combined(ax, "loss", "Mean Loss", legend_loc="upper right")

    fig.tight_layout()
    FIG_DIR_100EPOCH.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR_100EPOCH / "100epoch_mean_loss_combined.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR_100EPOCH / "100epoch_mean_loss_combined.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR_100EPOCH / "100epoch_mean_loss_combined.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved 100epoch_mean_loss_combined.png/svg/pdf")


def generate_combined_mean_accuracy():
    fig, ax = plt.subplots(figsize=(8, 5.2))
    _plot_combined(ax, "accuracy", "Mean Accuracy", legend_loc="lower right")

    fig.tight_layout()
    FIG_DIR_100EPOCH.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR_100EPOCH / "100epoch_mean_accuracy_combined.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR_100EPOCH / "100epoch_mean_accuracy_combined.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR_100EPOCH / "100epoch_mean_accuracy_combined.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved 100epoch_mean_accuracy_combined.png/svg/pdf")




def main():
    generate_combined_mean_loss()
    generate_combined_mean_accuracy()


if __name__ == "__main__":
    main()
