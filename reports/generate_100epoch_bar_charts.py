"""Generate 2 single-axes bar charts across all 9 models -- our 5 (the 4
classical backbones plus gpt_quantum, all from the 100-epoch run
four_model_100epoch_v1) and the source paper's 4 SLMs
(GPT-2/T5/GPT-Neo/SmolLM2):

1. Trainable LoRA parameter percentage
2. Mean VRAM usage

Companion to generate_100epoch_line_charts.py (mean loss / mean accuracy).
generate_nine_model_charts.py produces the same two charts keyed to the
earlier 16-epoch run and is left untouched.

PROVENANCE, do not strip when editing:
- Trainable %% for our 5 is computed from each 100-epoch run's own
  run.manifest.json (lora_trainable_parameters / backbone_parameters).
- VRAM for our 5 is the Apple MPS measurement from
  reports/measure_vram_usage.py, carried in nine_model_comparison.json. It was
  NOT measured on the RTX 4090 that actually ran the 100-epoch training --
  no VRAM profiling was instrumented there. It is hardware-transferred, and
  labelled as such on the chart.
- Paper rows are published reference values (Table II / Fig. 8b), measured on
  2x NVIDIA RTX 6000 over a 150-epoch budget.
Different hardware and different epoch budgets on both sides: read the two
colors as two separate groups, do not compute cross-group deltas.

Run: .venv/bin/python reports/generate_100epoch_bar_charts.py
"""

import json

import matplotlib.pyplot as plt

from under400m_chart_common import DISPLAY, ROOT

NINE_MODEL_DATA = ROOT / "reports" / "assets" / "nine_model_comparison.json"
FIG_DIR_100EPOCH = ROOT / "reports" / "figures" / "100epoch"

CLASSICAL_MODELS = ["granite_4_0_350m", "pleias_rag_350m", "lfm2_5_350m", "gemma_3_270m"]
MODELS_100EPOCH = CLASSICAL_MODELS + ["gpt_quantum"]

_RUN_ROOT = ROOT / "data/processed/lora_training/four_model_100epoch_v1"
RUN_MANIFESTS_100EPOCH = {
    model: _RUN_ROOT / f"{model}_classical/run.manifest.json" for model in CLASSICAL_MODELS
}
RUN_MANIFESTS_100EPOCH["gpt_quantum"] = _RUN_ROOT / "lfm2_5_350m_quantum/run.manifest.json"

# gpt_quantum reuses the LFM2.5-350M backbone and LoRA config unchanged, so its
# trainable %% and VRAM land on top of that model's bars. The VRAM measurement
# carried in nine_model_comparison.json was taken on the 16-epoch run; the head
# config is identical, so it transfers.

OURS_COLOR = "#4C72B0"
PAPER_COLOR = "#DD8452"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.edgecolor": "#444444",
    }
)


def _ours_rows():
    """Trainable %% straight from the 100-epoch manifests; VRAM joined in from
    the MPS measurement (see PROVENANCE above)."""
    vram_by_key = {
        row["model_key"]: row["mean_vram_usage_gb"]
        for row in json.loads(NINE_MODEL_DATA.read_text())["this_work"]
    }
    rows = []
    for model in MODELS_100EPOCH:
        manifest = json.loads(RUN_MANIFESTS_100EPOCH[model].read_text())
        percent = 100.0 * manifest["lora_trainable_parameters"] / manifest["backbone_parameters"]
        rows.append(
            {
                "display_name": DISPLAY[model],
                "lora_trainable_parameter_percent": percent,
                "mean_vram_usage_gb": vram_by_key[model],
            }
        )
    return rows


def _paper_rows():
    return json.loads(NINE_MODEL_DATA.read_text())["published_reference"]


def _bar_chart(values_key, ylabel, value_format, footnote, filename):
    rows = _ours_rows() + _paper_rows()
    names = [r["display_name"] for r in rows]
    values = [r[values_key] for r in rows]
    colors = [OURS_COLOR] * len(MODELS_100EPOCH) + [PAPER_COLOR] * 4

    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars = ax.bar(names, values, color=colors, width=0.6)
    for bar, value in zip(bars, values):
        ax.annotate(
            value_format.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.25)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=OURS_COLOR, label="This work (5 models, 100 epochs)"),
        plt.Rectangle((0, 0), 1, 1, color=PAPER_COLOR, label="Published reference (paper, 150 epochs)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.text(0.5, 0.0, footnote, ha="center", fontsize=8, style="italic", color="#555555")

    FIG_DIR_100EPOCH.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        kwargs = {"dpi": 200} if ext == "png" else {}
        fig.savefig(FIG_DIR_100EPOCH / f"{filename}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"saved {filename}.png/svg/pdf")


def generate_trainable_parameter_bar():
    _bar_chart(
        "lora_trainable_parameter_percent",
        "Trainable Parameters (%)",
        "{:.2f}%",
        "This work: all 5 runs share LoRA rank 128, so differences here are backbone architecture only.\n"
        "gpt_quantum reuses the LFM2.5-350M backbone and LoRA config unchanged, so its bar coincides with it.\n"
        "The paper does not publish its LoRA rank / target modules -- the two groups are not on a common\n"
        "configuration. Read them as two separate groups; do not compute cross-group deltas.",
        "100epoch_trainable_parameters_combined",
    )


def generate_vram_bar():
    _bar_chart(
        "mean_vram_usage_gb",
        "Mean VRAM Usage (GB)",
        "{:.2f} GB",
        "This work: peak memory over one real training step, measured on Apple MPS "
        "(reports/measure_vram_usage.py) -- NOT\nmeasured on the RTX 4090 that ran the 100-epoch training, "
        "where no VRAM profiling was instrumented.\nPaper: Fig. 8(b), 2x NVIDIA RTX 6000. Different hardware "
        "on both sides; do not compute cross-group deltas.",
        "100epoch_vram_combined",
    )


def main():
    generate_trainable_parameter_bar()
    generate_vram_bar()


if __name__ == "__main__":
    main()
