"""Shared config, styling, and data loading for the under-400M chart scripts.

Not runnable on its own — imported by `generate_under400m_line_charts.py` and
`generate_under400m_box_charts.py`.

Edit MODELS / DISPLAY / COLORS / LINE_STYLE below to add a 5th model (e.g.
gpt_quantum) once its Tokyo evaluation predictions.csv exists at the same
path pattern.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures" / "under400m"

MODELS = ["granite_4_0_350m", "pleias_rag_350m", "lfm2_5_350m", "gemma_3_270m"]
# The 4 frozen classical backbones, plus gpt_quantum as the 5th "new" model
# (same LFM2.5-350M backbone, Qiskit VQC head instead of a classical head).
MODELS_WITH_QUANTUM = MODELS + ["gpt_quantum"]
DISPLAY = {
    "granite_4_0_350m": "Granite-4.0-350M",
    "pleias_rag_350m": "Pleias-RAG-350M",
    "lfm2_5_350m": "LFM2.5-350M",
    "gemma_3_270m": "Gemma-3-270M",
    "gpt_quantum": "GPT Quantum (LFM2.5-350M+VQC)",
}
COLORS = {
    "granite_4_0_350m": "#1f77b4",
    "pleias_rag_350m": "#d62728",
    "lfm2_5_350m": "#2ca02c",
    "gemma_3_270m": "#17becf",
    "gpt_quantum": "#9467bd",
    "REAL": "#7f7f7f",
}

# Fig.7 line style in the source paper, by model position (GPT2/T5/SmolLM2/GPT-Neo):
# solid blue circle, dashed green square, dash-dot red diamond, dotted cyan triangle.
# Applied here to our 4 models in the same order/role, since there is no 1:1
# name correspondence between the paper's SLMs and these under-400M backbones.
# gpt_quantum has no paper counterpart, so it gets a 5th, visually distinct style.
LINE_STYLE = {
    "granite_4_0_350m": {"color": "blue", "marker": "o", "linestyle": "-"},
    "pleias_rag_350m": {"color": "green", "marker": "s", "linestyle": "--"},
    "lfm2_5_350m": {"color": "red", "marker": "D", "linestyle": "-."},
    "gemma_3_270m": {"color": "cyan", "marker": "^", "linestyle": ":"},
    "gpt_quantum": {"color": "purple", "marker": "x", "linestyle": "-"},
}

TRAIN_METRICS = {
    "granite_4_0_350m": ROOT
    / "data/processed/lora_training/under400_rank128_epochs16_seed100003_v1"
    / "granite_4_0_350m_rank128_epochs16_seed100003_train/metrics.json",
    "pleias_rag_350m": ROOT
    / "data/processed/lora_training/under400_rank128_epochs16_seed100003_v1"
    / "pleias_rag_350m_rank128_epochs16_seed100003_train/metrics.json",
    "lfm2_5_350m": ROOT
    / "data/processed/lora_training/under400_rank128_epochs16_seed100003_v1"
    / "lfm2_5_350m_rank128_epochs16_seed100003_train/metrics.json",
    "gemma_3_270m": ROOT
    / "data/processed/lora_training/gemma3_270m_rank128_epochs16_seed100003_v1"
    / "gemma_3_270m_rank128_epochs16_seed100003_train/metrics.json",
    "gpt_quantum": ROOT
    / "data/processed/lora_training/under400m_quantum_gpt_follow_on_v1"
    / "lfm2_5_350m_quantum/metrics.json",
}

PRED_CSV = {
    model: ROOT
    / "data/processed/evaluation/tokyo_8model_comparison_v1/results"
    / model
    / "predictions.csv"
    for model in MODELS
}
# gpt_quantum's results directory is named after its backbone ("lfm2_5_350m_quantum"),
# not "gpt_quantum", so it needs its own entry rather than fitting the comprehension above.
PRED_CSV["gpt_quantum"] = (
    ROOT
    / "data/processed/evaluation/tokyo_8model_comparison_v1/results"
    / "lfm2_5_350m_quantum"
    / "predictions.csv"
)

STREAM_GROUPS = [
    ("downlink-sequential-logs", "Downlink individual"),
    ("uplink-sequential-logs", "Uplink individual"),
    ("downlink-competitive-logs", "Downlink competing"),
    ("uplink-competitive-logs", "Uplink competing"),
]
PANEL_LETTERS = list("abcdefgh")

# --- Shared font + size settings for every chart under reports/. -----------
# This is the ONE place to edit to change font or text size across all 4
# kept charts (100epoch_mean_accuracy_combined, 100epoch_mean_loss_combined,
# 100epoch_trainable_parameters_combined, 100epoch_vram_combined):
#   - "font.family"/"font.serif": Times New Roman everywhere, falling back to
#     Liberation Serif (metrically identical) if the real font isn't
#     installed on this machine, then any other serif as a last resort.
#   - "font.size": base size for any text that doesn't set its own size.
#   - "axes.titlesize"/"axes.labelsize": chart title / axis-label size.
#   - "xtick.labelsize"/"ytick.labelsize": tick-number size.
#   - "legend.fontsize": legend text size.
# Editing any number below and re-running the two generate_100epoch_*.py
# scripts regenerates all 4 charts with the new setting -- no other file
# needs to change for a font/size-only tweak.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Times", "serif"],
        "font.size": 16,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.edgecolor": "#444444",
    }
)


def load_tokyo_series():
    """Load Tokyo predictions.csv for every model (including gpt_quantum) into
    REAL + per-model series."""
    data_by_group = {group_key: {"REAL": {"throughput": [], "retransmissions": []}} for group_key, _ in STREAM_GROUPS}
    for model in MODELS_WITH_QUANTUM:
        for group_key, _ in STREAM_GROUPS:
            data_by_group[group_key][model] = {"throughput": [], "retransmissions": []}

    for index, model in enumerate(MODELS_WITH_QUANTUM):
        with open(PRED_CSV[model], newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                group_key = row["stream_group"]
                if group_key not in data_by_group:
                    continue
                data_by_group[group_key][model]["throughput"].append(float(row["surrogate_throughput"]) / 1e6)
                data_by_group[group_key][model]["retransmissions"].append(float(row["surrogate_retransmissions"]))
                # observed (REAL) values are identical across models (same Tokyo samples) -> collect once
                if index == 0:
                    data_by_group[group_key]["REAL"]["throughput"].append(float(row["observed_throughput"]) / 1e6)
                    data_by_group[group_key]["REAL"]["retransmissions"].append(
                        float(row["observed_retransmissions"])
                    )
    return data_by_group
