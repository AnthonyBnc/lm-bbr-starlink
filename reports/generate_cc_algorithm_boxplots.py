"""2x3 grid of grouped box plots comparing 4 congestion-control algorithms
(Cubic, Hybla, Vegas, BBR2) across the 6 real Starlink measurement
locations in this project's raw dataset, in the same visual style as
Fig. 5 ("Summarized downlink observations ... for globally distributed
locations") in project_sources/2607.07133v1 -- the companion BBR-v3 paper
in this project's sources (De Silva, Pokhrel, Kua), NOT the main SLM-BBR
paper (2607.07142). That paper benchmarks BBR-v3 against eight CCAs
(Cubic, Hybla, Vegas, LeoCC, Copa, PCC, BBR-v1, BBR-v2) across the same
six cities; this figure reproduces 4 of those series from our own raw
captures.

Data source (real, no fabricated numbers): the raw iperf3 JSON captures
under tcp-cc-starlink-main/downlink-sequential-logs/<Location>/
iperf3-downlink-sequential-logs/<algo>_<Location>__REV_run<N>.json -- 10
runs per algorithm per location, 9 algorithms available (bbr, bbr1, bbr2,
ccp, cubic, hybla, leocc, pcc, vegas); this figure uses 4 of them
(cubic, hybla, vegas, bbr).

Which raw tag is "BBR-v3"? The dataset's own JSON field
(end.sender_tcp_congestion) just echoes the filename prefix, so there is
no explicit version tag. 2607.07133 studies BBR-v1, BBR-v2, and BBR-v3 as
three distinct algorithms, and its own text reports downlink dedicated
median throughput for BBR-v3 (Mbps): SaoPaulo 134.11, Ohio 180.37, London
157.14, Mumbai 197.21, Sydney 247.56. Checking each candidate tag's median
throughput on our own data against those numbers, the unsuffixed "bbr" tag
is consistently the closest match (e.g. Sydney 240.0 vs. the paper's
247.56, London 147.8 vs. 157.14) -- much closer than "bbr1" or "bbr2" on
the same locations. So "bbr" is used here and labeled "BBR" (not
overclaimed as an exact "BBR-v3" match, since this is inference from
matching statistics, not an explicit tag in the raw data).

Each captured JSON file is a per-second iperf3 log (300 s / ~300 intervals
per run); this script pools all interval-level samples across the 10 runs
of a given (algorithm, location) pair into one distribution per box (so
each box can represent up to ~3000 points), matching the dense box +
outlier look of the reference figure.

Note: some of these raw JSON files have a stray second JSON object
appended after the real one (an artifact of how they were logged, e.g. a
trailing "iperf_json_finish: ..." error line followed by another JSON blob)
-- this script only parses the FIRST JSON object in each file via
json.JSONDecoder().raw_decode(), which is the real, complete 300 s run.

Run: python3 reports/generate_cc_algorithm_boxplots.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "tcp-cc-starlink-main" / "downlink-sequential-logs"
FIG_DIR = ROOT / "reports" / "figures" / "cc_algorithms"

LOCATIONS = ["Tokyo", "SaoPaulo", "Ohio", "London", "Mumbai", "Sydney"]

ALGOS = ["cubic", "hybla", "vegas", "bbr"]
ALGO_LABELS = {"cubic": "Cubic", "hybla": "Hybla", "vegas": "Vegas", "bbr": "BBR"}
ALGO_COLORS = {
    "cubic": "#4C9F9F",
    "hybla": "#D9534F",
    "vegas": "#E8A33D",
    "bbr": "#8064A2",
}
N_RUNS = 10

# Same font family + size convention as the other kept charts under reports/
# (under400m_chart_common.py, generate_pacing_gain_timeseries.py,
# generate_model_comparison_boxplots.py) -- edit here to change font/size.
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


def load_run_intervals(algo, location, run):
    path = RAW_ROOT / location / "iperf3-downlink-sequential-logs" / f"{algo}_{location}__REV_run{run}.json"
    with open(path) as f:
        text = f.read()
    data, _ = _decoder.raw_decode(text)
    return data.get("intervals", [])


def collect(algo, location, field):
    """Pool one field's per-interval values across all N_RUNS runs."""
    values = []
    for run in range(1, N_RUNS + 1):
        try:
            intervals = load_run_intervals(algo, location, run)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        for interval in intervals:
            streams = interval.get("streams", [])
            if not streams:
                continue
            values.append(streams[0][field])
    return values


def grouped_boxplot(ax, field, transform):
    n_algo = len(ALGOS)
    width = 0.8 / n_algo
    positions = np.arange(len(LOCATIONS))

    for i, algo in enumerate(ALGOS):
        offset = (i - (n_algo - 1) / 2) * width
        color = ALGO_COLORS[algo]
        data = [
            [transform(v) for v in collect(algo, loc, field)]
            for loc in LOCATIONS
        ]
        bp = ax.boxplot(
            data,
            positions=positions + offset,
            widths=width * 0.9,
            patch_artist=True,
            # Outlier points hidden to reduce visual clutter -- box/whiskers
            # still summarize the complete pooled data (median, IQR, and the
            # 1.5xIQR whisker range), only the individual dots beyond the
            # whiskers are not drawn. Set back to True to see them again.
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.1),
            boxprops=dict(facecolor=color, alpha=0.7, edgecolor=color),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
        )
        bp["boxes"][0].set_label(ALGO_LABELS[algo])

    ax.set_xticks(positions)
    # Rotated (not fitting the bigger post-font-bump tick labels flush
    # horizontal caused adjacent 6-location clusters to run into each
    # other, e.g. "Tokyo" colliding with "SaoPaulo") -- same fix as the
    # other multi-panel charts in this project.
    ax.set_xticklabels(LOCATIONS, rotation=20, ha="right")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    panels = [
        (axes[0, 0], "bits_per_second", lambda v: v / 1e6, "(a) Throughput", "Mbps", "linear"),
        # Retransmissions are extremely skewed (mostly 0, rare large spikes),
        # exactly like the reference paper's own Fig. 5(b)/Fig. 21, which use
        # a "symmetric log scale" for this metric -- a linear zoom either
        # hides the boxes (if capped low) or hides the real spiky behavior
        # (if capped high). symlog shows both at once.
        (axes[0, 1], "retransmits", lambda v: v, "(b) Retransmissions", "Count (symmetric log scale)", "symlog"),
        (axes[0, 2], "snd_cwnd", lambda v: v / 1e6, "(c) Congestion Window", "MB", "linear"),
        (axes[1, 0], "snd_wnd", lambda v: v / 1e6, "(d) Receiver Advertised Window", "MB", "linear"),
        (axes[1, 1], "rtt", lambda v: v / 1000, "(e) RTT", "ms", "linear"),
        (axes[1, 2], "rttvar", lambda v: v / 1000, "(f) RTT Variance", "ms", "linear"),
    ]

    for ax, field, transform, title, ylabel, scale in panels:
        grouped_boxplot(ax, field, transform)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        all_vals = [
            transform(v)
            for algo in ALGOS
            for loc in LOCATIONS
            for v in collect(algo, loc, field)
        ]
        if scale == "symlog":
            # linthresh=1: values below 1 (i.e. 0) are shown linearly near
            # the origin, values above scale logarithmically -- this is what
            # lets a box sitting at 0 and a whisker reaching into the
            # thousands share one readable axis.
            ax.set_yscale("symlog", linthresh=1)
            if all_vals:
                ax.set_ylim(0, max(all_vals) * 1.3)
        else:
            # Zoom the y-axis to the 99.5th percentile (+10% headroom) so the
            # bulk of the real distribution (boxes/whiskers) is readable
            # instead of being squashed by one rare extreme sample. This
            # only changes the VIEW -- no data is removed or altered, and
            # boxes/whiskers/medians are still computed from the complete
            # pooled dataset; a handful of points simply fall outside the
            # visible range, same convention the reference figure itself
            # uses.
            if all_vals:
                cap = np.percentile(all_vals, 99.5) * 1.1
                lo = min(0, min(all_vals))
                ax.set_ylim(lo, max(cap, 1e-6))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=4, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(FIG_DIR / f"cc_algorithm_boxplots.{ext}", bbox_inches="tight", **kwargs)
    print("written:", FIG_DIR / "cc_algorithm_boxplots.pdf")


if __name__ == "__main__":
    main()
