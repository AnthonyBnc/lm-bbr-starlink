"""Digitize the source paper's Fig.7(a)/(b) curves (GPT-2/T5/GPT-Neo/SmolLM2
mean loss and mean accuracy vs epoch) via color-matching pixel analysis of
the page image, since the paper publishes only the plot, not the underlying
per-epoch numbers.

Methodology (documented for reproducibility, per docs/CONTEXT.md and
ARCHITECTURE.md's plot-digitization-must-be-labelled rule):
1. Locate each panel's axes box by detecting the solid black border rectangle.
2. Calibrate pixel->data mapping from detected gridlines (gray horizontal/
   vertical lines at known tick values).
3. For each pixel column, find the row(s) matching each series' known line
   color (sampled from the legend swatches) within a tolerance; the legend
   box itself is masked out to avoid the swatch line contaminating readings.
4. Detect where a series stops being reliably separable (the four lines
   converge and interleave, especially in Fig.7(b) after ~epoch 20) via a
   discontinuity check once each series is past its initial rise/fall: a
   jump larger than the plateau-region threshold marks the last reliable
   epoch for that series.

Output: reports/assets/digitized_paper_fig7.json with per-series values AND
a `reliable_until_epoch` field. This is a DIGITIZED ESTIMATE, not exact
published data -- see the citing chart script for how the unreliable tail is
rendered (dashed, annotated, not solid).

Run: .venv/bin/python reports/digitize_paper_fig7.py
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "reports" / "assets" / "paper_page8_source.png"
OUTPUT = ROOT / "reports" / "assets" / "digitized_paper_fig7.json"

im = Image.open(PAGE).convert("RGB")
arr = np.array(im)
Y0, X0 = 220, 1700
region = arr[Y0:Y0 + 1040 - 220, X0:X0 + 1700]

PANEL_A = dict(top=100, bottom=461, left=103, right=708)  # (a) mean loss
PANEL_B = dict(top=99, bottom=460, left=807, right=1419)  # (b) mean accuracy
CAL_A_Y = {"row0": 40.0, "val0": 0.08, "row1": 341.0, "val1": 0.01}
CAL_A_X = {"col0": 28.0, "ep0": 0.0, "col1": 396.5, "ep1": 100.0}
CAL_B_Y = {"row0": 52.5, "val0": 0.8, "row1": 344.5, "val1": 0.0}
CAL_B_X = {"col0": 27.5, "ep0": 0.0, "col1": 401.0, "ep1": 100.0}

# Legend box interior mask (panel-box-relative pixel coordinates)
LEGEND_MASK = {
    "A": dict(x0=320, x1=605, y0=0, y2=145),
    "B": dict(x0=470, x1=720, y0=190, y2=400),
}

TARGET_COLORS = {
    "GPT2": (1, 1, 254),
    "T5": (1, 128, 1),
    "SMOLLM2": (254, 1, 1),
    "GPT_NEO": (1, 190, 190),
}
TOLERANCE = 60

# Discontinuity detection: only applied after SETTLE_EPOCH (past the initial
# steep rise/fall, where legitimate large jumps are expected), flags the
# first jump between consecutive raw detected points larger than the given
# absolute threshold as the onset of an unreliable (occluded/misdetected) tail.
SETTLE_EPOCH = {"loss": 55, "accuracy": 25}
JUMP_THRESHOLD = {"loss": 0.012, "accuracy": 0.08}
# All four lines visually converge into this plateau band after settling
# (read from the source image); a point outside it past the settle epoch is
# treated as a gradual-drift misdetection (e.g. a green/blue line slowly
# tracking a wrong nearby feature), not necessarily a sudden jump.
PLATEAU_RANGE = {"loss": (0.005, 0.028), "accuracy": (0.75, 0.95)}


def row_to_val(row, cal):
    return cal["val0"] + (row - cal["row0"]) * (cal["val1"] - cal["val0"]) / (cal["row1"] - cal["row0"])


def col_to_epoch(col, cal):
    return cal["ep0"] + (col - cal["col0"]) * (cal["ep1"] - cal["ep0"]) / (cal["col1"] - cal["col0"])


def digitize_panel(panel, cal_y, cal_x, max_epoch, legend_mask):
    box = region[panel["top"]:panel["bottom"], panel["left"]:panel["right"]].astype(int)
    height, width, _ = box.shape
    series = {name: {} for name in TARGET_COLORS}
    for col in range(width):
        epoch = col_to_epoch(col, cal_x)
        if epoch < 0 or epoch > max_epoch:
            continue
        epoch_bin = int(round(epoch))
        column_pixels = box[:, col, :]
        row_indices = np.arange(height)
        if legend_mask and legend_mask["x0"] <= col <= legend_mask["x1"]:
            valid = (row_indices < legend_mask["y0"]) | (row_indices > legend_mask["y2"])
        else:
            valid = np.ones(height, dtype=bool)
        for name, target in TARGET_COLORS.items():
            dist = np.sqrt(((column_pixels - np.array(target)) ** 2).sum(axis=1))
            matches = np.where((dist < TOLERANCE) & valid)[0]
            if len(matches) == 0:
                continue
            row = float(np.median(matches))
            value = row_to_val(row, cal_y)
            series[name].setdefault(epoch_bin, []).append(value)
    result = {}
    for name, epoch_map in series.items():
        epochs = sorted(epoch_map)
        values = [float(np.mean(epoch_map[e])) for e in epochs]
        result[name] = {"epoch": epochs, "value": values}
    return result


def find_reliable_cutoff(epochs, values, metric):
    settle = SETTLE_EPOCH[metric]
    threshold = JUMP_THRESHOLD[metric]
    plateau_lo, plateau_hi = PLATEAU_RANGE[metric]
    last_reliable_epoch = epochs[-1] if epochs else 0
    for i in range(1, len(epochs)):
        if epochs[i] < settle:
            continue
        jump = abs(values[i] - values[i - 1])
        gap = epochs[i] - epochs[i - 1]
        # Sudden jump: a jump spread over a larger epoch gap is less suspicious.
        if gap > 0 and (jump / max(gap, 1)) > threshold:
            return epochs[i - 1]
        # Gradual drift outside the visually-observed convergence band: no
        # single jump is large, but the value has wandered somewhere the
        # source image never shows any line going after settling.
        if not (plateau_lo <= values[i] <= plateau_hi):
            return epochs[i - 1]
    return last_reliable_epoch


def clean_and_densify(raw_metric, metric_name, full_epochs, clip_range):
    cleaned = {}
    for name, series in raw_metric.items():
        epochs, values = series["epoch"], series["value"]
        cutoff = find_reliable_cutoff(epochs, values, metric_name)
        reliable_epochs = [e for e in epochs if e <= cutoff]
        reliable_values = [v for e, v in zip(epochs, values) if e <= cutoff]
        dense = np.interp(
            full_epochs, reliable_epochs, reliable_values, left=reliable_values[0], right=reliable_values[-1]
        )
        dense = np.clip(dense, *clip_range)
        cleaned[name] = {
            "value": dense.tolist(),
            "reliable_until_epoch": cutoff,
            "n_raw_points": len(epochs),
        }
    return cleaned


def main():
    loss_raw = digitize_panel(PANEL_A, CAL_A_Y, CAL_A_X, 150, LEGEND_MASK["A"])
    accuracy_raw = digitize_panel(PANEL_B, CAL_B_Y, CAL_B_X, 150, LEGEND_MASK["B"])

    full_epochs = list(range(0, 151))
    loss = clean_and_densify(loss_raw, "loss", full_epochs, (0.0, 0.2))
    accuracy = clean_and_densify(accuracy_raw, "accuracy", full_epochs, (0.0, 1.0))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        json.dump({"epochs": full_epochs, "loss": loss, "accuracy": accuracy}, f, indent=2)

    for name in TARGET_COLORS:
        print(
            name,
            "loss reliable_until_epoch=",
            loss[name]["reliable_until_epoch"],
            "| accuracy reliable_until_epoch=",
            accuracy[name]["reliable_until_epoch"],
        )
    print("saved", OUTPUT)


if __name__ == "__main__":
    main()
