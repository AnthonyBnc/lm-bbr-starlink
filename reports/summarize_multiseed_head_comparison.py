"""Mean +/- SD over training seeds: gpt_classical vs gpt_quantum on Tokyo.

Evaluation-only (evidence class: measured_final). Reads frozen predictions.csv
files; loads, trains, selects or tunes nothing.

Seeds: 100003 from four_model_100epoch_v1 (the run reported in the paper) plus
every seed evaluated by run_multiseed_head_comparison.py. A seed is included
only when BOTH heads have a Tokyo result, so every comparison is paired.

Note for the paper: the seed-100003 gpt_quantum run used the Qiskit
parameter-shift path for epochs 1-11 and the exact torch statevector path for
12-100; the new seeds use the torch path throughout (equal to 1e-5). All Tokyo
evaluations use the Qiskit path.

Run: python reports/summarize_multiseed_head_comparison.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UP = [1.05, 1.10, 1.15, 1.20, 1.25]
ROLES = {"classical": "gpt_classical", "quantum": "gpt_quantum"}
REFERENCE_SEED = 100003
REFERENCE_DIRS = {
    "classical": ROOT / "data/processed/evaluation/four_model_100epoch_v1/results/lfm2_5_350m",
    "quantum": ROOT / "data/processed/evaluation/four_model_100epoch_v1/results/lfm2_5_350m_quantum",
}
METRICS = [
    ("bw_up_accuracy", "BW_UP acc."),
    ("bw_up_macro_f1", "BW_UP macro-F1"),
    ("bw_up_balanced_accuracy", "BW_UP bal. acc."),
    ("bw_up_distinct_predictions", "BW_UP distinct"),
    ("accuracy", "Overall acc."),
    ("macro_phase_accuracy", "Macro-phase acc."),
    ("macro_f1_all_actions", "Macro-F1 (all)"),
]
# ms/action is kept per run but not averaged: seed 100003 was measured with the
# pre-speed-up code, so its latency is not comparable with the new seeds.


def f1_stats(y, p, labels):
    f1s, recalls = [], []
    for c in labels:
        support = np.sum(y == c)
        if support == 0:
            continue
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        recalls.append(recall)
    return float(np.mean(f1s)), float(np.mean(recalls))


def metrics_for(result_dir):
    result_dir = Path(result_dir)
    df = pd.read_csv(result_dir / "predictions.csv").sort_values("pool_index").reset_index(drop=True)
    manifest = json.loads((result_dir / "result.manifest.json").read_text(encoding="utf-8"))
    y = df.target_gain.round(2).to_numpy(); p = df.predicted_gain.round(2).to_numpy()
    up = (df.phase == "BW_UP").to_numpy()
    labels = sorted(set(y.tolist()) | set(UP))
    mf1_all, _ = f1_stats(y, p, labels)
    mf1_up, bacc_up = f1_stats(y[up], p[up], UP)
    per_phase = manifest["metrics"]["per_phase_accuracy"]
    return df.sample_id.to_numpy(), {
        "accuracy": float(np.mean(y == p)),
        "macro_phase_accuracy": float(np.mean([per_phase[k] for k in per_phase])),
        "macro_f1_all_actions": mf1_all,
        "bw_up_accuracy": float(np.mean(y[up] == p[up])),
        "bw_up_macro_f1": mf1_up,
        "bw_up_balanced_accuracy": bacc_up,
        "bw_up_distinct_predictions": int(len(np.unique(p[up]))),
        "ms_per_action": float(manifest["metrics"]["latency"]["mean_milliseconds_per_action"]),
        "bw_up_majority_accuracy": float(pd.Series(y[up]).value_counts().iloc[0] / up.sum()),
    }


def collect(eval_root, include_reference):
    runs = {}
    if include_reference and all((d / "predictions.csv").is_file() for d in REFERENCE_DIRS.values()):
        runs[REFERENCE_SEED] = dict(REFERENCE_DIRS)
    for seed_dir in sorted(Path(eval_root).glob("seed*")):
        seed = int(seed_dir.name[4:])
        dirs = {h: seed_dir / "lfm2_5_350m_{}".format(h) for h in ROLES}
        if all((d / "result.manifest.json").is_file() for d in dirs.values()):
            runs[seed] = dirs
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path,
                        default=ROOT / "data/processed/evaluation/multiseed_head_comparison_v1")
    parser.add_argument("--no-reference-seed", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/assets/multiseed_head_comparison_tokyo.json")
    args = parser.parse_args()

    runs = collect(args.eval_root, not args.no_reference_seed)
    if not runs:
        raise SystemExit("No seed has Tokyo results for both heads yet.")
    rows, sample_ids = [], None
    for seed, dirs in sorted(runs.items()):
        for head, directory in dirs.items():
            ids, m = metrics_for(directory)
            if sample_ids is None:
                sample_ids = ids
            elif not np.array_equal(ids, sample_ids):
                raise SystemExit("Tokyo sample ids differ for seed {} {}".format(seed, head))
            rows.append({"seed": seed, "head": head, "role": ROLES[head], **m})
    table = pd.DataFrame(rows)

    summary = {}
    for head in ROLES:
        sub = table[table["head"] == head]
        summary[ROLES[head]] = {
            key: {"mean": float(sub[key].mean()),
                  "sd": float(sub[key].std(ddof=1)) if len(sub) > 1 else None,
                  "min": float(sub[key].min()), "max": float(sub[key].max())}
            for key, _ in METRICS
        }
        summary[ROLES[head]]["seeds_with_single_bw_up_action"] = int((sub.bw_up_distinct_predictions == 1).sum())
        summary[ROLES[head]]["seeds_at_or_below_majority_baseline"] = int(
            (sub.bw_up_accuracy <= sub.bw_up_majority_accuracy + 1e-12).sum())

    wide = table.pivot(index="seed", columns="head")
    paired = {}
    for key in ("bw_up_accuracy", "bw_up_macro_f1", "bw_up_balanced_accuracy", "accuracy", "macro_phase_accuracy"):
        diff = wide[key]["classical"] - wide[key]["quantum"]
        paired[key] = {
            "per_seed": {int(s): float(v) for s, v in diff.items()},
            "mean": float(diff.mean()),
            "sd": float(diff.std(ddof=1)) if len(diff) > 1 else None,
            "seeds_classical_better": int((diff > 0).sum()),
            "n_seeds": int(len(diff)),
        }

    out = {
        "evidence_class": "measured_final",
        "experiment": "multiseed_head_comparison_v1 (+ seed 100003 from four_model_100epoch_v1)",
        "seeds": sorted(int(s) for s in runs),
        "sd": "sample standard deviation over training seeds (ddof=1)",
        "per_run": rows,
        "summary": summary,
        "paired_classical_minus_quantum": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("Seeds:", out["seeds"], "\n")
    print(table[["seed", "role"] + [k for k, _ in METRICS]].round(4).to_string(index=False), "\n")
    for role, stats in summary.items():
        print(role)
        for key, label in METRICS:
            s = stats[key]
            sd = "n/a" if s["sd"] is None else "{:.4f}".format(s["sd"])
            print("  {:<18} {:.4f} +/- {}  [{:.4f}, {:.4f}]".format(label, s["mean"], sd, s["min"], s["max"]))
        print("  seeds with a single BW_UP action: {}/{}".format(stats["seeds_with_single_bw_up_action"], len(runs)))
    print("\nPaired (classical - quantum) over seeds:")
    for key, d in paired.items():
        sd = "n/a" if d["sd"] is None else "{:.4f}".format(d["sd"])
        print("  {:<24} {:+.4f} +/- {}   classical better in {}/{}".format(
            key, d["mean"], sd, d["seeds_classical_better"], d["n_seeds"]))

    def cell(role, key, digits=4):
        s = summary[role][key]
        return "{:.{d}f}".format(s["mean"], d=digits) if s["sd"] is None else \
            "{:.{d}f} $\\pm$ {:.{d}f}".format(s["mean"], s["sd"], d=digits)
    print("\nLaTeX rows (Table II, mean $\\pm$ SD over {} seeds):".format(len(runs)))
    for role, name in (("gpt_classical", "LFM2.5-350M (\\texttt{gpt\\_classical})"),
                       ("gpt_quantum", "\\texttt{gpt\\_quantum}")):
        distinct = summary[role]["bw_up_distinct_predictions"]
        print("{} & {} & {} & {} & {:.1f} & {} & {} & {} \\\\".format(
            name, cell(role, "bw_up_accuracy"), cell(role, "bw_up_macro_f1"),
            cell(role, "bw_up_balanced_accuracy"), distinct["mean"],
            cell(role, "accuracy"), cell(role, "macro_phase_accuracy"), cell(role, "macro_f1_all_actions")))
    print("\nWrote", args.output)


if __name__ == "__main__":
    main()
