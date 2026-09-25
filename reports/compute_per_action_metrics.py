"""Per-action / class-balanced metrics for the frozen Tokyo evaluation.

Evaluation-only (evidence class: measured_final). Reads the existing frozen
predictions.csv files; no model is loaded, trained, selected or tuned. Added for
reviewer revision (per-action results, macro-F1, balanced accuracy, confusion
matrices, trace-level bootstrap CIs).

Run: python3 reports/compute_per_action_metrics.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data/processed/evaluation/four_model_100epoch_v1/results"
OUT = ROOT / "reports/assets/per_action_metrics_tokyo.json"
MODELS = {
    "granite_4_0_350m": "Granite-4.0-350M",
    "pleias_rag_350m": "Pleias-RAG-350M",
    "lfm2_5_350m": "gpt_classical (LFM2.5-350M)",
    "gemma_3_270m": "Gemma-3-270M",
    "lfm2_5_350m_quantum": "gpt_quantum (LFM2.5-350M+VQC)",
}
UP = [1.05, 1.10, 1.15, 1.20, 1.25]
N_BOOT = 10000
RNG = np.random.default_rng(100003)


def f1_stats(y, p, labels):
    per = {}
    for c in labels:
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = dict(support=int(np.sum(y == c)), predicted=int(np.sum(p == c)),
                      precision=float(prec), recall=float(rec), f1=float(f1))
    present = [c for c in labels if per[c]["support"] > 0]
    return per, float(np.mean([per[c]["f1"] for c in present])), float(np.mean([per[c]["recall"] for c in present]))


def trace_ids(df):
    # Raw-trace membership of each pool index, rebuilt from the Tokyo pool's
    # episode "dones" (40 primary BBR traces) and saved alongside this script.
    m = pd.read_csv(ROOT / "reports/assets/tokyo_pool_trace_ids.csv").set_index("pool_index").trace_id
    return m.loc[df.pool_index].to_numpy()


def main():
    dfs = {k: pd.read_csv(RES / k / "predictions.csv").sort_values("pool_index").reset_index(drop=True) for k in MODELS}
    base = dfs["lfm2_5_350m"]
    for k, d in dfs.items():
        assert (d.sample_id.values == base.sample_id.values).all(), k
    tid = trace_ids(base)
    out = {"evidence_class": "measured_final", "pool": "four_model_100epoch_v1 (Tokyo held-out, 100-epoch checkpoints)",
           "n_samples": int(len(base)), "n_traces_detected": int(tid.max() + 1), "models": {}}
    up_mask = (base.phase == "BW_UP").to_numpy()
    y_up = base.target_gain.to_numpy()[up_mask].round(2)
    maj = pd.Series(y_up).value_counts()
    out["bw_up_majority_baseline"] = {"label": float(maj.index[0]), "accuracy": float(maj.iloc[0] / len(y_up))}
    all_labels = sorted(base.target_gain.round(2).unique().tolist() + [g for g in UP if g not in base.target_gain.round(2).unique()])
    for k, name in MODELS.items():
        d = dfs[k]
        y = d.target_gain.round(2).to_numpy(); p = d.predicted_gain.round(2).to_numpy()
        per_all, mf1_all, bacc_all = f1_stats(y, p, all_labels)
        per_up, mf1_up, bacc_up = f1_stats(y[up_mask], p[up_mask], UP)
        cm = pd.crosstab(pd.Series(y[up_mask], name="target"), pd.Series(p[up_mask], name="pred")).reindex(index=UP, columns=UP, fill_value=0)
        out["models"][k] = dict(
            name=name,
            accuracy=float(np.mean(y == p)),
            bw_up_accuracy=float(np.mean(y[up_mask] == p[up_mask])),
            macro_f1_all_actions=mf1_all, balanced_accuracy_all_actions=bacc_all,
            bw_up_macro_f1=mf1_up, bw_up_balanced_accuracy=bacc_up,
            distinct_bw_up_predictions=int(len(np.unique(p[up_mask]))),
            bw_up_per_action={str(c): v for c, v in per_up.items()},
            bw_up_confusion={str(r): {str(c): int(cm.loc[r, c]) for c in UP} for r in UP},
        )
    # Paired trace-level bootstrap: classical vs quantum on BW_UP accuracy and macro-F1.
    c = dfs["lfm2_5_350m"]; q = dfs["lfm2_5_350m_quantum"]
    traces = np.unique(tid)
    yc = c.target_gain.round(2).to_numpy(); pc = c.predicted_gain.round(2).to_numpy(); pq = q.predicted_gain.round(2).to_numpy()
    rows_by_trace = [np.where((tid == t) & up_mask)[0] for t in traces]
    d_acc, d_f1 = [], []
    for _ in range(N_BOOT):
        pick = RNG.choice(len(traces), len(traces), replace=True)
        r = np.concatenate([rows_by_trace[i] for i in pick])
        if len(r) == 0:
            continue
        d_acc.append(np.mean(pc[r] == yc[r]) - np.mean(pq[r] == yc[r]))
        d_f1.append(f1_stats(yc[r], pc[r], UP)[1] - f1_stats(yc[r], pq[r], UP)[1])
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out["paired_trace_bootstrap_classical_minus_quantum"] = dict(
        n_boot=N_BOOT, seed=100003, unit="raw Tokyo trace",
        bw_up_accuracy_diff_ci95=ci(d_acc), bw_up_macro_f1_diff_ci95=ci(d_f1))
    # Paired trace-level bootstrap for every pair of classical backbones.
    import itertools
    classical = ["granite_4_0_350m", "gemma_3_270m", "lfm2_5_350m", "pleias_rag_350m"]
    corr = {k: dfs[k].correct.to_numpy() for k in classical}
    rows_all = [np.where(tid == t)[0] for t in traces]
    rng2 = np.random.default_rng(100003)
    picks = [np.concatenate([rows_all[i] for i in rng2.choice(len(traces), len(traces))]) for _ in range(5000)]
    pairs = {}
    for a, b in itertools.combinations(classical, 2):
        da = [corr[a][r].mean() - corr[b][r].mean() for r in picks]
        du = [corr[a][r][up_mask[r]].mean() - corr[b][r][up_mask[r]].mean() for r in picks]
        pairs[f"{a}_minus_{b}"] = dict(
            accuracy_diff=float(corr[a].mean() - corr[b].mean()), accuracy_diff_ci95=ci(da),
            bw_up_accuracy_diff=float(corr[a][up_mask].mean() - corr[b][up_mask].mean()), bw_up_accuracy_diff_ci95=ci(du))
    out["paired_trace_bootstrap_backbone_pairs"] = dict(n_boot=5000, seed=100003, pairs=pairs)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: {kk: v[kk] for kk in ["accuracy", "bw_up_accuracy", "macro_f1_all_actions", "balanced_accuracy_all_actions", "bw_up_macro_f1", "bw_up_balanced_accuracy", "distinct_bw_up_predictions"]} for k, v in out["models"].items()}, indent=1))
    print("traces", out["n_traces_detected"], "majority", out["bw_up_majority_baseline"])
    print(out["paired_trace_bootstrap_classical_minus_quantum"])
    for k in ["lfm2_5_350m", "lfm2_5_350m_quantum"]:
        print(k, json.dumps(out["models"][k]["bw_up_confusion"]))
        print({a: (round(v["precision"], 3), round(v["recall"], 3), round(v["f1"], 3)) for a, v in out["models"][k]["bw_up_per_action"].items()})


if __name__ == "__main__":
    main()
