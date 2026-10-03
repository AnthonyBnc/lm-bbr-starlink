"""Seen-location test: frozen checkpoints on the development validation traces.

The validation traces come from the five training locations (Ohio, SaoPaulo,
London, Mumbai, Sydney) but were never trained on; they were used only for
parent selection. This script evaluates the four classical backbones and the
LFM2.5 VQC head of one training seed on them, then compares chosen seen
locations with held-out Tokyo on BW_UP (metrics table + boxplots).

Lab machine (needs checkpoints and a GPU):
    python evaluate_seen_locations.py evaluate --seed 100003 --device cuda
    python evaluate_seen_locations.py evaluate --seed 400003 --device cuda

Any machine (needs only the predictions.csv files, no torch):
    python evaluate_seen_locations.py plot --seed 100003 --locations Mumbai SaoPaulo
    python evaluate_seen_locations.py plot --seed 400003 --locations Mumbai SaoPaulo
"""
import argparse
import gc
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BACKBONES = ["granite_4_0_350m", "pleias_rag_350m", "gemma_3_270m", "lfm2_5_350m"]
LABEL = {
    "granite_4_0_350m_classical": "Granite",
    "pleias_rag_350m_classical": "Pleias",
    "gemma_3_270m_classical": "Gemma3",
    "lfm2_5_350m_classical": "LFM2.5\nclassical",
    "lfm2_5_350m_quantum": "LFM2.5\nVQC",
}
GAINS = [0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25]
UP = [1.05, 1.10, 1.15, 1.20, 1.25]
VAL_POOL = Path("data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003/validation.pkl")
TOKYO_POOL = Path("data/processed/evaluation/tokyo_8model_comparison_v1/tokyo.pkl")
OUT = Path("data/processed/evaluation/seen_location_validation_v1")
TRAIN_100 = Path("data/processed/lora_training/four_model_100epoch_v1")
TOKYO_100 = Path("data/processed/evaluation/four_model_100epoch_v1/results")
MULTI = Path("data/processed/evaluation/multiseed_head_comparison_v1")


def models(seed):
    """(key, training run dir, Tokyo result dir) for every model of one seed."""
    rows = []
    for backbone in BACKBONES:
        heads = ["classical", "quantum"] if backbone == "lfm2_5_350m" else ["classical"]
        for head in heads:
            key = "{}_{}".format(backbone, head)
            if seed == 100003:
                run = TRAIN_100 / key
                tokyo = TOKYO_100 / (backbone if head == "classical" else "lfm2_5_350m_quantum")
            else:
                tokyo = MULTI / "seed{}".format(seed) / key
                frozen = tokyo / "frozen_checkpoint.json"
                run = Path(json.loads(frozen.read_text())["run_dir"]) if frozen.is_file() else None
            rows.append((key, run, tokyo))
    return rows


# ---------------------------------------------------------------- evaluation
def load_policy(run_dir, device, local_model_root):
    import torch
    from peft import PeftModel
    from config import cfg
    from plm_special.backbones import load_local_backbone, resolve_local_revision
    from plm_special.models.rl_policy import OfflineRLPolicy
    from plm_special.models.state_encoder import EncoderNetwork
    from utils.bbr import ACTION_LEVELS

    manifest = json.loads((run_dir / "run.manifest.json").read_text(encoding="utf-8"))
    model_key = manifest["model_key"]
    model_path = Path(local_model_root).expanduser() / cfg.get_registered_model(model_key)["local_dir"]
    if resolve_local_revision(model_path) != manifest["model_revision"]:
        raise ValueError("Local backbone revision changed for {}".format(model_key))
    backbone, model_config = load_local_backbone(model_path, device=device, dtype=getattr(torch, manifest["dtype"]))
    backbone = PeftModel.from_pretrained(backbone, run_dir / "checkpoint" / "adapter",
                                         is_trainable=False, local_files_only=True)
    task_state = torch.load(run_dir / "checkpoint" / "task_modules.pt", map_location=device, weights_only=True)
    hidden = getattr(model_config, "hidden_size", None) or model_config.text_config.hidden_size
    head = manifest.get("head_config") or {}
    quantum_config = None
    if manifest["head_type"] == "quantum":
        quantum_config = {
            "n_qubits": head.get("n_qubits"), "depth": head.get("depth"), "ansatz": head.get("ansatz"),
            "input_layernorm": head.get("input_layernorm", False),
            "temperature": head.get("temperature", 1.0), "angle_scale": head.get("angle_scale", "pi"),
        }
    policy = OfflineRLPolicy(
        state_feature_dim=manifest["state_feature_dim"], action_levels=ACTION_LEVELS,
        state_encoder=EncoderNetwork(embed_dim=manifest["state_feature_dim"]).to(device),
        plm=backbone, plm_embed_size=hidden, max_length=manifest["sequence_length"],
        max_ep_len=int(task_state["1.weight"].shape[0]) - 1, device=device, device_out=device,
        head_type=manifest["head_type"], quantum_config=quantum_config,
    )
    policy.modules_except_plm.load_state_dict(task_state, strict=True)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    return policy, manifest


def evaluate_pool(policy, manifest, pool, device):
    import torch
    from torch.utils.data import DataLoader
    from plm_special.data.dataset import ExperienceDataset
    from plm_special.utils.utils import process_bbr_batch
    from utils.bbr import action_index_to_gain
    from utils.tokyo_evaluation import paper_surrogate_episode
    from utils.training_metrics import BBRMetricAccumulator

    dataset = ExperienceDataset(pool, gamma=1.0, scale=1000, max_length=manifest["sequence_length"],
                                sample_step=manifest["sample_step"])
    metrics = BBRMetricAccumulator()
    loc_names = {v: k for k, v in pool.metadata["location_flags"].items()}
    stream_names = {v: k for k, v in pool.metadata["stream_flags"].items()}
    records, predicted = [], {}
    with torch.no_grad():
        for b, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
            states, actions, returns, timesteps, labels, phases = process_bbr_batch(batch, device=device)
            masked, _ = metrics.update(policy(states, actions, returns, timesteps), labels, phases)
            preds = masked.argmax(dim=-1).reshape(-1).cpu().tolist()
            targets = labels.reshape(-1).cpu().tolist()
            start = dataset.dataset_indices[b]
            for off, (phase, t, p) in enumerate(zip(phases, targets, preds)):
                i = start + off
                predicted[i] = p
                s = pool.states[i]
                records.append({
                    "sample_id": pool.sample_ids[i], "pool_index": i,
                    "location": loc_names[int(s[0])], "stream_group": stream_names[int(s[1])],
                    "phase": phase, "target_action": t, "target_gain": action_index_to_gain(t),
                    "predicted_action": p, "predicted_gain": action_index_to_gain(p), "correct": int(t == p),
                    "observed_throughput": float(s[3]), "observed_retransmissions": float(s[4]),
                })
    surrogate, start = {}, 0
    for end, done in enumerate(pool.dones):
        if not done:
            continue
        stop = end + 1
        out = paper_surrogate_episode(pool.states[start:stop],
                                      [predicted.get(i, pool.actions[i]) for i in range(start, stop)])
        for off in range(stop - start):
            surrogate[start + off] = (float(out["predicted_throughput"][off]),
                                      float(out["predicted_retransmissions"][off]))
        start = stop
    for r in records:
        r["surrogate_throughput"], r["surrogate_retransmissions"] = surrogate[r["pool_index"]]
    return metrics.compute(), records


def cmd_evaluate(args):
    import torch
    from config import cfg
    from evaluate_tokyo_models import write_records

    with open(VAL_POOL, "rb") as f:
        pool = pickle.load(f)
    if pool.metadata.get("split_role") != "validation":
        raise ValueError("Expected the development validation pool")
    root = args.local_model_root or os.environ.get("LM_BBR_LOCAL_MODEL_ROOT", cfg.local_model_root)
    for key, run_dir, _ in models(args.seed):
        out = OUT / "seed{}".format(args.seed) / key / "predictions.csv"
        if out.exists():
            print("skip (exists):", out)
            continue
        if run_dir is None or not (run_dir / "checkpoint").is_dir():
            print("MISSING checkpoint for", key, "seed", args.seed, "->", run_dir)
            continue
        print("evaluating seed {} {}".format(args.seed, key), flush=True)
        policy, manifest = load_policy(run_dir, args.device, root)
        if manifest.get("seed") not in (None, args.seed):
            raise ValueError("Run {} has seed {}, expected {}".format(run_dir, manifest.get("seed"), args.seed))
        m, recs = evaluate_pool(policy, manifest, pool, args.device)
        write_records(out, recs)
        up = [r for r in recs if r["phase"] == "BW_UP"]
        print("  accuracy {:.4f}  BW_UP accuracy {:.4f} (n={})".format(
            m["accuracy"], sum(r["correct"] for r in up) / len(up), len(up)), flush=True)
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- plotting
def _roll(v, h, op):
    return np.asarray([op(v[max(0, i - h):i + h + 1]) for i in range(len(v))])


def surrogate_retx(states, actions, h=10, sharp=5.0, growth=1.5, floor=1e-3, red=0.5):
    """NumPy copy of utils.tokyo_evaluation.paper_surrogate_episode (retransmissions)."""
    s = np.asarray(states, float)
    thr, rx, rtt = s[:, 3], s[:, 4], s[:, 7]
    bcap = _roll(thr, h, lambda v: np.percentile(v, 95))
    rmin = _roll(rtt, h, np.min)
    q = np.maximum(rtt - rmin, 0)
    qref = np.minimum(_roll(q, h, lambda v: np.percentile(v, 95)), 0.15 * rmin)
    ru = np.divide(thr, bcap, out=np.zeros_like(thr), where=bcap > 0)
    du = np.divide(q, qref, out=np.zeros_like(q), where=qref > 0)
    u = np.maximum(np.minimum(ru, 1), np.minimum(du, 1))
    tmin, tmax = float(np.min(rx)), _roll(rx, h, np.max)
    g = np.asarray([GAINS[int(a)] for a in actions])
    sp0 = float(np.logaddexp(0, 0))
    mx = max(float(np.logaddexp(0, sharp * 0.25)) - sp0, 0)
    st = np.maximum(np.logaddexp(0, sharp * (g - 1)) - sp0, 0)
    lf = u * (floor + (1 - floor) * np.power(st / mx, growth))
    below = g < 1
    lf[below] *= 1 - red * (1 - g[below])
    return tmin + (tmax - tmin) * np.clip(lf, 0, 1)


def pool_reference(path):
    """Expert surrogate retransmissions and raw-trace id for every pool index."""
    with open(path, "rb") as f:
        pool = pickle.load(f)
    retx = np.zeros(len(pool.dones))
    trace = np.zeros(len(pool.dones), dtype=int)
    start, tid = 0, 0
    for end, done in enumerate(pool.dones):
        if done:
            retx[start:end + 1] = surrogate_retx(pool.states[start:end + 1], pool.actions[start:end + 1])
            trace[start:end + 1] = tid
            start, tid = end + 1, tid + 1
    return retx, trace


def macro_f1(y, p):
    f1s = []
    for c in UP:
        tp = np.sum((y == c) & (p == c)); npred = np.sum(p == c); nsup = np.sum(y == c)
        prec = tp / npred if npred else 0.0
        rec = tp / nsup if nsup else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 12, "axes.titlesize": 12, "xtick.labelsize": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    val_retx, val_trace = pool_reference(VAL_POOL)
    tok_retx, tok_trace = pool_reference(TOKYO_POOL)
    rows = [(loc, "{} (seen)".format(loc)) for loc in args.locations] + [("Tokyo", "Tokyo (unseen)")]
    data = {}
    for key, _, tokyo_dir in models(args.seed):
        v = pd.read_csv(OUT / "seed{}".format(args.seed) / key / "predictions.csv")
        v["expert_retx"] = val_retx[v.pool_index]; v["trace"] = val_trace[v.pool_index]
        t = pd.read_csv(tokyo_dir / "predictions.csv").assign(location="Tokyo")
        t["expert_retx"] = tok_retx[t.pool_index]; t["trace"] = tok_trace[t.pool_index]
        data[key] = pd.concat([v, t], ignore_index=True)
    keys = list(data)
    col = {"e": "#4A4A4A", "c": "#8FA6BF", "q": "#D9622B"}

    def box(ax, vals, kinds, labels, title, ylabel, ref=None, log=False, whis=1.5):
        bp = ax.boxplot(vals, widths=0.6, patch_artist=True, showfliers=False, whis=whis,
                        medianprops=dict(color="black", linewidth=1.6))
        for patch, k in zip(bp["boxes"], kinds):
            patch.set_facecolor(col[k]); patch.set_edgecolor("#333")
        if ref is not None:
            ax.axhline(ref, ls="--", lw=1, color="#777", zorder=0)
        if log:
            ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 4, 8])
            ax.set_yticklabels(["0.5×", "1×", "2×", "4×", "8×"]); ax.minorticks_off()
        ax.set_xticks(range(1, len(vals) + 1)); ax.set_xticklabels(labels)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.25)

    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4.5 * len(rows)),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]}, squeeze=False)
    letters = iter("abcdefghijklmnop")
    print("\nseed {}: BW_UP results".format(args.seed))
    print("{:10s} {:18s} {:>5s} {:>8s} {:>8s} {:>8s} {:>9s}".format(
        "location", "model", "n", "acc", "majority", "macroF1", "distinct"))
    for r, (loc, title) in enumerate(rows):
        gains, bias, ratio, kinds, labels = [], [], [], [], []
        for i, key in enumerate(keys):
            d = data[key]
            u = d[(d.location == loc) & (d.phase == "BW_UP")]
            y = u.target_gain.round(2).to_numpy(); p = u.predicted_gain.round(2).to_numpy()
            if i == 0:
                gains.append(y)
            gains.append(p)
            g = pd.DataFrame({"t": u.trace, "d": p - y, "r": u.surrogate_retransmissions,
                              "e": u.expert_retx}).groupby("t")
            bias.append(g["d"].mean().to_numpy())
            ratio.append(((g["r"].sum() + 1) / (g["e"].sum() + 1)).to_numpy())
            kinds.append("q" if key.endswith("quantum") else "c"); labels.append(LABEL[key])
            majority = pd.Series(y).value_counts().iloc[0] / len(y)
            print("{:10s} {:18s} {:5d} {:8.4f} {:8.4f} {:8.4f} {:9d}".format(
                loc, LABEL[key].replace("\n", " "), len(y), float(np.mean(y == p)), majority,
                macro_f1(y, p), len(np.unique(p))))
        box(axes[r][0], gains, ["e"] + kinds, ["Expert"] + labels,
            "({}) {}: predicted gain\n(whiskers: min to max)".format(next(letters), title), "Pacing gain", whis=(0, 100))
        box(axes[r][1], bias, kinds, labels,
            "({}) {}: gain bias per trace".format(next(letters), title), "Predicted − expert", ref=0)
        box(axes[r][2], ratio, kinds, labels,
            "({}) {}: surrogate retransmissions\nper trace relative to expert".format(next(letters), title),
            "Ratio to expert (log scale)", ref=1, log=True)
        axes[r][0].set_ylim(1.04, 1.26)
    for c in (1, 2):
        lo = min(a[c].get_ylim()[0] for a in axes); hi = max(a[c].get_ylim()[1] for a in axes)
        for a in axes:
            a[c].set_ylim(lo, hi)
    fig.suptitle("Seed {}: seen locations (validation traces) vs. unseen Tokyo, BW_UP".format(args.seed), y=1.0)
    fig.tight_layout(h_pad=2.5)
    out = args.out or Path("reports/figures/under400m/seen_vs_tokyo_bwup_seed{}.png".format(args.seed))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("written:", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--seed", type=int, required=True, choices=[100003, 200003, 300003, 400003, 500003])
    e.add_argument("--device", default="cuda")
    e.add_argument("--local-model-root")
    p = sub.add_parser("plot")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--locations", nargs="+", default=["Mumbai", "SaoPaulo"],
                   choices=["Ohio", "SaoPaulo", "London", "Mumbai", "Sydney"])
    p.add_argument("--out", type=Path)
    args = parser.parse_args()
    cmd_evaluate(args) if args.cmd == "evaluate" else cmd_plot(args)


if __name__ == "__main__":
    main()
