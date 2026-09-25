"""Analyze where UP-action separability disappears in saved gate200 checkpoints."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader

from config import cfg
from plm_special.backbones import load_local_backbone, resolve_local_revision
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from plm_special.utils.utils import process_bbr_batch, set_random_seed
from train_modern_lora import hidden_size_from_config, verify_development_pool
from utils.bbr import ACTION_LEVELS, BW_UP


DEFAULT_GATE_ROOT = Path("data/processed/lora_training/trainability_gate200_ln_t4")
DEFAULT_OUTPUT = Path("data/processed/lora_training/up_separability_gate200")


def l2_normalize(features):
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def nearest_centroid_predict(train_x, train_y, test_x, classes):
    centroids = np.stack([train_x[train_y == label].mean(axis=0) for label in classes])
    centroids = l2_normalize(centroids)
    scores = l2_normalize(test_x) @ centroids.T
    return np.asarray(classes)[scores.argmax(axis=1)]


def leave_one_trace_out_accuracy(features, labels, trace_ids):
    features = l2_normalize(features)
    labels = np.asarray(labels)
    trace_ids = np.asarray(trace_ids)
    classes = sorted(np.unique(labels).tolist())
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    evaluated = np.zeros(labels.shape, dtype=bool)
    for trace_id in np.unique(trace_ids):
        test = trace_ids == trace_id
        train = ~test
        if not all(np.any(labels[train] == label) for label in classes):
            continue
        predictions[test] = nearest_centroid_predict(
            features[train], labels[train], features[test], classes
        )
        evaluated[test] = True
    if not evaluated.any():
        return {
            "status": "insufficient_trace_support",
            "accuracy": None,
            "evaluated_samples": 0,
            "total_samples": int(len(labels)),
            "prediction_distribution": {},
        }
    return {
        "status": "complete",
        "accuracy": float((predictions[evaluated] == labels[evaluated]).mean()),
        "evaluated_samples": int(evaluated.sum()),
        "total_samples": int(len(labels)),
        "prediction_distribution": {
            str(label): int((predictions[evaluated] == label).sum()) for label in classes
        },
    }


def centroid_separation_ratio(features, labels):
    features = l2_normalize(features)
    labels = np.asarray(labels)
    classes = sorted(np.unique(labels).tolist())
    centroids = {label: features[labels == label].mean(axis=0) for label in classes}
    between = []
    for index, left in enumerate(classes):
        for right in classes[index + 1 :]:
            between.append(float(np.linalg.norm(centroids[left] - centroids[right])))
    within = []
    for label in classes:
        distances = np.linalg.norm(features[labels == label] - centroids[label], axis=1)
        within.extend(distances.tolist())
    mean_between = float(np.mean(between))
    mean_within = float(np.mean(within))
    return {
        "mean_between_centroid_distance": mean_between,
        "mean_within_class_distance": mean_within,
        "between_to_within_ratio": mean_between / max(mean_within, 1e-12),
    }


def analyze_stage(features, labels, trace_ids):
    return {
        "dimensions": int(features.shape[1]),
        "samples": int(features.shape[0]),
        "centroid_separation": centroid_separation_ratio(features, labels),
        "leave_one_trace_out_nearest_centroid": leave_one_trace_out_accuracy(
            features, labels, trace_ids
        ),
    }


def episode_trace_ids(pool):
    sources = pool.metadata.get("source_files", ())
    trace_ids = np.empty(len(pool.dones), dtype=np.int64)
    episode = 0
    start = 0
    for index, done in enumerate(pool.dones):
        if done:
            trace_ids[start : index + 1] = episode
            episode += 1
            start = index + 1
    if start != len(pool.dones) or episode != len(sources):
        raise ValueError(
            "Episode/source mismatch: {} episodes, {} sources".format(
                episode, len(sources)
            )
        )
    return trace_ids, [item["path"] for item in sources]


def build_policy(run_dir, device):
    manifest = json.loads((run_dir / "run.manifest.json").read_text(encoding="utf-8"))
    if manifest.get("tokyo_isolation") != "PASS":
        raise ValueError("Run does not pass Tokyo isolation: {}".format(run_dir))
    model_path = cfg.get_registered_model_path(manifest["model_key"])
    if resolve_local_revision(model_path) != manifest["model_revision"]:
        raise ValueError("Local model revision differs from checkpoint manifest")
    dtype = getattr(torch, manifest["dtype"])
    backbone, model_config = load_local_backbone(model_path, device=device, dtype=dtype)
    backbone = PeftModel.from_pretrained(
        backbone,
        run_dir / "checkpoint" / "adapter",
        is_trainable=False,
        local_files_only=True,
    )
    task_state = torch.load(
        run_dir / "checkpoint" / "task_modules.pt",
        map_location=device,
        weights_only=True,
    )
    state_feature_dim = int(task_state["0.fc1.0.weight"].shape[0])
    max_ep_len = int(task_state["1.weight"].shape[0] - 1)
    head = manifest["head_config"]
    policy = OfflineRLPolicy(
        state_feature_dim=state_feature_dim,
        action_levels=ACTION_LEVELS,
        state_encoder=EncoderNetwork(embed_dim=state_feature_dim).to(device),
        plm=backbone,
        plm_embed_size=hidden_size_from_config(model_config),
        max_length=manifest["sequence_length"],
        max_ep_len=max_ep_len,
        device=device,
        device_out=device,
        head_type=manifest["head_type"],
        quantum_config={
            "n_qubits": head.get("n_qubits", head.get("bottleneck_dim", 8)),
            "depth": head.get("depth", 2),
            "ansatz": head.get("ansatz", "trainable_ry_layers"),
            "input_layernorm": head.get("input_layernorm", False),
            "temperature": head.get("temperature", 1.0),
            "angle_scale": head.get("angle_scale", "pi"),
        },
    )
    policy.modules_except_plm.load_state_dict(task_state)
    policy.eval()
    return policy, manifest


def register_representation_hooks(policy, captured):
    handles = []

    def capture_input(_module, inputs):
        captured["qwen_hidden"] = inputs[0].detach().float().cpu().reshape(-1, inputs[0].shape[-1])

    def capture(name):
        def hook(_module, _inputs, output):
            captured[name] = output.detach().float().cpu().reshape(-1, output.shape[-1])
        return hook

    handles.append(policy.action_head.register_forward_pre_hook(capture_input))
    if policy.head_type == "quantum":
        handles.append(policy.action_head.angle_projection.register_forward_hook(capture("raw_projection")))
        handles.append(policy.action_head.quantum_layer.register_forward_hook(capture("bottleneck")))
        handles.append(policy.action_head.output_projection.register_forward_hook(capture("logits")))
    else:
        handles.append(policy.action_head.net[0].register_forward_hook(capture("raw_projection")))
        handles.append(policy.action_head.net[1].register_forward_hook(capture("bottleneck")))
        handles.append(policy.action_head.net[2].register_forward_hook(capture("logits")))
    return handles


def collect_representations(policy, pool, sequence_length, sample_step, device, max_windows):
    dataset = ExperienceDataset(
        pool, gamma=1.0, scale=1000, max_length=sequence_length, sample_step=sample_step
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
    trace_by_position, source_paths = episode_trace_ids(pool)
    stage_values = {name: [] for name in ("qwen_hidden", "raw_projection", "bottleneck", "logits")}
    labels_out, traces_out = [], []
    captured = {}
    handles = register_representation_hooks(policy, captured)
    try:
        with torch.no_grad():
            for window_index, batch in enumerate(loader):
                if max_windows and window_index >= max_windows:
                    break
                states, actions, returns, timesteps, labels, phases = process_bbr_batch(
                    batch, device=device
                )
                captured.clear()
                policy(states, actions, returns, timesteps)
                up_mask = np.asarray([phase == BW_UP for phase in phases], dtype=bool)
                if not up_mask.any():
                    continue
                for name in stage_values:
                    stage_values[name].append(captured[name].numpy()[up_mask])
                labels_out.extend(labels.detach().cpu().reshape(-1).numpy()[up_mask].tolist())
                start = dataset.dataset_indices[window_index]
                traces_out.extend(trace_by_position[start : start + sequence_length][up_mask].tolist())
                if (window_index + 1) % 50 == 0:
                    print("{} validation windows processed".format(window_index + 1), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if not labels_out:
        raise ValueError("No UP positions found in selected validation windows")
    return (
        {name: np.concatenate(values, axis=0) for name, values in stage_values.items()},
        np.asarray(labels_out, dtype=np.int64),
        np.asarray(traces_out, dtype=np.int64),
        source_paths,
        min(len(dataset), max_windows) if max_windows else len(dataset),
    )


def analyze_run(run_dir, split_dir, output_dir, device, max_windows):
    started = time.time()
    policy, manifest = build_policy(run_dir, device)
    with (split_dir / "validation.pkl").open("rb") as stream:
        pool = pickle.load(stream)
    verify_development_pool(pool, "validation")
    representations, labels, trace_ids, source_paths, windows = collect_representations(
        policy,
        pool,
        manifest["sequence_length"],
        manifest["sample_step"],
        device,
        max_windows,
    )
    classes, counts = np.unique(labels, return_counts=True)
    result = {
        "status": "up_representation_separability_development_diagnostic",
        "reportable_result": False,
        "head_type": manifest["head_type"],
        "source_run": str(run_dir),
        "checkpoint_reload": manifest["checkpoint_reload"],
        "held_out_location": "Tokyo",
        "tokyo_isolation": "PASS",
        "validation_windows": windows,
        "up_samples": int(len(labels)),
        "up_label_distribution": {str(c): int(n) for c, n in zip(classes, counts)},
        "majority_baseline": float(counts.max() / counts.sum()),
        "trace_count": int(len(np.unique(trace_ids))),
        "trace_paths_sha256": sha256("\n".join(source_paths).encode("utf-8")).hexdigest(),
        "stages": {
            name: analyze_stage(values, labels, trace_ids)
            for name, values in representations.items()
        },
        "wall_clock_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "separability.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Separability result: {}".format(path))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-root", type=Path, default=DEFAULT_GATE_ROOT)
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-validation-windows", type=int, default=0)
    args = parser.parse_args()
    if args.max_validation_windows < 0:
        parser.error("max-validation-windows cannot be negative")
    set_random_seed(100003)
    runs = (
        args.gate_root / "twin_ln_t4",
        args.gate_root / "quantum_ln_t4_pi",
    )
    results = []
    for run_dir in runs:
        results.append(
            analyze_run(
                run_dir,
                args.split_dir,
                args.output_root / run_dir.name,
                args.device,
                args.max_validation_windows,
            )
        )
    summary = {
        "status": "up_representation_separability_summary",
        "reportable_result": False,
        "held_out_location": "Tokyo",
        "tokyo_isolation": "PASS",
        "runs": results,
    }
    summary_path = args.output_root / "separability.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
