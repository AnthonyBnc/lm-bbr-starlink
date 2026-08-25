"""Train one modern LoRA BBR policy on the shared development split."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import pickle
import platform
import subprocess
import time

import torch
import transformers
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from config import cfg
from plm_special.backbones import load_local_backbone, resolve_local_revision
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from plm_special.utils.utils import process_bbr_batch, set_random_seed
from utils.bbr import ACTION_LEVELS
from utils.training_metrics import (
    BBRMetricAccumulator,
    LOSS_WEIGHTING_OPTIONS,
    LOSS_WEIGHTING_NONE,
)
from utils.training_sampling import (
    TRAINING_SAMPLING_OPTIONS,
    TRAINING_SAMPLING_ORIGINAL,
    build_training_window_indices,
    load_training_window_plan,
)


DEFAULT_SPLIT_DIR = Path(
    "data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003"
)


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def sample_ids_sha256(pool):
    values = getattr(pool, "sample_ids", None)
    if values is None or any(value is None for value in values):
        raise ValueError("Every experience must have a stable sample ID")
    return sha256("\n".join(values).encode("utf-8")).hexdigest()


def verify_development_pool(pool, expected_role):
    metadata = pool.metadata
    if metadata.get("held_out_location") != "Tokyo":
        raise ValueError("Experience pool does not declare Tokyo as held out")
    if metadata.get("split_role") != expected_role:
        raise ValueError(
            "Expected {} pool, got {!r}".format(expected_role, metadata.get("split_role"))
        )
    source_paths = [item["path"] for item in metadata.get("source_files", ())]
    if any("tokyo" in path.lower() for path in source_paths):
        raise ValueError("Tokyo source found in development pool")
    if len(set(pool.sample_ids)) != len(pool.sample_ids):
        raise ValueError("Experience pool contains duplicate sample IDs")


def resolve_dtype(name, model_key):
    if name == "auto":
        name = cfg.get_registered_model(model_key)["preferred_dtype"]
    return name, getattr(torch, name)


def hidden_size_from_config(model_config):
    hidden_size = getattr(model_config, "hidden_size", None)
    if hidden_size is None and hasattr(model_config, "text_config"):
        hidden_size = model_config.text_config.hidden_size
    if not hidden_size:
        raise ValueError("Cannot resolve backbone hidden size")
    return hidden_size


def maximum_episode_length(pool):
    maximum = current = 0
    for done in pool.dones:
        current += 1
        if done:
            maximum = max(maximum, current)
            current = 0
    if current:
        raise ValueError("Final experience does not end an episode")
    return maximum


def make_loader(pool, sequence_length, sample_step, shuffle, seed):
    dataset = ExperienceDataset(
        pool,
        gamma=1.0,
        scale=1000,
        max_length=sequence_length,
        sample_step=sample_step,
    )
    generator = torch.Generator().manual_seed(seed)
    return dataset, DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        generator=generator,
        pin_memory=False,
    )


def make_training_loader(
    pool,
    sequence_length,
    sample_step,
    seed,
    sampling_strategy,
    target_windows,
    up_density_power,
    down_penalty,
    replacement_windows,
    sampling_plan_path=None,
):
    dataset = ExperienceDataset(
        pool,
        gamma=1.0,
        scale=1000,
        max_length=sequence_length,
        sample_step=sample_step,
    )
    if sampling_plan_path is None:
        selected_indices, sampling_record = build_training_window_indices(
            dataset,
            strategy=sampling_strategy,
            target_windows=target_windows,
            seed=seed,
            up_density_power=up_density_power,
            down_penalty=down_penalty,
            replacement_windows=replacement_windows,
        )
    else:
        selected_indices, sampling_record = load_training_window_plan(
            sampling_plan_path,
            dataset,
            strategy=sampling_strategy,
            target_windows=target_windows,
            seed=seed,
            up_density_power=up_density_power,
            down_penalty=down_penalty,
            replacement_windows=replacement_windows,
        )
    selected_dataset = Subset(dataset, selected_indices)
    generator = torch.Generator().manual_seed(seed)
    return selected_dataset, DataLoader(
        selected_dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        pin_memory=False,
    ), sampling_record


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    grad_accum_steps=1,
    max_steps=0,
    loss_weighting=LOSS_WEIGHTING_NONE,
    trainability_diagnostics=False,
):
    training = optimizer is not None
    model.train(training)
    metrics = BBRMetricAccumulator()
    optimizer_steps = 0
    gradient_norms = []
    component_gradient_norms = {}
    head_diagnostic_values = {}
    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        states, actions, returns, timesteps, labels, phases = process_bbr_batch(
            batch, device=device
        )
        with torch.set_grad_enabled(training):
            logits = model(states, actions, returns, timesteps)
            if logits.shape[-1] != ACTION_LEVELS:
                raise RuntimeError("Policy did not produce 11 action logits")
            _, loss = metrics.update(logits, labels, phases, loss_weighting=loss_weighting)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss at step {}".format(step))
            if training:
                (loss / grad_accum_steps).backward()
                should_step = (step + 1) % grad_accum_steps == 0
                is_last = step + 1 == len(loader) or (max_steps and step + 1 == max_steps)
                if should_step or is_last:
                    if trainability_diagnostics:
                        for name, parameter in model.action_head.named_parameters():
                            if parameter.grad is None:
                                continue
                            value = parameter.grad.detach().float().norm().cpu().item()
                            component_gradient_norms.setdefault(name, []).append(value)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    if not torch.isfinite(grad_norm):
                        raise RuntimeError("Non-finite gradient norm at step {}".format(step))
                    gradient_norms.append(float(grad_norm.detach().cpu()))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
            if trainability_diagnostics:
                snapshot = getattr(model.action_head, "last_diagnostics", None) or {}
                for name, value in snapshot.items():
                    head_diagnostic_values.setdefault(name, []).append(float(value))
        if training and (step == 0 or (step + 1) % 25 == 0):
            print("train step {} loss {:.6f}".format(step + 1, float(loss.detach().cpu())))

    result = metrics.compute()
    result["batches"] = min(len(loader), max_steps) if max_steps else len(loader)
    result["optimizer_steps"] = optimizer_steps
    result["gradient_norm_max"] = max(gradient_norms) if gradient_norms else None
    if trainability_diagnostics:
        result["trainability_diagnostics"] = {
            "component_gradient_norm": {
                name: {
                    "mean": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values),
                }
                for name, values in sorted(component_gradient_norms.items())
            },
            "head_activation": {
                name: {
                    "mean": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values),
                }
                for name, values in sorted(head_diagnostic_values.items())
            },
        }
    return result


def save_checkpoint(policy, optimizer, output_dir):
    checkpoint_dir = output_dir / "checkpoint"
    adapter_dir = checkpoint_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    policy.plm.save_pretrained(adapter_dir, safe_serialization=True)
    torch.save(policy.modules_except_plm.state_dict(), checkpoint_dir / "task_modules.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    return checkpoint_dir, adapter_dir


def verify_checkpoint_reload(policy, adapter_dir, task_state_path, batch, device):
    policy.eval()
    states, actions, returns, timesteps, _, _ = process_bbr_batch(batch, device=device)
    with torch.no_grad():
        expected = policy(states, actions, returns, timesteps).float().cpu()
    policy.plm.load_adapter(adapter_dir, adapter_name="reload_check", is_trainable=False)
    policy.plm.set_adapter("reload_check")
    task_state = torch.load(task_state_path, map_location=device, weights_only=True)
    policy.modules_except_plm.load_state_dict(task_state)
    with torch.no_grad():
        restored = policy(states, actions, returns, timesteps).float().cpu()
    if not torch.allclose(expected, restored, rtol=2e-3, atol=2e-3):
        raise RuntimeError("Checkpoint reload changed policy logits")
    return "PASS"


def git_metadata():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    return commit, dirty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="lfm2_5_2_6b", choices=tuple(cfg.modern_model_registry))
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--sample-step", type=int, default=20)
    parser.add_argument("--state-feature-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-validation-steps", type=int, default=0)
    parser.add_argument(
        "--training-sampling-strategy",
        choices=TRAINING_SAMPLING_OPTIONS,
        default=TRAINING_SAMPLING_ORIGINAL,
    )
    parser.add_argument(
        "--target-train-windows",
        type=int,
        default=0,
        help="Exact sampled training windows; required by UP-focused sampling.",
    )
    parser.add_argument("--up-density-power", type=float, default=3.0)
    parser.add_argument("--down-window-penalty", type=float, default=1.0)
    parser.add_argument("--replacement-train-windows", type=int, default=0)
    parser.add_argument("--training-sampling-plan", type=Path)
    parser.add_argument(
        "--trainability-diagnostics",
        action="store_true",
        help="Record head component gradients and bottleneck activation statistics.",
    )
    parser.add_argument(
        "--loss-weighting",
        choices=LOSS_WEIGHTING_OPTIONS,
        default=LOSS_WEIGHTING_NONE,
        help="Training loss weighting. Validation metrics remain unweighted.",
    )
    parser.add_argument(
        "--head-type",
        choices=("classical", "classical_twin", "quantum"),
        default="classical",
    )
    parser.add_argument("--n-qubits", type=int, default=cfg.quantum_defaults["n_qubits"])
    parser.add_argument("--quantum-depth", type=int, default=cfg.quantum_defaults["depth"])
    parser.add_argument(
        "--head-input-layernorm",
        action="store_true",
        help="Apply LayerNorm before the classical-twin or quantum projection.",
    )
    parser.add_argument(
        "--bottleneck-temperature",
        type=float,
        default=1.0,
        help="Divide projected features by this positive value before tanh.",
    )
    parser.add_argument(
        "--quantum-angle-scale",
        choices=("pi", "half_pi"),
        default="pi",
    )
    parser.add_argument(
        "--quantum-ansatz",
        choices=("trainable_ry_layers", "trainable_ry_rz_layers"),
        default=cfg.quantum_defaults.get("ansatz", "trainable_ry_layers"),
    )
    parser.add_argument(
        "--run-purpose",
        choices=(
            "phase2_pilot",
            "phase3_exploratory",
            "phase4_dev_multiseed",
            "quantum_smoke",
            "head_ablation_smoke",
            "head_ablation_pilot",
            "quantum_up_focus",
        ),
        default="phase2_pilot",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.sequence_length < 1 or args.sample_step < 1:
        parser.error("epochs, sequence-length, and sample-step must be positive")
    if args.bottleneck_temperature <= 0:
        parser.error("bottleneck-temperature must be positive")

    set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype_name, dtype = resolve_dtype(args.dtype, args.model_key)
    model_info = cfg.get_registered_model(args.model_key)
    model_path = cfg.get_registered_model_path(args.model_key)
    revision = resolve_local_revision(model_path)
    if revision != model_info["revision"]:
        raise RuntimeError("Local model revision does not match config")

    # Place the large weights before materialising both pool objects. On 16 GB
    # unified-memory Macs this avoids fragmentation in the Metal allocator.
    backbone, model_config = load_local_backbone(model_path, device=args.device, dtype=dtype)
    from plm_special.lora import attach_modern_lora, parameter_counts

    backbone, lora_config, adapter_modules = attach_modern_lora(
        backbone,
        args.model_key,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        gradient_checkpointing=True,
    )
    hidden_size = hidden_size_from_config(model_config)

    train_path = args.split_dir / "train.pkl"
    validation_path = args.split_dir / "validation.pkl"
    split_manifest_path = args.split_dir / "split.manifest.json"
    with train_path.open("rb") as stream:
        train_pool = pickle.load(stream)
    with validation_path.open("rb") as stream:
        validation_pool = pickle.load(stream)
    verify_development_pool(train_pool, "train")
    verify_development_pool(validation_pool, "validation")
    if set(train_pool.sample_ids) & set(validation_pool.sample_ids):
        raise ValueError("Train and validation sample IDs overlap")
    if args.training_sampling_plan is not None:
        sampling_plan = json.loads(
            args.training_sampling_plan.read_text(encoding="utf-8")
        )
        if sampling_plan.get("source_train", {}).get("sha256") != file_sha256(train_path):
            raise ValueError("Training sampling plan does not match train.pkl")
        if sampling_plan.get("source_validation", {}).get("sha256") != file_sha256(
            validation_path
        ):
            raise ValueError("Training sampling plan does not match validation.pkl")
        if sampling_plan.get("split_manifest", {}).get("sha256") != file_sha256(
            split_manifest_path
        ):
            raise ValueError("Training sampling plan does not match split.manifest.json")

    train_dataset, train_loader, training_sampling = make_training_loader(
        train_pool,
        args.sequence_length,
        args.sample_step,
        args.seed,
        args.training_sampling_strategy,
        args.target_train_windows,
        args.up_density_power,
        args.down_window_penalty,
        args.replacement_train_windows,
        args.training_sampling_plan,
    )
    validation_dataset, validation_loader = make_loader(
        validation_pool, args.sequence_length, args.sample_step, False, args.seed
    )
    policy = OfflineRLPolicy(
        state_feature_dim=args.state_feature_dim,
        action_levels=ACTION_LEVELS,
        state_encoder=EncoderNetwork(embed_dim=args.state_feature_dim).to(args.device),
        plm=backbone,
        plm_embed_size=hidden_size,
        max_length=args.sequence_length,
        max_ep_len=max(maximum_episode_length(train_pool), maximum_episode_length(validation_pool)),
        device=args.device,
        device_out=args.device,
        head_type=args.head_type,
        quantum_config={
            "n_qubits": args.n_qubits,
            "depth": args.quantum_depth,
            "ansatz": args.quantum_ansatz,
            "input_layernorm": args.head_input_layernorm,
            "temperature": args.bottleneck_temperature,
            "angle_scale": args.quantum_angle_scale,
        },
    )
    if args.trainability_diagnostics and hasattr(policy.action_head, "enable_diagnostics"):
        policy.action_head.enable_diagnostics(True)
    trainable_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_parameters, lora_trainable_parameters = parameter_counts(policy.plm)
    task_trainable_parameters = sum(
        parameter.numel()
        for parameter in policy.modules_except_plm.parameters()
        if parameter.requires_grad
    )

    started = time.time()
    epoch_metrics = []
    for epoch in range(args.epochs):
        print("epoch {}/{}".format(epoch + 1, args.epochs))
        train_metrics = run_epoch(
            policy,
            train_loader,
            args.device,
            optimizer=optimizer,
            grad_accum_steps=args.grad_accum_steps,
            max_steps=args.max_train_steps,
            loss_weighting=args.loss_weighting,
            trainability_diagnostics=args.trainability_diagnostics,
        )
        validation_metrics = run_epoch(
            policy,
            validation_loader,
            args.device,
            max_steps=args.max_validation_steps,
            trainability_diagnostics=args.trainability_diagnostics,
        )
        epoch_metrics.append(
            {"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics}
        )

    checkpoint_dir, adapter_dir = save_checkpoint(policy, optimizer, args.output_dir)
    reload_batch = next(iter(validation_loader))
    checkpoint_reload = verify_checkpoint_reload(
        policy,
        adapter_dir,
        checkpoint_dir / "task_modules.pt",
        reload_batch,
        args.device,
    )
    elapsed = time.time() - started
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({"epochs": epoch_metrics}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    limited = bool(args.max_train_steps or args.max_validation_steps)
    git_commit, git_dirty = git_metadata()
    status_by_purpose = {
        "phase2_pilot": "exploratory_lora_pilot",
        "phase3_exploratory": "exploratory_modern_baseline",
        "phase4_dev_multiseed": "phase4_development_multiseed",
        "quantum_smoke": "quantum_gpt_smoke",
        "head_ablation_smoke": "head_ablation_smoke",
        "head_ablation_pilot": "head_ablation_pilot",
        "quantum_up_focus": "quantum_up_focus_development_diagnostic",
    }
    status = (
        "exploratory_pipeline_check"
        if limited
        and args.run_purpose not in ("quantum_smoke", "head_ablation_smoke")
        else status_by_purpose[args.run_purpose]
    )
    if args.run_purpose.startswith("head_ablation"):
        model_role = {
            "classical": "gpt_classical",
            "classical_twin": "gpt_classical_twin",
            "quantum": "gpt_quantum",
        }[args.head_type]
    else:
        model_role = {
            "classical": "modern_baseline",
            "classical_twin": "gpt_classical_twin",
            "quantum": "gpt_quantum",
        }[args.head_type]
    head_config = (
        policy.action_head.manifest_config()
        if hasattr(policy.action_head, "manifest_config")
        else {"type": "linear"}
    )
    manifest = {
        "status": status,
        "run_purpose": args.run_purpose,
        "reportable_result": False,
        "run_completed_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "model_role": model_role,
        "head_type": args.head_type,
        "head_config": head_config,
        "model_key": args.model_key,
        "model_id": model_info["hf_id"],
        "model_revision": revision,
        "device": args.device,
        "dtype": dtype_name,
        "dataset_version": train_pool.metadata["parent_dataset_version"],
        "split_version": train_pool.metadata["split_version"],
        "split_status": "exploratory_pending_supervisor_approval",
        "held_out_location": "Tokyo",
        "tokyo_isolation": "PASS",
        "train_pool": train_path.as_posix(),
        "train_pool_sha256": file_sha256(train_path),
        "train_sample_ids_sha256": sample_ids_sha256(train_pool),
        "validation_pool": validation_path.as_posix(),
        "validation_pool_sha256": file_sha256(validation_path),
        "validation_sample_ids_sha256": sample_ids_sha256(validation_pool),
        "split_manifest_sha256": file_sha256(split_manifest_path),
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "sample_step": args.sample_step,
        "batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "effective_sequences_per_optimizer_step": args.grad_accum_steps,
        "epochs": args.epochs,
        "max_train_steps": args.max_train_steps or None,
        "max_validation_steps": args.max_validation_steps or None,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": 0.25,
        "trainability_diagnostics": args.trainability_diagnostics,
        "training_loss_weighting": args.loss_weighting,
        "validation_loss_weighting": LOSS_WEIGHTING_NONE,
        "training_sampling": training_sampling,
        "validation_sampling": "original_unmodified_distribution",
        "checkpoint_rule": "final epoch for {}".format(args.run_purpose),
        "lora_config": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": cfg.modern_lora_registry[args.model_key]["target_modules"],
            "adapter_module_count": len(adapter_modules),
        },
        "backbone_parameters": total_parameters,
        "lora_trainable_parameters": lora_trainable_parameters,
        "task_trainable_parameters": task_trainable_parameters,
        "total_trainable_parameters": sum(p.numel() for p in trainable_parameters),
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "wall_clock_seconds": elapsed,
        "metrics": metrics_path.as_posix(),
        "metrics_sha256": file_sha256(metrics_path),
        "checkpoint": checkpoint_dir.as_posix(),
        "checkpoint_reload": checkpoint_reload,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "hardware": platform.platform(),
        "quantum_config": policy.quantum_config,
    }
    manifest_path = args.output_dir / "run.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("LoRA {} PASS".format("pipeline check" if limited else args.run_purpose))
    print("Validation accuracy: {:.6f}".format(epoch_metrics[-1]["validation"]["accuracy"]))
    print("Checkpoint reload: {}".format(checkpoint_reload))
    print("Manifest: {}".format(manifest_path))


if __name__ == "__main__":
    main()
