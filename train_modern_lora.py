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
from torch.optim.lr_scheduler import LambdaLR
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
    evaluate_validation_criteria,
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


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def resolve_recorded_path(run_dir, recorded_path, fallback):
    if recorded_path:
        candidate = Path(recorded_path)
        if candidate.is_absolute():
            return candidate
        relative_candidate = run_dir / candidate
        if relative_candidate.exists():
            return relative_candidate
        if candidate.exists():
            return candidate
    return run_dir / fallback


def load_resume_record(run_dir):
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run.manifest.json"
    if not manifest_path.is_file():
        manifest_path = run_dir / "progress.manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "Resume source has neither run.manifest.json nor progress.manifest.json: {}".format(
                run_dir
            )
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_dir = resolve_recorded_path(
        run_dir, manifest.get("checkpoint"), "checkpoint"
    )
    metrics_path = resolve_recorded_path(run_dir, manifest.get("metrics"), "metrics.json")
    required = (
        checkpoint_dir / "adapter" / "adapter_config.json",
        checkpoint_dir / "task_modules.pt",
        checkpoint_dir / "optimizer.pt",
        metrics_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Resume source is incomplete: {}".format(", ".join(missing)))
    completed_epochs = int(manifest.get("completed_epochs", manifest.get("epochs", 0)))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if completed_epochs < 1 or len(metrics.get("epochs", ())) != completed_epochs:
        raise ValueError("Resume epoch count does not match metrics history")
    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "checkpoint_dir": checkpoint_dir,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "completed_epochs": completed_epochs,
    }


def validate_resume_compatibility(manifest, expected):
    """Reject methodology/config drift before loading a large resume checkpoint."""
    mismatches = []
    for field, expected_value in expected.items():
        observed = manifest.get(field)
        if field == "warmup_steps":
            observed = observed or 0
        if observed != expected_value:
            mismatches.append(
                "{}: resume={!r}, requested={!r}".format(
                    field, observed, expected_value
                )
            )
    if mismatches:
        raise ValueError("Resume configuration mismatch: {}".format("; ".join(mismatches)))


def validate_resume_lora(manifest, rank, alpha, dropout):
    observed = manifest.get("lora_config") or {}
    expected = {"rank": rank, "alpha": alpha, "dropout": dropout}
    mismatches = [
        "{}: resume={!r}, requested={!r}".format(name, observed.get(name), value)
        for name, value in expected.items()
        if observed.get(name) != value
    ]
    if mismatches:
        raise ValueError("Resume LoRA mismatch: {}".format("; ".join(mismatches)))


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
    lr_scheduler=None,
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
                    if lr_scheduler is not None:
                        lr_scheduler.step()
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


def save_checkpoint(
    policy,
    optimizer,
    checkpoint_dir,
    lr_scheduler=None,
    training_state=None,
):
    adapter_dir = checkpoint_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    policy.plm.save_pretrained(adapter_dir, safe_serialization=True)
    torch.save(policy.modules_except_plm.state_dict(), checkpoint_dir / "task_modules.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    if lr_scheduler is not None:
        torch.save(lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    if training_state is not None:
        write_json(checkpoint_dir / "training_state.json", training_state)
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Target total epochs. With --resume-from-run, continue up to this epoch.",
    )
    parser.add_argument(
        "--resume-from-run",
        type=Path,
        help="Completed or interrupted run directory whose checkpoint will be continued.",
    )
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--sample-step", type=int, default=20)
    parser.add_argument("--state-feature-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Linear optimizer-step warmup; 0 preserves the existing constant LR.",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-validation-steps", type=int, default=0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation-loss epochs; 0 disables it.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss reduction counted as improvement.",
    )
    parser.add_argument("--success-overall-accuracy", type=float)
    parser.add_argument("--success-up-accuracy", type=float)
    parser.add_argument("--success-macro-phase-accuracy", type=float)
    parser.add_argument("--max-up-prediction-share", type=float)
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
        choices=("trainable_ry_layers", "trainable_ry_rz_layers", "data_reuploading_ry"),
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
            "slm_bbr_modern_backbone_extension",
            "under400m_quantum_gpt_follow_on",
        ),
        default="phase2_pilot",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.sequence_length < 1 or args.sample_step < 1:
        parser.error("epochs, sequence-length, and sample-step must be positive")
    if args.warmup_steps < 0:
        parser.error("warmup-steps must be non-negative")
    if args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0:
        parser.error("early-stopping values must be non-negative")
    if args.bottleneck_temperature <= 0:
        parser.error("bottleneck-temperature must be positive")
    for name in (
        "success_overall_accuracy",
        "success_up_accuracy",
        "success_macro_phase_accuracy",
        "max_up_prediction_share",
    ):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error("{} must be between 0 and 1".format(name.replace("_", "-")))

    set_random_seed(args.seed)
    resume = load_resume_record(args.resume_from_run) if args.resume_from_run else None
    if resume and args.output_dir.resolve() == resume["run_dir"].resolve():
        parser.error("Resume output-dir must differ from resume-from-run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype_name, dtype = resolve_dtype(args.dtype, args.model_key)
    model_info = cfg.get_registered_model(args.model_key)
    model_path = cfg.get_registered_model_path(args.model_key)
    revision = resolve_local_revision(model_path)
    if revision != model_info["revision"]:
        raise RuntimeError("Local model revision does not match config")
    if resume:
        validate_resume_compatibility(
            resume["manifest"],
            {
                "model_key": args.model_key,
                "model_revision": revision,
                "dtype": dtype_name,
                "seed": args.seed,
                "sequence_length": args.sequence_length,
                "sample_step": args.sample_step,
                "gradient_accumulation_steps": args.grad_accum_steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "head_type": args.head_type,
            },
        )
        validate_resume_lora(
            resume["manifest"], args.rank, args.alpha, args.dropout
        )
        if args.epochs <= resume["completed_epochs"]:
            parser.error(
                "--epochs must be greater than the {} completed resume epochs".format(
                    resume["completed_epochs"]
                )
            )

    # Place the large weights before materialising both pool objects. On 16 GB
    # unified-memory Macs this avoids fragmentation in the Metal allocator.
    backbone, model_config = load_local_backbone(model_path, device=args.device, dtype=dtype)
    from plm_special.lora import (
        attach_modern_lora,
        load_modern_lora_checkpoint,
        parameter_counts,
    )

    if resume:
        backbone, lora_config, adapter_modules = load_modern_lora_checkpoint(
            backbone,
            args.model_key,
            resume["checkpoint_dir"] / "adapter",
            gradient_checkpointing=True,
        )
    else:
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
    if resume:
        validate_resume_compatibility(
            resume["manifest"],
            {
                "train_pool_sha256": file_sha256(train_path),
                "validation_pool_sha256": file_sha256(validation_path),
                "split_manifest_sha256": file_sha256(split_manifest_path),
            },
        )
        resume_loss_weighting = resume["manifest"].get(
            "training_loss_weighting", LOSS_WEIGHTING_NONE
        )
        if resume_loss_weighting != args.loss_weighting:
            raise ValueError("Resume training loss weighting does not match")
        resume_sampling = resume["manifest"].get("training_sampling")
        if resume_sampling is None:
            resume_sampling_strategy = TRAINING_SAMPLING_ORIGINAL
        else:
            resume_sampling_strategy = resume_sampling.get(
                "strategy", TRAINING_SAMPLING_ORIGINAL
            )
        if resume_sampling_strategy != args.training_sampling_strategy:
            raise ValueError("Resume training sampling strategy does not match")
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
    if resume:
        task_state = torch.load(
            resume["checkpoint_dir"] / "task_modules.pt",
            map_location=args.device,
            weights_only=True,
        )
        policy.modules_except_plm.load_state_dict(task_state, strict=True)
    trainable_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    lr_scheduler = None
    if args.warmup_steps:
        lr_scheduler = LambdaLR(
            optimizer,
            lambda step: min((step + 1) / args.warmup_steps, 1.0),
        )
    if resume:
        optimizer_state = torch.load(
            resume["checkpoint_dir"] / "optimizer.pt",
            map_location=args.device,
            weights_only=True,
        )
        optimizer.load_state_dict(optimizer_state)
        scheduler_path = resume["checkpoint_dir"] / "scheduler.pt"
        if lr_scheduler is not None:
            if not scheduler_path.is_file():
                raise ValueError(
                    "Warmup resume requires scheduler.pt in the source checkpoint"
                )
            scheduler_state = torch.load(
                scheduler_path, map_location="cpu", weights_only=True
            )
            lr_scheduler.load_state_dict(scheduler_state)
        if getattr(train_loader, "generator", None) is not None:
            train_loader.generator.manual_seed(args.seed + resume["completed_epochs"])
    total_parameters, lora_trainable_parameters = parameter_counts(policy.plm)
    task_trainable_parameters = sum(
        parameter.numel()
        for parameter in policy.modules_except_plm.parameters()
        if parameter.requires_grad
    )

    started = time.time()
    start_epoch = resume["completed_epochs"] if resume else 0
    epoch_metrics = list(resume["metrics"]["epochs"]) if resume else []
    metrics_path = args.output_dir / "metrics.json"
    best_validation_loss = min(
        (entry["validation"]["loss"] for entry in epoch_metrics),
        default=float("inf"),
    )
    non_improving_epochs = 0
    if epoch_metrics and args.early_stopping_patience:
        running_best = float("inf")
        for entry in epoch_metrics:
            loss = entry["validation"]["loss"]
            if loss < running_best - args.early_stopping_min_delta:
                running_best = loss
                non_improving_epochs = 0
            else:
                non_improving_epochs += 1
    stopped_early = False
    for epoch in range(start_epoch, args.epochs):
        print("epoch {}/{}".format(epoch + 1, args.epochs))
        train_metrics = run_epoch(
            policy,
            train_loader,
            args.device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
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
        validation_criteria = evaluate_validation_criteria(
            validation_metrics,
            overall_accuracy=args.success_overall_accuracy,
            up_accuracy=args.success_up_accuracy,
            macro_phase_accuracy=args.success_macro_phase_accuracy,
            max_up_prediction_share=args.max_up_prediction_share,
        )
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation_metrics,
                "validation_criteria": validation_criteria,
            }
        )
        validation_loss = validation_metrics["loss"]
        if validation_loss < best_validation_loss - args.early_stopping_min_delta:
            best_validation_loss = validation_loss
            non_improving_epochs = 0
        else:
            non_improving_epochs += 1

        epoch_checkpoint_dir = (
            args.output_dir / "epoch_checkpoints" / "epoch_{:04d}".format(epoch + 1)
        )
        training_state = {
            "completed_epochs": epoch + 1,
            "target_epochs": args.epochs,
            "resumed_from_epoch": start_epoch if resume else None,
            "data_order_resume_policy": (
                "deterministic_reseed_seed_plus_completed_epochs"
                if resume
                else "continuous_generator_from_seed"
            ),
            "best_validation_loss": best_validation_loss,
            "non_improving_epochs": non_improving_epochs,
        }
        save_checkpoint(
            policy,
            optimizer,
            epoch_checkpoint_dir,
            lr_scheduler=lr_scheduler,
            training_state=training_state,
        )
        write_json(metrics_path, {"epochs": epoch_metrics})
        progress_manifest = {
            "status": "training_in_progress",
            "model_key": args.model_key,
            "model_revision": revision,
            "dtype": dtype_name,
            "seed": args.seed,
            "sequence_length": args.sequence_length,
            "sample_step": args.sample_step,
            "gradient_accumulation_steps": args.grad_accum_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "head_type": args.head_type,
            "lora_config": {
                "rank": args.rank,
                "alpha": args.alpha,
                "dropout": args.dropout,
            },
            "training_loss_weighting": args.loss_weighting,
            "training_sampling": training_sampling,
            "train_pool_sha256": file_sha256(train_path),
            "validation_pool_sha256": file_sha256(validation_path),
            "split_manifest_sha256": file_sha256(split_manifest_path),
            "completed_epochs": epoch + 1,
            "target_epochs": args.epochs,
            "checkpoint": epoch_checkpoint_dir.relative_to(args.output_dir).as_posix(),
            "metrics": metrics_path.relative_to(args.output_dir).as_posix(),
            "validation_criteria": validation_criteria,
        }
        write_json(args.output_dir / "progress.manifest.json", progress_manifest)
        if (
            args.early_stopping_patience
            and non_improving_epochs >= args.early_stopping_patience
        ):
            stopped_early = True
            print(
                "early stopping after {} non-improving epochs".format(
                    non_improving_epochs
                )
            )
            break

    completed_epochs = epoch_metrics[-1]["epoch"]
    final_training_state = {
        "completed_epochs": completed_epochs,
        "target_epochs": args.epochs,
        "resumed_from_epoch": start_epoch if resume else None,
        "data_order_resume_policy": (
            "deterministic_reseed_seed_plus_completed_epochs"
            if resume
            else "continuous_generator_from_seed"
        ),
        "best_validation_loss": best_validation_loss,
        "non_improving_epochs": non_improving_epochs,
        "stopped_early": stopped_early,
    }
    checkpoint_dir, adapter_dir = save_checkpoint(
        policy,
        optimizer,
        args.output_dir / "checkpoint",
        lr_scheduler=lr_scheduler,
        training_state=final_training_state,
    )
    reload_batch = next(iter(validation_loader))
    checkpoint_reload = verify_checkpoint_reload(
        policy,
        adapter_dir,
        checkpoint_dir / "task_modules.pt",
        reload_batch,
        args.device,
    )
    elapsed = time.time() - started
    write_json(metrics_path, {"epochs": epoch_metrics})

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
        "slm_bbr_modern_backbone_extension": "slm_bbr_modern_backbone_extension",
        "under400m_quantum_gpt_follow_on": "under400m_quantum_gpt_follow_on",
    }
    status = (
        "exploratory_pipeline_check"
        if limited
        and args.run_purpose not in ("quantum_smoke", "head_ablation_smoke")
        else status_by_purpose[args.run_purpose]
    )
    if args.run_purpose.startswith("head_ablation") or args.run_purpose == "under400m_quantum_gpt_follow_on":
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
        "state_feature_dim": args.state_feature_dim,
        "batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "effective_sequences_per_optimizer_step": args.grad_accum_steps,
        "epochs": completed_epochs,
        "target_epochs": args.epochs,
        "epochs_executed_this_run": completed_epochs - start_epoch,
        "resumed_from_run": (
            resume["run_dir"].as_posix() if resume else None
        ),
        "resumed_from_epoch": start_epoch if resume else None,
        "resume_source_hardware": (
            resume["manifest"].get("hardware") if resume else None
        ),
        "resume_source_software_versions": (
            resume["manifest"].get("software_versions") if resume else None
        ),
        "resume_source_git_commit": (
            resume["manifest"].get("git_commit") if resume else None
        ),
        "resume_data_order_policy": (
            "deterministic_reseed_seed_plus_completed_epochs"
            if resume
            else None
        ),
        "max_train_steps": args.max_train_steps or None,
        "max_validation_steps": args.max_validation_steps or None,
        "optimizer": "AdamW",
        "learning_rate_scheduler": (
            "linear_warmup_then_constant" if args.warmup_steps else "constant"
        ),
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": 0.25,
        "trainability_diagnostics": args.trainability_diagnostics,
        "training_loss_weighting": args.loss_weighting,
        "validation_loss_weighting": LOSS_WEIGHTING_NONE,
        "training_sampling": training_sampling,
        "validation_sampling": "original_unmodified_distribution",
        "checkpoint_rule": "final completed epoch for {}".format(args.run_purpose),
        "early_stopping": {
            "patience": args.early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "stopped_early": stopped_early,
        },
        "validation_success_thresholds": {
            "overall_accuracy": args.success_overall_accuracy,
            "up_accuracy": args.success_up_accuracy,
            "macro_phase_accuracy": args.success_macro_phase_accuracy,
            "max_up_prediction_share": args.max_up_prediction_share,
        },
        "final_validation_criteria": epoch_metrics[-1]["validation_criteria"],
        "method_provenance": {
            "canonical_method": "Small Language Model-based Control for BBR over Low Earth Orbit Satellite Internet",
            "official_source_commit": "c0afba6521e62c09d4f558095fb83e577a1f7c80",
            "backbone_substitution": args.run_purpose in (
                "slm_bbr_modern_backbone_extension",
                "under400m_quantum_gpt_follow_on",
            ),
            "paper_reproduction_claim": False,
        },
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
        "metrics": metrics_path.relative_to(args.output_dir).as_posix(),
        "metrics_sha256": file_sha256(metrics_path),
        "checkpoint": checkpoint_dir.relative_to(args.output_dir).as_posix(),
        "epoch_checkpoint_root": "epoch_checkpoints",
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
    progress_path = args.output_dir / "progress.manifest.json"
    if progress_path.is_file():
        progress_manifest = json.loads(progress_path.read_text(encoding="utf-8"))
        progress_manifest.update(
            {
                "status": "training_completed",
                "completed_at": manifest["run_completed_at"],
                "checkpoint": manifest["checkpoint"],
                "checkpoint_reload": checkpoint_reload,
            }
        )
        write_json(progress_path, progress_manifest)
    print("LoRA {} PASS".format("pipeline check" if limited else args.run_purpose))
    print("Validation accuracy: {:.6f}".format(epoch_metrics[-1]["validation"]["accuracy"]))
    print(
        "Validation macro-phase accuracy: {:.6f}".format(
            epoch_metrics[-1]["validation"]["macro_phase_accuracy"]
        )
    )
    print(
        "Validation BW_UP accuracy: {:.6f}".format(
            epoch_metrics[-1]["validation"]["per_phase_accuracy"]["BW_UP"]
        )
    )
    print(
        "Declared validation criteria pass: {}".format(
            epoch_metrics[-1]["validation_criteria"]["all_declared_criteria_pass"]
        )
    )
    print("Checkpoint reload: {}".format(checkpoint_reload))
    print("Manifest: {}".format(manifest_path))


if __name__ == "__main__":
    main()
