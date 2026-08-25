"""Run one exploratory BBR forward/backward optimizer step on a local backbone."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import pickle

import torch
import transformers
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import cfg
from plm_special.backbones import (
    freeze_backbone,
    load_local_backbone,
    preferred_smoke_dtype,
    resolve_local_revision,
)
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from plm_special.utils.utils import process_bbr_batch
from utils.bbr import ACTION_LEVELS, BW_CRUISE, mask_action_logits


def _select_event_window(dataset):
    for index in range(len(dataset)):
        if any(phase != BW_CRUISE for phase in dataset[index][-1]):
            return index
    raise ValueError("No UP or DOWN event appears in a complete sequence window")


def _mask_sequence_logits(logits, phases):
    masked = [
        mask_action_logits(logits[:, index, :], phase)
        for index, phase in enumerate(phases)
    ]
    return torch.stack(masked, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="lfm2_5_2_6b", choices=tuple(cfg.modern_model_registry))
    parser.add_argument("--exp-pool-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--state-feature-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=100003)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    with Path(args.exp_pool_path).open("rb") as stream:
        pool = pickle.load(stream)
    dataset = ExperienceDataset(
        pool,
        gamma=1.0,
        scale=1000,
        max_length=args.sequence_length,
        sample_step=1,
    )
    window_index = _select_event_window(dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=[window_index],
        pin_memory=False,
    )
    batch = next(iter(loader))
    states, actions, returns, timesteps, labels, phases = process_bbr_batch(
        batch, device=args.device
    )

    model_info = cfg.get_registered_model(args.model_key)
    model_path = cfg.get_registered_model_path(args.model_key)
    dtype = (
        preferred_smoke_dtype(args.device)
        if args.dtype == "auto"
        else getattr(torch, args.dtype)
    )
    backbone, model_config = load_local_backbone(model_path, device=args.device, dtype=dtype)
    freeze_backbone(backbone)
    hidden_size = getattr(model_config, "hidden_size", None)
    if hidden_size is None and hasattr(model_config, "text_config"):
        hidden_size = model_config.text_config.hidden_size
    if not hidden_size:
        raise ValueError("Cannot resolve backbone hidden size")

    state_encoder = EncoderNetwork(embed_dim=args.state_feature_dim).to(args.device)
    policy = OfflineRLPolicy(
        state_feature_dim=args.state_feature_dim,
        action_levels=ACTION_LEVELS,
        state_encoder=state_encoder,
        plm=backbone,
        plm_embed_size=hidden_size,
        max_length=args.sequence_length,
        max_ep_len=max(pool.metadata["source_files"][0]["intervals"], 512),
        device=args.device,
        device_out=args.device,
    )
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    logits = policy(states, actions, returns, timesteps)
    masked_logits = _mask_sequence_logits(logits, phases)
    loss = CrossEntropyLoss()(masked_logits.permute(0, 2, 1), labels.reshape(1, -1))
    loss.backward()
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.detach().float().pow(2).sum()
            for parameter in trainable
            if parameter.grad is not None
        )
    )
    if not torch.isfinite(loss).item():
        raise RuntimeError("Smoke loss is not finite: {}".format(float(loss.detach().cpu())))
    if not torch.isfinite(gradient_norm).item():
        raise RuntimeError(
            "Smoke gradient norm is not finite: {}".format(float(gradient_norm.cpu()))
        )
    optimizer.step()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_state = {
        key: value.detach().cpu()
        for key, value in policy.state_dict().items()
        if not key.startswith("plm.")
    }
    torch.save(
        {
            "model_key": args.model_key,
            "hf_id": model_info["hf_id"],
            "dataset_version": pool.metadata.get("dataset_version"),
            "split_version": pool.metadata.get("split_version"),
            "seed": args.seed,
            "sequence_length": args.sequence_length,
            "phases": phases,
            "labels": labels.detach().cpu(),
            "task_state": task_state,
        },
        output_path,
    )
    restored = torch.load(output_path, map_location="cpu", weights_only=True)
    if set(restored["task_state"]) != set(task_state):
        raise RuntimeError("Smoke checkpoint task-state keys changed during reload")

    revision = resolve_local_revision(model_path)
    expected_revision = model_info.get("revision")
    if expected_revision and revision != expected_revision:
        raise RuntimeError(
            "Local model revision {} does not match configured revision {}".format(
                revision, expected_revision
            )
        )
    manifest = {
        "status": "exploratory_compatibility_smoke",
        "model_key": args.model_key,
        "hf_id": model_info["hf_id"],
        "revision": revision,
        "dataset_version": pool.metadata.get("dataset_version"),
        "split_version": pool.metadata.get("split_version"),
        "held_out_location": pool.metadata.get("held_out_location"),
        "experience_pool": Path(args.exp_pool_path).as_posix(),
        "experience_pool_sha256": sha256(Path(args.exp_pool_path).read_bytes()).hexdigest(),
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "phases": list(phases),
        "labels": labels.detach().cpu().tolist(),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "gradient_norm": float(gradient_norm.cpu()),
        "trainable_task_parameters": sum(p.numel() for p in trainable),
        "backbone_frozen": True,
        "optimizer_steps": 1,
        "device": args.device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "checkpoint": output_path.as_posix(),
        "checkpoint_sha256": sha256(output_path.read_bytes()).hexdigest(),
        "checkpoint_reload": "PASS",
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Exploratory smoke PASS")
    print("Model: {} ({})".format(args.model_key, model_info["hf_id"]))
    print("Device/dtype: {} / {}".format(args.device, dtype))
    print("Sequence phases: {}".format(phases))
    print("Logits shape: {}".format(tuple(logits.shape)))
    print("Loss: {:.8f}".format(float(loss.detach().cpu())))
    print("Gradient norm: {:.8f}".format(float(gradient_norm.cpu())))
    print("Trainable task parameters: {}".format(sum(p.numel() for p in trainable)))
    print("Checkpoint reload: PASS ({})".format(output_path))
    print("Manifest: {}".format(manifest_path))


if __name__ == "__main__":
    main()
