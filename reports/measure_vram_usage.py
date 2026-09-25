"""Measure peak MPS (Apple GPU) driver-allocated memory for one training
step (forward + backward + AdamW optimizer.step()), matching the real
training configuration (LoRA rank 128, gradient checkpointing, float16,
sequence length 20) used for the frozen under-400M runs.

This is the closest available analogue to the source paper's "mean VRAM
usage" (measured on NVIDIA GPUs via nvidia-smi/dedicated VRAM). Apple
Silicon uses unified memory, not dedicated VRAM, so this is reported
explicitly as "MPS driver-allocated memory", not claimed to be identical in
meaning to the paper's number -- only the closest measurable analogue on
this hardware.

Run once per model (isolated subprocess per model to avoid memory carrying
over between runs):
  .venv/bin/python reports/measure_vram_usage.py --model-key lfm2_5_350m --head-type classical
  .venv/bin/python reports/measure_vram_usage.py --model-key lfm2_5_350m --head-type quantum \
      --n-qubits 8 --quantum-depth 1 --quantum-ansatz trainable_ry_layers \
      --head-input-layernorm --bottleneck-temperature 4.0
"""

import argparse
import json
from pathlib import Path
import pickle

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import cfg
from plm_special.backbones import load_local_backbone, resolve_local_revision
from plm_special.data.dataset import ExperienceDataset
from plm_special.lora import attach_modern_lora
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from plm_special.utils.utils import process_bbr_batch
from utils.bbr import ACTION_LEVELS
from utils.training_metrics import BBRMetricAccumulator

GB = 1024 ** 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=tuple(cfg.modern_model_registry))
    parser.add_argument("--head-type", choices=("classical", "quantum"), default="classical")
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--quantum-depth", type=int, default=1)
    parser.add_argument("--quantum-ansatz", default="trainable_ry_layers")
    parser.add_argument("--quantum-angle-scale", default="pi")
    parser.add_argument("--head-input-layernorm", action="store_true")
    parser.add_argument("--bottleneck-temperature", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--state-feature-dim", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument(
        "--split-dir", type=Path, default=Path("data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003")
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument("--local-model-root", type=Path, default=Path(cfg.local_model_root))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)

    model_path = args.local_model_root.expanduser() / cfg.get_registered_model(args.model_key)["local_dir"]
    revision = resolve_local_revision(model_path)

    with (args.split_dir / "train.pkl").open("rb") as stream:
        pool = pickle.load(stream)
    dataset = ExperienceDataset(pool, gamma=1.0, scale=1000, max_length=args.sequence_length, sample_step=20)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
    batch = next(iter(loader))
    states, actions, returns, timesteps, labels, phases = process_bbr_batch(batch, device=args.device)

    if args.device == "mps":
        torch.mps.empty_cache()
    baseline_bytes = torch.mps.driver_allocated_memory() if args.device == "mps" else 0

    backbone, model_config = load_local_backbone(model_path, device=args.device, dtype=dtype)
    backbone, lora_config, adapter_modules = attach_modern_lora(
        backbone, args.model_key, rank=args.rank, alpha=args.alpha, dropout=args.dropout, gradient_checkpointing=True
    )
    hidden_size = getattr(model_config, "hidden_size", None) or model_config.text_config.hidden_size

    quantum_config = None
    if args.head_type == "quantum":
        quantum_config = {
            "n_qubits": args.n_qubits,
            "depth": args.quantum_depth,
            "ansatz": args.quantum_ansatz,
            "input_layernorm": args.head_input_layernorm,
            "temperature": args.bottleneck_temperature,
            "angle_scale": args.quantum_angle_scale,
        }

    policy = OfflineRLPolicy(
        state_feature_dim=args.state_feature_dim,
        action_levels=ACTION_LEVELS,
        state_encoder=EncoderNetwork(embed_dim=args.state_feature_dim).to(args.device),
        plm=backbone,
        plm_embed_size=hidden_size,
        max_length=args.sequence_length,
        max_ep_len=max(pool.metadata["source_files"][0]["intervals"], 512),
        device=args.device,
        device_out=args.device,
        head_type=args.head_type,
        quantum_config=quantum_config,
    )

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=1e-4)
    optimizer.zero_grad(set_to_none=True)

    readings = []

    def sample(label):
        if args.device == "mps":
            torch.mps.synchronize()
            readings.append((label, torch.mps.driver_allocated_memory()))

    sample("after_model_build")
    logits = policy(states, actions, returns, timesteps)
    metrics = BBRMetricAccumulator()
    _, loss = metrics.update(logits, labels, phases)
    sample("after_forward")
    loss.backward()
    sample("after_backward")
    optimizer.step()
    sample("after_optimizer_step")

    peak_bytes = max(v for _, v in readings) if readings else 0
    result = {
        "model_key": args.model_key,
        "head_type": args.head_type,
        "model_revision": revision,
        "device": args.device,
        "dtype": args.dtype,
        "rank": args.rank,
        "sequence_length": args.sequence_length,
        "baseline_driver_allocated_bytes": baseline_bytes,
        "readings_bytes": readings,
        "peak_driver_allocated_bytes": peak_bytes,
        "peak_driver_allocated_gb": peak_bytes / GB,
        "loss": float(loss.detach().cpu()),
        "note": (
            "MPS driver_allocated_memory() -- unified memory, closest available "
            "analogue on Apple Silicon to the paper's dedicated-GPU VRAM reading, "
            "not claimed to be numerically comparable to NVIDIA VRAM."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("{} ({}): peak={:.3f} GB".format(args.model_key, args.head_type, result["peak_driver_allocated_gb"]))
    print("saved", args.output)


if __name__ == "__main__":
    main()
