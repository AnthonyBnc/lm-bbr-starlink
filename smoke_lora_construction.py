"""Construct and save a validated exploratory LoRA adapter for one local model."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform

import torch
import transformers

from config import cfg
from plm_special.backbones import load_local_backbone, resolve_local_revision
from plm_special.lora import attach_modern_lora, parameter_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=tuple(cfg.modern_model_registry))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--rank", type=int, default=cfg.lora_defaults["construction_rank"])
    args = parser.parse_args()

    model_info = cfg.get_registered_model(args.model_key)
    model_path = cfg.get_registered_model_path(args.model_key)
    dtype_name = model_info["preferred_dtype"]
    dtype = getattr(torch, dtype_name)
    backbone, _ = load_local_backbone(model_path, device=args.device, dtype=dtype)
    model, lora_config, adapter_modules = attach_modern_lora(
        backbone,
        args.model_key,
        rank=args.rank,
        gradient_checkpointing=True,
    )
    total_parameters, trainable_parameters = parameter_counts(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    adapter_config_path = output_dir / "adapter_config.json"
    adapter_weights_path = output_dir / "adapter_model.safetensors"
    if not adapter_config_path.is_file() or not adapter_weights_path.is_file():
        raise RuntimeError("PEFT did not write the expected adapter files")
    saved_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if saved_config["r"] != args.rank:
        raise RuntimeError("Saved LoRA rank does not match the requested rank")

    revision = resolve_local_revision(model_path)
    if revision != model_info["revision"]:
        raise RuntimeError("Local revision does not match the configured revision")
    manifest = {
        "status": "exploratory_lora_construction_smoke",
        "training_started": False,
        "model_key": args.model_key,
        "hf_id": model_info["hf_id"],
        "revision": revision,
        "device": args.device,
        "dtype": dtype_name,
        "lora_rank": args.rank,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": cfg.modern_lora_registry[args.model_key]["target_modules"],
        "adapter_module_count": len(adapter_modules),
        "adapter_modules": adapter_modules,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percent": 100.0 * trainable_parameters / total_parameters,
        "adapter_config": adapter_config_path.as_posix(),
        "adapter_config_sha256": sha256(adapter_config_path.read_bytes()).hexdigest(),
        "adapter_weights": adapter_weights_path.as_posix(),
        "adapter_weights_sha256": sha256(adapter_weights_path.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "held_out_location": "Tokyo",
        "data_accessed": False,
    }
    manifest_path = output_dir / "construction.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("LoRA construction PASS")
    print("Model: {}".format(args.model_key))
    print("Revision: {}".format(revision))
    print("Adapter modules: {}".format(len(adapter_modules)))
    print("Trainable parameters: {} ({:.4f}%)".format(
        trainable_parameters, manifest["trainable_percent"]
    ))
    print("Manifest: {}".format(manifest_path))


if __name__ == "__main__":
    main()
