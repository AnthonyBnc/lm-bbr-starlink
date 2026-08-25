"""Shared local Hugging Face backbone loading for modern model smoke tests."""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel


def load_local_backbone(model_path, device="cpu", dtype=None):
    model_path = Path(model_path).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError("Local model config not found: {}".format(model_path))

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    load_kwargs = {
        "config": config,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    if device.startswith("cuda"):
        load_kwargs["device_map"] = {"": device}
    model = AutoModel.from_pretrained(model_path, **load_kwargs)
    if not device.startswith("cuda"):
        # Streaming shards directly to MPS can issue concurrent Metal copy/cast
        # kernels and crash the current macOS runtime. A sequential model move
        # is slower to load but deterministic and stable on unified memory.
        model = model.to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, config


def freeze_backbone(backbone):
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.eval()
    return backbone


def resolve_local_revision(model_path):
    metadata_path = (
        Path(model_path).expanduser().resolve()
        / ".cache"
        / "huggingface"
        / "download"
        / "config.json.metadata"
    )
    if not metadata_path.is_file():
        return None
    revision = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
    return revision or None


def preferred_smoke_dtype(device):
    if device == "mps":
        return torch.float16
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16
