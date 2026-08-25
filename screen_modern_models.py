import importlib.util
import json
from pathlib import Path

from config import cfg


SUPPORTED_MODEL_TYPES = {
    "qwen3_5": {
        "plm_type": "qwen3",
        "notes": "Current loader uses Qwen2Model; needs transformers support for Qwen3.5 conditional-generation checkpoints.",
    },
    "gemma3": {
        "plm_type": "gemma3",
        "notes": "Current loader maps to GemmaModel; verify installed transformers exposes a compatible text backbone.",
    },
    "llama": {
        "plm_type": "llama3",
        "notes": "Current loader path exists and hidden size metadata matches config.",
    },
}


def _has_module(name):
    return importlib.util.find_spec(name) is not None


def _read_model_metadata(model_path):
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    with config_path.open() as f:
        return json.load(f)


def main():
    print("Modern model screening")
    print(f"Local model root: {cfg.local_model_root}")
    print(
        "Dependencies:",
        {
            "torch": _has_module("torch"),
            "transformers": _has_module("transformers"),
            "peft": _has_module("peft"),
        },
    )
    print()

    for model_key, model_info in cfg.modern_model_registry.items():
        model_path = Path(cfg.get_registered_model_path(model_key))
        metadata = _read_model_metadata(model_path)

        print(f"[{model_key}]")
        print(f"  hf_id: {model_info['hf_id']}")
        print(f"  local_path: {model_path}")
        print(f"  release_date: {model_info['release_date']}")
        print(f"  configured_revision: {model_info.get('revision')}")
        print(f"  configured_loader_status: {model_info.get('loader_status')}")
        print(f"  preferred_dtype: {model_info.get('preferred_dtype')}")
        print(f"  exists: {model_path.exists()}")

        if metadata is None:
            print("  status: missing config.json")
            print()
            continue

        model_type = metadata.get("model_type")
        architectures = metadata.get("architectures", [])
        hidden_size = (
            metadata.get("hidden_size")
            or metadata.get("d_model")
            or metadata.get("text_config", {}).get("hidden_size")
        )
        num_layers = (
            metadata.get("num_hidden_layers")
            or metadata.get("num_layers")
            or metadata.get("text_config", {}).get("num_hidden_layers")
        )

        print(f"  model_type: {model_type}")
        print(f"  architectures: {architectures}")
        print(f"  hidden_size: {hidden_size}")
        print(f"  num_layers: {num_layers}")

        support = SUPPORTED_MODEL_TYPES.get(model_type)
        if support is None:
            loader_status = model_info.get("loader_status", "screen_only")
            print(f"  legacy_loader_status: {loader_status}")
            if loader_status == "generic_auto_model_smoke_passed":
                print("  notes: generic AutoModel BBR forward/backward/checkpoint smoke passed")
            else:
                print("  notes: no current legacy plm_type/model class mapping in repo")
            print()
            continue

        print(f"  legacy_loader_status: mapped to {support['plm_type']}")
        print(f"  notes: {support['notes']}")

        plm_type = support["plm_type"]
        embed_sizes = cfg.plm_embed_sizes.get(plm_type, {})
        layer_sizes = cfg.plm_layer_sizes.get(plm_type, {})
        print(f"  config_embed_size(base): {embed_sizes.get('base')}")
        print(f"  config_layer_size(base): {layer_sizes.get('base')}")

        if embed_sizes.get("base") != hidden_size:
            print("  warning: hidden size does not match current config")
        if layer_sizes.get("base") != num_layers:
            print("  warning: layer count does not match current config")
        print()


if __name__ == "__main__":
    main()
