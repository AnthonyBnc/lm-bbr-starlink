"""Validated family-specific LoRA construction for modern BBR backbones."""

from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from config import cfg


def attach_modern_lora(
    backbone,
    model_key,
    rank=None,
    alpha=None,
    dropout=None,
    gradient_checkpointing=True,
):
    if model_key not in cfg.modern_lora_registry:
        raise ValueError("No modern LoRA configuration for {}".format(model_key))
    family = cfg.modern_lora_registry[model_key]
    defaults = cfg.lora_defaults
    rank = defaults["construction_rank"] if rank is None else rank
    alpha = defaults["alpha"] if alpha is None else alpha
    dropout = defaults["dropout"] if dropout is None else dropout
    if rank < 1:
        raise ValueError("LoRA rank must be at least 1")

    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=family["target_modules"],
        lora_dropout=dropout,
        bias=defaults["bias"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(backbone, lora_config)
    adapter_modules = sorted(
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and len(module.lora_A) > 0
    )
    expected = family["expected_adapter_modules"]
    if len(adapter_modules) != expected:
        raise ValueError(
            "LoRA target mismatch for {}: expected {} modules, matched {}".format(
                model_key, expected, len(adapter_modules)
            )
        )
    return model, lora_config, adapter_modules


def load_modern_lora_checkpoint(
    backbone,
    model_key,
    adapter_dir,
    gradient_checkpointing=True,
    is_trainable=True,
):
    """Load a saved modern-model adapter for training or frozen evaluation."""
    if model_key not in cfg.modern_lora_registry:
        raise ValueError("No modern LoRA configuration for {}".format(model_key))
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()

    model = PeftModel.from_pretrained(
        backbone,
        adapter_dir,
        is_trainable=is_trainable,
    )
    adapter_modules = sorted(
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and len(module.lora_A) > 0
    )
    expected = cfg.modern_lora_registry[model_key]["expected_adapter_modules"]
    if len(adapter_modules) != expected:
        raise ValueError(
            "LoRA checkpoint target mismatch for {}: expected {} modules, matched {}".format(
                model_key, expected, len(adapter_modules)
            )
        )
    return model, model.peft_config[model.active_adapter], adapter_modules


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
