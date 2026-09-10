"""One-time Tokyo evaluation for the frozen gpt_quantum checkpoint.

Reuses utils.tokyo_evaluation.evaluate_frozen_policy (head-type-agnostic) and
writes output in the same predictions.csv / result.manifest.json shape as
evaluate_tokyo_models.py, so it can sit alongside the four gpt_classical
results under the same tokyo_8model_comparison_v1/results/ tree.

Run: LM_BBR_LOCAL_MODEL_ROOT=/Users/baoan/models .venv/bin/python \
  evaluate_tokyo_gpt_quantum.py --execute-final-held-out-evaluation
"""

import argparse
import json
from pathlib import Path
import pickle

import torch
from peft import PeftModel

from config import cfg
from plm_special.backbones import load_local_backbone, resolve_local_revision
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from utils.bbr import ACTION_LEVELS
from utils.checkpoint_freeze import file_sha256, write_json
from utils.tokyo_evaluation import evaluate_frozen_policy
from evaluate_tokyo_models import write_records


def _hidden_size(model_config):
    value = getattr(model_config, "hidden_size", None)
    if value is None and hasattr(model_config, "text_config"):
        value = model_config.text_config.hidden_size
    if not value:
        raise ValueError("Cannot resolve backbone hidden size")
    return value


def load_policy_from_run(run_dir, device, local_model_root):
    manifest = json.loads((run_dir / "run.manifest.json").read_text(encoding="utf-8"))
    if manifest.get("tokyo_isolation") != "PASS":
        raise ValueError("Run does not pass Tokyo isolation: {}".format(run_dir))
    if manifest.get("checkpoint_reload") != "PASS":
        raise ValueError("Run does not pass checkpoint reload: {}".format(run_dir))
    model_key = manifest["model_key"]
    model_path = Path(local_model_root).expanduser() / cfg.get_registered_model(model_key)["local_dir"]
    if resolve_local_revision(model_path) != manifest["model_revision"]:
        raise ValueError("Local backbone revision changed for {}".format(model_key))
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
    max_ep_len = int(task_state["1.weight"].shape[0]) - 1
    head = manifest["head_config"]
    policy = OfflineRLPolicy(
        state_feature_dim=manifest["state_feature_dim"],
        action_levels=ACTION_LEVELS,
        state_encoder=EncoderNetwork(embed_dim=manifest["state_feature_dim"]).to(device),
        plm=backbone,
        plm_embed_size=_hidden_size(model_config),
        max_length=manifest["sequence_length"],
        max_ep_len=max_ep_len,
        device=device,
        device_out=device,
        head_type=manifest["head_type"],
        quantum_config={
            "n_qubits": head.get("n_qubits"),
            "depth": head.get("depth"),
            "ansatz": head.get("ansatz"),
            "input_layernorm": head.get("input_layernorm", False),
            "temperature": head.get("temperature", 1.0),
            "angle_scale": head.get("angle_scale", "pi"),
        },
    )
    policy.modules_except_plm.load_state_dict(task_state, strict=True)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    return policy, manifest


def checkpoint_files_sha256(checkpoint_dir):
    files = {
        "adapter/adapter_config.json": checkpoint_dir / "adapter" / "adapter_config.json",
        "adapter/adapter_model.safetensors": checkpoint_dir / "adapter" / "adapter_model.safetensors",
        "task_modules.pt": checkpoint_dir / "task_modules.pt",
        "training_state.json": checkpoint_dir / "training_state.json",
    }
    return {name: file_sha256(path) for name, path in files.items() if path.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/processed/lora_training/under400m_quantum_gpt_follow_on_v1/lfm2_5_350m_quantum"),
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path("data/processed/evaluation/tokyo_8model_comparison_v1/frozen_checkpoints.manifest.json"),
    )
    parser.add_argument(
        "--tokyo-pool",
        type=Path,
        default=Path("data/processed/evaluation/tokyo_8model_comparison_v1/tokyo.pkl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/evaluation/tokyo_8model_comparison_v1/results/lfm2_5_350m_quantum"),
    )
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument(
        "--local-model-root",
        type=Path,
        default=Path(cfg.local_model_root),
    )
    parser.add_argument("--latency-warmup-batches", type=int, default=5)
    parser.add_argument(
        "--execute-final-held-out-evaluation",
        action="store_true",
        help="Required because results must not be used for tuning or checkpoint selection.",
    )
    args = parser.parse_args()
    if not args.execute_final_held_out_evaluation:
        parser.error("Final Tokyo inference requires --execute-final-held-out-evaluation")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Refusing to overwrite a non-empty Tokyo result directory")

    freeze_manifest = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if freeze_manifest.get("tokyo_opened") is not True:
        raise ValueError("Tokyo pool has not passed the frozen protocol gate")
    if file_sha256(args.tokyo_pool) != freeze_manifest.get("tokyo_pool_sha256"):
        raise ValueError("Tokyo pool checksum does not match the frozen protocol")
    with args.tokyo_pool.open("rb") as stream:
        pool = pickle.load(stream)
    if pool.metadata.get("split_role") != "held_out_test":
        raise ValueError("Pool is not marked held_out_test")

    print("Evaluating gpt_quantum ({})".format(args.run_dir), flush=True)
    policy, run_manifest = load_policy_from_run(args.run_dir, args.device, args.local_model_root)
    sequence_length = run_manifest["sequence_length"]
    sample_step = run_manifest["sample_step"]
    dataset = ExperienceDataset(pool, gamma=1.0, scale=1000, max_length=sequence_length, sample_step=sample_step)
    metrics, records = evaluate_frozen_policy(
        policy, dataset, pool, args.device, latency_warmup_batches=args.latency_warmup_batches
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "predictions.csv"
    write_records(records_path, records)
    checkpoint_dir = args.run_dir / "checkpoint"
    result = {
        "status": "held_out_tokyo_evaluation_completed",
        "model_key": run_manifest["model_key"],
        "model_role": "gpt_quantum",
        "head_type": run_manifest["head_type"],
        "head_config": run_manifest["head_config"],
        "model_id": run_manifest["model_id"],
        "model_revision": run_manifest["model_revision"],
        "checkpoint_files_sha256": checkpoint_files_sha256(checkpoint_dir),
        "tokyo_pool_sha256": freeze_manifest["tokyo_pool_sha256"],
        "sequence_length": sequence_length,
        "sample_step": sample_step,
        "inference_only": True,
        "phase_mask_applied_before_loss_and_argmax": True,
        "metrics": metrics,
        "predictions": records_path.as_posix(),
    }
    write_json(args.output_dir / "result.manifest.json", result)
    print("Tokyo result: {}".format(args.output_dir / "result.manifest.json"))
    print()
    print("accuracy", metrics["accuracy"])
    print("macro_phase_accuracy", metrics["macro_phase_accuracy"])
    print("per_phase_accuracy", metrics["per_phase_accuracy"])
    print("mean_milliseconds_per_action", metrics["latency"]["mean_milliseconds_per_action"])


if __name__ == "__main__":
    main()
