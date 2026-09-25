"""Run the frozen modern-model suite on the one-time Tokyo held-out pool."""

import argparse
import csv
import gc
import json
from pathlib import Path
import pickle

import torch

from config import cfg
from plm_special.backbones import load_local_backbone, resolve_local_revision
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from utils.bbr import ACTION_LEVELS
from utils.checkpoint_freeze import file_sha256, verify_frozen_entry, write_json
from utils.tokyo_evaluation import evaluate_frozen_policy


def _hidden_size(model_config):
    value = getattr(model_config, "hidden_size", None)
    if value is None and hasattr(model_config, "text_config"):
        value = model_config.text_config.hidden_size
    if not value:
        raise ValueError("Cannot resolve backbone hidden size")
    return value


def load_frozen_modern_policy(entry, device, local_model_root):
    verify_frozen_entry(entry)
    run_manifest = json.loads(Path(entry["run_manifest"]).read_text(encoding="utf-8"))
    model_key = entry["model_key"]
    model_path = Path(local_model_root).expanduser() / cfg.get_registered_model(
        model_key
    )["local_dir"]
    if resolve_local_revision(model_path) != entry["model_revision"]:
        raise ValueError("Local backbone revision changed for {}".format(model_key))
    dtype = getattr(torch, run_manifest["dtype"])
    backbone, model_config = load_local_backbone(model_path, device=device, dtype=dtype)
    from plm_special.lora import load_modern_lora_checkpoint

    backbone, _, _ = load_modern_lora_checkpoint(
        backbone,
        model_key,
        Path(entry["checkpoint"]) / "adapter",
        gradient_checkpointing=False,
        is_trainable=False,
    )
    task_state = torch.load(
        Path(entry["checkpoint"]) / "task_modules.pt",
        map_location=device,
        weights_only=True,
    )
    max_ep_len = int(task_state["1.weight"].shape[0]) - 1
    policy = OfflineRLPolicy(
        state_feature_dim=run_manifest["state_feature_dim"],
        action_levels=ACTION_LEVELS,
        state_encoder=EncoderNetwork(
            embed_dim=run_manifest["state_feature_dim"]
        ).to(device),
        plm=backbone,
        plm_embed_size=_hidden_size(model_config),
        max_length=run_manifest["sequence_length"],
        max_ep_len=max_ep_len,
        device=device,
        device_out=device,
        head_type=run_manifest["head_type"],
        quantum_config=run_manifest.get("quantum_config"),
    )
    policy.modules_except_plm.load_state_dict(task_state, strict=True)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy, run_manifest


def write_records(path, records):
    if not records:
        raise ValueError("No Tokyo prediction records were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path(
            "data/processed/evaluation/tokyo_8model_comparison_v1/"
            "frozen_checkpoints.manifest.json"
        ),
    )
    parser.add_argument(
        "--tokyo-pool",
        type=Path,
        default=Path(
            "data/processed/evaluation/tokyo_8model_comparison_v1/tokyo.pkl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/evaluation/tokyo_8model_comparison_v1/results"
        ),
    )
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument(
        "--local-model-root",
        type=Path,
        default=Path(cfg.local_model_root),
        help="Directory containing the downloaded backbone folders.",
    )
    parser.add_argument("--model-key", action="append", dest="model_keys")
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

    sequence_lengths = set()
    sample_hash = None
    summaries = []
    selected_keys = set(args.model_keys or ())
    entries = freeze_manifest["modern_models"]
    if selected_keys:
        entries = [entry for entry in entries if entry["model_key"] in selected_keys]
        if {entry["model_key"] for entry in entries} != selected_keys:
            raise ValueError("Requested model key is not in the frozen suite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        model_key = entry["model_key"]
        print("Evaluating {}".format(model_key), flush=True)
        policy, run_manifest = load_frozen_modern_policy(
            entry, args.device, args.local_model_root
        )
        sequence_length = run_manifest["sequence_length"]
        sample_step = run_manifest["sample_step"]
        sequence_lengths.add((sequence_length, sample_step))
        dataset = ExperienceDataset(
            pool,
            gamma=1.0,
            scale=1000,
            max_length=sequence_length,
            sample_step=sample_step,
        )
        metrics, records = evaluate_frozen_policy(
            policy,
            dataset,
            pool,
            args.device,
            latency_warmup_batches=args.latency_warmup_batches,
        )
        if sample_hash is None:
            sample_hash = metrics["sample_ids_sha256"]
        elif metrics["sample_ids_sha256"] != sample_hash:
            raise RuntimeError("Models did not receive identical Tokyo sample IDs")
        result_dir = args.output_dir / model_key
        records_path = result_dir / "predictions.csv"
        write_records(records_path, records)
        result = {
            "status": "held_out_tokyo_evaluation_completed",
            "model_key": model_key,
            "model_id": entry["model_id"],
            "model_revision": entry["model_revision"],
            "checkpoint_files_sha256": entry["checkpoint_files_sha256"],
            "tokyo_pool_sha256": freeze_manifest["tokyo_pool_sha256"],
            "sequence_length": sequence_length,
            "sample_step": sample_step,
            "inference_only": True,
            "phase_mask_applied_before_loss_and_argmax": True,
            "metrics": metrics,
            "predictions": records_path.as_posix(),
        }
        write_json(result_dir / "result.manifest.json", result)
        summaries.append(
            {
                "model_key": model_key,
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "macro_phase_accuracy": metrics["macro_phase_accuracy"],
                "mean_milliseconds_per_action": metrics["latency"][
                    "mean_milliseconds_per_action"
                ],
                "sample_ids_sha256": metrics["sample_ids_sha256"],
            }
        )
        del policy
        gc.collect()
        if args.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(sequence_lengths) != 1:
        raise RuntimeError("Frozen models use different sequence protocols")
    write_json(
        args.output_dir / "comparison.manifest.json",
        {
            "status": "modern_tokyo_comparison_completed",
            "held_out_location": "Tokyo",
            "models": summaries,
            "shared_sample_ids_sha256": sample_hash,
            "paper_models": freeze_manifest["paper_models"],
            "paper_model_comparison_status": "published_reference_only",
        },
    )
    print("Tokyo comparison: {}".format(args.output_dir / "comparison.manifest.json"))


if __name__ == "__main__":
    main()
