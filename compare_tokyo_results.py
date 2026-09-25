"""Combine four measured Tokyo runs with four paper-published references."""

import argparse
import csv
import json
from pathlib import Path

from utils.checkpoint_freeze import write_json


NEW_MODELS = (
    "granite_4_0_350m",
    "pleias_rag_350m",
    "lfm2_5_350m",
    "gemma_3_270m",
)
PAPER_MODELS = ("gpt2", "t5", "gpt_neo", "smollm2")


def load_measured_results(paths):
    results = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    keys = [item.get("model_key") for item in results]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate measured model result supplied")
    if set(keys) != set(NEW_MODELS):
        raise ValueError(
            "Measured set must be exactly {}; got {}".format(NEW_MODELS, keys)
        )
    sample_hashes = {item["metrics"]["sample_ids_sha256"] for item in results}
    pool_hashes = {item["tokyo_pool_sha256"] for item in results}
    if len(sample_hashes) != 1 or len(pool_hashes) != 1:
        raise ValueError("Measured results do not use identical Tokyo records")
    if not all(item.get("inference_only") is True for item in results):
        raise ValueError("Every measured Tokyo result must be inference-only")
    return sorted(results, key=lambda item: NEW_MODELS.index(item["model_key"]))


def load_published_reference(path):
    reference = json.loads(Path(path).read_text(encoding="utf-8"))
    models = reference.get("models", [])
    keys = [item.get("model_key") for item in models]
    if set(keys) != set(PAPER_MODELS) or len(keys) != len(PAPER_MODELS):
        raise ValueError("Published reference must contain the four paper models")
    if reference.get("source_type") != "published_paper_reference":
        raise ValueError("Old-model values must be marked published_paper_reference")
    return reference, sorted(models, key=lambda item: PAPER_MODELS.index(item["model_key"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_manifests", nargs="+", type=Path)
    parser.add_argument(
        "--published-reference",
        type=Path,
        default=Path("docs/reference/slm_bbr_published_results.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    measured = load_measured_results(args.result_manifests)
    reference, paper_models = load_published_reference(args.published_reference)

    rows = []
    for result in measured:
        metrics = result["metrics"]
        rows.append(
            {
                "model_key": result["model_key"],
                "source_type": "measured_frozen_tokyo_run",
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "macro_phase_accuracy": metrics["macro_phase_accuracy"],
                "bw_down_accuracy": metrics["per_phase_accuracy"]["BW_DOWN"],
                "bw_cruise_accuracy": metrics["per_phase_accuracy"]["BW_CRUISE"],
                "bw_up_accuracy": metrics["per_phase_accuracy"]["BW_UP"],
                "mean_milliseconds_per_action": metrics["latency"][
                    "mean_milliseconds_per_action"
                ],
            }
        )
    for item in paper_models:
        rows.append(
            {
                "model_key": item["model_key"],
                "source_type": "published_paper_reference",
                "accuracy": None,
                "loss": None,
                "macro_phase_accuracy": None,
                "bw_down_accuracy": None,
                "bw_cruise_accuracy": None,
                "bw_up_accuracy": None,
                "mean_milliseconds_per_action": item["inference_latency_ms_per_action"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "tokyo_new_vs_published_paper.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args.output_dir / "tokyo_new_vs_published_paper.manifest.json",
        {
            "status": "new_models_measured_vs_old_models_published_reference",
            "held_out_location": "Tokyo",
            "measured_model_order": list(NEW_MODELS),
            "published_model_order": list(PAPER_MODELS),
            "shared_measured_sample_ids_sha256": measured[0]["metrics"][
                "sample_ids_sha256"
            ],
            "measured_tokyo_pool_sha256": measured[0]["tokyo_pool_sha256"],
            "published_reference_source": reference["source"],
            "rows": rows,
            "comparison_limit": (
                "The two groups have different backbones, epoch budgets, hardware, "
                "and result provenance. Do not report same-sample deltas or statistical "
                "significance between groups."
            ),
        },
    )
    print("Comparison CSV: {}".format(csv_path))


if __name__ == "__main__":
    main()
