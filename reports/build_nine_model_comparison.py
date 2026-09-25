"""Build a single JSON comparing all 9 models on trainable LoRA parameters,
inference latency, and VRAM usage:

- 5 "this work" models (repository-measured, Tokyo-evaluated where available):
  granite_4_0_350m, pleias_rag_350m, lfm2_5_350m, gemma_3_270m (gpt_classical
  role) and gpt_quantum (lfm2_5_350m + Qiskit VQC head).
- 4 "published reference" models from the source paper: GPT-2, T5, GPT-Neo,
  SmolLM2 (not retrained; numbers come only from what the paper publishes
  exactly -- Table II / Fig.8 / Fig.10).

VRAM for the 5 "this work" models is measured separately by
reports/measure_vram_usage.py (one real training step: forward + backward +
AdamW optimizer.step(), same LoRA rank/gradient-checkpointing/dtype as the
frozen runs) and read here from reports/assets/vram_measurements/*.json.
This reports MPS driver-allocated memory (Apple unified memory), the closest
available analogue to the paper's dedicated NVIDIA VRAM reading -- not
claimed to be numerically comparable across the two hardware platforms.

Run: .venv/bin/python reports/build_nine_model_comparison.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_REF = ROOT / "docs" / "reference" / "slm_bbr_published_results.json"
TOKYO_COMPARISON = ROOT / "data/processed/evaluation/tokyo_8model_comparison_v1/results/comparison.manifest.json"
GPT_QUANTUM_RESULT = ROOT / "data/processed/evaluation/tokyo_8model_comparison_v1/results/lfm2_5_350m_quantum/result.manifest.json"
OUTPUT = ROOT / "reports" / "assets" / "nine_model_comparison.json"
VRAM_MEASUREMENTS_DIR = ROOT / "reports" / "assets" / "vram_measurements"

CLASSICAL_RUN_MANIFESTS = {
    "granite_4_0_350m": ROOT
    / "data/processed/lora_training/under400_rank128_epochs16_seed100003_v1"
    / "granite_4_0_350m_rank128_epochs16_seed100003_train/run.manifest.json",
    "pleias_rag_350m": ROOT
    / "data/processed/lora_training/under400_rank128_epochs16_seed100003_v1"
    / "pleias_rag_350m_rank128_epochs16_seed100003_train/run.manifest.json",
    "lfm2_5_350m": ROOT
    / "data/processed/lora_training/under400_rank128_epochs16_seed100003_v1"
    / "lfm2_5_350m_rank128_epochs16_seed100003_train/run.manifest.json",
    "gemma_3_270m": ROOT
    / "data/processed/lora_training/gemma3_270m_rank128_epochs16_seed100003_v1"
    / "gemma_3_270m_rank128_epochs16_seed100003_train/run.manifest.json",
}
GPT_QUANTUM_RUN_MANIFEST = (
    ROOT / "data/processed/lora_training/under400m_quantum_gpt_follow_on_v1/lfm2_5_350m_quantum/run.manifest.json"
)

DISPLAY = {
    "granite_4_0_350m": "Granite-4.0-350M",
    "pleias_rag_350m": "Pleias-RAG-350M",
    "lfm2_5_350m": "LFM2.5-350M",
    "gemma_3_270m": "Gemma-3-270M",
    "gpt_quantum": "gpt_quantum (LFM2.5-350M+VQC)",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_vram_gb(model_key):
    path = VRAM_MEASUREMENTS_DIR / "{}.json".format(model_key)
    if not path.is_file():
        return None, "not measured/logged"
    measured = load_json(path)
    return round(measured["peak_driver_allocated_gb"], 2), (
        "measured: peak MPS driver_allocated_memory() over one real training "
        "step (forward+backward+optimizer.step()), rank {} LoRA, gradient "
        "checkpointing on, dtype {} -- see reports/measure_vram_usage.py".format(
            measured["rank"], measured["dtype"]
        )
    )


def this_work_entry(model_key, run_manifest_path, latency_ms):
    manifest = load_json(run_manifest_path)
    backbone_params = manifest["backbone_parameters"]
    lora_trainable = manifest["lora_trainable_parameters"]
    vram_gb, vram_note = load_vram_gb(model_key)
    return {
        "model_key": model_key,
        "display_name": DISPLAY[model_key],
        "role": manifest.get("model_role", model_key),
        "backbone_parameters": backbone_params,
        "lora_trainable_parameters": lora_trainable,
        "lora_trainable_parameter_percent": round(100.0 * lora_trainable / backbone_params, 2),
        "inference_latency_ms_per_action": latency_ms,
        "mean_vram_usage_gb": vram_gb,
        "vram_note": vram_note,
        "source": "measured (this repository); see run.manifest.json / result.manifest.json",
    }


def main():
    tokyo = load_json(TOKYO_COMPARISON)
    latency_by_key = {m["model_key"]: m["mean_milliseconds_per_action"] for m in tokyo["models"]}
    quantum_result = load_json(GPT_QUANTUM_RESULT)
    latency_by_key["gpt_quantum"] = quantum_result["metrics"]["latency"]["mean_milliseconds_per_action"]

    this_work = [
        this_work_entry(key, path, latency_by_key[key]) for key, path in CLASSICAL_RUN_MANIFESTS.items()
    ]
    this_work.append(this_work_entry("gpt_quantum", GPT_QUANTUM_RUN_MANIFEST, latency_by_key["gpt_quantum"]))

    published = load_json(PUBLISHED_REF)
    published_reference = []
    for m in published["models"]:
        published_reference.append(
            {
                "model_key": m["model_key"],
                "display_name": m["display_name"],
                "parameters": m["parameters"],
                "lora_trainable_parameter_percent": m["trainable_parameter_percent"],
                "inference_latency_ms_per_action": m["inference_latency_ms_per_action"],
                "mean_vram_usage_gb": m["mean_vram_usage_gb"],
                "source": "published_reference: {} / {} / {}".format(
                    "Table II", m["latency_source"], m["vram_source"]
                ),
            }
        )

    result = {
        "comparison": "9-model trainable-LoRA%% / latency / VRAM comparison",
        "note": (
            "'this_work' rows are repository-measured (Tokyo-evaluated where "
            "listed). 'published_reference' rows are the source paper's own "
            "GPT-2/T5/GPT-Neo/SmolLM2 -- not retrained, different hardware "
            "(2x NVIDIA RTX 6000 vs this repository's Apple MPS), different "
            "epoch budget (150 vs 16). Do not compute deltas or statistical "
            "tests across the two groups."
        ),
        "this_work": this_work,
        "published_reference": published_reference,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", OUTPUT)
    for row in this_work + published_reference:
        print(
            row["display_name"],
            "| trainable%=",
            row["lora_trainable_parameter_percent"],
            "| latency_ms=",
            round(row["inference_latency_ms_per_action"], 2),
            "| vram_gb=",
            row["mean_vram_usage_gb"],
        )


if __name__ == "__main__":
    main()
