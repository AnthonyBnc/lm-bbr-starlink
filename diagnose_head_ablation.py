"""Diagnose a completed Qwen head-ablation pilot.

The input is the audited summary produced by ``summarize_head_ablation.py``.
This script does not read Tokyo data and does not recompute labels; it only
explains the already-recorded validation metrics and action distributions.
"""

import argparse
import json
from pathlib import Path

from utils.bbr import BBR_PHASES, PACING_GAINS, PHASE_ACTION_INDICES


DEFAULT_RUN_ROOT = Path(
    "data/processed/lora_training/qwen_head_ablation_pilot_epoch1_qiskit"
)


def _int_distribution(distribution):
    return {int(key): int(value) for key, value in distribution.items()}


def _phase_total(distribution, phase):
    return sum(distribution.get(index, 0) for index in PHASE_ACTION_INDICES[phase])


def _top_actions(distribution, phase, limit=3):
    items = [
        {
            "action_index": index,
            "gain": PACING_GAINS[index],
            "count": distribution.get(index, 0),
        }
        for index in PHASE_ACTION_INDICES[phase]
    ]
    items.sort(key=lambda item: (-item["count"], item["action_index"]))
    return items[:limit]


def _format_percent(value):
    if value is None:
        return "n/a"
    return "{:.2f}%".format(100.0 * value)


def _format_seconds(seconds):
    return "{:.2f}h".format(float(seconds) / 3600.0)


def load_summary(path):
    with path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    if "heads" not in summary:
        raise ValueError("Expected a head-ablation summary with a 'heads' field")
    return summary


def diagnose_head(head):
    label_distribution = _int_distribution(head["validation_label_distribution"])
    prediction_distribution = _int_distribution(
        head["validation_prediction_distribution"]
    )
    phase_details = {}
    for phase in BBR_PHASES:
        label_total = _phase_total(label_distribution, phase)
        prediction_total = _phase_total(prediction_distribution, phase)
        dominant_predictions = _top_actions(prediction_distribution, phase)
        top_prediction = dominant_predictions[0] if dominant_predictions else None
        collapse_ratio = None
        if label_total and top_prediction:
            collapse_ratio = top_prediction["count"] / label_total
        phase_details[phase] = {
            "label_total": label_total,
            "prediction_total": prediction_total,
            "accuracy": head["validation_per_phase_accuracy"].get(phase),
            "dominant_labels": _top_actions(label_distribution, phase),
            "dominant_predictions": dominant_predictions,
            "dominant_prediction_share_of_phase_labels": collapse_ratio,
        }
    return {
        "head_type": head["head_type"],
        "model_role": head["model_role"],
        "validation_accuracy": head["validation_accuracy"],
        "validation_loss": head["validation_loss"],
        "validation_macro_phase_accuracy": head["validation_macro_phase_accuracy"],
        "wall_clock_seconds": head["wall_clock_seconds"],
        "checkpoint_reload": head["checkpoint_reload"],
        "tokyo_isolation": head["tokyo_isolation"],
        "phase_details": phase_details,
    }


def build_diagnostic(summary):
    heads = [diagnose_head(head) for head in summary["heads"]]
    by_type = {head["head_type"]: head for head in heads}
    classical = by_type.get("classical")
    quantum = by_type.get("quantum")
    findings = []

    best_macro = max(heads, key=lambda item: item["validation_macro_phase_accuracy"])
    best_loss = min(heads, key=lambda item: item["validation_loss"])
    findings.append(
        "Best validation macro-phase accuracy is {} ({:.6f}).".format(
            best_macro["head_type"], best_macro["validation_macro_phase_accuracy"]
        )
    )
    findings.append(
        "Lowest validation loss is {} ({:.6f}).".format(
            best_loss["head_type"], best_loss["validation_loss"]
        )
    )

    if classical and quantum:
        macro_gap = (
            quantum["validation_macro_phase_accuracy"]
            - classical["validation_macro_phase_accuracy"]
        )
        accuracy_gap = quantum["validation_accuracy"] - classical["validation_accuracy"]
        findings.append(
            "Quantum trails the same-backbone classical head by {:.6f} overall accuracy and {:.6f} macro-phase accuracy.".format(
                abs(accuracy_gap), abs(macro_gap)
            )
        )
        quantum_down = quantum["phase_details"]["BW_DOWN"]
        if quantum_down["accuracy"] == 0.0:
            top_down = quantum_down["dominant_predictions"][0]
            findings.append(
                "Quantum predicts a valid DOWN action but collapses to action {} (gain {:.2f}) while DOWN labels are dominated by action 0 (gain 0.90).".format(
                    top_down["action_index"], top_down["gain"]
                )
            )
        quantum_up = quantum["phase_details"]["BW_UP"]
        top_up = quantum_up["dominant_predictions"][0]
        if quantum_up["label_total"] and top_up["count"] == quantum_up["label_total"]:
            findings.append(
                "Quantum collapses all UP predictions to action {} (gain {:.2f}), so UP accuracy is limited to that action's label frequency.".format(
                    top_up["action_index"], top_up["gain"]
                )
            )

    recommendations = [
        "Do not run larger or multi-seed Quantum-GPT training yet.",
        "First run one controlled quantum-capacity sensitivity smoke/pilot, such as 8 qubits with the same Qwen parent and shared split.",
        "If changing loss weighting for the DOWN/UP imbalance, apply the same weighting to classical, classical_twin, and quantum heads in a named ablation.",
        "Keep Tokyo unused until the architecture and selection policy are frozen.",
    ]

    return {
        "status": "phase4_qiskit_head_ablation_diagnostic",
        "reportable_result": False,
        "held_out_location": summary["held_out_location"],
        "model_key": summary["model_key"],
        "model_id": summary["model_id"],
        "model_revision": summary["model_revision"],
        "seed": summary["seed"],
        "findings": findings,
        "recommendations": recommendations,
        "heads": heads,
    }


def build_markdown(diagnostic):
    lines = [
        "# Phase 4 Qiskit Head-Ablation Diagnostic",
        "",
        "This is a development diagnostic, not a reportable result. It reads the audited Qwen head-ablation summary only; Tokyo remains held out.",
        "",
        "## Run Scope",
        "",
        "- Model: `{}`".format(diagnostic["model_id"]),
        "- Revision: `{}`".format(diagnostic["model_revision"]),
        "- Seed: `{}`".format(diagnostic["seed"]),
        "- Held-out location: `{}`".format(diagnostic["held_out_location"]),
        "- Reportable result: `{}`".format(str(diagnostic["reportable_result"]).lower()),
        "",
        "## Head Results",
        "",
        "| Head | Validation accuracy | Macro-phase accuracy | Loss | DOWN | CRUISE | UP | Runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for head in diagnostic["heads"]:
        phase = head["phase_details"]
        lines.append(
            "| `{}` | {} | {} | {:.6f} | {} | {} | {} | {} |".format(
                head["head_type"],
                _format_percent(head["validation_accuracy"]),
                _format_percent(head["validation_macro_phase_accuracy"]),
                head["validation_loss"],
                _format_percent(phase["BW_DOWN"]["accuracy"]),
                _format_percent(phase["BW_CRUISE"]["accuracy"]),
                _format_percent(phase["BW_UP"]["accuracy"]),
                _format_seconds(head["wall_clock_seconds"]),
            )
        )
    lines.extend(["", "## Key Findings", ""])
    lines.extend("- {}".format(item) for item in diagnostic["findings"])
    lines.extend(["", "## Action-Distribution Clues", ""])
    for head in diagnostic["heads"]:
        lines.append("### `{}`".format(head["head_type"]))
        lines.append("")
        for phase_name in BBR_PHASES:
            detail = head["phase_details"][phase_name]
            pred = detail["dominant_predictions"][0]
            label = detail["dominant_labels"][0]
            lines.append(
                "- `{}`: labels mostly action {} / gain {:.2f} (`{}` samples); predictions mostly action {} / gain {:.2f} (`{}` samples); accuracy {}.".format(
                    phase_name,
                    label["action_index"],
                    label["gain"],
                    label["count"],
                    pred["action_index"],
                    pred["gain"],
                    pred["count"],
                    _format_percent(detail["accuracy"]),
                )
            )
        lines.append("")
    lines.extend(["## Recommended Next Step", ""])
    lines.extend("- {}".format(item) for item in diagnostic["recommendations"])
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    summary_path = args.summary or args.run_root / "head_ablation.summary.json"
    markdown_output = (
        args.markdown_output
        or args.run_root / "phase4_qiskit_head_ablation_diagnostic.md"
    )
    json_output = args.json_output or args.run_root / "head_ablation.diagnostic.json"

    diagnostic = build_diagnostic(load_summary(summary_path))
    markdown = build_markdown(diagnostic)
    markdown_output.write_text(markdown, encoding="utf-8")
    json_output.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(markdown)
    print("Markdown diagnostic: {}".format(markdown_output))
    print("JSON diagnostic: {}".format(json_output))


if __name__ == "__main__":
    main()
