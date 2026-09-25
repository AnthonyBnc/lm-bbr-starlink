# ADR-0012: Canonical Paper Naming and Reproduction Gate

## Status

Accepted and extended by user confirmation on 2026-09-01.

## Context

New research directions had accumulated internal labels that could be mistaken
for names used by the main SLM-BBR paper. The modern backbone comparison also
used four locally available checkpoints, creating a risk that a four-model run
would be described as equivalent to the paper's four-SLM comparison.

The paper evaluates GPT-2, T5, GPT-Neo, and SmolLM2. It reports cross-entropy
training for 150 epochs, mini-batches of 20 with gradient accumulation, and
gradient clipping. It does not fully specify exact model revisions, LoRA
rank/alpha/dropout/target modules, gradient-accumulation count, optimizer
schedule, or checkpoint-selection rule.

The local complete modern set is LFM2.5-2.6B, Llama-3.2-3B,
Qwen3.5-4B-Base, and gemma-3-4b-pt. Only GPT-2 from the paper set is visible in
the local Hugging Face cache. The paper authors' official GitHub source is
available as `upstream/main` at commit
`c0afba6521e62c09d4f558095fb83e577a1f7c80`, and the local `run_all.py` matches
its runner. That source includes training modules and an Experience Pool reader,
but not the generated Experience Pool, the raw-to-pool preprocessing script, or
the model weights. `tcp-cc-starlink-main/` is a separate raw-trace data source.

## Decision

1. Use the main paper's canonical scientific terms in new documentation,
   manifests, figures, tables, and experiment descriptions. Existing code
   aliases may remain only for backward compatibility.
2. Do not rename existing identifiers or introduce a replacement method name
   without explicit human approval and a compatibility plan.
3. Reserve the paper's four-SLM comparison for GPT-2, T5, GPT-Neo, and SmolLM2.
   A run with the four local modern checkpoints is a modern-backbone extension,
   regardless of whether it copies paper-reported hyperparameters.
4. Permit the claims `exact`, `same as the paper`, or `paper reproduction` only
   after every item in the reproduction gate in `docs/ARCHITECTURE.md` has been
   verified and written to a run manifest.
5. Use the verified `upstream/main` commit as the source-code reference, but do
   not launch it as exact reproduction while its 12-action constant, hard-coded
   data paths, unmasked training loss, checkpoint-selection defect, missing
   paper checkpoints, and missing Experience Pool remain unresolved.
6. Record whether each reused function is byte-identical to upstream, a
   paper-faithful correction of an upstream defect, or a project extension.
7. The user-approved experiment keeps the SLM-BBR method and substitutes exactly
   four local backbones: LFM2.5-2.6B, Llama-3.2-3B, Qwen3.5-4B-Base, and
   gemma-3-4b-pt. This is a backbone-only experiment variable; it does not permit
   model-specific data, labels, masks, losses, metrics, or Tokyo access.
8. The user owns training execution. Agents may prepare the reproducible command
   plan and audit results, but may not start or resume training without a new
   explicit instruction to execute it.

## Consequences

The completed modern four-model results remain valid only as exploratory
modern-backbone comparisons. They do not answer whether this repository exactly
reproduces the paper's model selection or training results.

Before an expensive paper-comparison launch, the repository needs the exact
four checkpoints and revisions, the paper preprocessing/Experience Pool,
a phase-safe shared loss path, a corrected and declared checkpoint rule, and a
complete manifest of every known and unknown training parameter. If an omitted
paper detail must be chosen, that choice must be recorded as an implementation
assumption and the run must be labelled partial rather than exact.
