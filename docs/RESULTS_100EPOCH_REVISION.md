# Results Section Revision — Epoch-100 Correction

**For:** *Quantum-Enhanced Language-Model-Based Control for BBR over Low-Earth-Orbit Satellite Internet* (An Nguyen)
**Date:** 2026-09-11
**Sources used:** `lm-bbr-starlink` repo — `docs/RESULTS_100EPOCH_SUMMARY.md`, `docs/reference/slm_bbr_published_results.json`, and the frozen manifests under `data/processed/evaluation/four_model_100epoch_v1/results/*/result.manifest.json` (Tokyo checksum `800f1a44…d012f27b` for all five 100-epoch models) and `data/processed/evaluation/tokyo_8model_comparison_v1/results/*/result.manifest.json` (the older 16-epoch runs).

---

## 0. What I found before rewriting anything

I compared every number in your current draft's Section IV against the manifests in the repo. **Table III and the `gpt_classical` row of Table IV in your current draft are not the 100-epoch results — they are the old 16-epoch run**, even though the text around them (and Table II) already correctly uses the 100-epoch numbers. Evidence:

| Backbone | Your Table III (Loss) | `tokyo_8model_comparison_v1` (16 ep) Loss | `four_model_100epoch_v1` (100 ep) Loss |
|---|---:|---:|---:|
| Granite-4.0-350M | 0.1292 | **0.1292** ✓ matches old | 0.4233 |
| Pleias-RAG-350M | 0.1244 | **0.1244** ✓ matches old | 0.3180 |
| LFM2.5-350M | 0.0995 | **0.0995** ✓ matches old | 0.5131 |
| Gemma-3-270M | 0.1143 | **0.1143** ✓ matches old | 0.4156 |

Every cell in your current Table III matches the *old* 16-epoch evaluation exactly, not the 100-epoch one. The same is true for your Table IV's `gpt_classical` row (0.9575 / 0.8205 / 0.4615 / 0.0995 / 3.597 ms — all old-run values). Your Table IV's `gpt_quantum` row, by contrast, **already is** the 100-epoch value — this isn't a mistake on that side, it's because `gpt_quantum`'s prediction is frozen from epoch 1 onward (see Section V-E of your draft / `RESULTS_100EPOCH_SUMMARY.md`), so its epoch-16 and epoch-100 Tokyo numbers are identical. That accidental agreement is probably why the mismatch on the classical side went unnoticed.

Net effect: your current Table IV is **not an epoch-matched comparison** — it compares a 100-epoch quantum checkpoint against a 16-epoch classical checkpoint. Below is the corrected version, with both heads read from their own frozen epoch-100 checkpoints (`four_model_100epoch_v1`), evaluated on the identical Tokyo pool.

I did not touch anything about `gpt_quantum`'s own analysis (the constant-predictor finding, gate-200 diagnostics, CRUISE-80 rebalancing) — that content is already consistent with the 100-epoch manifests and needs no numeric changes. I also did not find any place where "Shiva's" SLM-BBR paper's own published numbers (GPT-2/T5/GPT-Neo/SmolLM2, from `2607.07142v1`) needed correction — your existing contextual-reference framing and caveats (different hardware, different epoch budget, no cross-group statistical test) are accurate and unaffected by this fix.

---

## 1. Corrected Table III — held-out Tokyo, four under-400M backbones, epoch 100 (contextual)

**Table III. Held-out Tokyo evaluation of the four under-400M backbone candidates, 100-epoch training, final-epoch checkpoint (contextual; played no role in backbone selection).**

| Backbone | Accuracy | Macro | `BW_UP` | Loss |
|---|---:|---:|---:|---:|
| Granite-4.0-350M | **0.9607** | **0.8339** | **0.5016** | 0.4233 |
| Gemma-3-270M | 0.9602 | 0.8317 | 0.4952 | 0.4156 |
| LFM2.5-350M | 0.9593 | 0.8282 | 0.4847 | 0.5131 |
| Pleias-RAG-350M | 0.9588 | 0.8261 | 0.4784 | **0.3180** |

*(All five 100-epoch models predict 7 distinct action labels each on Tokyo, except `gpt_quantum` at 3 — see Table IV.)*

**Prose to replace your current paragraph under Table III:**

> Table III reports the same four candidates on held-out Tokyo, evaluated from each backbone's own frozen epoch-100 checkpoint. Unlike the development-split ranking in Table II — where LFM2.5-350M led on all three metrics — **LFM2.5-350M is the weakest of the four on Tokyo along every metric except is not the worst only nowhere: it trails Granite-4.0-350M and Gemma-3-270M on overall accuracy, macro-phase accuracy, and `BW_UP` accuracy, and records the highest (worst) loss of the four.** This is a materially different — and more consequential — finding than the "small, real gap" language used for the 16-epoch comparison: at epoch 100, the backbone selected by development-only validation is not merely edged out slightly on Tokyo, it is Tokyo's worst performer on three of four metrics. Section III-F's selection remains methodologically valid (Tokyo was correctly excluded from the decision), but this result should be stated plainly as a limitation of development-only backbone selection under extended training, not smoothed over as a small ranking reshuffle.

> The elevated loss values across all four models relative to the 16-epoch run (mean loss roughly 3–5× higher) are traced in Section IV-C to training-set overfitting past each model's peak-validation epoch (Table VI); they do not reflect a change in the phase-masking or loss implementation.

---

## 2. Corrected Table IV — primary comparison, both heads at their own frozen epoch-100 checkpoint

**Table IV. `gpt_classical` versus `gpt_quantum` on held-out Tokyo, same `LFM2.5-350M` backbone, both variants read from their epoch-100 checkpoint (`four_model_100epoch_v1`).**

| Metric | `gpt_classical` | `gpt_quantum` | Δ |
|---|---:|---:|---:|
| Accuracy | **0.9593** | 0.9550 | −0.0043 |
| Macro-phase acc. | **0.8282** | 0.8099 | −0.0183 |
| `BW_DOWN` acc. | 1.0000 | 1.0000 | 0 |
| `BW_CRUISE` acc. | 1.0000 | 1.0000 | 0 |
| `BW_UP` acc. | **0.4847** | 0.4298 | −0.0549 |
| Loss | 0.5131 | **0.1228** | +0.3903 |
| Latency (ms/action) | **1.1735** | 5.7395 | +4.5660 |

**Prose to replace your current paragraph under Table IV** (the sentence-level deltas that change from your draft: "3.17 points" → **5.49 points**, "1.61× faster / 3.597 vs. 5.794 ms" → **4.89× faster / 1.174 vs. 5.740 ms**; the loss sentence needs to flip direction and gain a caveat — see below):

> Table IV reports the controlled comparison defined in Section III-F, with both `gpt_classical` and `gpt_quantum` read from their own frozen epoch-100 checkpoint (`LFM2.5-350M` backbone), evaluated once on Tokyo under identical data, labels, phase masks, LoRA configuration, and seed. `gpt_classical` outperforms `gpt_quantum` on every accuracy metric in Table IV, with the largest gap on `BW_UP` accuracy (**5.49 points** — the only phase with more than one feasible action, and consequently the most informative phase for genuine action discrimination). Both variants remain at 100% on `BW_DOWN` and `BW_CRUISE`, consistent with the label degeneracy noted in Section III-F. The classical head is also **4.89× faster per action (1.174 ms vs. 5.740 ms)**, reflecting the added cost of statevector simulation in the quantum forward pass (Algorithm 2).
>
> **Loss requires a caveat that did not apply in the 16-epoch comparison.** At epoch 100, `gpt_classical`'s Tokyo loss (0.5131) is *higher* than `gpt_quantum`'s (0.1228) — the reverse of the 16-epoch result (0.0995 vs. 0.1373), where the classical head had the lower loss. This is not evidence that the quantum head is better calibrated. Section IV-C's training curves show `gpt_classical`'s validation loss rising well past its epoch-15 peak-accuracy point while training loss collapses toward zero (99.88% train accuracy at epoch 100) — a standard overfitting signature. `gpt_quantum`'s lower loss instead reflects the constant-predictor behaviour documented in Section V-B: it emits one label per phase with high confidence, which minimises cross-entropy on the two phases (`BW_CRUISE`, `BW_DOWN`) that dominate the sample count, independent of whether the constant label is informative. Cross-entropy loss should therefore not be read as a head-quality ranking in this table; accuracy, macro-phase accuracy, and prediction diversity (Section V-B) remain the primary evidence.

---

## 3. Corrected Table V — surrogate throughput/retransmissions, both heads at epoch 100

**Table V. Mean surrogate throughput and retransmissions by Tokyo stream group, `gpt_classical` vs. `gpt_quantum`, both at their frozen epoch-100 checkpoint (3,000 positions per group).**

| Stream group | Throughput (Mbps) | Retx (classical) | Retx (quantum) |
|---|---:|---:|---:|
| Downlink, sequential | 228.99 | 49.94 | 125.92 |
| Downlink, competitive | 9.41 | 3.29 | 4.98 |
| Uplink, sequential | 45.56 | 10.88 | 14.68 |
| Uplink, competitive | 10.75 | 1.83 | 2.48 |

**Prose update:** the qualitative claim in your draft ("retransmissions are consistently higher for `gpt_quantum` across all four stream groups, most sharply on downlink-sequential traffic") still holds and is now slightly *stronger*: the downlink-sequential gap widens from 2.07× (16-epoch classical) to **2.52×** (100-epoch classical: 125.92 vs. 49.94), because the longer-trained classical head reduces its own retransmissions on that stream group while `gpt_quantum`'s are unchanged (its predictions never move). Two of the four groups (uplink-sequential, uplink-competitive) show a small *regression* for the 100-epoch classical head relative to its 16-epoch self (10.88 vs. 10.25; 1.83 vs. 1.60) — worth a one-sentence mention rather than silently dropping it, since it's a real, small counter-trend inside an otherwise consistent result.

---

## 4. New Table VI (suggested addition) — peak vs. final epoch, explaining the loss jump

Since the loss reversal in Table IV needs support, consider adding this small table (data already computed in `docs/RESULTS_100EPOCH_SUMMARY.md`):

**Table VI. Development-split peak vs. final-epoch validation accuracy (checkpoint rule: final epoch, fixed before training).**

| Model | Best epoch | Best val. acc. | Epoch-100 val. acc. | Regression |
|---|---:|---:|---:|---:|
| Gemma-3-270M | 41 | 0.9583 | 0.9573 | −0.09 pt |
| Granite-4.0-350M | 11 | 0.9599 | 0.9557 | −0.42 pt |
| LFM2.5-350M | 15 | 0.9615 | 0.9578 | −0.37 pt |
| Pleias-RAG-350M | 13 | 0.9585 | 0.9567 | −0.18 pt |
| `gpt_quantum` | 1 | 0.9452 | 0.9452 | 0.00 pt |

For `gpt_classical` (`LFM2.5-350M`) specifically, its own training manifest (`data/processed/lora_training/four_model_100epoch_v1/lfm2_5_350m_classical/metrics.json`) shows the mechanism directly: training loss falls from 0.127 (epoch 1) to **0.0031** (epoch 100, train accuracy 99.89%), while development-split *validation* loss rises from 0.119 (epoch 1) to **0.460** (epoch 100) — a textbook overfitting curve that the accuracy-only regression numbers above understate. `gpt_quantum`'s validation loss instead falls monotonically (0.184 → 0.111) despite its accuracy never moving, because its predictions get more confident, not more correct.

**Figure note:** your draft references "Figs. ??–??" twice on page 6 (broken cross-references) for the training curves and VRAM/parameter-count bars. The regenerated, correctly epoch-100 figures already exist in the repo and should be cited there: `reports/figures/100epoch/100epoch_mean_loss_combined.pdf`, `100epoch_mean_accuracy_combined.pdf`, `100epoch_vram_combined.pdf`, `100epoch_trainable_parameters_combined.pdf`.

---

## 5. Consistency check-list for Section V (Discussion) — not rewritten here, but these will now be wrong if left as-is

1. **Numeric echoes of the old Table IV** — Section V-A's opening and any other prose that repeats "3.17 points," "1.61×," "3.597 vs. 5.794 ms," or the old loss direction (classical lower than quantum) need the same replacements as in §2 above.
2. **Section V-A, first paragraph — factual contradiction, unrelated to the epoch fix.** It currently reads: *"The modern-backbone comparison identifies Qwen3.5-4B-Base as the strongest GPT-style candidate... This selection... supports its use as the common parent model for the head comparison."* This contradicts Section III-F and Table II, where **`LFM2.5-350M`** — not Qwen3.5-4B-Base — is the selected parent backbone. Your Conclusion (Section VIII) correctly frames Qwen3.5-4B-Base as a *supplementary, earlier-development diagnostic on a different, larger backbone* (consistent with `docs/CONTEXT.md`'s project history). Section V-A's opening paragraph appears to be left over from that earlier phase and should be reworded to match the Conclusion's framing — e.g.: *"A supplementary diagnostic on a larger, discontinued backbone (Qwen3.5-4B-Base, Section VIII) motivated the LayerNorm/temperature-4 fix and the eight-qubit architecture used here, but is not part of the primary backbone comparison; `LFM2.5-350M` (Table II) is the parent backbone for both `gpt_classical` and `gpt_quantum`."* This is a pre-existing issue I found while cross-checking — flagging it since a reviewer would catch it immediately.
3. **Section V-B–V-D (quantum-collapse, ablation, CRUISE-80)** — these already match the 100-epoch manifests (`gpt_quantum`'s numbers didn't change), so no numeric edits needed there.

---

## 6. Source files (for your own re-verification)

- `docs/RESULTS_100EPOCH_SUMMARY.md` — narrative summary this revision is built from.
- `docs/reference/slm_bbr_published_results.json` — Shiva's (Pokhrel et al.) SLM-BBR published reference numbers, unaffected by this fix.
- `data/processed/evaluation/four_model_100epoch_v1/results/{granite_4_0_350m,gemma_3_270m,lfm2_5_350m,lfm2_5_350m_quantum,pleias_rag_350m}/result.manifest.json` — epoch-100 Tokyo ground truth used for every corrected number above (`tokyo_pool_sha256 = 800f1a444fcf1610720b83f6feadf0743fb2acaca81ed8d251f264fad012f27b`, identical across all five).
- `data/processed/evaluation/tokyo_8model_comparison_v1/results/*/result.manifest.json` — the older 16-epoch runs your current Table III/IV rows actually match.
- `data/processed/lora_training/four_model_100epoch_v1/lfm2_5_350m_classical/metrics.json` and `.../lfm2_5_350m_quantum/metrics.json` — per-epoch train/validation curves behind §4.
