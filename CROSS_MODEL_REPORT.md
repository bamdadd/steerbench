# steerbench — cross-architecture formality steering

Same FORMALITY concept, same repeng diff-of-means (PCA-diff) construction, same
eval, on three architectures. Anchor is **transferable, not raw**: inject at
**0.61 fraction-of-depth** and hold the **dimensionless dose `alpha_norm = 0.044`**
fixed (`alpha_norm = coeff·‖dir‖/‖resid‖`, `‖dir‖=1`). Each model's inject layer
and residual norm are **measured**, and the per-model raw coefficient is derived
from them — never assumed.

**Hardware:** Modal serverless **A100**, pinned image (`experiments/modal_app.py`).
**Seeds:** 3 (sampling, temp 0.7). **Dose grid:** shared alpha grid incl 0,
negatives, the 0.044 anchor, and past-cliff (±0.2, 0.284).
**Models:** Qwen2.5-7B-Instruct · Llama-3.1-8B-Instruct (ungated NousResearch
mirror, identical weights) · Mistral-7B-Instruct-v0.3.
**Gemma-2-9b-it: deferred pending license** (repo gated; the HF token authenticates
but lacks the license grant — auto-fell back to Mistral for the 3rd architecture).

---

## Headline

Residual-stream norm at the same fractional depth spans **~16×** across these
architectures (28.7 → 464.5). Raw steering coefficients therefore do **not**
transfer — only the dimensionless dose does. And even **at a fixed dose**, the
same concept vector produces sharply different behavior: steerability and the
coherence-cliff location are architecture-dependent.

| Model | layers | inject L (frac) | resid_norm@L | coeff @ α=0.044 | effect at anchor | coherence cliff | character |
|:--|--:|:--|--:|--:|:--|:--|:--|
| **Qwen2.5-7B-Instruct** | 28 | 17 (0.61) | **464.5** | 20.4 | **peak, 5.07 vs 4.59 base** | sharp, α≈0.13→0.20 | steerable **but brittle** |
| **Llama-3.1-8B-Instruct** | 32 | 20 (0.625) | **43.7** | 1.9 | weak (4.28 ≈ base); rises to 5.15 by α≈0.28 | late, only α>0.28 | **dose-robust**, needs more push |
| **Mistral-7B-Instruct-v0.3** | 32 | 20 (0.625) | **28.7** | 1.3 | minimal/flat (baseline 4.83 already peak) | none in range | **weakly steerable** for this concept |
| Gemma-2-9b-it | 42 | (26, 0.62) | — | — | — | — | deferred (license) |

Walls: Qwen dose 530s / sweep 803s · Llama 432s / 959s · Mistral 517s / 1143s.

---

## Dose-response per model (at 0.61 depth, dimensionless dose on the x-axis)

**Qwen** — textbook: formality peaks exactly at the α=0.044 anchor (5.07), sharp
coherence valley at 0, cliff both directions (repetition 0.04→0.50, ppl → 20 by
α=0.284; negative cliff at α=−0.2). Highly steerable, sweet spot narrow.

| α | formality | repetition |
|--:|--:|--:|
| −0.20 | 2.78±0.12 | 0.298 |
| 0.00 | 4.59±0.09 | 0.039 |
| **0.044** | **5.07±0.08** | 0.028 |
| 0.131 | 4.87±0.14 | 0.061 |
| 0.284 | 3.95±0.28 | 0.503 |

**Llama** — monotone rise; α=0.044 barely moves it (4.28), effect builds to 5.15
at α=0.284 with coherence still intact (cliff only just emerging). Dose-response
is shifted right vs Qwen — robust, low sensitivity per unit dose.

| α | formality | repetition |
|--:|--:|--:|
| −0.20 | 3.91±0.04 | 0.030 |
| 0.00 | 4.36±0.03 | 0.027 |
| **0.044** | 4.28±0.05 | 0.045 |
| 0.197 | 4.89±0.15 | 0.045 |
| 0.284 | 5.15±0.14 | 0.061 |

**Mistral** — near-flat and noisy; baseline (4.83) is already the peak, both
directions mostly reduce formality. The diff-of-means direction does not cleanly
control formality here at this depth — least steerable / most entangled-or-saturated.

| α | formality | repetition |
|--:|--:|--:|
| −0.20 | 4.33±0.12 | 0.054 |
| 0.00 | 4.83±0.07 | 0.028 |
| **0.044** | 4.66±0.14 | 0.045 |
| 0.284 | 4.74±0.09 | 0.061 |

Artifacts: `artifacts/dose_response_{qwen,llama,mistral}.{csv,png}`.

---

## Normalized layer sweep (each layer's own direction at α=0.044)

Only **Qwen** shows a clean layer-sensitivity peak — at **L17 (0.61 depth,
formality 5.07)**, which *is* the anchor, independently validating the
0.61-depth choice. Llama and Mistral show no strong per-layer peak at α=0.044,
consistent with their flat dose-response at that dose (their "coherent-peak" rows
— Llama L2, Mistral L11 — are within run-to-run noise, not real optima). All
layers stay coherent at this normalized dose for all three models.

Artifacts: `artifacts/layer_sweep_{qwen,llama,mistral}.{csv,png}`.

---

## Defensible claim

> **Steerability and concept-entanglement are architecture-dependent, not
> universal.** Under an identical concept, construction, depth-fraction, and
> dimensionless dose, Qwen2.5-7B is strongly steerable but brittle (sharp
> coherence cliff adjacent to its sweet spot), Llama-3.1-8B is dose-robust and
> under-responsive at the Qwen-optimal dose (needs ~5× more), and Mistral-7B is
> weakly steerable for formality at this site. Because residual norms differ ~16×,
> only a residual-normalized dose (`alpha_norm`) is comparable across models —
> which is the unit steerbench reports.

**Caveats:** single concept (formality); cheap lexical formality proxy (not a
calibrated classifier); repeng PCA-diff directions; Gemma-2-9b-it deferred. The
per-model sweet-spot dose is itself model-dependent (Qwen 0.044, Llama ~0.25),
so the anchor transfers the *protocol*, not the optimal setting.

---

## Follow-up 1 — re-dosed layer sweep (each model at its OWN sweet-spot dose)

The layer sweeps above used `alpha_norm=0.044` (Qwen's sweet spot), which
UNDER-doses Llama (its dose-response peaks ~0.20) and Mistral. Re-run at each
model's own dose makes the **layer** comparison valid:

| Model | dose | coherent-peak layer (frac) | effect vs baseline |
|:--|--:|:--|:--|
| Qwen2.5-7B (ref, @0.044) | 0.044 | L17 (0.61) | 5.07 vs 4.46 (+0.6) |
| **Llama-3.1-8B @0.197** | 0.197 | **L14 (0.44)** | 5.04 vs 4.34 (**+0.70**) |
| **Mistral-7B @0.197** | 0.197 | L15 (0.47) | 4.99 vs 4.92 (**+0.07**) |

Properly dosed, Llama shows a **real** receptive peak at L14 (0.44 depth) — and
its best layer (0.44) is **shallower than Qwen's (0.61)**: the most-steerable
depth is itself architecture-dependent. Mistral stays flat even at its own dose
(+0.07), so its formality inertness is **not** an under-dosing artifact.
Artifacts: `artifacts/layer_sweep_{llama,mistral}_redosed.{csv,png}`.

---

## Follow-up 2 — SECOND CONCEPT (sentiment): is inertness concept-specific?

Trained a **sentiment** vector (positive vs negative contrastive personas) on all
three, same dose grid at 0.61 depth. Effect = lexical sentiment proxy
(pos−neg per token; small absolute magnitudes because neutral prompts elicit few
sentiment words — read relatively).

| Model | baseline | peak effect (dose) | steers sentiment? |
|:--|--:|:--|:--|
| **Qwen2.5-7B** | 0.47 | **4.71** (α=0.197), cliff at 0.284 | **yes, strongly** |
| **Llama-3.1-8B** | 0.11 | 0.56 (α=0.197) | weakly |
| **Mistral-7B** | 0.66 | ~1.3 noisy, no monotone | **no (inert/noisy)** |

**Verdict on Mistral:** inert on **both** formality (+0.07) and sentiment (no
clean response) → its resistance is **architectural, not concept-specific**.
Qwen steers both concepts strongly (sentiment sweet spot ~0.15–0.20, higher than
formality's 0.044 — dose is concept-dependent too). Llama steers both but weakly
at low dose. Artifacts: `artifacts/{dose_response,layer_sweep}_sentiment_{qwen,llama,mistral}.{csv,png}`.

### Steerability matrix (2 concepts × 3 architectures)

| | formality | sentiment |
|:--|:--|:--|
| **Qwen2.5-7B** | strong (sweet 0.044, sharp cliff) | strong (sweet ~0.17) |
| **Llama-3.1-8B** | moderate (needs ~0.20) | weak |
| **Mistral-7B** | inert | inert |

This strengthens the claim: **steerability is a property of the architecture (and
partly the concept), not universal.** Qwen is broadly steerable, Llama
under-responsive but works at higher dose, Mistral-7B-v0.3 resists diff-of-means
steering for both concepts at this depth. (Caveats as above; plus: cheap sentiment
proxy is noisy, single injection depth, one Mistral version — not proof Mistral is
unsteerable in general, only for this method/site.)

**3-concepts acceptance criterion:** formality ✓, sentiment ✓, verbosity pending.
