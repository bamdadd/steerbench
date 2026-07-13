# steerbench M0 — dose-response + per-layer sweep

**Concept:** FORMALITY steering vector on **Qwen/Qwen2.5-7B-Instruct**.
**Hardware:** Modal serverless **A100** (pinned image, see `modal_app.py`).
**Method:** repeng `ControlVector` (PCA-diff over 69 formal↔casual contrastive
pairs), exported to repeng-native **gguf**, reloaded off disk through
`steerbench.load_vector` and injected with `ControlModel(model, [L])`.
Directions are unit-normalized by repeng (`||dir_L|| = 1.0` for every layer),
so **coefficient == raw injection magnitude**; `alpha_norm = coeff / resid_norm`
where `resid_norm` is the mean L2 norm of the layer-L residual stream on the
eval prompts.

Decoding: `do_sample=True, temp=0.7, top_p=0.9`, 96 new tokens, **3 seeds**
(0/1/2), 4 fixed neutral eval prompts. Metrics per point:
- **EFFECT** = lexical formality proxy (higher = more formal; cheap, inline).
- **COHERENCE** = n-gram repetition rate (1 − distinct-2) **and** perplexity of
  the generated text under the **unsteered** model (control reset before scoring).

Repeng-vector ingest is exercised on the real path: trained → gguf on a Modal
Volume → `import_gguf` in a separate function. No conversion step.

---

## STEP 3 — dose-response (fixed layer 14 / 28 = 0.50 depth)

`||dir||=1.0`, `resid_norm=457.5`, **wall 426 s**, 3 seeds.
Grid includes 0 (baseline), negatives (steer casual), and past-the-cliff
(coeff 90, 130).

| coeff (raw) | alpha_norm | formality (mean±std) | repetition | ppl(unsteered) |
|---:|---:|---:|---:|---:|
| −60 | −0.131 | 2.31±0.38 | 0.228 | 12.29 |
| −40 | −0.087 | 3.26±0.30 | 0.057 | 5.72 |
| −25 | −0.055 | 3.64±0.19 | 0.030 | 3.62 |
| −15 | −0.033 | 4.09±0.18 | 0.062 | 2.85 |
| −8  | −0.017 | 4.34±0.04 | 0.040 | 2.64 |
| **0** | 0.000 | 4.59±0.09 | 0.039 | 2.63 |
| +8  | +0.017 | 4.83±0.17 | 0.046 | 2.75 |
| +15 | +0.033 | 5.06±0.13 | 0.042 | 2.99 |
| +25 | +0.055 | 5.06±0.08 | 0.041 | 3.50 |
| +40 | +0.087 | 4.88±0.06 | 0.046 | 4.18 |
| +60 | +0.131 | 4.81±0.11 | 0.073 | 8.56 |
| +90 | +0.197 | 4.31±0.13 | 0.435 | 8.12 |
| +130| +0.284 | 3.85±0.13 | 0.663 | 5.77 |

- **Effect** rises monotone through 0, **peaks at coeff 15–25**
  (alpha_norm 0.033–0.055), then declines.
- **Cliff:** coherence collapses past coeff ≈ 60→90 — repetition 0.04→0.44→0.66,
  ppl spikes; past-cliff formality drops (degenerate text). Negative side has its
  own cliff at −60.
- **Sweet-spot coeff ≈ 20** (alpha_norm ≈ 0.044): max effect, coherence at
  baseline.

Artifacts: `artifacts/dose_response.{csv,png}`.

---

## STEP 4 — per-layer sweep (each layer's OWN direction)

Each layer L injects its own trained direction `vec.directions[L]` at layer L.
Run TWO ways; the fixed-alpha run is primary.

### 4a. PRIMARY — fixed alpha_norm = 0.044 (equal normalized strength per layer)

Per-layer `coeff_L = alpha · resid_norm_L` (dir norm = 1), so every layer is
compared at the SAME normalized injection strength. **wall 1162 s**, 3 seeds,
baseline formality = 4.46. This is the deconfounded layer-sensitivity curve.

| layer | frac depth | coeff_L (raw) | formality (mean±std) | repetition | ppl |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.04 | 0.9 | 4.53±0.08 | 0.047 | 2.59 |
| 2 | 0.07 | 1.2 | 4.61±0.10 | 0.040 | 2.61 |
| 3 | 0.11 | 13.5 | 4.77±0.13 | 0.063 | 2.61 |
| 5 | 0.18 | 17.3 | 4.93±0.12 | 0.028 | 2.68 |
| **13** | **0.46** | 20.0 | 5.00±0.09 | 0.039 | 3.41 |
| 16 | 0.57 | 20.3 | 5.01±0.08 | 0.041 | 3.01 |
| **17** | **0.61** | 20.4 | **5.09±0.10** | 0.036 | 3.39 |
| 18 | 0.64 | 20.7 | 4.04±0.10 | 0.055 | 2.92 |
| 21 | 0.75 | 22.7 | 4.98±0.10 | 0.035 | 2.88 |
| 25 | 0.89 | 29.2 | 4.66±0.15 | 0.042 | 2.63 |
| 27 | 0.96 | 12.2 | 4.67±0.12 | 0.027 | 2.75 |

(full 27 rows: `artifacts/layer_sweep.csv`)

**Deconfounded read.** At equal alpha every layer stays coherent (repetition
0.03–0.06, ppl 2.6–3.4 across ALL 27 layers). The effect is a broad plateau:
weakly receptive at the extremes (layers 1–2 barely move; layers 23–27 fade to
~4.67) and **peaking layers 13–21 (0.46–0.75), max at layer 17 (0.61 depth,
5.09)**. Layer 18 has a real, reproducible dip (both runs).
Per-layer residual norms rise with depth (coeff to reach alpha 0.044 goes
0.9 → ~29 → back to 12 at the last layers) — handed to orch-1 in the CSV.

### 4b. SECONDARY — fixed RAW coeff = 25 (a confound, kept as a cautionary result)

**wall 802 s.** Here layers 1–3 collapse into degenerate repetition
(rep 0.5–0.62, ppl up to 81) and the naive formality argmax falsely points at
layer 1. This is **not** a layer property — coeff 25 lands at alpha ≈ 0.8 at
layer 2 (well past the dose cliff of 0.197) but ≈ 0.055 at layer 14. **Lesson:
normalize the coefficient to each layer's residual norm before comparing layers,
or the early-layer cliff masquerades as a peak.** (`artifacts/layer_sweep_coeff.*`)

---

## STEP 5 — confirmatory dose-response on Qwen2.5-7B-**BASE** (for orch-1)

Same vector construction + same sweep, on the base (non-instruct) model.
Layer 14 (0.50 depth), **resid_norm@L14 = 469.0** (instruct: 457.5 — close),
`||dir||=1.0`, **wall 544 s**, 3 seeds.

| coeff | alpha_norm | formality (mean±std) | repetition | ppl |
|---:|---:|---:|---:|---:|
| −40 | −0.085 | 3.71±0.27 | 0.269 | 9.48 |
| −8  | −0.017 | 4.45±0.05 | 0.144 | 4.92 |
| **0** | 0.000 | 4.32±0.09 | 0.113 | 4.74 |
| +8  | +0.017 | 4.48±0.20 | 0.056 | 3.20 |
| +25 | +0.053 | 4.44±0.13 | 0.129 | 4.38 |
| +60 | +0.128 | 4.61±0.39 | 0.625 | 7.34 |
| +90 | +0.192 | 4.39±0.47 | 0.853 | 2.85 |

**Base vs instruct:** base residual norm is nearly identical (469 vs 457), so
`alpha_norm` transfers 1:1. But the base model's formality **effect is weak and
noisy** (flat ~4.3–4.6, large std) and its baseline coherence is worse
(repetition 0.11 at coeff 0 vs 0.04 instruct); the coherence cliff arrives sooner
and harder in both directions. Reference for orch-1: **base resid_norm@L14 ≈ 469,
usable alpha band ~0.017–0.05, degrades past ±0.09.** Artifacts:
`artifacts/dose_response_base.{csv,png}`.

---

## Sweet-spot summary (for downstream / orch-1)

| axis | Qwen2.5-7B-Instruct | Qwen2.5-7B-Base |
|:--|:--|:--|
| layers | 28 | 28 |
| hardware / seeds | A100 / 3 (sampling, temp 0.7) | A100 / 3 |
| resid_norm @ L14 (0.50 depth) | **457.5** | **469.0** |
| vector norm | 1.0 (repeng unit-normalizes) | 1.0 |
| **sweet-spot layer** | **17 · frac 0.61** (band 13–21, 0.46–0.75) | — (used L14) |
| **sweet-spot coeff (raw) @ L14** | **≈ 20** (15–25) | ≈ 8–25 (weak) |
| **sweet-spot alpha (normalized)** | **≈ 0.044** | ≈ 0.017–0.05 |
| effect cliff (positive) | coeff ≈ 60→90 (rep 0.04→0.44) | coeff ≈ 40→60 (earlier) |
| effect cliff (negative) | coeff ≈ −60 | coeff ≈ −40 (earlier) |
| effect strength | strong, clean | weak, noisy |
| walls | dose 426 s · layer-sweep 1162 s | dose 544 s |

---

## Side effects — capability vs steering dose (cross-model)

Held-out **MMLU** (40) + **GSM8K** (30) slice accuracy, formality vector at each
model's 0.61-depth layer. **Unsteered vs steered at the sweet spot (α_norm=0.044)
vs past-the-cliff (α_norm=0.284).** Hardware **A100**, **3 seeds** (sampling,
temp 0.7), mean±std; batched chat-templated generation, slices cached on the
Modal Volume. Full detail + analysis in `CROSS_MODEL_REPORT.md` (Follow-up 5).

| Model | benchmark | unsteered | sweet α=0.044 | past-cliff α=0.284 | wall |
|:--|:--|--:|--:|--:|--:|
| **Qwen2.5-7B** | MMLU | 0.567±0.012 | 0.608±0.012 | **0.000±0.000** | 311 s |
| | GSM8K | 0.544±0.016 | 0.467±0.054 | **0.000±0.000** | |
| **Llama-3.1-8B** | MMLU | 0.575±0.041 | 0.517±0.051 | 0.358±0.031 | 291 s |
| | GSM8K | 0.556±0.042 | 0.556±0.031 | 0.367±0.072 | |
| **Mistral-7B** | MMLU | 0.575±0.000 | 0.567±0.012 | 0.525±0.054 | 532 s |
| | GSM8K | 0.378±0.057 | 0.367±0.125 | 0.333±0.027 | |

**Read:** the sweet spot preserves capability on every model; past the cliff Qwen
**collapses to 0%** on both benchmarks (over-steering → gibberish), while Llama
only drops to ~0.36 (its coherence cliff is further out) and Mistral barely moves
(~0.53/0.33 — its formality vector is un-extractable noise, so injection doesn't
coherently perturb it). Side-effect degradation tracks each model's coherence
cliff. This is the 4th report-card panel; it renders end-to-end via
`report.py::build_report` (example: `artifacts/example_report_card/`).

Artifacts: `artifacts/side_effects_{qwen,llama,mistral}.csv` (report schema:
`benchmark,unsteered_acc,steered_acc`) + `_dose.csv` (per-dose mean±std).
