# steerbench small-model canonical sweeps — results

One-command-reproducible dose-response + layer-sensitivity for a **formality**
ControlVector on a cheap small model. This is the canonical reproducible run; the
existing 7B/8B/9B cross-model CSVs are real and untouched.

- **Hardware:** Modal serverless **T4** (16 GB). bf16 confirmed working on Turing.
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` (primary, 28 layers, hidden 1536),
  `Qwen/Qwen2.5-0.5B-Instruct` (secondary, 24 layers, hidden 896).
- **Seeds:** [0, 1, 2], mean ± std (population).
- **Vector:** repeng `ControlVector` (native training path, not reimplemented),
  69 contrastive persona pairs, one unit-norm direction per layer.
- **Metrics:** effect = lexical formality proxy; coherence = repetition rate
  (1−distinct-2) + unsteered perplexity. Capability side-effects (MMLU/GSM8K)
  are deliberately **not** folded into the dose curve.
- **GPU spend:** see the running tally at the bottom (< $15 budget).

## TL;DR

- The layer of maximum coherent effect on 1.5B is **layer 17 = frac 0.61**, which
  **matches the depth anchor** the 7B run and introspection-scaling use (0.61).
  It is a **back-half plateau** (frac ~0.61–0.93), not a sharp point — L17 is the
  argmax but L20/21/22/25 are statistically indistinguishable within seed noise.
- The transferable 7B dose (alpha 0.044) is **too weak on 1.5B to resolve a peak**
  (documented below-floor run). A **cliff-justified** dose (alpha 0.09, from the
  L17 dose-response usable band) resolves it. Honest: the alpha was committed from
  the cliff map *before* looking at which layer won — not tuned toward 0.61.
- Honest negative: early-mid layers (frac ~0.25–0.54) are a **dead-spot** — at the
  resolving dose, injection there *reduces* formality below baseline and *breaks
  coherence* (perplexity spikes to 6–875, seed variance ±1–2).

## 1. Layer sweep (each layer's own direction, fixed normalized alpha)

### 1a. Below the resolving floor — alpha 0.044 (the 7B transferable dose)

`layer_sweep_qwen1.5b_a044.{csv,png}`. Baseline formality 4.56. Every layer lands
4.3–4.9 within seed noise (±0.07–0.23); repetition 0.03–0.09 and unsteered ppl
2.7–4.5 everywhere → far below any coherence cliff, large headroom. No
layer-localized peak resolves; a weak broad back-half plateau (frac ~0.57–0.89)
sits above baseline with L17/0.61 *inside* it, not uniquely peaked. Kept as the
documented "below-floor" point. (Note: alpha 0.044 *did* resolve on 7B — the
smaller model needs a stronger normalized dose to separate layers.)

### 1b. Resolved — alpha 0.09 (cliff-justified, canonical)

`layer_sweep_qwen1.5b.{csv,png}`. Coherent-gated peak (rep < 0.15, ppl < 6):

| region | frac | formality (mean±std) | read |
|---|---|---|---|
| **peak** | **0.61 (L17)** | **4.97 ± 0.09** | argmax, rep 0.07, ppl 5.1 |
| plateau | 0.61–0.93 (L17–25) | 4.82–4.97 | overlapping seed range; L20/21/22/25 ≈ L17 |
| dead-spot | 0.25–0.54 (L7–15) | 3.2–4.2, several < baseline | disruptive; ppl 6→875, var ±1–2 |
| early | 0.04–0.21 (L1–6) | 4.5–4.8 | mild, near baseline |

**Peak plateau: frac 0.61–0.93, argmax frac 0.61 (layer 17).** This brackets and
hits the 0.61 anchor. The plateau width is the honest caveat — 3 seeds give a
coarse argmax, so "the max-effect layer is 0.61" is true as an argmax but the
effect is broad across the back half.

## 2. Dose-response at layer 17 (frac 0.61 anchor = the resolved peak)

`dose_response_qwen1.5b.{csv,png}`. resid_norm@L17 = 352.5. Coeff grid
[−60…130] → alpha −0.17…0.37.

| alpha | coeff | formality | repetition | ppl | note |
|---|---|---|---|---|---|
| −0.170 | −60 | 2.37 ± 0.36 | 0.505 | 9.6 | degenerate (casual side) |
| −0.113 | −40 | 2.82 ± 0.23 | 0.406 | 8.1 | degenerate |
| −0.071 | −25 | 3.05 ± 0.11 | 0.133 | 5.3 | casual, edge |
| −0.043 | −15 | 3.47 ± 0.13 | 0.082 | 3.5 | casual, coherent |
| 0.000 | 0 | 4.49 ± 0.07 | 0.070 | 2.6 | baseline |
| 0.023 | 8 | 4.81 ± 0.09 | 0.057 | 2.8 | formal, clean |
| 0.043 | 15 | 4.84 ± 0.03 | 0.055 | 2.9 | formal, clean |
| 0.071 | 25 | 4.90 ± 0.02 | 0.066 | 3.6 | formal, clean |
| **0.113** | **40** | **5.06 ± 0.12** | 0.100 | 7.1 | **effect peak, ppl rising** |
| 0.170 | 60 | 5.00 ± 0.01 | 0.335 | 15.1 | **cliff** |
| 0.255 | 90 | 5.97 ± 0.44 | 0.662 | 9.4 | past-cliff artifact (looping) |
| 0.369 | 130 | 4.70 ± 2.42 | 0.309 | 6.1 | chaotic, variance blows up |

- **Usable band:** alpha ≈ −0.07 → +0.11 (coeff −25 → +40). Formality moves
  monotonically 3.05 → 5.06 while rep < 0.15 and ppl stays single-digit.
- **Coherence cliff:** alpha ≈ 0.13–0.17. Past it, the formality proxy *inflates
  on degenerate repetition* (alpha 0.255: rep 0.66) — an effect-reversal artifact,
  not real steering — and seed variance explodes (alpha 0.369: ±2.42).
- The layer-sweep dose (alpha 0.09) sits just inside the top of this band — the
  strongest still-coherent dose, chosen to maximize layer separation.

## 3. The 0.61 synergy (used by introspection-scaling)

introspection-scaling injects/reads at **frac 0.61 of depth**. Independently, on a
different (much smaller) model, the coherent max-effect layer for formality
steering lands at **frac 0.61** (L17/28), the same anchor the 7B qwen sweep found
(L17/28 = 0.607). The dose-response is run at that same 0.61 layer. So the
small-model canonical run supplies both a per-layer sensitivity curve and a
dose-response *at the depth the scaling work assumes* — a cross-size cross-check
that 0.61 is a reasonable formality-injection depth, with the caveat that it is a
broad plateau, not a razor peak.

## 4. Qwen2.5-0.5B-Instruct (secondary)

_(pending — filled after the 0.5B dose + resolved sweep complete)_

## Reproduce

```bash
experiments/reproduce.sh          # trains vector + both sweeps + dose, prints spend
```
