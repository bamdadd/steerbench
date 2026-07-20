# steerbench — small-model canonical sweeps (RESULTS)

One-command-reproducible dose-response + layer-sensitivity for a **formality**
ControlVector on cheap small models. This is the **canonical reproducible** run
(`./experiments/reproduce.sh`); the existing 7B/8B/9B cross-model CSVs in
`results/` are real and untouched.

- **Hardware:** Modal serverless **T4** (16 GB). bf16 confirmed working on Turing.
- **Models:** `Qwen/Qwen2.5-1.5B-Instruct` (primary, 28 layers, hidden 1536),
  `Qwen/Qwen2.5-0.5B-Instruct` (secondary, 24 layers, hidden 896).
- **Seeds:** [0, 1, 2], mean ± std (population std).
- **Vector:** repeng `ControlVector` (native training path, not reimplemented),
  69 contrastive persona pairs, one unit-norm direction per layer.
- **Metrics:** effect = lexical formality proxy; coherence = repetition rate
  (1−distinct-2) + unsteered perplexity. Capability side-effects (MMLU/GSM8K) are
  deliberately **not** folded into the dose curve — separate work.

## Headline — 0.61 depth is a high-effect coherent injection layer at every scale

| model | params | depth | coherent-effect argmax | frac | high-effect plateau (frac) | 0.61 status |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.5B | 24 L | L23 (last) | 0.96 | ~0.58–0.96 | plateau start, not argmax |
| Qwen2.5-1.5B-Instruct | 1.5B | 28 L | L17 | **0.61** | ~0.61–0.93 | argmax |
| Qwen2.5-7B-Instruct (prior, real) | 7B | 28 L | L17 | **0.607** | back-half | argmax |

The honest claim: on all three sizes, **frac ≈ 0.61 is a high-effect, coherent
formality-injection depth** — it sits on the coherent back-half plateau at every
scale, and on 1.5B and 7B it is the **argmax**. This justifies the depth
**introspection-scaling** injects/reads at (0.61) — measured, not assumed, across
a ~14× size range.

The honest caveat (do **not** overclaim a universal 0.61 argmax): on **0.5B the
numeric argmax is the last layer (frac 0.96)**, and it is **real coherent formal
prose**, not an artifact — a 1-seed inspection at L23/coeff 33 produced fluent,
more-formal-than-baseline text (verified, not asserted). So 0.5B's effect keeps
climbing gently to the output end; 0.61 is the **start of its coherent plateau**,
not its peak. Reported as a plateau, not razor-precision — with 3 seeds the argmax
is coarse and the back half is a broad high band.

**Total GPU spend: ~$0.70** (measured dose+sweep wall 3,464 s across 5 sweeps +
~600 s container/cold-start overhead across 8 T4 runs; T4 @ ~$0.59/hr). Well
under the $15 budget.

| run | wall-clock |
|---|---|
| 1.5B layer sweep α=0.044 (below-floor) | 995 s |
| 1.5B layer sweep α=0.09 (resolved) | 954 s |
| 1.5B dose-response @ L17 | 483 s |
| 0.5B layer sweep α=0.124 (resolved) | 684 s |
| 0.5B dose-response @ L15 | 348 s |
| train (1.5B, 0.5B) + smoke | ~short, container-dominated |

---

## Qwen2.5-1.5B-Instruct (primary / canonical)

### Layer sweep — resolving the dose

The transferable 7B dose (α 0.044) is **below the resolving floor on 1.5B**
(`layer_sweep_qwen1.5b_a044.{csv,png}`): every layer lands 4.3–4.9 within seed
noise (±0.07–0.23), rep 0.03–0.09, ppl 2.7–4.5 — huge headroom, no localized peak.
(α 0.044 *did* resolve on 7B; the smaller model needs a stronger normalized dose
to separate layers.) So the sweep re-ran at a **cliff-justified** dose read from
the L17 dose-response, **α 0.09** — committed before looking at which layer won,
to avoid nudging toward 0.61.

Resolved (`layer_sweep_qwen1.5b.{csv,png}`), coherent-gated (rep < 0.15, ppl < 6):

| region | frac | formality (mean±std) | read |
|---|---|---|---|
| **peak** | **0.61 (L17)** | **4.97 ± 0.09** | argmax, rep 0.07, ppl 5.1 |
| plateau | 0.61–0.93 (L17–25) | 4.82–4.97 | overlapping seed range; L20/21/22/25 ≈ L17 |
| **dead-spot** | 0.25–0.54 (L7–15) | 3.2–4.2, several < baseline | disruptive: ppl 6→875, var ±1–2 |
| early | 0.04–0.21 (L1–6) | 4.5–4.8 | mild, near baseline |

**Peak plateau frac 0.61–0.93, argmax frac 0.61 (L17).** Brackets and hits the
anchor.

### Dose-response @ L17 (frac 0.61 anchor = the resolved peak)

`dose_response_qwen1.5b.{csv,png}`, resid_norm@L17 = 352.5.

| alpha | coeff | formality | rep | ppl | note |
|---|---|---|---|---|---|
| −0.113 | −40 | 2.82 ± 0.23 | 0.406 | 8.1 | degenerate (casual) |
| −0.071 | −25 | 3.05 ± 0.11 | 0.133 | 5.3 | casual, edge |
| −0.043 | −15 | 3.47 ± 0.13 | 0.082 | 3.5 | casual, coherent |
| 0.000 | 0 | 4.49 ± 0.07 | 0.070 | 2.6 | baseline |
| 0.043 | 15 | 4.84 ± 0.03 | 0.055 | 2.9 | formal, clean |
| 0.071 | 25 | 4.90 ± 0.02 | 0.066 | 3.6 | formal, clean |
| **0.113** | **40** | **5.06 ± 0.12** | 0.100 | 7.1 | **effect peak, ppl rising** |
| 0.170 | 60 | 5.00 ± 0.01 | 0.335 | 15.1 | **cliff** |
| 0.255 | 90 | 5.97 ± 0.44 | 0.662 | 9.4 | past-cliff artifact (looping) |
| 0.369 | 130 | 4.70 ± 2.42 | 0.309 | 6.1 | chaotic, variance blows up |

- **Usable band:** α ≈ −0.07 → +0.11 (coeff −25 → +40); formality 3.05 → 5.06
  monotone, rep < 0.15, ppl single-digit.
- **Coherence cliff:** α ≈ 0.13–0.17. Past it the formality proxy **inflates on
  degenerate repetition** (α 0.255: rep 0.66) — an effect-reversal artifact, not
  real steering — and seed variance explodes (α 0.369: ±2.42).

---

## Qwen2.5-0.5B-Instruct (secondary)

Same protocol. resid_norm@L15 = 64.5 (~5× smaller than 1.5B) → the shared coeff
grid overshoots hard, so the model is far more fragile.

### Dose-response @ L15 (frac 0.62 anchor)

`dose_response_qwen0.5b.{csv,png}`:

| alpha | coeff | formality | rep | ppl | note |
|---|---|---|---|---|---|
| −0.124 | −8 | 3.06 ± 0.16 | 0.087 | 5.9 | casual, coherent |
| 0.000 | 0 | 4.55 ± 0.12 | 0.051 | 3.1 | baseline |
| 0.124 | 8 | 5.02 ± 0.03 | 0.078 | 4.9 | formal, clean |
| 0.233 | 15 | 5.24 ± 0.05 | 0.160 | 10.7 | effect peak, cliff edge |
| 0.388 | 25 | 4.74 ± 0.37 | 0.548 | 10.2 | degenerate (looping) |
| 0.620 | 40 | 0.81 ± 0.62 | 0.000 | 4.4 | **collapse to near-empty** |
| 0.930 | 60 | 0.00 ± 0.00 | 0.000 | nan | **total collapse** |
| 1.395 | 90 | 1.40 ± 1.98 | 0.000 | 347490 | garbage |

- **Usable band (narrow):** α ≈ −0.12 → +0.23 (coeff −8 → +15); formality
  3.06 → 5.24.
- **Sharp cliff at α ≈ 0.23–0.39**, then — the honest negative — a **catastrophic
  collapse**: the 0.5B model emits near-empty output (formality → 0), unlike
  1.5B's graceful degradation into repetition.

### Layer sweep @ α 0.124 (strongest clean dose)

`layer_sweep_qwen0.5b.{csv,png}`, baseline 4.53, coherent-gated (rep < 0.15, ppl < 6):

| region | frac | formality | read |
|---|---|---|---|
| **argmax** | **0.96 (L23, last)** | **5.77 ± 0.17** | rep 0.05, ppl 5.5 — real formal prose (verified) |
| plateau | 0.58–0.96 (L14–23) | 4.87–5.77 | broad coherent high band; climbs to the output end |
| 0.61 anchor | 0.62–0.67 (L15–16) | 5.02–5.06 | on the plateau, not its peak |
| dead-spot | 0.29–0.54 (L7–13) | 3.2–4.25 | ppl 6–15, rep spikes |

**On the argmax (verified, not assumed):** the coherent-gated argmax is
**L23/frac 0.96 (5.77)** — the last layer, where fixed α maps to coeff 33 (vs ~8
mid-stack). I initially suspected an output-layer logit-boost artifact, but a
1-seed inspection at L23/coeff 33 disproved that: the generation is fluent,
coherent, and more formal than baseline ("…created by Alibaba Cloud… including but
not limited to general knowledge, cultural customs, scientific discoveries…") — a
real steering effect, not gamed tokens or truncation. So 0.5B genuinely peaks at
the output end; **0.61 is the start of its coherent plateau, not its peak.** (By
contrast, on 1.5B the last layers sit at/below baseline — e.g. L25/coeff 43.8 =
4.88, L27 = 4.56 — so this is a 0.5B-specific late-layer climb, not a general
high-coeff effect.) 0.5B is also noisier than 1.5B (seed std up to ±0.7 mid-stack).

---

## Honest negatives (summary)

1. **Dose transfer breaks down at small scale.** α 0.044 (fine on 7B) is below
   the layer-resolving floor on 1.5B; α needs re-derivation per model from the
   cliff, not blind transfer.
2. **Mid-network dead-spot.** frac ~0.25–0.54 injection *reduces* formality and
   *breaks coherence* on both small models (ppl spikes, high variance).
3. **Past-cliff effect is an artifact.** Beyond the cliff the formality proxy can
   rise, but on degenerate repetition (1.5B) or collapsed/near-empty output
   (0.5B) — not real steering; report the usable band, not the raw max.
4. **Peak is a plateau, not a point — and 0.61 is not the argmax everywhere.**
   3 seeds give a coarse argmax; the back half is broadly high. 0.61 is the clean
   argmax on 1.5B and 7B, but on 0.5B the argmax is the last layer (0.96, verified
   real prose) and 0.61 is only the plateau *start*. The defensible claim is
   "0.61 is a high-effect coherent injection depth at every scale," not "0.61 is
   the universal peak."

## Reproduce

```bash
# from the repo root (so the src/ mount resolves), Modal authed:
./experiments/reproduce.sh              # 1.5B primary: train + dose + both sweeps
MODEL=0.5b ./experiments/reproduce.sh   # 0.5B secondary
```

Renders `results/{dose_response,layer_sweep}_qwen{1.5b,0.5b}.{csv,png}` and prints
the GPU spend. `steer-report` then builds the four-part card from these CSVs on
CPU (no GPU, no download).
