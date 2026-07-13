# Steerability is not one number per model

> **DRAFT — do not publish.** One claim in this post (the Mistral formality
> result) is gated on an extraction-stability check that is still running. Hold
> until that lands; see *Open confounds* at the end.

## The short version

If you inject a concept direction into a language model's residual stream, how
steerable is the model? The tempting answer is a single number per model — "Qwen
is steerable, model X is stubborn." Running one extraction recipe across a few
models and a couple of concepts, that framing breaks in two ways:

1. **Steerability is concept-specific, not a per-model scalar.** The *same*
   model, under the *same* extraction, can be readily steerable for one concept
   and near-inert for another. Our cleanest instance: **Mistral-7B** takes
   **sentiment** steering but is **flat** to **formality**.
2. **Architectures differ in the *shape* of the dose-response curve.** **Qwen**
   has an interior optimum — a peak, then a cliff, with the effect reversing if
   you push past it. **Llama** is monotonic-to-cliff — its best usable effect is
   the last dose before coherence breaks, and it needs a markedly higher
   normalized dose to get there.

These are **existence / dissociation** claims, not statistical interactions. We
have one vector per (model, concept) cell and coarse behavioural proxies. We are
saying *this pattern occurs*, not estimating an interaction term.

## How to read the numbers (and how not to)

Two rules we hold ourselves to, because they are easy to get wrong:

- **Never compare effect sizes across concepts.** The formality proxy is a
  0–5 lexical score; the sentiment proxy is on a different scale entirely. A
  formality delta and a sentiment delta are not commensurable. Every number
  below is a change **within one concept, against that cell's own unsteered
  baseline** — never one concept's delta versus another's.
- **Dose is measured in `alpha_norm`, not raw coefficient.** `alpha_norm` is the
  injection magnitude as a fraction of the residual-stream norm (`alpha · ‖dir‖ /
  ‖residual‖`). Raw coefficients don't transfer across models — residual norms
  differ — so any cross-model dose comparison uses `alpha_norm`.

## Finding 1 — the same model, two concepts, opposite outcomes

**Mistral-7B, sentiment:** injecting the sentiment direction moves the sentiment
proxy **+0.65 over the model's own sentiment baseline**. It steers.

**Mistral-7B, formality:** nothing usable. The formality proxy is **flat in both
directions**:

- Positive `alpha` never lifts formality **above the 4.83 baseline**.
- Negative `alpha` **barely moves it** — even though there is plenty of downward
  headroom for it to fall into.

The two-sided flatness matters. If formality had simply saturated at the top, you
could dismiss it as a ceiling artifact of the proxy. But the negative side is
flat *with room to move*, so this is not a measurement ceiling — under this
extraction, the formality direction just doesn't drive Mistral.

So: same model, same recipe, one concept steers and the other doesn't. That is
the dissociation. "How steerable is Mistral?" has no single answer.

## Finding 2 — the dose-response *shape* differs by architecture

This is the more robust half, because it rides on `alpha_norm` rather than the
proxy scale.

**Qwen2.5-7B — interior optimum.** Formality effect rises with dose, peaks around
`alpha_norm` ≈ 0.04–0.055, and then the coherence cliff takes over: push harder
and the effect **reverses** as the text degenerates. There is a genuine best dose
in the interior of the range.

**Llama-3.x-8B — monotonic-to-cliff.** No interior turn. The effect keeps
climbing right up to the coherence cliff, so the best usable dose is simply the
**last coherent one** (`alpha_norm` ≈ 0.197). And that dose is high: **~3–4×**
Qwen's — about **3.6×** Qwen's formality optimum (~0.055), and about **2.3×**
Qwen's sentiment optimum (0.087). Llama needs to be pushed substantially harder,
in normalized units, before the concept lands.

Practically: a dose you tuned on Qwen will *underdrive* Llama, and a search
strategy that assumes an interior peak (stop when the effect stops rising) will
stop too early on Llama — its effect doesn't stop rising until coherence does.

## The coherence gate is doing real work

Every "peak" here is coherence-gated, and we deliberately **do not** headline
depth/layer patterns, because the raw effect numbers lie without the gate:

- **Mistral, formality, layer 2** shows an apparent effect of **15.6** — far
  above anything else. It lands at **perplexity ≈ 8.5**: degenerate, repetitive
  text. The gate rejects it. Reported as a "peak" it would be pure artifact;
  rejected, it is the gate *working*.
- **Llama's formality peak** sits at **perplexity ≈ 5.0** — borderline. We flag
  it rather than celebrate it.

This is the whole reason steerbench pairs every effect measurement with a
coherence measurement. An effect number without a coherence number next to it is
not a result.

## Open confounds — why this is a draft

The dose-response *shape* finding (Qwen interior optimum vs Llama
monotonic-to-cliff) we are fairly confident in: it uses the model-independent
`alpha_norm` axis and reproduces the expected cliff behaviour.

The Mistral **formality** dissociation has one confound we have **not yet ruled
out**: the formality *direction* we extracted for Mistral may be a **low-SNR
extraction** — a noisy vector — rather than evidence that the model resists
formality steering. A flat dose-response is exactly what a near-random direction
would also produce. Agent 1 is running an **extraction-stability check** now
(re-extract across seeds / data splits, compare direction agreement). Until that
comes back:

- We do **not** publish the Mistral formality claim.
- We do **not** upgrade "dissociation exists" into any statement about *why*
  (model architecture vs extraction quality).

## What we are **not** claiming

- Not a statistical interaction. n = 1 vector per cell, coarse proxies. This is
  an existence proof of dissociation, nothing stronger.
- Not a cross-concept effect-size comparison. A formality delta and a sentiment
  delta live on different proxy scales; we never put them side by side. Every
  number in this post is a within-concept change against that cell's own
  baseline.
- Not a per-model steerability ranking. The whole point is that no such single
  ranking exists.

## Method, briefly

repeng PCA-diff contrastive extraction (not reimplemented), exported to
repeng-native gguf, reloaded and injected via `ControlModel`. 3 seeds per point;
effect = concept proxy, coherence = repetition rate + perplexity under the
unsteered model. Dose reported as `alpha_norm`. Full Qwen M0 run and the
per-model CSVs are in the repo; the cross-model sweep is ongoing.
