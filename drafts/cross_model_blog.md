# Same recipe, different curves — and a vector that lied

> **DRAFT — do not publish yet.** Awaiting review sign-off. The extraction-
> stability confound flagged in the earlier draft is now **resolved** (see *The
> cautionary case*), so the science is settled; this hold is editorial, not
> scientific.

## The short version

Extract a concept direction with one recipe, inject it into a model's residual
stream, and sweep the dose. Two things came out of running that across a few
models:

1. **Dose-response *shape* is architecture-specific.** **Qwen2.5-7B** has a true
   **interior optimum** — the effect peaks at a middling dose and then *reverses*
   as you push past the coherence cliff. **Llama-3.x-8B** is
   **monotonic-to-cliff** — no interior turn; its best usable effect is the last
   dose before coherence breaks, and it needs **~3–4× the normalized dose** to get
   there. This is the defensible, non-obvious result.
2. **A flat dose-response curve does not mean the model is stubborn — it can mean
   the vector is broken.** On Mistral-7B a formality vector produced a dead-flat
   curve. It *looked* like "this model resists formality steering." A stability
   check said otherwise: the vector was noise. steerbench flagged the symptom;
   the stability check found the cause. That is the methods story, and it is more
   useful than the false headline it replaced.

Both are **existence** claims: one vector per (model, concept) cell, coarse
proxies. We are pointing at phenomena, not estimating interaction terms.

## How to read the numbers (and how not to)

- **Never compare effect sizes across concepts.** Formality is a 0–5 lexical
  proxy; sentiment is a different scale. Every number here is a change **within
  one concept, against that cell's own unsteered baseline** — never one concept's
  delta versus another's.
- **Dose is `alpha_norm`, not raw coefficient** — the injection magnitude as a
  fraction of the residual-stream norm (`alpha · ‖dir‖ / ‖residual‖`). Raw
  coefficients don't transfer across models; `alpha_norm` does. The cross-model
  dose comparison below is a comparison of these scale-free doses, not of
  effects.

## The finding — dose-response shape differs by architecture

**Qwen2.5-7B — interior optimum.** Formality effect rises with dose, peaks around
`alpha_norm` ≈ 0.04–0.055, and then the coherence cliff takes over: push harder
and the effect **reverses** as the text degenerates. There is a genuine best dose
in the interior of the range.

**Llama-3.x-8B — monotonic-to-cliff.** No interior turn. The effect climbs right
up to the coherence cliff, so the best usable dose is simply the **last coherent
one** (`alpha_norm` ≈ 0.197). And that dose is high — **~3–4× Qwen's**: about
**3.6×** Qwen's formality optimum (~0.055), and about **2.3×** Qwen's sentiment
optimum (0.087). Llama has to be pushed substantially harder, in normalized
units, before the concept lands.

Practically: a dose tuned on Qwen will *underdrive* Llama, and a search that
assumes an interior peak — stop when the effect stops rising — stops too early on
Llama, whose effect doesn't stop rising until coherence does. This is the one
claim here we would defend: it rides on the model-independent `alpha_norm` axis
and on coherence-gated peaks, not on any proxy-scale number.

We hold this claim to the same gate the cautionary case below is about: it is
only as trustworthy as the vectors under it, so we stability-checked them. Both
formality directions are **stable at adequate data** — injection-layer cosine
across re-extractions **0.97 (Qwen)** and **0.83 (Llama)** at a 90% subsample of
the 69 contrastive pairs (Llama's spread is modestly wider, one pair at 0.71).
The shape difference therefore reflects the architectures, not extraction
variance. We applied the lesson to our own headline before publishing it.

One honest caveat that *strengthens* the piece rather than weakening it:
formality extraction is **data-hungry**. Drop to a 70% subsample of those 69
pairs and the same cosine falls to **0.55–0.64** — the vector's stability depends
on how much contrastive data you gave it. Qwen's 0.64 at 70% is data-subsample
sensitivity (only 69 pairs), not algorithmic noise; it recovers to 0.97 with the
full set. Vector stability is itself a thing to measure, not assume — another
face of the same repeng failure mode below.

## The cautionary case — telling a bad vector from a stubborn model

On **Mistral-7B**, the formality dose-response came back **flat in both
directions**: positive `alpha` never lifted formality above the 4.83 baseline,
and negative `alpha` barely moved it despite plenty of downward headroom. The
obvious read is "Mistral won't take formality steering."

**That read would have been wrong.** A re-extraction **stability check** — run the
same extraction on different subsamples and measure how much the resulting
direction agrees with itself — showed the formality direction is **unstable**:

- injection-layer cosine similarity across re-extractions ≈ **−0.13**, and
  **sign-flipping** run to run (−0.81, −0.29, +0.69). A vector that points a
  different way every time you extract it is not measuring anything.
- and this is **not** just the data-hunger from the last section. At the *same*
  70% subsample, the **sentiment** direction on the **same model** is rock-stable
  (cosine **0.95**) and steers fine (**+0.65** over its own sentiment baseline).
  Same model, same data fraction, same recipe — sentiment holds, formality
  collapses. So it is not "small data broke everything"; Mistral's *formality*
  extraction fails specifically.

And it is a *different failure than data-hunger*, not just a worse degree of it —
which matters, because we measured Mistral formality at the same data-starved 70%
where Qwen formality also dips. But those two dips are different animals. Qwen's
ss70 direction is **consistent and positive** (0.64) and **recovers to 0.97**
with the full data — a weak-but-real signal starved of pairs. Mistral's
**sign-flips** (−0.81, −0.29, +0.69): a direction that reverses run to run is not
a weak signal, it is **noise**. Degradation you can fix with more data; a
sign-flipping direction is not measuring anything to begin with.

So Mistral's flat formality curve is a **low-SNR extraction**, not architectural
resistance. One concept produced a stable, working vector and the other produced
noise. The flatness was the vector, not the model.

The lesson is the product. A single steered generation can look "meh" for either
reason — a genuinely stubborn model or a broken vector — and you **cannot tell
them apart by eyeballing one sample**. A flat, coherence-gated dose-response
*plus* an extraction-stability check can: stable-but-flat would implicate the
model; unstable-and-flat implicates the extraction. Here it was the extraction.
This is exactly the repeng extraction-instability failure mode
([vgel/repeng#78](https://github.com/vgel/repeng/issues/78)) — the report card
surfaces it instead of shipping a confident wrong conclusion.

## The coherence gate is doing real work

Every "peak" here is coherence-gated, and we deliberately **do not** headline
depth/layer patterns, because the raw effect numbers lie without the gate:

- **Mistral, formality, layer 2** shows an apparent effect of **15.6** — far
  above anything else — at **perplexity ≈ 8.5**: degenerate, repetitive text. The
  gate rejects it; reported as a "peak" it would be pure artifact. (This is a
  layer of the *unstable* vector, so doubly not a result.)
- **Llama's formality peak** sits at **perplexity ≈ 5.0** — borderline. We flag
  it rather than celebrate it.

An effect number without a coherence number beside it is not a result.

## What we are **not** claiming

- **Not** that Mistral architecturally resists formality steering, and **not** a
  decode-vs-steer dissociation. The formality vector was noise; there is no
  model-level claim to make from it.
- **Not** a statistical interaction. n = 1 vector per cell, coarse proxies —
  existence claims only.
- **Not** a cross-concept effect-size comparison. A formality delta and a
  sentiment delta are on different proxy scales; we never put them side by side.
- **Not** a single per-model steerability scalar in either direction — the
  defensible cross-model statement is about dose-response *shape*, above.

## Method, briefly

repeng PCA-diff contrastive extraction (not reimplemented), exported to
repeng-native gguf, reloaded and injected via `ControlModel`. 3 seeds per point;
effect = concept proxy, coherence = repetition rate + perplexity under the
unsteered model; dose reported as `alpha_norm`. Extraction stability = cosine
agreement of the injection-layer direction across independent re-extractions.
Full Qwen M0 run and the per-model CSVs are in the repo; the cross-model sweep is
ongoing.
