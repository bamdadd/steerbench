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

One caveat we hold ourselves to — the same one the cautionary case below is
about. This shape claim is only as trustworthy as the vectors under it. Qwen's is
3-seed stable and its interior optimum is visible across the sweep; the honest
gate before calling Llama's *shape* settled (rather than strong) is the identical
extraction-stability check we run on Mistral below, applied to the Llama formality
direction. We apply the lesson to our own headline, not just to the cautionary
tale.

## The cautionary case — telling a bad vector from a stubborn model

On **Mistral-7B**, the formality dose-response came back **flat in both
directions**: positive `alpha` never lifted formality above the 4.83 baseline,
and negative `alpha` barely moved it despite plenty of downward headroom. The
obvious read is "Mistral won't take formality steering."

**That read would have been wrong.** A 3× re-extraction **stability check** — run
the same extraction on different seeds/splits and measure how much the resulting
direction agrees with itself — showed the formality direction is **unstable**:

- injection-layer cosine similarity across re-extractions ≈ **−0.13**, and
  **sign-flipping** run to run (−0.81, −0.29, +0.69). A vector that points a
  different way every time you extract it is not measuring anything.
- the **sentiment** direction on the **same model** is rock-stable (cosine ≈
  **0.95**) and steers fine (**+0.65** over its own sentiment baseline).

So Mistral's flat formality curve is a **low-SNR extraction**, not architectural
resistance. Same model, same recipe: one concept produced a stable, working
vector and the other produced noise. The flatness was the vector, not the model.

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
