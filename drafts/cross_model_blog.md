# Same recipe, different curves: a vector that lied

> **DRAFT, do not publish yet.** Awaiting review sign-off. The extraction-
> stability confound flagged in the earlier draft is now resolved (see *The
> cautionary case*), so the science is settled. This hold is editorial, not
> scientific.

## The short version

Extract a concept direction with one recipe, inject it into a model's residual
stream, and sweep the dose. Running that across a few models turned up two
things.

First, the *shape* of the dose-response curve depends on the architecture.
Qwen2.5-7B has a true interior optimum: the effect peaks at a middling dose and
then reverses once you push past the coherence cliff. Llama-3.1-8B never turns
over. Its effect climbs right up to the cliff, so the best usable dose is the
last one before coherence breaks, and it takes roughly 3-4x the normalized dose
to get there. That is the defensible, non-obvious result.

Second, a flat dose-response curve does not have to mean the model is stubborn.
It can mean the vector is broken. On Mistral-7B a formality vector produced a
dead-flat curve that looked like "this model resists formality steering." A
stability check said otherwise: the vector was noise. steerbench flagged the
symptom, and the stability check found the cause. That turned out to be more
useful than the false headline it replaced.

Both are existence claims. One vector per (model, concept) cell, coarse proxies.
I am pointing at phenomena, not estimating interaction terms.

## How to read the numbers (and how not to)

Never compare effect sizes across concepts. Formality is a 0-5 lexical proxy;
sentiment sits on a different scale. Every number here is a change within one
concept, measured against that cell's own unsteered baseline, never one concept's
delta against another's.

Dose is `alpha_norm`, not the raw coefficient. It is the injection magnitude as a
fraction of the residual-stream norm (`alpha · ‖dir‖ / ‖residual‖`). Raw
coefficients do not transfer across models; `alpha_norm` does. The cross-model
dose comparison below compares these scale-free doses, not effects.

## The finding: dose-response shape differs by architecture

On Qwen2.5-7B the formality effect rises with dose, peaks around `alpha_norm`
0.04-0.055, and then the coherence cliff takes over: push harder and the effect
reverses as the text degenerates. There is a genuine best dose in the interior of
the range.

Llama-3.1-8B has no interior turn. The effect keeps climbing until coherence
breaks, so the best usable dose is just the last coherent one (`alpha_norm`
0.197). That dose is high, roughly 3-4x Qwen's: about 3.6x Qwen's formality
optimum (0.055) and about 2.3x its sentiment optimum (0.087). Llama has to be
pushed a lot harder, in normalized units, before the concept lands.

In practice that means a dose tuned on Qwen will underdrive Llama, and a search
that assumes an interior peak (stop when the effect stops rising) will stop too
early on Llama, whose effect does not stop rising until coherence does. This is
the one claim I would defend without hedging. It rides on the model-independent
`alpha_norm` axis and on coherence-gated peaks, not on any proxy-scale number.

That claim is only as good as the vectors under it, so I ran the same gate the
cautionary case below is about. Both formality directions are stable once the
data is adequate: injection-layer cosine across re-extractions is 0.97 for Qwen
and 0.83 for Llama at a 90% subsample of the 69 contrastive pairs (Llama's spread
is a bit wider, with one pair at 0.71). So the shape difference reflects the
architectures, not extraction noise. I checked my own headline before writing it
up.

One caveat, and it makes the piece stronger rather than weaker: formality
extraction is data-hungry. Subsample those 69 pairs down to 70% and the same
cosine drops to 0.55-0.64. How stable the vector is depends on how much
contrastive data it saw. Qwen's 0.64 at 70% is data-subsample sensitivity from
having only 69 pairs, not algorithmic noise; it climbs back to 0.97 on the full
set. Vector stability is something you have to measure, not assume, which is
another face of the repeng failure mode below.

## The cautionary case: telling a bad vector from a stubborn model

On Mistral-7B the formality dose-response came back flat in both directions.
Positive `alpha` never lifted formality above the 4.83 baseline, and negative
`alpha` barely moved it despite plenty of downward headroom. The obvious read is
that Mistral will not take formality steering.

That read would have been wrong. A re-extraction stability check (run the same
extraction on different subsamples, then measure how much the resulting direction
agrees with itself) showed the formality direction is unstable, and it showed it
three ways.

The direction itself does not hold. Injection-layer cosine across re-extractions
is about -0.13, and it sign-flips run to run: -0.81, -0.29, +0.69. A vector that
points a different way each time you extract it is not measuring anything.

It is also not just the data-hunger from the last section. At the same 70%
subsample, the sentiment direction on the same model is rock-stable (cosine 0.95)
and steers fine, +0.65 over its own sentiment baseline. Same model, same data
fraction, same recipe, and sentiment holds where formality collapses. So this is
not small data breaking everything. Mistral's formality extraction is the part
that fails.

And it is not that Mistral is hard to extract from in general. Verbosity extracts
stably on all three models: cosine 0.94 for Qwen, 0.92 for Llama, 0.93 for
Mistral. Mistral gets two of the three concepts cleanly, sentiment at 0.95 and
verbosity at 0.93, and misses only formality. The defect is specific to one
concept on one model, which is exactly the case a single flat curve leaves you
guessing about.

The failure is also a different kind from data-hunger, not just a worse degree of
it, and that distinction is what carries the argument. I measured Mistral
formality at the same starved 70% where Qwen formality also dips, but the two
dips are not the same animal. Qwen's 70% direction stays consistent and positive
(0.64) and recovers to 0.97 on the full data: a weak but real signal short on
pairs. Mistral's sign-flips, -0.81, -0.29, +0.69, and a direction that reverses
run to run is noise, not a weak signal. More data fixes degradation. It cannot
fix a direction that was never pointing anywhere.

So Mistral's flat formality curve is a low-SNR extraction, not architectural
resistance. One concept gave a stable, working vector; the other gave noise. The
flatness came from the vector, not the model.

That distinction is the whole point of the report card. A single steered
generation can look mediocre for either reason, a genuinely stubborn model or a
broken vector, and you cannot tell them apart by eyeballing one sample. A flat,
coherence-gated dose-response plus an extraction-stability check can: stable but
flat would point at the model, unstable and flat points at the extraction. This
time it was the extraction. It is the repeng extraction-instability failure mode
([vgel/repeng#78](https://github.com/vgel/repeng/issues/78)), and the report card
surfaces it instead of shipping a confident wrong conclusion.

## Coherence gates every peak

Every peak here is coherence-gated, and I deliberately do not headline depth or
layer patterns, because the raw effect numbers lie without the gate.

Take Mistral formality at layer 2. It shows an apparent effect of 15.6, far above
anything else, at a perplexity of about 8.5: degenerate, repetitive text. The
gate rejects it, and called a "peak" it would be pure artifact. It is also a
layer of the unstable vector, so it is doubly not a result. Llama's formality
peak is milder but still worth flagging: it sits at a perplexity of about 5.0,
which is borderline, so I flag it rather than celebrate it.

An effect number with no coherence number next to it is not a result.

## What I am not claiming

I am not claiming Mistral architecturally resists formality steering, and I am
not claiming a decode-versus-steer dissociation. The formality vector was noise,
so there is no model-level claim to draw from it.

I am not claiming a statistical interaction. One vector per cell and coarse
proxies make these existence claims, nothing stronger.

I am not comparing effect sizes across concepts. A formality delta and a
sentiment delta live on different proxy scales, and they never go side by side.

And I am not putting a single steerability score on any model. The defensible
cross-model statement is the one about dose-response shape, above.

## Method, briefly

Models: Qwen2.5-7B, Llama-3.1-8B (the NousResearch mirror of Llama 3.1), and
Mistral-7B. Extraction is repeng PCA-diff contrastive extraction, not
reimplemented, exported to repeng-native gguf and then reloaded and injected via
`ControlModel`. Three seeds per point. Effect is the concept proxy; coherence is
repetition rate plus perplexity under the unsteered model; dose is reported as
`alpha_norm`. Extraction stability is the cosine agreement of the injection-layer
direction across independent re-extractions. The cross-model sweep is complete:
three concepts (formality, sentiment, verbosity) across all three models, with
the full Qwen M0 run and the per-model CSVs in the repo.
