# steerbench

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bamdadd/steerbench/blob/main/notebooks/steerbench_quickstart.ipynb)
&nbsp;·&nbsp; MIT &nbsp;·&nbsp; no GPU needed for the report card

> **The report card for steering vectors** — effect size, side effects, a dose-response curve, and per-layer sensitivity for any concept direction.

![Dose-response for a real Qwen2.5-7B formality vector: effect vs coherence, with the sweet spot and coherence cliff marked](assets/hero_dose_response.png)

*A real Qwen2.5-7B formality vector. Effect changes with the injection
coefficient while perplexity holds — then falls off the **coherence cliff**.
steerbench marks the sweet spot and the cliff automatically.*

## Quickstart

Render the four-part card from a repeng vector — CPU-only, no model download:

```bash
git clone https://github.com/bamdadd/steerbench.git && cd steerbench
uv sync --extra vectors --extra report
uv run steer-report data/examples/formality_qwen2.5-7b.gguf --out report_out
open report_out/report.html          # the four-part card (self-contained HTML)
```

`steer-report` summarises the vector (per-layer norms), then reads the committed
sweep CSVs in `artifacts/` and renders the card on CPU. Drop the vector argument
to render from the CSVs alone. Zero-install alternative: the
[Colab notebook](notebooks/steerbench_quickstart.ipynb) (badge above) does the
same in well under 10 minutes.

The GPU sweep that produces the CSVs lives behind `steerbench[gpu]` (Modal),
reached via `steer-report --run <function>`; the core never imports it.

## Pipeline

```mermaid
flowchart LR
    subgraph S["steerbench"]
        direction TB
        D["dose sweep (α grid)"]
        L["layer sweep (every depth)"]
        X["side-effect slices (MMLU / GSM8K)"]
    end

    subgraph R["report card"]
        direction TB
        R1["effect size"]
        R2["side effects"]
        R3["dose-response: sweet spot + cliff"]
        R4["layer sensitivity"]
    end

    V["steering vector<br/>repeng .gguf or .pt"] --> S
    S --> R
```

## How it works (plain language)

**1. A concept is a direction.** As a model reads text, every layer keeps its
running "thoughts" as a big list of numbers — the *residual stream*, a conveyor
belt each layer adds to. A concept like *formality* or *ocean* shows up as a
**direction** in that space. steerbench consumes directions extracted by
[repeng](https://github.com/vgel/repeng) — it does **not** reimplement
extraction.

**2. The nudge has two dials.** You steer by **adding** that direction into the
model's live internal state mid-generation — plain vector addition:

```
current thoughts  +  α · (formality direction)  →  nudged thoughts
```

Two dials: **where** (which layer — depth) and **how much** (`α` — strength).
Pick them wrong and you either get no effect, or you push `α` too high and the
output collapses into repetitive nonsense — the *coherence cliff*.

**3. The problem steerbench solves.** Anyone injecting a concept must choose
**where** and **how much** to nudge *without breaking the model* — and "looks
fine" is not a measurement. steerbench answers it objectively: **dose-response**
sweeps `α` to find the sweet spot and the cliff; the **layer sweep** injects at
every depth to find where the vector actually works (a coherent plateau, versus
the degenerate-trap layers that look strong but are just broken text); and
**side effects** re-check held-out benchmarks (MMLU / GSM8K) so you know the
nudge didn't quietly break everything else.

**The tool exists because the science needed it.**
[introspection-scaling](https://github.com/bamdadd/introspection-scaling) — the
project steerbench was built for — asks whether models can introspect on
*injected* concepts as they scale. To run that cleanly it must inject at a
known-good layer and strength on every model in the ladder. steerbench produces
that report card — the sweet spot, the cliff, the safe layer — so the injection
is **calibrated, not guessed**.

## Results

One extraction recipe (repeng PCA-diff, 3 seeds, A100 via Modal) run across
models and concepts. Each (model, concept) cell is **n = 1 vector** with coarse
behavioural proxies, so these are **existence** claims, not statistical
interactions. Every effect is measured **within-concept against that cell's own
baseline**; the formality and sentiment proxies are on different scales and are
never compared to each other.

**The finding: architectures differ in dose-response *shape*** — measured on the
coherence-gated `alpha_norm` axis (the residual-fraction dose; model-independent,
unlike the proxy scale):

- **Qwen2.5-7B — interior optimum.** Formality effect rises with dose, peaks
  around `alpha_norm` ≈ 0.04–0.055, then the coherence cliff takes over and the
  effect *reverses* past it (the hero above).
- **Llama-3.x-8B — monotonic-to-cliff.** No interior turn; the best usable effect
  is the *last coherent dose* (`alpha_norm` ≈ 0.197). It needs a **~3–4× higher
  normalized dose** than Qwen (≈ 3.6× vs Qwen formality at ~0.055; ≈ 2.3× vs
  Qwen sentiment at 0.087). A dose tuned on Qwen underdrives Llama.

**A cautionary case — the report card catches a bad vector.** On **Mistral-7B**,
the formality dose-response came back **flat in both directions** (positive
`alpha` never clears the 4.83 baseline; negative `alpha` barely moves it despite
downward headroom). That *looks* like "this model won't take formality steering"
— but a 3× re-extraction **stability check** showed why: the formality direction
is **unstable** (injection-layer cosine ≈ **−0.13**, sign-flipping across
re-extractions: −0.81, −0.29, +0.69), i.e. a low-SNR vector, not a model
property. The **sentiment** vector on the *same model* is rock-stable
(cosine ≈ **0.95**) and steers fine. Lesson: a flat curve can mean *bad vector*
or *stubborn model*, and you cannot tell by eyeballing one generation — the
report card plus a stability check distinguishes them. This is the repeng
extraction-instability failure mode ([vgel/repeng#78](https://github.com/vgel/repeng/issues/78)).

**Coherence gates every peak.** We deliberately do *not* headline layer/depth
patterns: Llama's formality peak sits at perplexity ≈ 5.0 (borderline), and
Mistral's apparent formality spike at layer 2 (effect 15.6) lands at perplexity
≈ 8.5 — degenerate text the gate rejects rather than reports as a peak.

Qwen M0 full run: [M0_REPORT.md](M0_REPORT.md). Cross-model runs ongoing on
Llama-3.x-8B and Mistral-7B (Gemma 2 9B gated, substituted).

## Scope

Consumes vectors produced by [repeng](https://github.com/vgel/repeng) —
extraction is not reimplemented. Ships benign example concepts (formality,
sentiment, verbosity).

## Related reading

[Two bugs in repeng](https://bamdad.substack.com/p/two-bugs-in-repeng) — the
non-determinism bug found while building steerbench, and the one-line fix
([vgel/repeng#79](https://github.com/vgel/repeng/pull/79)).

## License

MIT.
