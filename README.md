# steerbench

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bamdadd/steerbench/blob/main/notebooks/steerbench_quickstart.ipynb)
&nbsp; MIT-licensed &nbsp;·&nbsp; no GPU needed for the report card

> **The steering-vector report card.** Extraction is crowded; evaluation is
> empty. Give steerbench any concept vector and get: effect size, side
> effects, a dose-response curve, and a per-layer sensitivity chart.

![Dose-response: effect vs coherence, with the sweet spot and coherence cliff annotated](assets/hero_dose_response.png)

> The hero: a real Qwen2.5 formality vector. Effect climbs with the injection
> coefficient while perplexity holds — then falls off the **coherence cliff**.

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
fine" is not a measurement. steerbench is the instrument that answers it,
objectively:

- **Dose-response** sweeps `α` and charts effect against coherence — locating
  the **sweet spot** (the strongest push that stays fluent) and the **cliff**
  (where fluency dies).
- **Layer sweep** injects at every depth to find *where* the vector actually
  works: a coherent plateau, plus the degenerate-trap layers that look strong
  but are just broken text.
- **Side effects** re-check held-out benchmarks (MMLU / GSM8K) so you know the
  nudge didn't quietly break everything else.

**The tool exists because the science needed it.**
[introspection-scaling](https://github.com/bamdadd/introspection-scaling) — the
project steerbench was built for — asks whether models can introspect on
*injected* concepts as they scale. To run that cleanly it must inject at a
known-good layer and strength on every model in the ladder. steerbench produces
that report card — the sweet spot, the cliff, the safe layer — so the injection
is **calibrated, not guessed**.

## Quickstart (target: <10 min to first report on Colab)

**Fastest — zero install, click the badge above.** The [Colab notebook](notebooks/steerbench_quickstart.ipynb)
renders the committed M0 sweep artifacts into the full card, CPU-only, in well
under 10 minutes.

Locally:
```bash
uv sync
# Render the card from committed sweep artifacts (no GPU):
uv run steer-report --out report_out
# ...or summarise a repeng vector too:
uv run steer-report path/to/vector.gguf --model Qwen/Qwen2.5-7B --out report_out
```

## What it measures
1. **Effect size** on the target behaviour, objectively.
2. **Side effects** — capability degradation on held-out benchmark slices.
3. **Dose-response** — effect & coherence vs injection strength (the cliff).
4. **Layer sensitivity** — the same vector injected at every layer.

## Scope (said out loud)
Consumes vectors produced by **repeng** — extraction is not reimplemented.
Ships benign example concepts (formality, sentiment, verbosity).

## The finding it aims at
Run one concept vector across Llama 3.x 8B / Qwen2.5 7B / Gemma 2 9B and
chart the differences — steerability varies by architecture.

## License
MIT.
