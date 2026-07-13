# steerbench

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bamdadd/steerbench/blob/main/notebooks/steerbench_quickstart.ipynb)
&nbsp; MIT-licensed &nbsp;·&nbsp; no GPU needed for the report card

> **The steering-vector report card.** Extraction is crowded; evaluation is
> empty. Give steerbench any concept vector and get: effect size, side
> effects, a dose-response curve, and a per-layer sensitivity chart.

![Dose-response: effect vs coherence, with the sweet spot and coherence cliff annotated](assets/hero_dose_response.png)

> The hero: a real Qwen2.5 formality vector. Effect climbs with the injection
> coefficient while perplexity holds — then falls off the **coherence cliff**.

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
