# steerbench

> **The steering-vector report card.** Extraction is crowded; evaluation is
> empty. Give steerbench any concept vector and get: effect size, side
> effects, a dose-response curve, and a per-layer sensitivity chart.

<!-- HERO FIGURE: a dose-response curve. Put it above the fold. -->

## Quickstart (target: <10 min to first report on Colab)
```bash
uv sync
uv run steer-report path/to/vector.pt --model Qwen/Qwen2.5-7B --out report.md
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
