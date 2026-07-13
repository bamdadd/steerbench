"""report.py — render the steering report card from sweep outputs.

Consumes three CSVs produced by the M0 sweep and emits the four-part card as
both markdown and a self-contained HTML file:

1. **Dose-response** — effect and coherence vs injection coefficient, with the
   sweet spot and the coherence cliff annotated. This is the hero figure and is
   rendered *above the fold* in HTML.
2. **Effect size** — peak behaviour change and the effect at the sweet spot.
3. **Side effects** — steered vs unsteered accuracy on held-out benchmarks.
4. **Layer sensitivity** — the same vector injected at every layer.

CSV schema (the contract with the M0 ``m0-sweep`` worktree). Rows are RAW —
one sample per seed, NOT pre-aggregated — and this module computes mean±std
across seeds itself:

* dose-response (``artifacts/dose_response.csv``) — one row per (coeff, seed)::

      coeff,seed,alpha_norm,formality,repetition,ppl

  ``effect`` = ``formality``; coherence = the pair ``ppl`` (lower = better) and
  ``repetition`` (higher = worse). ``alpha_norm`` is ignored. Extra columns are
  tolerated; the required set is ``{coeff, seed, formality, repetition, ppl}``.
* layer-sweep (``artifacts/layer_sweep_coeff.csv``) — one row per (layer, seed)::

      layer,layer_pos,seed,dir_norm,formality,repetition,ppl

  Same effect/coherence tail keyed on ``layer``. ``dir_norm`` (per-layer
  direction L2, ~1.0 since repeng unit-normalises) and ``layer_pos`` are
  ignored. :func:`analyze_layers` flags coherent-peak vs degenerate-trap layers.
* side-effects — one row per benchmark::

      benchmark,unsteered_acc,steered_acc

Coherence is carried as **two raw axes matching**
:class:`steerbench.metrics.CoherenceScore`: perplexity (lower = better) and
repetition (higher = worse). No sign convention is imposed on the producer;
:func:`analyze_dose` owns direction explicitly (see ``COHERENCE_DIRECTION``), so
the sweep can dump exactly what ``metrics.py`` returns.

Layering: CSV parsing + aggregation + sweet-spot/cliff analysis are pure and
CPU-testable; plotting is gated behind an optional ``matplotlib`` import and
returns PNG bytes; the markdown/HTML string assembly takes those bytes as input
so it is testable with a fake PNG.
"""

from __future__ import annotations

import base64
import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# How to pick the sweet spot from the effect shift off baseline.
SweetSpotDirection = Literal["increase", "decrease", "abs"]

# Documented coherence direction — analyze_dose keys its floor logic off this
# instead of assuming a higher-is-better score.
COHERENCE_DIRECTION = {
    "perplexity": "lower_is_better",
    "repetition": "higher_is_worse",
}

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MetricStat:
    """Central value and its spread (error-band half-width) for one metric."""

    mean: float
    spread: float


@dataclass(frozen=True)
class SweepPoint:
    """One aggregated grid point. ``x`` is the strength (dose) or layer index."""

    x: float
    effect: MetricStat
    perplexity: MetricStat
    repetition: MetricStat
    n_seeds: int


@dataclass(frozen=True)
class SideEffect:
    """Steered vs unsteered accuracy on one benchmark slice."""

    benchmark: str
    unsteered_acc: float
    steered_acc: float

    @property
    def delta(self) -> float:
        """Signed accuracy change; negative means the vector degraded capability."""
        return self.steered_acc - self.unsteered_acc


@dataclass(frozen=True)
class DoseAnalysis:
    """Sweet spot and cliff located on a dose-response curve.

    ``baseline`` is the unsteered reference (coeff nearest 0); "strength" means
    distance from it, so the analysis works for a two-sided coefficient sweep.
    ``sweet_spot`` is the coherent steered point whose effect shift off baseline
    is most in the requested direction (effect-increasing by default; see
    :func:`analyze_dose`); ``cliff_x`` is the coeff of the nearest-to-baseline point
    at which coherence breaks (the onset of collapse). Either may be ``None`` (no
    coherent steered point / never collapses). Floors are reported so the
    annotation is reproducible.
    """

    baseline: SweepPoint
    sweet_spot: SweepPoint | None
    cliff_x: float | None
    perplexity_floor: float
    repetition_floor: float


@dataclass(frozen=True)
class LayerAnalysis:
    """Layers classified by the effect/coherence tradeoff across the sweep.

    Steerable models tend to show a broad *coherent plateau* of usable layers
    rather than a few sharp peaks, so the principled output is the plateau itself
    (``coherent``, ``plateau_span``) and its ``best_layer``. ``top_coherent`` is a
    convenience *display* headline — the ``top_n`` coherent layers by effect — not
    a claim that only those layers work; on a flat plateau its exact membership is
    seed-noise sensitive, so treat it as "layers worth showing", not "the peaks".

    ``degenerate_traps`` are the seductive failures: incoherent layers whose
    effect looks as strong as the best coherent layer but whose perplexity has
    blown up, so the behaviour is an artifact of broken text (e.g. an early block
    with ppl in the tens). Floors and ``max_layer`` are reported for
    reproducibility and depth.
    """

    coherent: list[SweepPoint]
    top_coherent: list[SweepPoint]
    degenerate_traps: list[SweepPoint]
    perplexity_floor: float
    repetition_floor: float
    max_layer: float

    def depth(self, point: SweepPoint) -> float:
        """Relative depth of ``point`` in the swept range (``layer / max_layer``)."""
        return point.x / self.max_layer if self.max_layer else 0.0

    @property
    def best_layer(self) -> SweepPoint | None:
        """The single most-effective coherent layer (the one to inject at)."""
        return max(self.coherent, key=lambda p: p.effect.mean) if self.coherent else None

    @property
    def plateau_span(self) -> tuple[float, float] | None:
        """(lowest, highest) coherent layer index, or ``None`` if none coherent."""
        if not self.coherent:
            return None
        xs = [p.x for p in self.coherent]
        return min(xs), max(xs)


@dataclass(frozen=True)
class ReportData:
    """Everything the renderer needs, already parsed and analysed."""

    dose: list[SweepPoint]
    analysis: DoseAnalysis
    layer: list[SweepPoint]
    layer_analysis: LayerAnalysis
    side_effects: list[SideEffect]


# --------------------------------------------------------------------------- #
# CSV parsing + aggregation  (pure)
# --------------------------------------------------------------------------- #

# Column names in the M0 sweep CSV. The producer emits one RAW row per seed
# (no pre-aggregated mean/std); this reader computes mean±std across seeds.
# ``effect`` = formality; coherence is the pair (ppl, repetition).
_COL_EFFECT = "formality"
_COL_PERPLEXITY = "ppl"
_COL_REPETITION = "repetition"
_SWEEP_COLUMNS = ("seed", _COL_EFFECT, _COL_REPETITION, _COL_PERPLEXITY)


@dataclass(frozen=True)
class _RawRow:
    """One raw per-seed measurement (a single sample, not aggregated)."""

    x: float
    seed: int
    effect: float
    perplexity: float
    repetition: float


def _read_sweep_rows(path: Path, x_column: str) -> list[_RawRow]:
    """Parse a raw sweep CSV keyed on ``x_column`` (``coeff`` for dose)."""
    rows: list[_RawRow] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {x_column, *_SWEEP_COLUMNS} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(
                _RawRow(
                    x=float(row[x_column]),
                    seed=int(row["seed"]),
                    effect=float(row[_COL_EFFECT]),
                    perplexity=float(row[_COL_PERPLEXITY]),
                    repetition=float(row[_COL_REPETITION]),
                )
            )
    return rows


def _combine(values: list[float]) -> MetricStat:
    """Aggregate raw per-seed samples into mean and seed-to-seed std.

    ``spread`` is the population std across seeds (``0.0`` for a single seed) —
    the error band the report draws.
    """
    mean = statistics.fmean(values)
    spread = statistics.pstdev(values) if len(values) > 1 else 0.0
    return MetricStat(mean=mean, spread=spread)


def _aggregate(rows: list[_RawRow]) -> list[SweepPoint]:
    """Group raw rows by ``x`` and aggregate across seeds; sorted by ``x``."""
    groups: dict[float, list[_RawRow]] = defaultdict(list)
    for row in rows:
        groups[row.x].append(row)
    points = [
        SweepPoint(
            x=x,
            effect=_combine([r.effect for r in group]),
            perplexity=_combine([r.perplexity for r in group]),
            repetition=_combine([r.repetition for r in group]),
            n_seeds=len({r.seed for r in group}),
        )
        for x, group in groups.items()
    ]
    return sorted(points, key=lambda p: p.x)


def load_dose_curve(path: Path) -> list[SweepPoint]:
    """Load and aggregate the dose-response CSV (keyed on ``coeff``)."""
    return _aggregate(_read_sweep_rows(path, "coeff"))


def load_layer_curve(path: Path, x_column: str = "layer") -> list[SweepPoint]:
    """Load and aggregate the layer-sweep CSV (keyed on ``layer``).

    Columns ``layer,layer_pos,seed,dir_norm,formality,repetition,ppl``;
    ``layer_pos``/``dir_norm`` are ignored (extra columns tolerated). Pair with
    :func:`analyze_layers` to locate coherent peaks and degenerate traps.
    """
    return _aggregate(_read_sweep_rows(path, x_column))


def load_side_effects(path: Path) -> list[SideEffect]:
    """Parse the side-effect CSV (benchmark, unsteered_acc, steered_acc)."""
    effects: list[SideEffect] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"benchmark", "unsteered_acc", "steered_acc"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            effects.append(
                SideEffect(
                    benchmark=row["benchmark"],
                    unsteered_acc=float(row["unsteered_acc"]),
                    steered_acc=float(row["steered_acc"]),
                )
            )
    return effects


# --------------------------------------------------------------------------- #
# Dose-response analysis  (pure) — owns coherence direction
# --------------------------------------------------------------------------- #


def analyze_dose(
    curve: list[SweepPoint],
    perplexity_tol: float = 0.5,
    repetition_cap: float = 0.5,
    direction: SweetSpotDirection = "increase",
) -> DoseAnalysis:
    """Locate the sweet spot and the coherence cliff on a dose-response curve.

    The baseline is the unsteered point (coeff nearest 0); it sets the reference
    perplexity. "Strength" is distance from baseline, so a two-sided coefficient
    sweep is handled symmetrically. A point is *coherent* when its perplexity
    stays within ``perplexity_tol`` of baseline (``ppl <= baseline * (1 + tol)``,
    since lower is better) **and** its repetition stays at or below
    ``repetition_cap`` (since higher is worse). Both axes are checked because the
    cliff can appear in either.

    * sweet spot — the coherent steered point whose effect shift off baseline is
      most in the requested ``direction``:

      - ``"increase"`` (default) — largest **positive** shift (steer toward the
        concept). This is the usual intent: on a near-symmetric curve the older
        direction-agnostic rule could mark the effect-*decreasing* side.
      - ``"decrease"`` — largest **negative** shift (steer away).
      - ``"abs"`` — largest shift either way (the previous behaviour).

    * cliff — the coeff of the nearest-to-baseline steered point that is *not*
      coherent (the onset of collapse).
    """
    if not curve:
        raise ValueError("cannot analyse an empty dose-response curve")
    if direction not in ("increase", "decrease", "abs"):
        raise ValueError(f"unknown sweet-spot direction: {direction!r}")

    baseline = min(curve, key=lambda p: abs(p.x))
    perplexity_floor = baseline.perplexity.mean * (1.0 + perplexity_tol)

    def coherent(point: SweepPoint) -> bool:
        return point.perplexity.mean <= perplexity_floor and point.repetition.mean <= repetition_cap

    def strength(point: SweepPoint) -> float:
        return abs(point.x - baseline.x)

    def shift(point: SweepPoint) -> float:
        return point.effect.mean - baseline.effect.mean

    steered = [p for p in curve if p.x != baseline.x]
    coherent_steered = [p for p in steered if coherent(p)]
    incoherent_steered = [p for p in steered if not coherent(p)]

    if not coherent_steered:
        sweet_spot = None
    elif direction == "decrease":
        sweet_spot = min(coherent_steered, key=shift)
    elif direction == "abs":
        sweet_spot = max(coherent_steered, key=lambda p: abs(shift(p)))
    else:  # "increase"
        sweet_spot = max(coherent_steered, key=shift)
    cliff = min(incoherent_steered, key=strength, default=None)

    return DoseAnalysis(
        baseline=baseline,
        sweet_spot=sweet_spot,
        cliff_x=cliff.x if cliff is not None else None,
        perplexity_floor=perplexity_floor,
        repetition_floor=repetition_cap,
    )


def peak_effect_shift(curve: list[SweepPoint], baseline: SweepPoint) -> SweepPoint | None:
    """Point with the largest effect shift from ``baseline``, or ``None`` if empty."""
    return max(curve, key=lambda p: abs(p.effect.mean - baseline.effect.mean)) if curve else None


def analyze_layers(
    curve: list[SweepPoint],
    perplexity_tol: float = 1.0,
    repetition_cap: float = 0.5,
    top_n: int = 3,
) -> LayerAnalysis:
    """Find the coherent layer plateau and flag degenerate-trap layers.

    The best-case coherence in the sweep (minimum perplexity across layers) sets
    the reference; a layer is *coherent* when its perplexity stays within
    ``perplexity_tol`` of it and its repetition stays at or below
    ``repetition_cap``. The tolerance is deliberately generous (default 1.0) so
    fully-fluent layers on the plateau are not misfiled as broken — a trap means
    catastrophic blowup, not a hair above the tightest floor.

    * ``coherent`` — the usable plateau (all coherent layers).
    * ``top_coherent`` — the ``top_n`` coherent layers by effect, a *display*
      headline only (membership is noise-sensitive on a flat plateau).
    * ``degenerate_traps`` — incoherent layers whose effect equals or exceeds the
      best coherent effect: high effect bought with broken text. This is the
      generalising signal, independent of ``top_n``.
    """
    if not curve:
        raise ValueError("cannot analyse an empty layer-sweep curve")

    perplexity_floor = min(p.perplexity.mean for p in curve) * (1.0 + perplexity_tol)

    def coherent(point: SweepPoint) -> bool:
        return point.perplexity.mean <= perplexity_floor and point.repetition.mean <= repetition_cap

    coherent_points = sorted((p for p in curve if coherent(p)), key=lambda p: p.x)
    best_coherent_effect = (
        max(p.effect.mean for p in coherent_points)
        if coherent_points
        else max(p.effect.mean for p in curve)
    )

    top_coherent = sorted(
        sorted(coherent_points, key=lambda p: p.effect.mean, reverse=True)[:top_n],
        key=lambda p: p.x,
    )
    degenerate_traps = sorted(
        (p for p in curve if not coherent(p) and p.effect.mean >= best_coherent_effect),
        key=lambda p: p.x,
    )

    return LayerAnalysis(
        coherent=coherent_points,
        top_coherent=top_coherent,
        degenerate_traps=degenerate_traps,
        perplexity_floor=perplexity_floor,
        repetition_floor=repetition_cap,
        max_layer=max(p.x for p in curve),
    )


# --------------------------------------------------------------------------- #
# Plotting  (gated behind matplotlib; returns PNG bytes)
# --------------------------------------------------------------------------- #


def plot_dose_response(data: ReportData) -> bytes:
    """Render the hero figure: effect and perplexity vs strength as PNG bytes.

    Effect on the left axis, perplexity (the coherence axis) on the right, with
    the sweet spot and cliff drawn as vertical guides so the tradeoff is legible.
    ``matplotlib`` is imported lazily (optional ``report`` extra).
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed on Modal or CI
    import matplotlib.pyplot as plt

    curve = data.dose
    xs = [p.x for p in curve]
    effects = [p.effect.mean for p in curve]
    effect_err = [p.effect.spread for p in curve]
    ppl = [p.perplexity.mean for p in curve]
    ppl_err = [p.perplexity.spread for p in curve]

    fig, ax_effect = plt.subplots(figsize=(7.0, 4.2))
    ax_effect.errorbar(xs, effects, yerr=effect_err, color="#1f77b4", marker="o", label="effect")
    ax_effect.set_xlabel("injection coefficient")
    ax_effect.set_ylabel("effect (behaviour score)", color="#1f77b4")
    ax_effect.tick_params(axis="y", labelcolor="#1f77b4")

    ax_ppl = ax_effect.twinx()
    ax_ppl.errorbar(xs, ppl, yerr=ppl_err, color="#d62728", marker="s", label="perplexity")
    ax_ppl.set_ylabel("perplexity (lower = more coherent)", color="#d62728")
    ax_ppl.tick_params(axis="y", labelcolor="#d62728")

    if data.analysis.sweet_spot is not None:
        ax_effect.axvline(
            data.analysis.sweet_spot.x, color="#2ca02c", linestyle="--", label="sweet spot"
        )
    if data.analysis.cliff_x is not None:
        ax_effect.axvline(
            data.analysis.cliff_x, color="#7f7f7f", linestyle=":", label="coherence cliff"
        )

    lines_effect, labels_effect = ax_effect.get_legend_handles_labels()
    lines_ppl, labels_ppl = ax_ppl.get_legend_handles_labels()
    ax_effect.legend(lines_effect + lines_ppl, labels_effect + labels_ppl, loc="best", fontsize=8)
    ax_effect.set_title("Dose-response: effect vs coherence")
    fig.tight_layout()
    return _figure_to_png(fig)


def plot_layer_sensitivity(data: ReportData) -> bytes:
    """Render layer sensitivity: effect and perplexity vs layer as PNG bytes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = data.layer
    xs = [p.x for p in curve]
    effects = [p.effect.mean for p in curve]
    effect_err = [p.effect.spread for p in curve]
    ppl = [p.perplexity.mean for p in curve]

    fig, ax_effect = plt.subplots(figsize=(7.0, 4.2))
    ax_effect.errorbar(xs, effects, yerr=effect_err, color="#1f77b4", marker="o", label="effect")
    ax_effect.set_xlabel("injection layer")
    ax_effect.set_ylabel("effect (behaviour score)", color="#1f77b4")
    ax_effect.tick_params(axis="y", labelcolor="#1f77b4")

    ax_ppl = ax_effect.twinx()
    ax_ppl.plot(xs, ppl, color="#d62728", marker="s", label="perplexity")
    ax_ppl.set_ylabel("perplexity (lower = more coherent)", color="#d62728")
    ax_ppl.tick_params(axis="y", labelcolor="#d62728")

    # Shade the coherent plateau, mark the top coherent layers, flag the traps.
    layers = data.layer_analysis
    span = layers.plateau_span
    if span is not None:
        ax_effect.axvspan(span[0], span[1], color="#2ca02c", alpha=0.07, label="coherent plateau")
    top_xs = [p.x for p in layers.top_coherent]
    top_ys = [p.effect.mean for p in layers.top_coherent]
    if top_xs:
        ax_effect.scatter(
            top_xs,
            top_ys,
            s=140,
            facecolors="none",
            edgecolors="#2ca02c",
            linewidths=2,
            zorder=5,
            label="top coherent",
        )
    for trap in layers.degenerate_traps:
        ax_effect.scatter(
            [trap.x],
            [trap.effect.mean],
            marker="x",
            s=90,
            color="#d62728",
            zorder=6,
            label="degenerate trap",
        )
        ax_effect.annotate(
            f"trap (ppl {trap.perplexity.mean:.0f})",
            (trap.x, trap.effect.mean),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            color="#d62728",
        )

    lines_effect, labels_effect = ax_effect.get_legend_handles_labels()
    # de-duplicate repeated trap labels
    seen: dict[str, object] = {}
    for line, label in zip(lines_effect, labels_effect, strict=True):
        seen.setdefault(label, line)
    ax_effect.legend(list(seen.values()), list(seen.keys()), loc="best", fontsize=8)
    ax_effect.set_title("Layer sensitivity")
    fig.tight_layout()
    return _figure_to_png(fig)


def _figure_to_png(fig: object) -> bytes:
    """Serialise a matplotlib figure to PNG bytes and close it."""
    from io import BytesIO

    import matplotlib.pyplot as plt

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)  # type: ignore[attr-defined]
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Rendering  (pure string assembly; PNGs injected as input)
# --------------------------------------------------------------------------- #


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _effect_summary_lines(data: ReportData) -> list[str]:
    """Human sentences describing baseline, peak shift + sweet spot, for md/html."""
    lines: list[str] = []
    baseline = data.analysis.baseline
    lines.append(
        f"Baseline (coeff {_fmt(baseline.x)}): effect {_fmt(baseline.effect.mean)}, "
        f"perplexity {_fmt(baseline.perplexity.mean)}."
    )
    peak = peak_effect_shift(data.dose, baseline)
    if peak is not None and peak.x != baseline.x:
        shift = peak.effect.mean - baseline.effect.mean
        lines.append(
            f"Peak effect shift {_fmt(shift)} (±{_fmt(peak.effect.spread)}) "
            f"at coeff {_fmt(peak.x)}."
        )
    spot = data.analysis.sweet_spot
    if spot is not None:
        shift = spot.effect.mean - baseline.effect.mean
        lines.append(
            f"Sweet spot at coeff {_fmt(spot.x)}: effect shift {_fmt(shift)}, "
            f"perplexity {_fmt(spot.perplexity.mean)}."
        )
    else:
        lines.append("No coherent steered point found — every dose breaks coherence.")
    if data.analysis.cliff_x is not None:
        lines.append(f"Coherence cliff onset at coeff {_fmt(data.analysis.cliff_x)}.")
    else:
        lines.append("No coherence cliff within the swept range.")
    return lines


def _layer_peak_lines(data: ReportData) -> list[str]:
    """Sentences describing the coherent plateau, best layer + top-N, for md/html."""
    layers = data.layer_analysis
    span = layers.plateau_span
    best = layers.best_layer
    if span is None or best is None:
        return ["No coherent layer found — every layer breaks coherence."]
    lo, hi = span
    lines = [
        f"Coherent plateau: layers {int(lo)}–{int(hi)} "
        f"({len(layers.coherent)} of {len(data.layer)} layers); best effect at layer "
        f"{int(best.x)} (depth {_fmt(layers.depth(best), 2)}, effect {_fmt(best.effect.mean)}, "
        f"perplexity {_fmt(best.perplexity.mean)})."
    ]
    top = ", ".join(f"layer {int(p.x)}" for p in layers.top_coherent)
    lines.append(f"Top {len(layers.top_coherent)} coherent layers by effect: {top}.")
    return lines


def _layer_trap_lines(data: ReportData) -> list[str]:
    """Warning sentences naming degenerate-trap layers, for md/html."""
    layers = data.layer_analysis
    return [
        f"Degenerate trap at layer {int(p.x)} (depth {_fmt(layers.depth(p), 2)}): "
        f"effect {_fmt(p.effect.mean)} looks strong but perplexity {_fmt(p.perplexity.mean)} "
        f"means it is an artifact of broken text, not real steering."
        for p in layers.degenerate_traps
    ]


def render_markdown(data: ReportData, dose_png_name: str, layer_png_name: str) -> str:
    """Assemble the markdown report; images referenced as sidecar PNG files."""
    out: list[str] = ["# Steering report card", ""]

    # Hero first — dose-response above the fold.
    out += ["## Dose-response (hero)", "", f"![dose-response]({dose_png_name})", ""]
    out += [f"- {line}" for line in _effect_summary_lines(data)]
    out.append("")

    out += ["## Effect size", ""]
    out += ["| coeff | effect | ± | perplexity | repetition |", "|---|---|---|---|---|"]
    for p in data.dose:
        out.append(
            f"| {_fmt(p.x)} | {_fmt(p.effect.mean)} | {_fmt(p.effect.spread)} "
            f"| {_fmt(p.perplexity.mean)} | {_fmt(p.repetition.mean)} |"
        )
    out.append("")

    out += ["## Side effects (steered vs unsteered)", ""]
    out += ["| benchmark | unsteered | steered | Δ |", "|---|---|---|---|"]
    for eff in data.side_effects:
        out.append(
            f"| {eff.benchmark} | {_fmt(eff.unsteered_acc)} | {_fmt(eff.steered_acc)} "
            f"| {_fmt(eff.delta)} |"
        )
    out.append("")

    out += ["## Layer sensitivity", "", f"![layer sensitivity]({layer_png_name})", ""]
    out += [f"- {line}" for line in _layer_peak_lines(data)]
    out += [f"- ⚠️ {line}" for line in _layer_trap_lines(data)]
    out.append("")
    return "\n".join(out)


_HTML_CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
       max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
h1 { margin-bottom: 0.2rem; }
.hero { border: 1px solid #ccc; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
.hero img, section img { max-width: 100%; height: auto; }
.summary { margin: 0.5rem 0 0; padding-left: 1.2rem; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.neg { color: #d62728; }
.pos { color: #2ca02c; }
.peaks { margin: 0.5rem 0 0; padding-left: 1.2rem; }
.trap { border-left: 4px solid #d62728; background: rgba(214,39,40,0.08);
        padding: 0.5rem 0.8rem; margin: 0.5rem 0; }
""".strip()


def _png_data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_html(data: ReportData, dose_png: bytes, layer_png: bytes) -> str:
    """Assemble a self-contained HTML report (inline CSS, base64-embedded PNGs).

    No external stylesheet or CDN. The dose-response hero sits directly under the
    title, above the fold; the PNGs passed in are embedded as data URIs.
    """
    summary = "".join(f"<li>{_html_escape(line)}</li>" for line in _effect_summary_lines(data))

    dose_rows = "".join(
        f"<tr><td>{_fmt(p.x)}</td><td>{_fmt(p.effect.mean)}</td>"
        f"<td>{_fmt(p.effect.spread)}</td><td>{_fmt(p.perplexity.mean)}</td>"
        f"<td>{_fmt(p.repetition.mean)}</td></tr>"
        for p in data.dose
    )
    side_rows = "".join(
        f"<tr><td>{_html_escape(e.benchmark)}</td><td>{_fmt(e.unsteered_acc)}</td>"
        f"<td>{_fmt(e.steered_acc)}</td>"
        f'<td class="{"neg" if e.delta < 0 else "pos"}">{_fmt(e.delta)}</td></tr>'
        for e in data.side_effects
    )
    peaks = "".join(f"<li>{_html_escape(line)}</li>" for line in _layer_peak_lines(data))
    traps = "".join(
        f'<div class="trap">⚠️ {_html_escape(line)}</div>' for line in _layer_trap_lines(data)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Steering report card</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<h1>Steering report card</h1>
<div class="hero">
<h2>Dose-response</h2>
<img alt="dose-response curve" src="{_png_data_uri(dose_png)}">
<ul class="summary">{summary}</ul>
</div>
<section>
<h2>Effect size</h2>
<table>
<thead><tr><th>coeff</th><th>effect</th><th>&plusmn;</th>
<th>perplexity</th><th>repetition</th></tr></thead>
<tbody>{dose_rows}</tbody>
</table>
</section>
<section>
<h2>Side effects (steered vs unsteered)</h2>
<table>
<thead><tr><th>benchmark</th><th>unsteered</th><th>steered</th><th>&Delta;</th></tr></thead>
<tbody>{side_rows}</tbody>
</table>
</section>
<section>
<h2>Layer sensitivity</h2>
<img alt="layer sensitivity" src="{_png_data_uri(layer_png)}">
<ul class="peaks">{peaks}</ul>
{traps}
</section>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def build_report(
    dose_csv: Path,
    layer_csv: Path,
    side_csv: Path,
    out_dir: Path,
    stem: str = "report",
) -> dict[str, Path]:
    """Load CSVs, plot, and write ``<stem>.md`` + ``<stem>.html`` (+ PNGs).

    Requires ``matplotlib``. Returns the written paths keyed by artefact
    (``markdown``, ``html``, ``dose_png``, ``layer_png``). The markdown
    references the sidecar PNGs; the HTML embeds them so it travels as one file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dose = load_dose_curve(dose_csv)
    layer = load_layer_curve(layer_csv)
    side_effects = load_side_effects(side_csv)
    data = ReportData(
        dose=dose,
        analysis=analyze_dose(dose),
        layer=layer,
        layer_analysis=analyze_layers(layer),
        side_effects=side_effects,
    )

    dose_png = plot_dose_response(data)
    layer_png = plot_layer_sensitivity(data)
    dose_png_path = out_dir / f"{stem}_dose.png"
    layer_png_path = out_dir / f"{stem}_layer.png"
    dose_png_path.write_bytes(dose_png)
    layer_png_path.write_bytes(layer_png)

    md_path = out_dir / f"{stem}.md"
    html_path = out_dir / f"{stem}.html"
    md_path.write_text(render_markdown(data, dose_png_path.name, layer_png_path.name))
    html_path.write_text(render_html(data, dose_png, layer_png))

    return {
        "markdown": md_path,
        "html": html_path,
        "dose_png": dose_png_path,
        "layer_png": layer_png_path,
    }
