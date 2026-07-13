"""CPU-only, fast tests for the report-card renderer.

Parsing, aggregation and sweet-spot/cliff analysis run on canned CSV fixtures;
the markdown/HTML assembly is tested with fake PNG bytes. matplotlib is never
imported (the plot functions are exercised only if it happens to be installed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from steerbench.report import (
    MetricStat,
    ReportData,
    SweepPoint,
    analyze_dose,
    analyze_layers,
    load_dose_curve,
    load_layer_curve,
    load_side_effects,
    peak_effect_shift,
    render_html,
    render_markdown,
)

# Real M0 schema: raw one-row-per-seed. effect=formality; coherence=(ppl, rep).
# coeff 0 is the unsteered baseline; coeffs are two-sided. Effect shifts away
# from baseline both ways; perplexity holds until coeff 40, the cliff.
DOSE_CSV = """coeff,seed,alpha_norm,formality,repetition,ppl
-20.0,0,-0.04,1.9,0.09,11.5
-20.0,1,-0.04,2.0,0.10,12.0
-20.0,2,-0.04,2.1,0.11,12.5
0.0,0,0.0,4.4,0.04,9.8
0.0,1,0.0,4.5,0.05,10.0
0.0,2,0.0,4.6,0.06,10.2
20.0,0,0.04,7.9,0.07,11.5
20.0,1,0.04,8.0,0.08,12.0
20.0,2,0.04,8.1,0.09,12.5
40.0,0,0.08,7.4,0.58,40.0
40.0,1,0.08,7.5,0.60,42.0
40.0,2,0.08,7.6,0.62,44.0
"""

# Real M0 layer schema. Layer 1 is a degenerate trap (highest formality but
# ppl blown up); layers 13/14/20 are coherent peaks; layer 2 is coherent but
# lower effect. Extra cols layer_pos/dir_norm tolerated.
LAYER_CSV = """layer,layer_pos,seed,dir_norm,formality,repetition,ppl
1,1,0,1.0,5.7,0.62,81.0
1,1,1,1.0,6.0,0.60,90.0
1,1,2,1.0,5.5,0.61,72.0
2,2,0,1.0,4.4,0.12,4.8
2,2,1,1.0,4.3,0.10,4.9
2,2,2,1.0,4.5,0.11,5.0
13,13,0,1.0,5.0,0.04,3.9
13,13,1,1.0,5.1,0.05,3.9
13,13,2,1.0,5.0,0.05,4.0
14,14,0,1.0,5.0,0.03,3.5
14,14,1,1.0,5.1,0.04,3.4
14,14,2,1.0,5.0,0.04,3.6
20,20,0,1.0,5.0,0.04,2.8
20,20,1,1.0,5.1,0.05,2.9
20,20,2,1.0,5.0,0.05,2.9
"""

SIDE_CSV = """benchmark,unsteered_acc,steered_acc
mmlu,0.62,0.55
gsm8k,0.40,0.41
"""


@pytest.fixture
def csvs(tmp_path: Path) -> tuple[Path, Path, Path]:
    dose = tmp_path / "dose.csv"
    layer = tmp_path / "layer.csv"
    side = tmp_path / "side.csv"
    dose.write_text(DOSE_CSV)
    layer.write_text(LAYER_CSV)
    side.write_text(SIDE_CSV)
    return dose, layer, side


@pytest.fixture
def data(csvs: tuple[Path, Path, Path]) -> ReportData:
    dose_csv, layer_csv, side_csv = csvs
    dose = load_dose_curve(dose_csv)
    layer = load_layer_curve(layer_csv)
    return ReportData(
        dose=dose,
        analysis=analyze_dose(dose),
        layer=layer,
        layer_analysis=analyze_layers(layer),
        side_effects=load_side_effects(side_csv),
    )


# --------------------------------------------------------------------------- #
# Parsing + aggregation
# --------------------------------------------------------------------------- #


def test_dose_aggregation_across_seeds(csvs: tuple[Path, Path, Path]) -> None:
    dose = load_dose_curve(csvs[0])
    assert [p.x for p in dose] == [-20.0, 0.0, 20.0, 40.0]  # sorted, deduped by coeff
    baseline = dose[1]  # coeff 0
    assert baseline.n_seeds == 3
    assert baseline.effect.mean == pytest.approx(4.5)  # mean of 4.4, 4.5, 4.6
    assert baseline.effect.spread == pytest.approx(0.081649658)  # pstdev across seeds
    assert baseline.perplexity.mean == pytest.approx(10.0)


def test_layer_and_side_parsing(csvs: tuple[Path, Path, Path]) -> None:
    layer = load_layer_curve(csvs[1])  # extra cols layer_pos/dir_norm tolerated
    assert [p.x for p in layer] == [1.0, 2.0, 13.0, 14.0, 20.0]
    assert all(p.n_seeds == 3 for p in layer)
    side = load_side_effects(csvs[2])
    assert side[0].benchmark == "mmlu"
    assert side[0].delta == pytest.approx(-0.07)


def test_missing_column_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("coeff,seed,formality\n0.0,0,0.1\n")  # lacks ppl + repetition
    with pytest.raises(ValueError, match="missing columns"):
        load_dose_curve(bad)


# --------------------------------------------------------------------------- #
# Analysis — sweet spot + cliff
# --------------------------------------------------------------------------- #


def test_analyze_finds_sweet_spot_and_cliff(data: ReportData) -> None:
    analysis = data.analysis
    assert analysis.baseline.x == 0.0  # coeff nearest 0, not the min coeff
    # baseline ppl 10.0, floor = *1.5 = 15.0; coeff 40 (ppl ~42) breaks it
    assert analysis.cliff_x == 40.0
    # largest coherent effect shift from baseline -> coeff 20 (|8.0-4.5|=3.5)
    assert analysis.sweet_spot is not None
    assert analysis.sweet_spot.x == 20.0


def test_analyze_baseline_is_nearest_zero_not_min() -> None:
    def pt(x: float, eff: float, ppl: float) -> SweepPoint:
        return SweepPoint(x, MetricStat(eff, 0.0), MetricStat(ppl, 0.0), MetricStat(0.05, 0.0), 1)

    # all coherent, two-sided: baseline must be coeff 0 despite -30 being smallest
    curve = [pt(-30.0, 1.0, 10.5), pt(0.0, 4.0, 10.0), pt(30.0, 7.0, 11.0)]
    analysis = analyze_dose(curve)
    assert analysis.baseline.x == 0.0
    assert analysis.cliff_x is None
    assert analysis.sweet_spot is not None  # coeff 30: |7-4|=3 > coeff -30: |1-4|=3 tie->first max
    assert analysis.sweet_spot.x in (30.0, -30.0)


def test_analyze_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        analyze_dose([])


def test_peak_effect_shift(data: ReportData) -> None:
    baseline = data.analysis.baseline
    peak = peak_effect_shift(data.dose, baseline)
    assert peak is not None
    assert peak.x == 20.0  # largest |effect - baseline| shift
    assert peak_effect_shift([], baseline) is None


# --------------------------------------------------------------------------- #
# Analysis — layer coherent peaks vs degenerate traps
# --------------------------------------------------------------------------- #


def test_analyze_layers_properties(data: ReportData) -> None:
    la = data.layer_analysis
    coherent_x = {int(p.x) for p in la.coherent}
    trap_x = {int(p.x) for p in la.degenerate_traps}

    # Property: the trap is the catastrophic blowup (layer 1, ppl >> plateau),
    # NOT a fluent layer near the floor. Layer 13 (ppl ~3.9) must never be a trap.
    assert 13 not in trap_x
    assert 13 in coherent_x
    assert trap_x == {1}
    trap = la.degenerate_traps[0]
    assert trap.perplexity.mean > la.perplexity_floor  # incoherent
    best = la.best_layer
    assert best is not None
    assert trap.effect.mean >= best.effect.mean  # seductively as-strong-as-best

    # Property: traps are disjoint from the coherent plateau; best is coherent.
    assert not (trap_x & coherent_x)
    assert int(best.x) in coherent_x
    # top_coherent is a subset of the plateau (display headline, not detection)
    assert {int(p.x) for p in la.top_coherent} <= coherent_x
    assert len(la.top_coherent) == 3


def test_analyze_layers_plateau_and_depth(data: ReportData) -> None:
    la = data.layer_analysis
    assert la.plateau_span == (2.0, 20.0)  # coherent layers span 2..20
    # max layer in the fixture is 20, so layer 20 sits at depth 1.0
    depths = {int(p.x): round(la.depth(p), 2) for p in la.coherent}
    assert depths[20] == 1.0
    assert 0.0 < depths[13] < 1.0
    with pytest.raises(ValueError, match="empty"):
        analyze_layers([])


# --------------------------------------------------------------------------- #
# Rendering — fake PNG bytes, no matplotlib
# --------------------------------------------------------------------------- #

FAKE_PNG = b"\x89PNG\r\n\x1a\nFAKEPNGDATA"


def test_render_markdown_structure(data: ReportData) -> None:
    md = render_markdown(data, "report_dose.png", "report_layer.png")
    # hero appears before the effect-size table (above the fold)
    assert md.index("Dose-response (hero)") < md.index("## Effect size")
    assert "![dose-response](report_dose.png)" in md
    assert "![layer sensitivity](report_layer.png)" in md
    assert "| mmlu |" in md
    assert "Coherence cliff onset at coeff 40.000" in md
    assert "Baseline (coeff 0.000)" in md
    # layer annotations: plateau + top coherent named, trap flagged
    assert "Coherent plateau: layers 2–20" in md
    assert "Top 3 coherent layers by effect:" in md
    assert "Degenerate trap at layer 1" in md
    assert md.index("## Side effects") < md.index("## Layer sensitivity")  # section order


def test_render_html_self_contained(data: ReportData) -> None:
    html = render_html(data, FAKE_PNG, FAKE_PNG)
    import base64

    b64 = base64.b64encode(FAKE_PNG).decode("ascii")
    assert f"data:image/png;base64,{b64}" in html  # embedded, not linked
    assert "http://" not in html and "https://" not in html  # no CDN / external assets
    assert "<link" not in html  # inline CSS only
    # hero above the fold: the dose img precedes the side-effects section
    assert html.index('alt="dose-response curve"') < html.index("Side effects")
    assert 'class="neg"' in html  # mmlu delta is negative
    assert 'class="trap"' in html  # degenerate-trap callout present
    assert "Degenerate trap at layer 1" in html


def test_no_matplotlib_imported() -> None:
    # The parse/analysis/markdown+html path must stay matplotlib-free. Checked in
    # a fresh interpreter: a global sys.modules check is unreliable in a suite
    # where sibling tests (cli) legitimately plot.
    import subprocess
    import sys
    import textwrap

    artifacts = Path(__file__).resolve().parent.parent / "artifacts"
    code = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from steerbench import report as r

        dose = r.load_dose_curve(Path(sys.argv[1]))
        layer = r.load_layer_curve(Path(sys.argv[2]))
        data = r.ReportData(
            dose=dose,
            analysis=r.analyze_dose(dose),
            layer=layer,
            layer_analysis=r.analyze_layers(layer),
            side_effects=[],
        )
        r.render_markdown(data, "d.png", "l.png")
        r.render_html(data, b"png-bytes", b"png-bytes")
        assert "matplotlib" not in sys.modules, "render/analysis path imported matplotlib"
        """
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(artifacts / "dose_response.csv"),
            str(artifacts / "layer_sweep.csv"),
        ],
        check=True,
    )
