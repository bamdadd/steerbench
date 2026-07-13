"""CLI smoke test: the CPU read-existing-CSV path renders a card end to end."""

from __future__ import annotations

from pathlib import Path

import pytest

from steerbench import cli

_ROOT = Path(__file__).resolve().parent.parent
_DOSE = _ROOT / "artifacts" / "dose_response.csv"
_LAYER = _ROOT / "artifacts" / "layer_sweep.csv"


def test_cli_renders_from_real_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "card"
    rc = cli.main(
        [
            "--dose-csv",
            str(_DOSE),
            "--layer-csv",
            str(_LAYER),
            "--out",
            str(out),
            "--stem",
            "report",
        ]
    )
    assert rc == 0
    assert (out / "report.md").exists()
    assert (out / "report.html").exists()
    assert (out / "report_dose.png").exists()
    assert (out / "report_layer.png").exists()
    # side-effects stub was synthesised (no benchmark CSV in M0 artifacts).
    assert (out / "side_effects.csv").read_text().startswith("benchmark,")


def test_cli_errors_on_missing_csv(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--dose-csv", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "o")])
