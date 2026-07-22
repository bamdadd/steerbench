"""CLI smoke test: the CPU read-existing-CSV path renders a card end to end."""

from __future__ import annotations

import json
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


def test_cli_json_emits_artifact_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
            "--json",
        ]
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    # human "wrote:" lines are suppressed under --json.
    assert "[steerbench] wrote:" not in stdout
    payload = json.loads(stdout)
    for key in ("markdown", "html", "dose_png", "layer_png"):
        assert key in payload, f"missing artifact key: {key}"
        assert Path(payload[key]).exists(), f"artifact path does not exist: {payload[key]}"


def test_cli_errors_on_missing_csv(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--dose-csv", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "o")])


def test_cli_errors_on_missing_named_side_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # An explicitly named --side-csv that doesn't exist must error, not
    # silently fall back to the header-only stub (that's for unspecified only).
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--dose-csv",
                str(_DOSE),
                "--layer-csv",
                str(_LAYER),
                "--side-csv",
                str(tmp_path / "typo.csv"),
                "--out",
                str(tmp_path / "o"),
            ]
        )
    assert exc.value.code != 0
    assert "side-effects CSV not found" in capsys.readouterr().err


@pytest.mark.parametrize("stem", ["", "   "])
def test_cli_errors_on_empty_stem(
    stem: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # An empty or whitespace-only --stem would write cryptic dotfiles
    # (".md", ".html", ".png"); it must fail fast instead.
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--dose-csv",
                str(_DOSE),
                "--layer-csv",
                str(_LAYER),
                "--out",
                str(tmp_path / "o"),
                "--stem",
                stem,
            ]
        )
    assert exc.value.code != 0
    assert "output stem must not be empty" in capsys.readouterr().err
