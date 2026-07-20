"""steer-report — emit an HTML/markdown report card for a steering vector.

Primary path is CPU-only and reads *existing* sweep CSVs (produced by the
GPU/Modal harness in ``experiments/modal_app.py``): load the vector for its
provenance + per-layer norms, parse the dose-response and layer-sweep CSVs,
analyse them, and render the four-part card as markdown + self-contained HTML.

Only the CSV → report path is needed for a report; it requires
``steerbench[report]`` (matplotlib). Passing a vector to summarise additionally
needs ``steerbench[vectors]`` (torch + gguf). The heavy model/Modal work lives
behind ``steerbench[gpu]`` and is reached only via ``--run`` (which shells out
to ``modal``; core never imports the harness).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import report

_DEFAULT_DOSE_CSV = Path("artifacts/dose_response.csv")
_DEFAULT_LAYER_CSV = Path("artifacts/layer_sweep.csv")


def _summarise_vector(path: Path) -> None:
    """Load a steering vector and print provenance + per-layer L2 norms.

    Imported lazily so the CSV → report path works without torch/gguf.
    """
    from .vectors import load_vector

    vec = load_vector(path)
    print(f"[steerbench] vector: {path}")
    if vec.model_id:
        print(f"  source model : {vec.model_id}")
    if vec.concept:
        print(f"  concept      : {vec.concept}")
    norms = vec.layer_norms()
    print(f"  layers       : {len(norms)} ({min(norms)}..{max(norms)})")
    for layer in sorted(norms):
        print(f"    L{layer:<3d} ‖dir‖ = {norms[layer]:.4f}")


def _ensure_side_csv(side_csv: Path | None, out_dir: Path) -> Path:
    """Return a side-effects CSV, synthesising a header-only stub if absent.

    The M0 sweep does not yet emit a side-effects benchmark file; ``build_report``
    requires one, so we write an empty (header-only) CSV that renders as an empty
    "Side effects" table rather than failing.
    """
    if side_csv is not None and side_csv.exists():
        return side_csv
    out_dir.mkdir(parents=True, exist_ok=True)
    stub = out_dir / "side_effects.csv"
    stub.write_text("benchmark,unsteered_acc,steered_acc\n")
    return stub


def _run_modal(function: str, extra: list[str]) -> int:
    """Shell out to the Modal harness (``steerbench[gpu]``). Core never imports it."""
    app = Path("experiments/modal_app.py")
    if not app.exists():
        print(f"[steerbench] {app} not found", file=sys.stderr)
        return 1
    cmd = ["modal", "run", f"{app}::{function}", *extra]
    print(f"[steerbench] $ {' '.join(cmd)}")
    return subprocess.call(cmd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steer-report",
        description="Render a steering-vector report card from existing sweep CSVs.",
    )
    parser.add_argument(
        "vector",
        nargs="?",
        type=Path,
        help="optional path to a steering vector (.gguf or .pt) to summarise",
    )
    parser.add_argument("--model", help="HF model id (recorded in output; informational)")
    parser.add_argument(
        "--dose-csv", type=Path, default=_DEFAULT_DOSE_CSV, help="dose-response sweep CSV"
    )
    parser.add_argument(
        "--layer-csv", type=Path, default=_DEFAULT_LAYER_CSV, help="layer-sweep CSV"
    )
    parser.add_argument(
        "--side-csv",
        type=Path,
        default=None,
        help="side-effects CSV (benchmark,unsteered_acc,steered_acc); optional",
    )
    parser.add_argument("--out", type=Path, default=Path("report_out"), help="output directory")
    parser.add_argument("--stem", default="report", help="output filename stem")
    parser.add_argument(
        "--run",
        metavar="FUNCTION",
        help="run a Modal harness function (needs steerbench[gpu]) instead of rendering",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the written artifact paths as a single JSON object (suppresses human text)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, extra = parser.parse_known_args(argv)

    if args.run:
        return _run_modal(args.run, extra)

    if args.vector is not None:
        _summarise_vector(args.vector)

    for label, csv_path in (("dose-response", args.dose_csv), ("layer-sweep", args.layer_csv)):
        if not csv_path.exists():
            parser.error(f"{label} CSV not found: {csv_path}")

    # An explicitly named side CSV must exist; only an *unspecified* one may
    # fall back to the header-only stub.
    if args.side_csv is not None and not args.side_csv.exists():
        parser.error(f"side-effects CSV not found: {args.side_csv}")

    side_csv = _ensure_side_csv(args.side_csv, args.out)
    outputs = report.build_report(
        dose_csv=args.dose_csv,
        layer_csv=args.layer_csv,
        side_csv=side_csv,
        out_dir=args.out,
        stem=args.stem,
    )
    if args.json:
        print(json.dumps({kind: str(path) for kind, path in outputs.items()}))
    else:
        print("[steerbench] wrote:")
        for kind, path in outputs.items():
            print(f"  {kind:9s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
