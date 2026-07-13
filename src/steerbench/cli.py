"""steer-report — emit an HTML/markdown report card for a steering vector."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="steer-report")
    parser.add_argument("vector", help="path to a steering vector (layer index + tensor)")
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--out", default="report.md", help="output report path")
    args = parser.parse_args(argv)
    # TODO: effect size, side effects (MMLU/GSM8K slices), dose-response,
    # layer sweep -> render report card. Consume repeng vectors as-is.
    print(f"[steerbench] TODO report for {args.vector} on {args.model} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
