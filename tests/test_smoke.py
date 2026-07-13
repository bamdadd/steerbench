"""Smoke test: the package imports and exposes its version."""

import steerbench


def test_import() -> None:
    assert steerbench.__version__
