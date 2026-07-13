"""CPU-only, model-free tests for steerbench.vectors.

Uses tiny synthetic directions. No GPU, no model download, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# torch is an optional dep (steerbench[vectors]); skip the whole module without
# it. steerbench.vectors imports torch at import time, so guard before importing.
pytest.importorskip("torch")

import torch  # noqa: E402

from steerbench.vectors import (  # noqa: E402
    SteeringVector,
    dose,
    load_vector,
    normalize_alpha,
    save_vector,
)


def _synthetic() -> SteeringVector:
    return SteeringVector(
        directions={
            5: torch.tensor([3.0, 4.0]),  # L2 norm 5
            6: torch.tensor([1.0, 0.0, 0.0, 0.0]),  # L2 norm 1
        },
        model_id="gpt2",
        concept="honesty",
        repeng_version="0.19.0",
    )


def _assert_same_directions(a: SteeringVector, b: SteeringVector) -> None:
    assert a.directions.keys() == b.directions.keys()
    for layer in a.directions:
        assert torch.equal(a.directions[layer], b.directions[layer].to(a.directions[layer].dtype))


def test_layer_norms() -> None:
    norms = _synthetic().layer_norms()
    assert norms[5] == pytest.approx(5.0)
    assert norms[6] == pytest.approx(1.0)


def test_round_trip_gguf(tmp_path: Path) -> None:
    pytest.importorskip("gguf")
    original = _synthetic()
    path = tmp_path / "vec.gguf"
    save_vector(original, path)

    loaded = load_vector(path)
    _assert_same_directions(original, loaded)
    assert loaded.model_id == "gpt2"
    assert loaded.concept == "honesty"
    assert loaded.repeng_version == "0.19.0"


def test_load_native_repeng_gguf(tmp_path: Path) -> None:
    """Headline claim: ingest a file written with repeng's exact calls.

    Constructs the GGUF the way ``ControlVector.export_gguf`` does — no
    ``steerbench.*`` KV — and checks ``load_vector`` reads it.
    """
    gguf = pytest.importorskip("gguf")
    import numpy as np

    path = tmp_path / "native.gguf"
    writer = gguf.GGUFWriter(str(path), "controlvector")
    writer.add_string("controlvector.model_hint", "llama")
    writer.add_uint32("controlvector.layer_count", 2)
    writer.add_tensor("direction.10", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    writer.add_tensor("direction.11", np.array([-1.0, 0.0], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    loaded = load_vector(path)
    assert loaded.layers == [10, 11]
    assert torch.equal(loaded.directions[10], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(loaded.directions[11], torch.tensor([-1.0, 0.0]))
    assert loaded.model_id == "llama"
    # repeng has no concept / version fields → stay None on native load.
    assert loaded.concept is None
    assert loaded.repeng_version is None


def test_pt_fallback(tmp_path: Path) -> None:
    path = tmp_path / "vec.pt"
    torch.save({5: torch.tensor([3.0, 4.0]), 6: torch.tensor([1.0, 0.0, 0.0, 0.0])}, path)

    loaded = load_vector(path)
    assert loaded.layers == [5, 6]
    assert loaded.layer_norms()[5] == pytest.approx(5.0)
    assert loaded.model_id is None


def test_pt_fallback_rejects_non_dict(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(torch.tensor([1.0, 2.0]), path)
    with pytest.raises(ValueError, match="expected a dict"):
        load_vector(path)


def test_normalize_alpha_by_vector_norm() -> None:
    vec = _synthetic()
    result = normalize_alpha(vec, layer=5, alpha=2.0)
    assert result.raw == 2.0
    assert result.vector_norm == pytest.approx(5.0)
    assert result.by_vector_norm == pytest.approx(10.0)  # 2 * ‖d‖=5
    assert result.by_residual_norm is None


def test_normalize_alpha_dose_transfers_across_models() -> None:
    """Same dose + different ‖d‖ and ‖residual‖ → equal normalized alpha.

    This pins the formula direction: dose = alpha * ‖d‖ / ‖residual‖.
    """
    dose = 0.25

    # Model A: ‖d‖=5 (layer 5), residual norm 20 → raw alpha to hit `dose`.
    vec_a = _synthetic()
    norm_a, resid_a = 5.0, 20.0
    alpha_a = dose * resid_a / norm_a

    # Model B: different scales entirely.
    vec_b = SteeringVector(directions={0: torch.tensor([2.0, 0.0])})  # ‖d‖=2
    norm_b, resid_b = 2.0, 7.0
    alpha_b = dose * resid_b / norm_b

    got_a = normalize_alpha(vec_a, layer=5, alpha=alpha_a, residual_norm=resid_a)
    got_b = normalize_alpha(vec_b, layer=0, alpha=alpha_b, residual_norm=resid_b)

    assert got_a.by_residual_norm == pytest.approx(dose)
    assert got_b.by_residual_norm == pytest.approx(dose)
    assert got_a.by_residual_norm == pytest.approx(got_b.by_residual_norm)


def test_dose_scalar_matches_normalize_alpha() -> None:
    vec = _synthetic()  # layer 5 ‖d‖=5
    got = dose(vec, layer=5, alpha=2.0, residual_norm=20.0)
    assert got == pytest.approx(2.0 * 5.0 / 20.0)  # alpha * ‖d‖ / ‖residual‖
    assert got == pytest.approx(
        normalize_alpha(vec, layer=5, alpha=2.0, residual_norm=20.0).by_residual_norm
    )


def test_gguf_superset_loads_in_repeng(tmp_path: Path) -> None:
    """Files we save must remain natively loadable by repeng's reader shape.

    We can't import repeng here, but we replicate its ``import_gguf`` decode:
    arch, model_hint, and ``direction.{n}`` tensors must all be present and
    correct despite our extra ``steerbench.*`` KV.
    """
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "vec.gguf"
    save_vector(_synthetic(), path)

    reader = gguf.GGUFReader(str(path))
    arch = str(bytes(reader.get_field("general.architecture").parts[-1]), encoding="utf-8")
    assert arch == "controlvector"
    hint = str(bytes(reader.get_field("controlvector.model_hint").parts[-1]), encoding="utf-8")
    assert hint == "gpt2"
    names = {t.name for t in reader.tensors}
    assert names == {"direction.5", "direction.6"}
