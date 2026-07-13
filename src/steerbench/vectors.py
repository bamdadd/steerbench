"""Durable steering-vector I/O for steerbench.

Ingests repeng's native ``ControlVector`` save format (a llama.cpp ``.gguf``
file) with no conversion pain, and also accepts a plain ``.pt`` mapping of
``layer -> tensor`` as a fallback. See ``load_vector`` for format detection.

repeng stores directions as ``dict[int, np.ndarray]`` inside a GGUF file with:

* architecture (``general.architecture``) == ``"controlvector"``
* KV field ``controlvector.model_hint`` — the source model's ``model_type``
* KV field ``controlvector.layer_count`` — number of directions
* one tensor per layer named ``direction.{layer}``

We keep directions as ``dict[int, torch.Tensor]`` to match the rest of
steerbench, and write a *superset* of the native format on save: extra
``steerbench.*`` KV fields carry metadata repeng doesn't have (concept, the
repeng version that produced the vector). repeng's ``import_gguf`` ignores
unknown KV, so files we save remain natively loadable by repeng.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import NamedTuple

import torch

# GGUF constants, mirrored from repeng's ``extract.py`` so a round-trip stays
# byte-compatible with ``ControlVector.export_gguf`` / ``import_gguf``.
_ARCH = "controlvector"
_MODEL_HINT_KEY = f"{_ARCH}.model_hint"
_LAYER_COUNT_KEY = f"{_ARCH}.layer_count"
_DIRECTION_PREFIX = "direction."
_CONCEPT_KEY = "steerbench.concept"
_REPENG_VERSION_KEY = "steerbench.repeng_version"

_GGUF_MAGIC = b"GGUF"


@dataclasses.dataclass(frozen=True, eq=False)
class SteeringVector:
    """A concept steering vector: per-layer directions plus provenance.

    ``eq=False`` because the default ``__eq__`` would compare the ``directions``
    dict element-wise and torch raises on ``bool(tensor == tensor)``. Compare
    layers explicitly with :func:`torch.equal` instead.
    """

    directions: dict[int, torch.Tensor]
    model_id: str | None = None
    concept: str | None = None
    repeng_version: str | None = None

    @property
    def layers(self) -> list[int]:
        """Layer indices in ascending order."""
        return sorted(self.directions)

    def layer_norms(self) -> dict[int, float]:
        """Per-layer L2 norm of each direction."""
        return layer_norms(self)


class AlphaNorm(NamedTuple):
    r"""A raw steering coefficient expressed in transferable units.

    Layer index + raw alpha don't transfer across models: repeng directions are
    un-normalized PCA components with arbitrary scale, and residual streams have
    different magnitudes per model. The transferable quantity is the *dose* —
    how hard the residual stream is pushed, relative to its own scale.

    Steering applies ``h += alpha * d`` at a layer, where ``d`` is the raw
    direction. The three units, in order of increasing model-independence::

        raw            = alpha                          # only valid for THIS d
        by_vector_norm = alpha * ‖d‖                    # injected perturbation magnitude
        by_residual_norm = alpha * ‖d‖ / ‖residual‖     # dimensionless dose

    Note the multiplication by ``‖d‖``: ``by_vector_norm`` is the coefficient
    you'd apply to a *unit-normalized* direction ``d/‖d‖`` for the identical
    effect (``alpha * d == (alpha * ‖d‖) * d/‖d‖``), which is why it — not
    ``alpha`` — carries across vectors of differing scale.
    """

    raw: float
    vector_norm: float
    #: ``raw * ‖direction‖`` — magnitude of the injected perturbation, i.e. the
    #: coefficient for a unit-normalized direction. Transfers across vectors of
    #: differing scale.
    by_vector_norm: float
    #: ``by_vector_norm / ‖residual‖`` — dimensionless dose relative to the
    #: residual stream. ``None`` when no residual norm is supplied. This is the
    #: quantity to hold fixed when transferring a dose across models.
    by_residual_norm: float | None


def layer_norms(vector: SteeringVector) -> dict[int, float]:
    """Per-layer L2 norm of a steering vector's directions."""
    return {layer: float(torch.linalg.vector_norm(d)) for layer, d in vector.directions.items()}


def normalize_alpha(
    vector: SteeringVector,
    layer: int,
    alpha: float,
    residual_norm: float | None = None,
) -> AlphaNorm:
    r"""Express a raw steering coefficient in transferable units.

    ``alpha`` scales ``vector.directions[layer]`` at inference time
    (``h += alpha * direction``). Because the direction has an arbitrary norm,
    the injected perturbation magnitude is::

        by_vector_norm = alpha * ‖direction‖

    i.e. we **multiply** by the vector norm (the direction is un-normalized, so
    ``alpha`` alone is meaningless across vectors). Supplying ``residual_norm``
    additionally divides by the residual-stream norm to yield the dimensionless,
    model-agnostic dose::

        by_residual_norm = alpha * ‖direction‖ / ‖residual‖

    Returns all three units. See :func:`dose` for the scalar full dose.
    """
    norm = float(torch.linalg.vector_norm(vector.directions[layer]))
    by_vector = alpha * norm
    by_residual = by_vector / residual_norm if residual_norm is not None else None
    return AlphaNorm(
        raw=alpha,
        vector_norm=norm,
        by_vector_norm=by_vector,
        by_residual_norm=by_residual,
    )


def dose(vector: SteeringVector, layer: int, alpha: float, residual_norm: float) -> float:
    r"""Full transferable dose for a raw coefficient at a layer.

    ``dose = alpha * ‖direction‖ / ‖residual‖`` — the injected perturbation
    magnitude relative to the residual stream. This is the quantity to hold
    fixed when moving a steering setting across models; residual-norm division
    happens here (at report time), not at vector-load time.
    """
    result = normalize_alpha(vector, layer, alpha, residual_norm=residual_norm)
    assert result.by_residual_norm is not None  # residual_norm is required here
    return result.by_residual_norm


def load_vector(path: str | Path) -> SteeringVector:
    """Load a steering vector, detecting the format by content.

    A leading ``GGUF`` magic → repeng's native format. Otherwise the file is
    treated as a plain ``.pt`` mapping of ``layer -> tensor`` (the fallback).
    """
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(4)
    if magic == _GGUF_MAGIC:
        return _load_gguf(path)
    return _load_pt(path)


def save_vector(vector: SteeringVector, path: str | Path) -> None:
    """Write a steering vector as a repeng-compatible GGUF superset.

    Files written here load with both :func:`load_vector` and repeng's
    ``ControlVector.import_gguf`` (which ignores our extra ``steerbench.*`` KV).
    """
    import gguf  # lazy: keep the .pt fallback usable without gguf installed

    path = Path(path)
    writer = gguf.GGUFWriter(str(path), _ARCH)
    writer.add_string(_MODEL_HINT_KEY, vector.model_id or "")
    writer.add_uint32(_LAYER_COUNT_KEY, len(vector.directions))
    if vector.concept is not None:
        writer.add_string(_CONCEPT_KEY, vector.concept)
    if vector.repeng_version is not None:
        writer.add_string(_REPENG_VERSION_KEY, vector.repeng_version)
    for layer in sorted(vector.directions):
        array = vector.directions[layer].detach().cpu().to(torch.float32).contiguous().numpy()
        writer.add_tensor(f"{_DIRECTION_PREFIX}{layer}", array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _load_gguf(path: Path) -> SteeringVector:
    import gguf  # lazy: only needed on the native path

    reader = gguf.GGUFReader(str(path))

    arch = _read_str_field(reader, "general.architecture")
    if arch is not None and arch != _ARCH:
        import warnings

        warnings.warn(
            f"{path} has architecture {arch!r}, not {_ARCH!r}; may not be a control vector.",
            stacklevel=2,
        )

    directions: dict[int, torch.Tensor] = {}
    for tensor in reader.tensors:
        if not tensor.name.startswith(_DIRECTION_PREFIX):
            continue
        try:
            layer = int(tensor.name[len(_DIRECTION_PREFIX) :])
        except ValueError as exc:
            raise ValueError(f"invalid direction tensor name: {tensor.name!r}") from exc
        # GGUFReader hands back read-only mmap views; copy before from_numpy so
        # the tensor owns writable memory.
        directions[layer] = torch.from_numpy(tensor.data.copy())

    model_hint = _read_str_field(reader, _MODEL_HINT_KEY)
    return SteeringVector(
        directions=directions,
        model_id=model_hint or None,
        concept=_read_str_field(reader, _CONCEPT_KEY),
        repeng_version=_read_str_field(reader, _REPENG_VERSION_KEY),
    )


def _load_pt(path: Path) -> SteeringVector:
    raw = torch.load(path, weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a dict[int, Tensor], got {type(raw).__name__}")
    directions: dict[int, torch.Tensor] = {}
    for key, value in raw.items():
        layer = int(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{path}: layer {layer} is {type(value).__name__}, not a Tensor")
        directions[layer] = value
    return SteeringVector(directions=directions)


def _read_str_field(reader: object, key: str) -> str | None:
    """Read a GGUF string KV field, or ``None`` if absent/empty.

    Mirrors repeng's decode: the value lives in the field's last ``part``.
    """
    import gguf

    assert isinstance(reader, gguf.GGUFReader)
    field = reader.get_field(key)
    if field is None or not field.parts:
        return None
    return str(bytes(field.parts[-1]), encoding="utf-8", errors="replace")
