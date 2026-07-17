"""CPU-only, fast tests for the measurement primitives.

No model or dataset download: effect/coherence scorers run on canned strings,
side-effect accuracy runs on hand-built examples with a fake generator. The
optional deps (torch/transformers/datasets) are never imported.
"""

from __future__ import annotations

import json
import math
import socket
from pathlib import Path

import pytest

from steerbench.metrics import (
    EFFECT_SCORERS,
    CoherenceScore,
    GSM8KExample,
    MMLUExample,
    coherence_score,
    effect_score,
    evaluate_gsm8k,
    evaluate_mmlu,
    extract_gsm8k_answer,
    extract_mmlu_answer,
    formality_score,
    format_mmlu_prompt,
    load_gsm8k_slice,
    load_mmlu_slice,
    perplexity_from_token_nlls,
    repetition_rate,
    score_gsm8k,
    score_mmlu,
    sentiment_score,
    verbosity_score,
)

FORMAL_TEXT = (
    "Furthermore, the aforementioned recommendation shall be implemented; "
    "consequently, the organisation's productivity demonstrates measurable "
    "improvement."
)
CASUAL_TEXT = "Hey! Yeah I'm gonna grab some stuff, it's super cool, wanna come? lol"


# --------------------------------------------------------------------------- #
# EFFECT scorers
# --------------------------------------------------------------------------- #


def test_formality_direction() -> None:
    assert formality_score(FORMAL_TEXT) > 0.0
    assert formality_score(CASUAL_TEXT) < 0.0
    assert formality_score(FORMAL_TEXT) > formality_score(CASUAL_TEXT)


def test_formality_range_and_empty() -> None:
    assert formality_score("") == 0.0
    assert -1.0 <= formality_score(FORMAL_TEXT) <= 1.0
    assert -1.0 <= formality_score(CASUAL_TEXT) <= 1.0


def test_sentiment_direction() -> None:
    assert sentiment_score("This is wonderful, I love this beautiful happy day") > 0.0
    assert sentiment_score("A terrible, awful, miserable and painful failure") < 0.0
    assert sentiment_score("") == 0.0


def test_sentiment_negation_flips() -> None:
    assert sentiment_score("not good") < 0.0
    assert sentiment_score("not bad") > 0.0


def test_sentiment_range_and_unknown_backend() -> None:
    assert -1.0 <= sentiment_score("love love love") <= 1.0
    with pytest.raises(ValueError):
        sentiment_score("hi", backend="nope")


def test_verbosity_monotonic() -> None:
    terse = "Yes."
    verbose = (
        "Well, to be perfectly honest with you, I would say that the answer, "
        "when considered from every conceivable angle and at considerable length, "
        "is affirmative in nature."
    )
    assert verbosity_score(verbose) > verbosity_score(terse)
    assert verbosity_score("") == 0.0


def test_effect_registry_dispatch() -> None:
    assert set(EFFECT_SCORERS) == {"formality", "sentiment", "verbosity"}
    assert effect_score("formality", FORMAL_TEXT) == formality_score(FORMAL_TEXT)
    with pytest.raises(ValueError):
        effect_score("nonexistent", "x")


# --------------------------------------------------------------------------- #
# COHERENCE
# --------------------------------------------------------------------------- #


def test_repetition_rate_extremes() -> None:
    assert repetition_rate("the cat sat on the warm mat today") == 0.0
    looping = "i am i am i am i am i am"
    assert repetition_rate(looping) > 0.5
    assert 0.0 <= repetition_rate(looping) <= 1.0


def test_repetition_rate_short_text() -> None:
    assert repetition_rate("hi", n=3) == 0.0
    with pytest.raises(ValueError):
        repetition_rate("hi there friend", n=0)


def test_perplexity_from_nlls() -> None:
    assert perplexity_from_token_nlls([0.0, 0.0, 0.0]) == 1.0
    assert math.isclose(perplexity_from_token_nlls([1.0, 1.0]), math.e)
    assert perplexity_from_token_nlls([]) == math.inf
    # a fluent (low-NLL) sequence must score below a confused (high-NLL) one
    assert perplexity_from_token_nlls([0.1, 0.2]) < perplexity_from_token_nlls([3.0, 4.0])


def test_coherence_score_cpu_only() -> None:
    result = coherence_score("the cat sat on the warm mat today")
    assert isinstance(result, CoherenceScore)
    assert result.perplexity == math.inf  # no model supplied
    assert result.repetition_rate == 0.0


# --------------------------------------------------------------------------- #
# SIDE EFFECTS — MMLU
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A", 0),
        ("(B)", 1),
        ("C.", 2),
        ("The answer is B", 1),
        ("Answer: C", 2),
        ("answer: d", 3),
        ("I first thought A but the answer is C", 2),  # last explicit wins
        ("Well maybe A, then B, finally D", 3),  # last standalone wins
        ("I have no idea", None),
        ("", None),
    ],
)
def test_extract_mmlu_answer_formats(text: str, expected: int | None) -> None:
    assert extract_mmlu_answer(text, num_choices=4) == expected


def test_extract_mmlu_answer_respects_num_choices() -> None:
    # 'E' is out of range for a 4-way question -> ignored
    assert extract_mmlu_answer("The answer is E", num_choices=4) is None
    assert extract_mmlu_answer("The answer is E", num_choices=5) == 4


def test_score_and_evaluate_mmlu() -> None:
    examples = [
        MMLUExample("2+2?", ["3", "4", "5", "6"], answer=1),
        MMLUExample("Sky colour?", ["Red", "Green", "Blue", "Pink"], answer=2),
    ]
    assert score_mmlu(examples, [1, 2]) == 1.0
    assert score_mmlu(examples, [1, None]) == 0.5
    assert score_mmlu([], []) == 0.0
    with pytest.raises(ValueError):
        score_mmlu(examples, [1])

    def fake_generate(prompts: list[str]) -> list[str]:
        # first right, second wrong
        return ["The answer is B", "Answer: A"]

    assert evaluate_mmlu(examples, fake_generate) == 0.5


def test_format_mmlu_prompt_shape() -> None:
    prompt = format_mmlu_prompt(MMLUExample("Q?", ["w", "x", "y", "z"], answer=0))
    assert "A. w" in prompt and "D. z" in prompt and prompt.rstrip().endswith("Answer:")


# --------------------------------------------------------------------------- #
# SIDE EFFECTS — GSM8K
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The result is 42", "42"),
        ("#### 42", "42"),
        ("It costs $1,000 total", "1000"),
        ("So the answer is 42.", "42"),
        ("We get 3.5 litres", "3.5"),
        ("First 10 apples then 25 oranges", "25"),  # last number
        ("#### 7\nsome trailing prose 99", "7"),  # marker beats trailing number
        ("no numbers here", None),
        ("", None),
    ],
)
def test_extract_gsm8k_answer_formats(text: str, expected: str | None) -> None:
    assert extract_gsm8k_answer(text) == expected


def test_score_and_evaluate_gsm8k() -> None:
    examples = [
        GSM8KExample("How many?", answer="18"),
        GSM8KExample("Total cost?", answer="1000"),
    ]
    # comma / prose formatting still matches numerically
    assert score_gsm8k(examples, ["18", "1,000"]) == 1.0
    assert score_gsm8k(examples, ["18", None]) == 0.5
    assert score_gsm8k(examples, ["17", "1000"]) == 0.5
    assert score_gsm8k([], []) == 0.0
    with pytest.raises(ValueError):
        score_gsm8k(examples, ["18"])

    def fake_generate(prompts: list[str]) -> list[str]:
        return ["...therefore 18 apples", "the bill is $1,000 exactly"]

    assert evaluate_gsm8k(examples, fake_generate) == 1.0


# --------------------------------------------------------------------------- #
# SLICE LOADERS — n validation (guard fires before any cache/datasets access)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_n", [0, -3])
def test_load_mmlu_slice_rejects_n_below_1(bad_n: int) -> None:
    with pytest.raises(ValueError):
        load_mmlu_slice(n=bad_n)


@pytest.mark.parametrize("bad_n", [0, -3])
def test_load_gsm8k_slice_rejects_n_below_1(bad_n: int) -> None:
    with pytest.raises(ValueError):
        load_gsm8k_slice(n=bad_n)


# --------------------------------------------------------------------------- #
# SLICE LOADERS — accept path with pre-seeded cache (n=1, no network)
# --------------------------------------------------------------------------- #


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any socket construction raise, so a cache miss cannot download."""

    def _no_socket(*args: object, **kwargs: object) -> object:
        raise OSError("network access disabled in test")

    monkeypatch.setattr(socket, "socket", _no_socket)


def test_load_mmlu_slice_accepts_n_equal_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _block_network(monkeypatch)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fixture = cache_dir / "mmlu_1_0.json"
    fixture.write_text(
        json.dumps([{"question": "What is 2+2?", "choices": ["1", "2", "3", "4"], "answer": 3}])
    )

    examples = load_mmlu_slice(n=1, cache_dir=cache_dir)

    assert len(examples) == 1
    ex = examples[0]
    assert isinstance(ex, MMLUExample)
    assert isinstance(ex.question, str)
    assert isinstance(ex.choices, list)
    assert all(isinstance(c, str) for c in ex.choices)
    assert isinstance(ex.answer, int)


def test_load_gsm8k_slice_accepts_n_equal_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_network(monkeypatch)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fixture = cache_dir / "gsm8k_1_0.json"
    fixture.write_text(json.dumps([{"question": "How many apples?", "answer": "42"}]))

    examples = load_gsm8k_slice(n=1, cache_dir=cache_dir)

    assert len(examples) == 1
    ex = examples[0]
    assert isinstance(ex, GSM8KExample)
    assert isinstance(ex.question, str)
    assert isinstance(ex.answer, str)


def test_no_optional_deps_imported() -> None:
    # Importing steerbench.metrics must not pull in the heavy optional deps.
    # Checked in a fresh interpreter: a global sys.modules check is unreliable
    # now that a sibling module (steerbench.vectors) legitimately imports torch.
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        import steerbench.metrics  # noqa: F401
        leaked = [m for m in ("torch", "transformers", "datasets") if m in sys.modules]
        assert not leaked, leaked
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
