"""metrics.py — measurement primitives for the steering report card.

Three families of measurement, each addressable by name so a caller can swap an
inline proxy for the real thing without touching the rest of the pipeline:

* **EFFECT** — how strongly a generation exhibits a target behaviour. One
  ``str -> float`` scorer per shipped concept (``formality``, ``sentiment``,
  ``verbosity``), collected in :data:`EFFECT_SCORERS`. Defaults are cheap,
  transparent and stdlib-only; sentiment has an optional HF backend.
* **COHERENCE** — does steering wreck the text? Two axes: ``repetition_rate``
  (stdlib, ``str -> float``) and perplexity under an *unsteered* LM (needs
  ``torch``/``transformers``, split so the arithmetic is unit-testable on CPU).
  Together they expose the dose-response cliff.
* **SIDE EFFECTS** — capability degradation on held-out MMLU / GSM8K slices.
  Loaders are gated behind ``datasets``; the accuracy logic (prompt formatting,
  answer extraction, scoring) is pure Python and runs on CPU with fake
  generations.

Every scorer documents its direction and range so the dose-response curve is
interpretable.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_RE = re.compile(r"[.!?]+")


def _words(text: str) -> list[str]:
    """Lowercased word tokens (letters/digits/apostrophes)."""
    return _WORD_RE.findall(text.lower())


def _sentence_count(text: str) -> int:
    """Number of sentences, floored at 1 for any non-empty text."""
    parts = [p for p in _SENTENCE_RE.split(text) if p.strip()]
    return max(1, len(parts))


# --------------------------------------------------------------------------- #
# EFFECT scorers  (str -> float)
# --------------------------------------------------------------------------- #

# Casual markers: contractions, slang, hedges, direct address.
_CONTRACTION_RE = re.compile(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", re.IGNORECASE)
_CASUAL_WORDS = frozenset(
    {
        "gonna",
        "wanna",
        "gotta",
        "kinda",
        "sorta",
        "yeah",
        "yep",
        "nope",
        "hey",
        "hi",
        "ok",
        "okay",
        "cool",
        "stuff",
        "thing",
        "things",
        "lot",
        "lots",
        "guy",
        "guys",
        "kid",
        "kids",
        "awesome",
        "super",
        "totally",
        "really",
        "basically",
        "actually",
        "just",
        "so",
        "like",
        "lol",
        "wow",
    }
)
# Formal markers: discourse connectives + nominalisation suffixes.
_FORMAL_WORDS = frozenset(
    {
        "however",
        "therefore",
        "furthermore",
        "moreover",
        "consequently",
        "thus",
        "hence",
        "nevertheless",
        "nonetheless",
        "accordingly",
        "notwithstanding",
        "whereas",
        "herein",
        "thereof",
        "aforementioned",
        "subsequently",
        "additionally",
        "regarding",
        "pursuant",
        "shall",
    }
)
_NOMINALISATION_RE = re.compile(r"\b\w{5,}(?:tion|ment|ness|ity|ance|ence)\b", re.IGNORECASE)


def formality_score(text: str) -> float:
    """Formality of ``text`` in ``[-1.0, 1.0]``. Higher = more formal.

    Net of formal signals (discourse connectives, nominalisations) against
    casual signals (contractions, slang, exclamation marks). ``0.0`` for empty
    or perfectly balanced text. Purely lexical — no model required.
    """
    words = _words(text)
    if not words:
        return 0.0

    casual = sum(1 for w in words if w in _CASUAL_WORDS)
    casual += len(_CONTRACTION_RE.findall(text))
    casual += text.count("!")

    formal = sum(1 for w in words if w in _FORMAL_WORDS)
    formal += len(_NOMINALISATION_RE.findall(text))
    formal += sum(1 for w in words if len(w) >= 8)  # long words read as formal

    total = formal + casual
    if total == 0:
        return 0.0
    return (formal - casual) / total


# Compact sentiment lexicon — enough to separate clearly-toned text on CPU.
_POSITIVE_WORDS = frozenset(
    {
        "good",
        "great",
        "excellent",
        "wonderful",
        "amazing",
        "fantastic",
        "love",
        "loved",
        "loves",
        "happy",
        "joy",
        "joyful",
        "delightful",
        "beautiful",
        "best",
        "brilliant",
        "perfect",
        "awesome",
        "nice",
        "pleasant",
        "positive",
        "success",
        "successful",
        "win",
        "won",
        "glad",
        "grateful",
        "hope",
        "hopeful",
        "kind",
        "warm",
        "bright",
    }
)
_NEGATIVE_WORDS = frozenset(
    {
        "bad",
        "terrible",
        "awful",
        "horrible",
        "hate",
        "hated",
        "hates",
        "sad",
        "unhappy",
        "miserable",
        "ugly",
        "worst",
        "poor",
        "disappointing",
        "disappointed",
        "negative",
        "fail",
        "failed",
        "failure",
        "lose",
        "lost",
        "angry",
        "afraid",
        "fear",
        "pain",
        "painful",
        "cruel",
        "dark",
        "wrong",
        "broken",
        "sick",
        "dead",
        "cold",
        "bitter",
    }
)
_NEGATORS = frozenset({"not", "no", "never", "n't", "hardly", "barely"})


def sentiment_score(text: str, backend: str = "lexicon") -> float:
    """Sentiment polarity in ``[-1.0, 1.0]``. Higher = more positive.

    ``backend="lexicon"`` (default) is a stdlib bag-of-words net polarity with
    simple negation flipping over a one-word window; ``0.0`` when neutral or
    empty. ``backend="hf"`` runs a small HF sentiment classifier (requires
    ``transformers``+``torch``; installed via the ``model`` extra) and maps
    its label to a signed score. The two share direction and range so a report
    can switch backends without rescaling.
    """
    if backend == "hf":
        return _hf_sentiment_score(text)
    if backend != "lexicon":
        raise ValueError(f"unknown sentiment backend: {backend!r}")

    words = _words(text)
    if not words:
        return 0.0

    score = 0
    hits = 0
    for i, word in enumerate(words):
        polarity = 0
        if word in _POSITIVE_WORDS:
            polarity = 1
        elif word in _NEGATIVE_WORDS:
            polarity = -1
        if polarity == 0:
            continue
        if i > 0 and words[i - 1] in _NEGATORS:
            polarity = -polarity
        score += polarity
        hits += 1

    if hits == 0:
        return 0.0
    return score / hits


def verbosity_score(text: str) -> float:
    """Verbosity as mean words per sentence (``>= 0.0``). Higher = more verbose.

    Captures elaboration/padding: a curt reply scores low, a run-on essay scores
    high. Mean sentence length is length-normalised so it is not dominated by raw
    output length, which the coherence axis already tracks. ``0.0`` for empty
    text.
    """
    words = _words(text)
    if not words:
        return 0.0
    return len(words) / _sentence_count(text)


EFFECT_SCORERS: dict[str, Callable[[str], float]] = {
    "formality": formality_score,
    "sentiment": sentiment_score,
    "verbosity": verbosity_score,
}


def effect_score(concept: str, text: str) -> float:
    """Score ``text`` for the behaviour named ``concept`` via :data:`EFFECT_SCORERS`."""
    try:
        scorer = EFFECT_SCORERS[concept]
    except KeyError:
        raise ValueError(
            f"no effect scorer for concept {concept!r}; known: {sorted(EFFECT_SCORERS)}"
        ) from None
    return scorer(text)


# --------------------------------------------------------------------------- #
# COHERENCE metrics
# --------------------------------------------------------------------------- #


def repetition_rate(text: str, n: int = 3) -> float:
    """Fraction of repeated word ``n``-grams in ``[0.0, 1.0]``. Higher = worse.

    ``1 - distinct_ngrams / total_ngrams``: ``0.0`` when every ``n``-gram is
    unique, approaching ``1.0`` as the model loops. Returns ``0.0`` when the text
    is shorter than ``n`` tokens (no repetition observable).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    words = _words(text)
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def perplexity_from_token_nlls(nlls: Sequence[float]) -> float:
    """Perplexity from per-token negative log-likelihoods (natural log).

    ``exp(mean(nlls))``. Pure arithmetic, unit-tested on CPU. Empty input
    returns ``inf`` (no evidence of coherence). This is the boundary the
    model-backed :func:`perplexity` funnels its logits through.
    """
    if len(nlls) == 0:
        return math.inf
    return math.exp(sum(nlls) / len(nlls))


@dataclass(frozen=True)
class CoherenceScore:
    """Coherence of a generation on two axes.

    ``perplexity``: fluency under an unsteered LM (lower = more fluent).
    ``repetition_rate``: degenerate looping in ``[0, 1]`` (lower = better).
    The dose-response cliff shows up as perplexity and repetition rising
    together past the useful steering strength.
    """

    perplexity: float
    repetition_rate: float


def perplexity(text: str, model: object, tokenizer: object) -> float:
    """Perplexity of ``text`` under an unsteered causal LM.

    Modal-ready: ``model``/``tokenizer`` are an already-loaded HF
    ``AutoModelForCausalLM`` and tokenizer. The scoring math lives in
    :func:`perplexity_from_token_nlls`, which is what tests exercise; this
    wrapper only turns text into per-token NLLs. ``torch`` is imported lazily so
    importing this module stays dependency-free.
    """
    import torch  # local import: keeps module import CPU/stdlib-only

    enc = tokenizer(text, return_tensors="pt")  # type: ignore[operator]
    input_ids = enc["input_ids"]
    with torch.no_grad():
        logits = model(input_ids).logits  # type: ignore[operator]
    # shift so token t is predicted from tokens < t
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_ll = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    nlls: list[float] = (-token_ll).squeeze(0).tolist()
    return perplexity_from_token_nlls(nlls)


def coherence_score(
    text: str,
    model: object | None = None,
    tokenizer: object | None = None,
    n: int = 3,
) -> CoherenceScore:
    """Both coherence axes for ``text``.

    ``repetition_rate`` is always computed. Perplexity is computed when a
    ``model``+``tokenizer`` pair is supplied, else recorded as ``inf`` (unknown)
    so callers can run the cheap axis alone on CPU.
    """
    have_model = model is not None and tokenizer is not None
    ppl = perplexity(text, model, tokenizer) if have_model else math.inf
    return CoherenceScore(perplexity=ppl, repetition_rate=repetition_rate(text, n=n))


# --------------------------------------------------------------------------- #
# SIDE EFFECTS — MMLU / GSM8K slices
# --------------------------------------------------------------------------- #

# A generator maps a batch of prompts to a batch of completions. Injected so the
# accuracy logic is testable with a fake (canned) generator on CPU.
Generator = Callable[[list[str]], list[str]]

_MMLU_LETTERS = "ABCDEFGHIJ"


@dataclass(frozen=True)
class MMLUExample:
    """A multiple-choice MMLU item. ``answer`` is the index of the correct choice."""

    question: str
    choices: list[str]
    answer: int


@dataclass(frozen=True)
class GSM8KExample:
    """A grade-school math word problem. ``answer`` is the canonical numeric string."""

    question: str
    answer: str


# ---- MMLU ---------------------------------------------------------------- #


def format_mmlu_prompt(example: MMLUExample) -> str:
    """Render an MMLU item as a lettered multiple-choice prompt."""
    lines = [example.question, ""]
    for i, choice in enumerate(example.choices):
        lines.append(f"{_MMLU_LETTERS[i]}. {choice}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


_MMLU_EXPLICIT_RE = re.compile(r"answer\s*(?:is|:)?\s*\(?([A-J])\)?", re.IGNORECASE)
_MMLU_STANDALONE_RE = re.compile(r"\(?\b([A-J])\b\)?[.):]?")


def extract_mmlu_answer(text: str, num_choices: int = 4) -> int | None:
    """Parse a choice index (``0`` = A) from a model completion, or ``None``.

    Robust to the common formats: bare ``A``, ``(A)``, ``A.``, ``Answer: C``,
    ``The answer is B``, lowercase, and a stray letter earlier in the text
    followed by a different final choice. Precedence:

    1. the *last* explicit ``answer is/:`` statement, else
    2. the *last* standalone letter token,

    each restricted to the first ``num_choices`` letters. Returns ``None`` when
    no valid letter appears.
    """
    limit = min(max(num_choices, 1), len(_MMLU_LETTERS))
    valid = set(_MMLU_LETTERS[:limit])

    explicit = [m.group(1).upper() for m in _MMLU_EXPLICIT_RE.finditer(text)]
    for letter in reversed(explicit):
        if letter in valid:
            return _MMLU_LETTERS.index(letter)

    standalone = [m.group(1).upper() for m in _MMLU_STANDALONE_RE.finditer(text)]
    for letter in reversed(standalone):
        if letter in valid:
            return _MMLU_LETTERS.index(letter)

    return None


def score_mmlu(examples: Sequence[MMLUExample], predictions: Sequence[int | None]) -> float:
    """Accuracy of predicted choice indices against gold. ``None`` counts wrong."""
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions differ in length")
    if not examples:
        return 0.0
    correct = sum(1 for ex, pred in zip(examples, predictions, strict=True) if pred == ex.answer)
    return correct / len(examples)


def evaluate_mmlu(examples: Sequence[MMLUExample], generate: Generator) -> float:
    """Steered-vs-unsteered MMLU accuracy given a batch ``generate`` function.

    Formats prompts, runs the injected generator, extracts answers and scores.
    Pass a model-backed generator on Modal, or a canned one in tests.
    """
    prompts = [format_mmlu_prompt(ex) for ex in examples]
    completions = generate(prompts)
    if len(completions) != len(examples):
        raise ValueError("generator returned the wrong number of completions")
    preds = [
        extract_mmlu_answer(c, num_choices=len(ex.choices))
        for c, ex in zip(completions, examples, strict=True)
    ]
    return score_mmlu(examples, preds)


# ---- GSM8K --------------------------------------------------------------- #

_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_GSM8K_MARKER_RE = re.compile(r"####\s*(-?\$?\d[\d,]*(?:\.\d+)?)")


def _normalise_number(raw: str) -> str | None:
    """Strip ``$``/commas/trailing dot from a matched number; ``None`` if unparseable."""
    cleaned = raw.replace("$", "").replace(",", "").rstrip(".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    # canonicalise: drop a pointless trailing ``.0`` so ints compare as strings too
    return str(int(value)) if value.is_integer() else str(value)


def format_gsm8k_prompt(example: GSM8KExample) -> str:
    """Render a GSM8K item as a question prompt."""
    return f"{example.question}\n\nAnswer:"


def extract_gsm8k_answer(text: str) -> str | None:
    """Parse the final numeric answer from a completion, or ``None``.

    Prefers an explicit ``#### <n>`` marker (the dataset's gold format); failing
    that, takes the *last* number in the text. Tolerates ``$``, thousands
    commas, and a trailing period. Returns a canonical string (``"42"``,
    ``"3.5"``) suitable for numeric comparison, or ``None`` when no number
    appears.
    """
    marker = _GSM8K_MARKER_RE.search(text)
    if marker is not None:
        return _normalise_number(marker.group(1))
    numbers = _NUMBER_RE.findall(text)
    if not numbers:
        return None
    return _normalise_number(numbers[-1])


def score_gsm8k(examples: Sequence[GSM8KExample], predictions: Sequence[str | None]) -> float:
    """Accuracy of predicted numbers against gold, compared numerically."""
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions differ in length")
    if not examples:
        return 0.0
    correct = 0
    for ex, pred in zip(examples, predictions, strict=True):
        if pred is None:
            continue
        pred_num = _normalise_number(pred)
        gold = _normalise_number(ex.answer)
        if pred_num is None or gold is None:
            continue
        if math.isclose(float(pred_num), float(gold), rel_tol=0.0, abs_tol=1e-6):
            correct += 1
    return correct / len(examples)


def evaluate_gsm8k(examples: Sequence[GSM8KExample], generate: Generator) -> float:
    """Steered-vs-unsteered GSM8K accuracy given a batch ``generate`` function."""
    prompts = [format_gsm8k_prompt(ex) for ex in examples]
    completions = generate(prompts)
    if len(completions) != len(examples):
        raise ValueError("generator returned the wrong number of completions")
    preds = [extract_gsm8k_answer(c) for c in completions]
    return score_gsm8k(examples, preds)


# ---- Loaders (gated behind ``datasets``, disk-cached) -------------------- #

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "steerbench"


def _cache_path(cache_dir: Path | None, name: str) -> Path:
    root = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def load_mmlu_slice(
    n: int = 50,
    seed: int = 0,
    cache_dir: Path | None = None,
) -> list[MMLUExample]:
    """Load a small, deterministic MMLU slice, caching it to disk as JSON.

    Downloads via ``datasets`` (installed with the ``data`` extra) only on a
    cache miss. Slice size and seed are part of the cache key so different
    configs coexist.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    path = _cache_path(cache_dir, f"mmlu_{n}_{seed}.json")
    if path.exists():
        rows = json.loads(path.read_text())
        return [
            MMLUExample(question=r["question"], choices=list(r["choices"]), answer=int(r["answer"]))
            for r in rows
        ]

    from datasets import load_dataset  # local import: gated optional dependency

    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    examples = [
        MMLUExample(
            question=str(row["question"]),
            choices=[str(c) for c in row["choices"]],
            answer=int(row["answer"]),
        )
        for row in ds
    ]
    path.write_text(
        json.dumps(
            [{"question": e.question, "choices": e.choices, "answer": e.answer} for e in examples]
        )
    )
    return examples


def load_gsm8k_slice(
    n: int = 50,
    seed: int = 0,
    cache_dir: Path | None = None,
) -> list[GSM8KExample]:
    """Load a small, deterministic GSM8K slice, caching it to disk as JSON.

    Gold answers are normalised from the dataset's ``#### <n>`` format at load
    time so scoring never re-parses them.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    path = _cache_path(cache_dir, f"gsm8k_{n}_{seed}.json")
    if path.exists():
        rows = json.loads(path.read_text())
        return [GSM8KExample(question=r["question"], answer=str(r["answer"])) for r in rows]

    from datasets import load_dataset  # local import: gated optional dependency

    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    examples = []
    for row in ds:
        gold = extract_gsm8k_answer(str(row["answer"]))
        examples.append(GSM8KExample(question=str(row["question"]), answer=gold if gold else ""))
    path.write_text(json.dumps([{"question": e.question, "answer": e.answer} for e in examples]))
    return examples


# --------------------------------------------------------------------------- #
# Optional HF sentiment backend
# --------------------------------------------------------------------------- #

_HF_SENTIMENT_PIPELINE: object | None = None


def _hf_sentiment_score(text: str) -> float:
    """Signed sentiment in ``[-1, 1]`` from a cached HF pipeline. Lazy + memoised."""
    global _HF_SENTIMENT_PIPELINE
    if not text.strip():
        return 0.0
    if _HF_SENTIMENT_PIPELINE is None:
        from transformers import pipeline  # local import: gated optional dependency

        _HF_SENTIMENT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    result = _HF_SENTIMENT_PIPELINE(text[:512])[0]  # type: ignore[operator]
    label = str(result["label"]).upper()
    confidence = float(result["score"])
    return confidence if label.startswith("POS") else -confidence
