"""Modal app for steerbench M0: repeng ControlVector dose-response + layer sweep.

Serverless A100. Pinned image so runs are reproducible.

Pipeline:
  1. train_and_export   — train a FORMALITY ControlVector on Qwen2.5-7B-Instruct,
                          save in repeng's native gguf to a Volume, print keys+norms.
  2. smoke              — reload gguf via steerbench.load_vector, steer one mid layer,
                          generate at coeff 0 and a big coeff (de-risk everything cheap).
  3. dose_response      — fix a mid layer, sweep coeff (incl 0, negatives, past-the-cliff),
                          3 seeds, metrics: formality proxy + coherence (repetition + ppl).
  4. layer_sweep        — inject each layer's OWN direction at that layer, fixed alpha,
                          3 seeds, same metrics.

Entrypoints:
  modal run modal_app.py::introspect
  modal run modal_app.py::train_and_export
  modal run modal_app.py::smoke
  modal run modal_app.py::run_dose          # writes results/dose_response.{csv,png}
  modal run modal_app.py::run_layer_sweep   # writes results/layer_sweep.{csv,png}
"""

from __future__ import annotations

import modal

# Pinned image. repeng + torch + transformers + matplotlib. Commit this file.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate==0.34.2",
        "repeng==0.4.0",
        "matplotlib==3.9.2",
        "numpy==1.26.4",
        "gguf==0.10.0",
        "sentencepiece==0.2.0",
        "protobuf==5.28.2",
    )
    .add_local_dir("src", "/root/src")
)

app = modal.App("steerbench-m0", image=image)

# Persist HF cache + trained gguf across function calls / cold starts.
vol = modal.Volume.from_name("steerbench-m0-vol", create_if_missing=True)
VOL = "/vol"
HF_CACHE = f"{VOL}/hf"
GGUF_PATH = f"{VOL}/formality.gguf"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_BASE = "Qwen/Qwen2.5-7B"
GPU = "A100"

# HF token for gated models (Llama, Gemma). Modal secret "huggingface".
HF_SECRET = modal.Secret.from_name("huggingface")

# Cross-model registry. inject_frac fixes the concept anchor at a constant
# fraction-of-depth; the absolute layer + per-model resid_norm are MEASURED,
# not assumed. sweet_alpha is the transferable dose (held constant across models).
SWEET_ALPHA = 0.044
INJECT_FRAC = 0.61
MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    # ungated mirror of Llama-3.1-8B-Instruct (identical weights)
    "llama": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-2-9b-it",
}
# If gemma-2-9b-it is gated (license not accepted), fall back to this ungated
# instruct model for the 3rd architecture. Report notes "Gemma deferred".
GEMMA_FALLBACK = "mistralai/Mistral-7B-Instruct-v0.3"


def _inject_layer(n_layers: int, frac: float = INJECT_FRAC) -> int:
    """Absolute layer index closest to `frac` of depth (1..n_layers-1)."""
    return max(1, min(n_layers - 1, round(frac * n_layers)))


def _gguf_path(model_id: str) -> str:
    slug = model_id.split("/")[-1].replace(".", "_")
    return f"{VOL}/formality_{slug}.gguf"

# ---------------------------------------------------------------------------
# Dataset: formal vs casual contrastive pairs (repeng persona-suffix pattern).
# ---------------------------------------------------------------------------

# Qwen2.5 chat scaffold. repeng reads the hidden state at the last token, so we
# end each string mid-assistant-turn with a shared truncated suffix; the pair
# differs ONLY in the persona instruction -> the PCA diff isolates formality.
_QWEN_TMPL = "<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n{suffix}"

_POS_PERSONAS = [
    "an extremely formal, professional person",
    "a highly eloquent scholar writing for publication",
    "a polished diplomat drafting official correspondence",
]
_NEG_PERSONAS = [
    "an extremely casual, laid-back person",
    "a slangy teenager texting a friend",
    "a chatty buddy shooting the breeze",
]

# Neutral seed text; we truncate it at many points to get many suffixes.
_SEED_TEXT = (
    "I think we should talk about the plan for the weekend and what everyone "
    "wants to do about the food and the drinks and the music and the games "
    "because there is a lot to sort out before people start showing up at the "
    "door and it would be good to have things ready ahead of time so nobody "
    "has to scramble at the last minute when the guests arrive at the house"
)


def build_dataset(tok=None):
    """Formal/casual contrastive pairs.

    If `tok` has a chat template (instruct models) each side is wrapped in THAT
    model's own chat scaffold via apply_chat_template — correct for Qwen, Llama,
    Gemma alike (not a hardcoded Qwen scaffold). Base models (no template) get
    plain completion text. repeng reads the last-token hidden state, so we append
    a shared truncated suffix; the pair differs ONLY in the persona instruction.
    """
    from repeng import DatasetEntry

    words = _SEED_TEXT.split()
    suffixes = [" ".join(words[:i]) for i in range(1, min(len(words), 24))]
    persona_msg = "Act as if you are {persona}. Write a short message."
    has_chat = tok is not None and bool(getattr(tok, "chat_template", None))

    def scaffold(persona: str) -> str:
        msg = persona_msg.format(persona=persona)
        if has_chat:
            return tok.apply_chat_template(
                [{"role": "user", "content": msg}],
                tokenize=False, add_generation_prompt=True)
        return msg + " "

    dataset: list[DatasetEntry] = []
    for suffix in suffixes:
        for pos, neg in zip(_POS_PERSONAS, _NEG_PERSONAS):
            dataset.append(
                DatasetEntry(
                    positive=scaffold(pos) + suffix,
                    negative=scaffold(neg) + suffix,
                )
            )
    return dataset


# ---------------------------------------------------------------------------
# Cheap inline metrics.
# ---------------------------------------------------------------------------

_CASUAL = {
    "gonna", "wanna", "gotta", "yeah", "yep", "nope", "lol", "haha", "hey",
    "hi", "stuff", "kinda", "sorta", "cool", "awesome", "gimme", "dunno",
    "ok", "okay", "totally", "super", "guys", "gutted", "vibe", "chill",
    "buddy", "dude", "yay", "wow", "nah", "bruh", "gotcha",
}
_FORMAL = {
    "therefore", "however", "furthermore", "moreover", "regarding",
    "consequently", "additionally", "nevertheless", "thus", "hence",
    "accordingly", "herein", "aforementioned", "subsequently", "respectfully",
    "kindly", "shall", "regards", "sincerely", "utilize", "facilitate",
    "demonstrate", "significant", "substantial", "concerning", "pursuant",
    "endeavor", "commence", "ascertain", "notwithstanding", "esteemed",
}


def formality_score(text: str) -> float:
    """Cheap lexical formality proxy. Higher = more formal. Not calibrated units."""
    import re

    words = re.findall(r"[A-Za-z']+", text.lower())
    n = max(len(words), 1)
    formal = sum(w in _FORMAL for w in words)
    casual = sum(w in _CASUAL for w in words)
    contractions = sum("'" in w for w in words)
    mean_len = sum(len(w) for w in words) / n
    return (
        mean_len
        + 20.0 * formal / n
        - 20.0 * casual / n
        - 8.0 * contractions / n
    )


def repetition_rate(text: str) -> float:
    """1 - distinct-2. High = degenerate looping (a cliff signature). 0..1."""
    import re

    words = re.findall(r"[A-Za-z']+", text.lower())
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams)


# Fixed neutral eval prompts (open-ended so style can move).
_EVAL_PROMPTS = [
    "Tell me about your weekend.",
    "Write a note to a colleague about a delayed project.",
    "Describe how to make a cup of coffee.",
    "Give some advice to someone starting a new job.",
]


# ---------------------------------------------------------------------------
# Model helpers.
# ---------------------------------------------------------------------------


def _load_model_and_tokenizer(model_id: str = MODEL_ID):
    import os

    os.environ["HF_HOME"] = HF_CACHE
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    return model, tok


def _prompt_ids(tok, prompt: str):
    # base models have no chat template -> plain completion prompt
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": prompt}]
        return tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        )
    return tok(prompt + "\n", return_tensors="pt").input_ids


def _generate(cmodel, tok, input_ids, seed: int, max_new_tokens: int = 96) -> str:
    import torch

    torch.manual_seed(seed)
    with torch.no_grad():
        out = cmodel.generate(
            input_ids=input_ids.to(cmodel.device),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            repetition_penalty=1.0,  # keep OFF so the cliff can show naturally
        )
    gen = out[0][input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def _ppl_unsteered(cmodel, tok, text: str) -> float:
    """Perplexity of `text` under the UNSTEERED model. Control MUST be reset first."""
    import math

    import torch

    cmodel.reset()  # critical: measure coherence under base model, not steered
    ids = tok(text, return_tensors="pt").input_ids.to(cmodel.device)
    if ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        loss = cmodel(input_ids=ids, labels=ids).loss
    return math.exp(min(float(loss), 20.0))


# ---------------------------------------------------------------------------
# STEP 1 — introspect (kept for the record).
# ---------------------------------------------------------------------------


@app.function(gpu=GPU, timeout=1800, volumes={VOL: vol}, secrets=[HF_SECRET])
def introspect() -> str:
    import inspect

    import repeng
    from repeng import ControlModel, ControlVector, DatasetEntry

    out = []
    out.append(f"repeng file: {repeng.__file__}")
    out.append(f"train sig: {inspect.signature(ControlVector.train)}")
    out.append(f"CV fields: {list(ControlVector.__dataclass_fields__)}")
    out.append(f"DatasetEntry fields: {list(DatasetEntry.__dataclass_fields__)}")
    out.append(f"CM init sig: {inspect.signature(ControlModel.__init__)}")
    out.append(f"set_control sig: {inspect.signature(ControlModel.set_control)}")
    text = "\n".join(out)
    print(text)
    return text


# ---------------------------------------------------------------------------
# STEP 2 — train + export to gguf on the Volume.
# ---------------------------------------------------------------------------


@app.function(gpu=GPU, timeout=3600, volumes={VOL: vol}, secrets=[HF_SECRET])
def train_and_export(model_id: str = MODEL_ID, gguf_path: str | None = None) -> dict:
    import numpy as np
    from repeng import ControlVector

    gguf_path = gguf_path or GGUF_PATH
    model, tok = _load_model_and_tokenizer(model_id)
    from repeng import ControlModel

    # ControlModel wrapping all layers is repeng's recommended training target.
    n_layers = model.config.num_hidden_layers
    wrapped = ControlModel(model, list(range(-1, -n_layers, -1)))

    chat = bool(getattr(tok, "chat_template", None))
    dataset = build_dataset(tok)
    print(f"model {model_id}  chat={chat}  pairs {len(dataset)}  n_layers {n_layers}")

    vec = ControlVector.train(wrapped, tok, dataset, batch_size=16)

    keys = sorted(vec.directions.keys())
    norms = {int(k): float(np.linalg.norm(vec.directions[k])) for k in keys}
    print(f"direction keys: {keys}")
    print(f"hidden_size: {model.config.hidden_size}")

    vec.export_gguf(gguf_path)
    vol.commit()  # make the gguf visible to other functions
    print(f"exported gguf -> {gguf_path}")

    return {
        "model": model_id,
        "n_layers": n_layers,
        "hidden_size": int(model.config.hidden_size),
        "keys": keys,
        "norms": norms,
        "n_pairs": len(dataset),
    }


# ---------------------------------------------------------------------------
# Smoke — reload from disk, steer one layer, generate 2 completions.
# ---------------------------------------------------------------------------


@app.function(gpu=GPU, timeout=1800, volumes={VOL: vol}, secrets=[HF_SECRET])
def smoke(layer: int = 14, big_coeff: float = 8.0) -> dict:
    import sys

    sys.path.insert(0, "/root/src")
    vol.reload()  # see the freshly committed gguf
    from steerbench.vectors import load_vector

    vec = load_vector(GGUF_PATH)  # exercise the real reload path
    print(f"reloaded keys: {sorted(vec.directions.keys())}")

    import numpy as np

    dnorm = float(np.linalg.norm(vec.directions[layer]))
    print(f"layer {layer} ||dir|| = {dnorm:.4f}")

    model, tok = _load_model_and_tokenizer()
    from repeng import ControlModel

    cmodel = ControlModel(model, [layer])
    ids = _prompt_ids(tok, _EVAL_PROMPTS[0])

    results = {}
    for tag, coeff in [("baseline_0", 0.0), (f"steer_{big_coeff}", big_coeff)]:
        cmodel.reset()
        cmodel.set_control(vec, coeff)
        txt = _generate(cmodel, tok, ids, seed=0, max_new_tokens=80)
        results[tag] = {
            "text": txt,
            "formality": round(formality_score(txt), 3),
            "repetition": round(repetition_rate(txt), 3),
        }
        print(f"\n=== {tag} (coeff={coeff}) ===\n{txt}")
    return results


# ---------------------------------------------------------------------------
# STEP 3 — dose-response at a fixed layer.
# ---------------------------------------------------------------------------


@app.function(gpu=GPU, timeout=5400, volumes={VOL: vol}, secrets=[HF_SECRET])
def dose_response(
    layer: int,
    seeds: list[int],
    coeffs: list[float] | None = None,
    alphas: list[float] | None = None,
    model_id: str = MODEL_ID,
    gguf_path: str | None = None,
) -> dict:
    """Dose-response at a fixed layer. Provide EITHER absolute `coeffs` OR
    `alphas` (dimensionless dose); alphas convert to per-model coeffs via the
    MEASURED residual norm, so the same dose grid is comparable across models."""
    assert (coeffs is None) != (alphas is None), "give exactly one of coeffs/alphas"
    import sys
    import time

    sys.path.insert(0, "/root/src")
    vol.reload()
    from steerbench.vectors import load_vector

    import numpy as np

    vec = load_vector(gguf_path or GGUF_PATH)
    dnorm = float(np.linalg.norm(vec.directions[layer]))

    model, tok = _load_model_and_tokenizer(model_id)
    from repeng import ControlModel

    cmodel = ControlModel(model, [layer])  # wrap ONCE, reset between points

    # measure residual-stream norm at this layer (baseline) for alpha normalization
    resid_norm = _resid_norm_at_layer(model, tok, layer)
    if alphas is not None:
        coeffs = [a * resid_norm / dnorm for a in alphas]  # dose -> raw coeff

    prompt_ids = [_prompt_ids(tok, p) for p in _EVAL_PROMPTS]

    t0 = time.time()
    rows = []
    for coeff in coeffs:
        for seed in seeds:
            forms, reps, ppls = [], [], []
            for ids in prompt_ids:
                cmodel.reset()
                cmodel.set_control(vec, coeff)
                txt = _generate(cmodel, tok, ids, seed=seed)
                forms.append(formality_score(txt))
                reps.append(repetition_rate(txt))
                ppls.append(_ppl_unsteered(cmodel, tok, txt))  # resets inside
            rows.append(
                {
                    "coeff": coeff,
                    "seed": seed,
                    "alpha_norm": coeff * dnorm / resid_norm,
                    "formality": float(np.mean(forms)),
                    "repetition": float(np.mean(reps)),
                    "ppl": float(np.nanmean(ppls)),
                }
            )
        print(f"coeff {coeff:+.2f} done")
    wall = time.time() - t0
    return {
        "layer": layer,
        "dir_norm": dnorm,
        "resid_norm": resid_norm,
        "rows": rows,
        "wall_s": wall,
        "n_layers": model.config.num_hidden_layers,
    }


def _resid_norm_at_layer(model, tok, layer: int) -> float:
    """Mean L2 norm of the residual-stream hidden state at `layer` on eval prompts."""
    import numpy as np
    import torch

    n_layers = model.config.num_hidden_layers
    pos = layer if layer >= 0 else n_layers + layer
    norms = []
    for p in _EVAL_PROMPTS:
        ids = _prompt_ids(tok, p).to(model.device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        # hidden_states[0] is embeddings; layer L output is hidden_states[L+1]
        hs = out.hidden_states[pos + 1][0]  # (seq, hidden)
        norms.append(float(hs.norm(dim=-1).mean()))
    return float(np.mean(norms))


# ---------------------------------------------------------------------------
# STEP 4 — layer sweep: each layer's own direction at that layer.
# ---------------------------------------------------------------------------


@app.function(gpu=GPU, timeout=7200, volumes={VOL: vol}, secrets=[HF_SECRET])
def layer_sweep(
    seeds: list[int],
    layers: list[int],
    coeff: float | None = None,
    target_alpha: float | None = None,
    model_id: str = MODEL_ID,
    gguf_path: str | None = None,
) -> dict:
    """Inject each layer's OWN direction at that layer.

    Two modes:
      - coeff=X          : fixed RAW coeff at every layer (unequal alpha -> confounded).
      - target_alpha=A   : fixed NORMALIZED alpha; per-layer coeff = A * resid_norm_L
                           (dir norm is 1). Equal injection strength across layers.
    Exactly one of coeff / target_alpha must be set.
    """
    assert (coeff is None) != (target_alpha is None), "set exactly one of coeff/alpha"
    import sys
    import time

    sys.path.insert(0, "/root/src")
    vol.reload()
    from steerbench.vectors import load_vector

    import numpy as np

    vec = load_vector(gguf_path or GGUF_PATH)
    model, tok = _load_model_and_tokenizer(model_id)
    from repeng import ControlModel

    n_layers = model.config.num_hidden_layers
    prompt_ids = [_prompt_ids(tok, p) for p in _EVAL_PROMPTS]

    # baseline (coeff 0) reference for formality delta
    base_model = ControlModel(model, [layers[0]])
    base_forms = []
    for ids in prompt_ids:
        base_model.reset()
        txt = _generate(base_model, tok, ids, seed=seeds[0])
        base_forms.append(formality_score(txt))
    base_form = float(np.mean(base_forms))
    base_model.unwrap()

    t0 = time.time()
    rows = []
    for layer in layers:
        dnorm = float(np.linalg.norm(vec.directions[layer]))
        resid = _resid_norm_at_layer(model, tok, layer)
        if target_alpha is not None:
            layer_coeff = target_alpha * resid / dnorm  # equal normalized strength
        else:
            layer_coeff = coeff
        alpha = layer_coeff * dnorm / resid
        cmodel = ControlModel(model, [layer])
        for seed in seeds:
            forms, reps, ppls = [], [], []
            for ids in prompt_ids:
                cmodel.reset()
                cmodel.set_control(vec, layer_coeff)
                txt = _generate(cmodel, tok, ids, seed=seed)
                forms.append(formality_score(txt))
                reps.append(repetition_rate(txt))
                ppls.append(_ppl_unsteered(cmodel, tok, txt))
            rows.append(
                {
                    "layer": layer,
                    "layer_pos": layer if layer >= 0 else n_layers + layer,
                    "seed": seed,
                    "dir_norm": dnorm,
                    "resid_norm": resid,
                    "coeff": layer_coeff,
                    "alpha_norm": alpha,
                    "formality": float(np.mean(forms)),
                    "repetition": float(np.mean(reps)),
                    "ppl": float(np.nanmean(ppls)),
                }
            )
        cmodel.unwrap()  # undo mutation before wrapping the next layer
        print(f"layer {layer} coeff {layer_coeff:.1f} alpha {alpha:.3f} done")
    wall = time.time() - t0
    return {
        "mode": "alpha" if target_alpha is not None else "coeff",
        "coeff": coeff,
        "target_alpha": target_alpha,
        "base_formality": base_form,
        "rows": rows,
        "wall_s": wall,
        "n_layers": n_layers,
    }


# ---------------------------------------------------------------------------
# Local orchestration entrypoints (write CSV + PNG locally).
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def train_export() -> None:
    import json

    meta = train_and_export.remote()
    print(json.dumps(meta, indent=2))


@app.function(timeout=600, volumes={VOL: vol})  # CPU only
def export_example() -> dict:
    """Emit the M0 Qwen formality vector as compact example assets (gguf + pt)
    into the Volume for local download. Agent 3's Colab loads a real vector."""
    import sys

    sys.path.insert(0, "/root/src")
    import torch

    from steerbench.vectors import load_vector, save_vector

    vec = load_vector(GGUF_PATH)  # SteeringVector from the M0 training run
    out_gguf = f"{VOL}/examples/formality_qwen2.5-7b.gguf"
    out_pt = f"{VOL}/examples/formality_qwen2.5-7b.pt"
    import os

    os.makedirs(f"{VOL}/examples", exist_ok=True)
    save_vector(vec, out_gguf)  # repeng-compatible gguf superset
    # plain dict[layer -> float32 tensor] for the .pt fallback path
    torch.save({int(k): v.to(torch.float32).cpu() for k, v in vec.directions.items()},
               out_pt)
    vol.commit()
    sizes = {p: os.path.getsize(p) for p in (out_gguf, out_pt)}
    print(f"layers {len(vec.directions)}  sizes {sizes}")
    return {"gguf": out_gguf, "pt": out_pt, "n_layers": len(vec.directions),
            "sizes": sizes}


@app.function(timeout=600, volumes={VOL: vol}, secrets=[HF_SECRET])  # CPU only
def gate_check() -> dict:
    """Confirm the HF token can pull gated Llama + Gemma configs/tokenizers
    (small download) before spending A100 minutes on the full model."""
    import os

    os.environ["HF_HOME"] = HF_CACHE
    from transformers import AutoConfig, AutoTokenizer

    out = {}
    for key, mid in MODELS.items():
        try:
            cfg = AutoConfig.from_pretrained(mid)
            AutoTokenizer.from_pretrained(mid)
            out[key] = {"ok": True, "n_layers": cfg.num_hidden_layers,
                        "hidden": cfg.hidden_size,
                        "inject_layer": _inject_layer(cfg.num_hidden_layers)}
        except Exception as e:  # noqa: BLE001
            out[key] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    vol.commit()
    for k, v in out.items():
        print(f"{k}: {v}")
    return out


@app.function(timeout=600, volumes={VOL: vol})  # CPU only
def pt_ingest_test(hidden: int = 3584, n_layers: int = 28) -> dict:
    """De-risk task (b): fabricate a unit-L2 dict[block->tensor], save .pt,
    reload through steerbench.load_vector, verify it becomes a usable
    SteeringVector. Proves the estimator-vector path before orch-1's file lands."""
    import sys

    sys.path.insert(0, "/root/src")
    import numpy as np
    import torch

    fake = {L: torch.nn.functional.normalize(torch.randn(hidden), dim=0)
            for L in range(1, n_layers)}
    path = f"{VOL}/orch1_test.pt"
    torch.save(fake, path)
    vol.commit()
    vol.reload()

    from steerbench.vectors import load_vector

    vec = load_vector(path)
    keys = sorted(vec.directions.keys())
    norms = {k: float(np.linalg.norm(vec.directions[k])) for k in keys[:3]}
    shape = vec.directions[keys[0]].shape
    dtype = str(vec.directions[keys[0]].dtype)
    print(f"ingested .pt -> keys {keys}")
    print(f"shape {shape}  dtype {dtype}  sample norms {norms}")
    # durable SteeringVector duck-types into repeng ControlModel.set_control,
    # which only reads .directions[layer]
    from steerbench.vectors import SteeringVector
    assert isinstance(vec, SteeringVector)
    assert vec.directions[keys[0]].shape == (hidden,)
    return {"keys": keys, "shape": list(shape), "dtype": dtype,
            "is_steering_vector": True}


def _agg(rows: list[dict], key: str, metrics: tuple[str, ...]) -> dict:
    """Group rows by `key`, return {kval: {metric: (mean, std)}} over seeds."""
    import numpy as np

    groups: dict = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    out = {}
    for kval, rs in sorted(groups.items()):
        out[kval] = {
            m: (float(np.mean([r[m] for r in rs])), float(np.std([r[m] for r in rs])))
            for m in metrics
        }
    return out


@app.local_entrypoint()
def probe(layer: int = 14) -> None:
    """Coarse 1-seed wide sweep to bracket the cliff. Prints, writes nothing."""
    coeffs = [-40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 120.0]
    res = dose_response.remote(layer=layer, coeffs=coeffs, seeds=[0])
    print(f"\nlayer {layer}  dir_norm {res['dir_norm']:.3f}  "
          f"resid_norm {res['resid_norm']:.2f}  wall {res['wall_s']:.0f}s")
    print(f"{'coeff':>7} {'alpha_n':>8} {'formality':>10} {'repeat':>8} {'ppl':>9}")
    for r in res["rows"]:
        print(f"{r['coeff']:>7.1f} {r['alpha_norm']:>8.3f} "
              f"{r['formality']:>10.3f} {r['repetition']:>8.3f} {r['ppl']:>9.2f}")


@app.local_entrypoint()
def train_export_base() -> None:
    import json

    meta = train_and_export.remote(
        model_id=MODEL_BASE, gguf_path=_gguf_path(MODEL_BASE))
    print(json.dumps(meta, indent=2))


@app.local_entrypoint()
def run_dose_base(layer: int = 14) -> None:
    run_dose(layer=layer, model_id=MODEL_BASE, stem="dose_response_base")


@app.local_entrypoint()
def run_dose(
    layer: int = 14, model_id: str = MODEL_ID, stem: str = "dose_response"
) -> None:
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gguf_path = _gguf_path(model_id) if model_id != MODEL_ID else GGUF_PATH
    coeffs = [-60.0, -40.0, -25.0, -15.0, -8.0, 0.0, 8.0, 15.0,
              25.0, 40.0, 60.0, 90.0, 130.0]
    seeds = [0, 1, 2]
    res = dose_response.remote(
        layer=layer, coeffs=coeffs, seeds=seeds,
        model_id=model_id, gguf_path=gguf_path)
    rows = res["rows"]

    import os

    os.makedirs("results", exist_ok=True)
    with open(f"results/{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["coeff", "seed", "alpha_norm", "formality",
                           "repetition", "ppl"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    agg = _agg(rows, "coeff", ("formality", "repetition", "ppl", "alpha_norm"))
    cs = sorted(agg)
    form_m = [agg[c]["formality"][0] for c in cs]
    form_s = [agg[c]["formality"][1] for c in cs]
    rep_m = [agg[c]["repetition"][0] for c in cs]
    ppl_m = [agg[c]["ppl"][0] for c in cs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.errorbar(cs, form_m, yerr=form_s, marker="o", capsize=3, color="C0")
    ax1.axvline(0, color="gray", ls=":", lw=1)
    ax1.set_xlabel("coefficient (raw)")
    ax1.set_ylabel("formality proxy (higher = more formal)")
    ax1.set_title(f"EFFECT — dose-response @ layer {layer} (0.50 depth)")
    ax1.grid(alpha=0.3)

    ax2.plot(cs, rep_m, marker="s", color="C3", label="repetition (1-distinct2)")
    ax2.set_xlabel("coefficient (raw)")
    ax2.set_ylabel("repetition rate", color="C3")
    ax2.axvline(0, color="gray", ls=":", lw=1)
    ax2b = ax2.twinx()
    ax2b.plot(cs, ppl_m, marker="^", color="C2", label="ppl (unsteered)")
    ax2b.set_ylabel("perplexity (unsteered model)", color="C2")
    ax2.set_title("COHERENCE — the cliff")
    ax2.grid(alpha=0.3)
    fig.suptitle(
        f"steerbench dose-response · {model_id.split('/')[-1]} · A100 · "
        f"3 seeds · layer {layer}/{res['n_layers']} · "
        f"||dir||={res['dir_norm']:.2f} resid_norm={res['resid_norm']:.1f} · "
        f"wall {res['wall_s']:.0f}s")
    fig.tight_layout()
    fig.savefig(f"results/{stem}.png", dpi=130)
    print(f"wrote results/{stem}.{{csv,png}}  resid_norm={res['resid_norm']:.2f}")

    print(f"\n{'coeff':>7} {'alpha_n':>8} {'formality':>16} {'repeat':>8} {'ppl':>8}")
    for c in cs:
        a = agg[c]
        print(f"{c:>7.1f} {a['alpha_norm'][0]:>8.3f} "
              f"{a['formality'][0]:>8.2f}±{a['formality'][1]:<5.2f} "
              f"{a['repetition'][0]:>8.3f} {a['ppl'][0]:>8.2f}")


@app.local_entrypoint()
def run_layer_sweep(target_alpha: float = 0.044, coeff: float = 0.0) -> None:
    """Primary: fixed-alpha (equal normalized strength per layer).
    Pass coeff>0 to instead run the secondary fixed-raw-coeff mode.
    """
    import csv
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(1, 28))  # each layer's own direction
    seeds = [0, 1, 2]
    if coeff > 0:
        res = layer_sweep.remote(seeds=seeds, layers=layers, coeff=coeff)
        tag, stem = f"coeff={coeff}", "layer_sweep_coeff"
    else:
        res = layer_sweep.remote(seeds=seeds, layers=layers, target_alpha=target_alpha)
        tag, stem = f"alpha_norm={target_alpha}", "layer_sweep"
    rows = res["rows"]
    n_layers = res["n_layers"]

    os.makedirs("results", exist_ok=True)
    fields = ["layer", "layer_pos", "seed", "dir_norm", "resid_norm", "coeff",
              "alpha_norm", "formality", "repetition", "ppl"]
    with open(f"results/{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    agg = _agg(rows, "layer",
               ("formality", "repetition", "ppl", "alpha_norm", "coeff"))
    ls = sorted(agg)
    form_m = [agg[layer]["formality"][0] for layer in ls]
    form_s = [agg[layer]["formality"][1] for layer in ls]
    rep_m = [agg[layer]["repetition"][0] for layer in ls]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.errorbar(ls, form_m, yerr=form_s, marker="o", capsize=3, color="C0",
                 label="formality")
    ax1.axhline(res["base_formality"], color="gray", ls="--", lw=1,
                label="baseline (coeff 0)")
    ax1.set_xlabel("layer index (absolute)")
    ax1.set_ylabel("formality proxy", color="C0")
    ax1b = ax1.twinx()
    ax1b.plot(ls, rep_m, marker="s", color="C3", alpha=0.6, label="repetition")
    ax1b.set_ylabel("repetition rate", color="C3")
    ax1.set_title(
        f"steerbench layer sweep · Qwen2.5-7B-Instruct · A100 · {tag} · "
        f"3 seeds · own-direction-per-layer · wall {res['wall_s']:.0f}s")
    ax1.grid(alpha=0.3)
    ax2 = ax1.secondary_xaxis(
        "top", functions=(lambda x: x / n_layers, lambda x: x * n_layers))
    ax2.set_xlabel("fraction of depth")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"results/{stem}.png", dpi=130)
    print(f"wrote results/{stem}.{{csv,png}}")

    # coherence-gated peak (exclude degenerate points)
    coherent = [layer for layer in ls
                if agg[layer]["repetition"][0] < 0.15 and agg[layer]["ppl"][0] < 6]
    best = max(coherent, key=lambda layer: agg[layer]["formality"][0])
    print(f"\nmode={res['mode']}  baseline formality: {res['base_formality']:.3f}")
    print(f"{'layer':>6} {'frac':>6} {'coeff':>8} {'alpha':>7} "
          f"{'formality':>16} {'repeat':>8} {'ppl':>8}")
    for layer in ls:
        a = agg[layer]
        mark = " <-- coherent peak" if layer == best else ""
        print(f"{layer:>6} {layer / n_layers:>6.2f} {a['coeff'][0]:>8.1f} "
              f"{a['alpha_norm'][0]:>7.3f} "
              f"{a['formality'][0]:>8.2f}±{a['formality'][1]:<5.2f} "
              f"{a['repetition'][0]:>8.3f} {a['ppl'][0]:>8.2f}{mark}")


# ---------------------------------------------------------------------------
# Cross-model comparison (Llama-3.1-8B-Instruct, Gemma-2-9b-it vs Qwen).
# Anchor: inject at INJECT_FRAC of depth, dose held fixed at SWEET_ALPHA.
# ---------------------------------------------------------------------------

# Dose grid in transferable ALPHA units (Qwen's coeff grid / its resid_norm).
# Includes 0, negatives, the SWEET_ALPHA anchor, and past-cliff values.
ALPHA_GRID = [-0.20, -0.131, -0.087, -0.055, -0.033, 0.0,
              0.033, SWEET_ALPHA, 0.055, 0.087, 0.131, 0.197, 0.284]


def _dose_artifacts(res, model_id, layer, stem):
    import csv
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["rows"]
    frac = layer / res["n_layers"]
    os.makedirs("results", exist_ok=True)
    with open(f"results/{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["coeff", "seed", "alpha_norm",
                                          "formality", "repetition", "ppl"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    agg = _agg(rows, "coeff", ("formality", "repetition", "ppl", "alpha_norm"))
    cs = sorted(agg)
    xs = [agg[c]["alpha_norm"][0] for c in cs]  # x-axis in transferable dose
    form_m = [agg[c]["formality"][0] for c in cs]
    form_s = [agg[c]["formality"][1] for c in cs]
    rep_m = [agg[c]["repetition"][0] for c in cs]
    ppl_m = [agg[c]["ppl"][0] for c in cs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.errorbar(xs, form_m, yerr=form_s, marker="o", capsize=3, color="C0")
    ax1.axvline(0, color="gray", ls=":", lw=1)
    ax1.axvline(SWEET_ALPHA, color="C1", ls="--", lw=1, label=f"anchor α={SWEET_ALPHA}")
    ax1.set_xlabel("alpha_norm (dimensionless dose)")
    ax1.set_ylabel("formality proxy (higher = more formal)")
    ax1.set_title(f"EFFECT — dose-response @ layer {layer} ({frac:.2f} depth)")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3)

    ax2.plot(xs, rep_m, marker="s", color="C3", label="repetition")
    ax2.set_xlabel("alpha_norm (dimensionless dose)")
    ax2.set_ylabel("repetition rate", color="C3")
    ax2.axvline(0, color="gray", ls=":", lw=1)
    ax2b = ax2.twinx()
    ax2b.plot(xs, ppl_m, marker="^", color="C2")
    ax2b.set_ylabel("perplexity (unsteered)", color="C2")
    ax2.set_title("COHERENCE — the cliff")
    ax2.grid(alpha=0.3)
    fig.suptitle(
        f"steerbench dose-response · {model_id.split('/')[-1]} · A100 · 3 seeds · "
        f"layer {layer}/{res['n_layers']} ({frac:.2f}) · "
        f"resid_norm={res['resid_norm']:.1f} · wall {res['wall_s']:.0f}s")
    fig.tight_layout()
    fig.savefig(f"results/{stem}.png", dpi=130)
    print(f"wrote results/{stem}.{{csv,png}}")


def _layer_artifacts(res, model_id, stem, tag):
    import csv
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["rows"]
    n_layers = res["n_layers"]
    os.makedirs("results", exist_ok=True)
    fields = ["layer", "layer_pos", "seed", "dir_norm", "resid_norm", "coeff",
              "alpha_norm", "formality", "repetition", "ppl"]
    with open(f"results/{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    agg = _agg(rows, "layer", ("formality", "repetition", "ppl", "coeff"))
    ls = sorted(agg)
    form_m = [agg[layer]["formality"][0] for layer in ls]
    form_s = [agg[layer]["formality"][1] for layer in ls]
    rep_m = [agg[layer]["repetition"][0] for layer in ls]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.errorbar(ls, form_m, yerr=form_s, marker="o", capsize=3, color="C0",
                 label="formality")
    ax1.axhline(res["base_formality"], color="gray", ls="--", lw=1,
                label="baseline")
    ax1.axvline(_inject_layer(n_layers), color="C1", ls=":", lw=1,
                label=f"{INJECT_FRAC} depth anchor")
    ax1.set_xlabel("layer index (absolute)")
    ax1.set_ylabel("formality proxy", color="C0")
    ax1b = ax1.twinx()
    ax1b.plot(ls, rep_m, marker="s", color="C3", alpha=0.6)
    ax1b.set_ylabel("repetition rate", color="C3")
    ax1.set_title(
        f"steerbench layer sweep · {model_id.split('/')[-1]} · A100 · {tag} · "
        f"3 seeds · own-direction-per-layer · wall {res['wall_s']:.0f}s")
    ax1.grid(alpha=0.3)
    ax1.secondary_xaxis(
        "top", functions=(lambda x: x / n_layers, lambda x: x * n_layers)
    ).set_xlabel("fraction of depth")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"results/{stem}.png", dpi=130)
    print(f"wrote results/{stem}.{{csv,png}}")

    coherent = [layer for layer in ls
                if agg[layer]["repetition"][0] < 0.15 and agg[layer]["ppl"][0] < 6]
    best = max(coherent or ls, key=lambda layer: agg[layer]["formality"][0])
    return {"peak_layer": best, "peak_frac": best / n_layers,
            "peak_formality": agg[best]["formality"][0],
            "baseline": res["base_formality"]}


@app.local_entrypoint()
def run_cross(model: str = "llama", skip_train: bool = False) -> None:
    """Full cross-model run for one model key: train -> dose (alpha grid at
    INJECT_FRAC depth) -> normalized layer sweep. Emits per-model artifacts."""
    import json

    assert model in MODELS, f"model must be one of {list(MODELS)}"
    model_id = MODELS[model]

    # Gemma is gated: probe access, auto-fall back to an ungated model.
    if model == "gemma":
        access = gate_check.remote().get("gemma", {}).get("ok", False)
        if not access:
            model_id = GEMMA_FALLBACK
            model = model_id.split("/")[-1].split("-")[0].lower()  # "mistral"
            print(f"gemma-2-9b-it gated -> falling back to {model_id} "
                  f"(report: Gemma deferred pending license)")

    gguf = _gguf_path(model_id)
    seeds = [0, 1, 2]

    if not skip_train:
        meta = train_and_export.remote(model_id=model_id, gguf_path=gguf)
        n_layers = meta["n_layers"]
        print(f"trained {model_id}: {json.dumps({k: meta[k] for k in ('n_layers', 'hidden_size', 'n_pairs')})}")
    else:
        n_layers = {"qwen": 28, "llama": 32, "gemma": 42}[model]

    layer = _inject_layer(n_layers)
    print(f"{model}: n_layers={n_layers}  inject layer={layer} "
          f"(frac {layer / n_layers:.3f}, target {INJECT_FRAC})")

    # dose-response at the anchored layer, alpha grid (past-cliff both ways)
    dres = dose_response.remote(
        layer=layer, seeds=seeds, alphas=ALPHA_GRID,
        model_id=model_id, gguf_path=gguf)
    _dose_artifacts(dres, model_id, layer, f"dose_response_{model}")

    # normalized layer sweep, each layer's own direction at fixed dose
    sres = layer_sweep.remote(
        seeds=seeds, layers=list(range(1, n_layers)), target_alpha=SWEET_ALPHA,
        model_id=model_id, gguf_path=gguf)
    peak = _layer_artifacts(sres, model_id, f"layer_sweep_{model}",
                            f"alpha_norm={SWEET_ALPHA}")

    dagg = _agg(dres["rows"], "coeff", ("formality", "alpha_norm", "repetition"))
    print(f"\n===== {model_id} =====")
    print(f"n_layers={n_layers}  inject L{layer} (frac {layer / n_layers:.3f})  "
          f"resid_norm@L={dres['resid_norm']:.1f}  "
          f"coeff@α{SWEET_ALPHA}={SWEET_ALPHA * dres['resid_norm']:.1f}")
    print(f"dose wall {dres['wall_s']:.0f}s  sweep wall {sres['wall_s']:.0f}s")
    print(f"layer-sweep coherent peak: L{peak['peak_layer']} "
          f"(frac {peak['peak_frac']:.2f})  formality {peak['peak_formality']:.2f} "
          f"vs baseline {peak['baseline']:.2f}")
    print(f"{'alpha':>8} {'formality':>16} {'repeat':>8}")
    for c in sorted(dagg):
        a = dagg[c]
        print(f"{a['alpha_norm'][0]:>8.3f} "
              f"{a['formality'][0]:>8.2f}±{a['formality'][1]:<5.2f} "
              f"{a['repetition'][0]:>8.3f}")
