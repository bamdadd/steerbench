#!/usr/bin/env bash
# Reproduce the steerbench small-model canonical sweeps (Qwen2.5-1.5B, T4).
# One command: trains the formality vector, runs the dose-response + both layer
# sweeps, renders CSV+PNG into results/. Prints the Modal GPU spend at the end.
#
# Prereqs: Modal authed (`modal token new`), run from the repo root so the
# `src/` mount resolves. HF cache + trained gguf persist on the Modal volume.
#
# Usage:
#   experiments/reproduce.sh            # 1.5B primary run
#   MODEL=0.5b experiments/reproduce.sh # 0.5B secondary run
set -euo pipefail

MODAL="${MODAL:-modal}"
APP="experiments/modal_app.py"
MODEL="${MODEL:-1.5b}"

if [[ "$MODEL" == "1.5b" ]]; then
  MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; NLAYERS=28; ANCHOR=17; STEM="qwen1.5b"
  GGUF=""                       # 1.5B is the default MODEL_ID -> uses GGUF_PATH
elif [[ "$MODEL" == "0.5b" ]]; then
  MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"; NLAYERS=24; ANCHOR=15; STEM="qwen0.5b"
  GGUF="/vol/formality_Qwen2_5-0_5B-Instruct.gguf"
else
  echo "MODEL must be 1.5b or 0.5b" >&2; exit 1
fi

# alpha for the resolved layer sweep. 0.044 (the transferable 7B dose) is below
# the resolving floor on small models; 0.09 is the strongest-still-coherent dose
# read from the anchor-layer dose-response usable band (see RESULTS_small_model.md).
SWEEP_ALPHA="${SWEEP_ALPHA:-0.09}"

echo "== 1/4 train formality ControlVector ($MODEL_ID) =="
if [[ "$MODEL" == "1.5b" ]]; then
  "$MODAL" run "$APP"::train_and_export --model-id "$MODEL_ID"
else
  "$MODAL" run "$APP"::train_and_export --model-id "$MODEL_ID" --gguf-path "$GGUF"
fi

echo "== 2/4 dose-response at the 0.61 anchor layer (L$ANCHOR) — maps cliff =="
"$MODAL" run "$APP"::run_dose --layer "$ANCHOR" --model-id "$MODEL_ID" \
  --stem "dose_response_$STEM"

echo "== 3/4 layer sweep @ alpha 0.044 (documented below-floor point) =="
"$MODAL" run "$APP"::run_layer_sweep --target-alpha 0.044 --model-id "$MODEL_ID" \
  --n-layers "$NLAYERS" --stem "layer_sweep_${STEM}_a044"

echo "== 4/4 layer sweep @ alpha $SWEEP_ALPHA (resolved, canonical) =="
"$MODAL" run "$APP"::run_layer_sweep --target-alpha "$SWEEP_ALPHA" \
  --model-id "$MODEL_ID" --n-layers "$NLAYERS" --stem "layer_sweep_$STEM"

echo
echo "== done. artifacts in results/{dose_response,layer_sweep}_$STEM.{csv,png} =="
echo "GPU spend: see https://modal.com/apps/$($MODAL profile current 2>/dev/null || echo bamdad) "
echo "(Modal dashboard shows per-app T4 seconds x rate; this run is a handful of"
echo " minutes of T4 at ~\$0.59/hr — well under a dollar.)"
