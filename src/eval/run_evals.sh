#!/usr/bin/env bash
# Run lm-eval-harness across all Seer checkpoints (base / sft / dpo) and build
# a before/after comparison table + figure.
#
# Buckets:
#   capability control : hellaswag, arc_e/c, piqa, winogrande, lambada  (0-shot)
#   capability (5-shot): mmlu                                           (optional)
#   instruction        : ifeval                                        (generative)
#   alignment-adjacent : truthfulqa_mc2  (real alignment = reward-acc/AlpacaEval, separate)
#
# Usage (Colab):
#   INSTALL=1 bash src/eval/run_evals.sh        # first run: also pip-installs deps
#   bash src/eval/run_evals.sh                  # subsequent runs
#   RUN_MMLU=1 bash src/eval/run_evals.sh       # include the slow 5-shot MMLU
#
# Edit the MODELS map below to point at your checkpoint dirs (HF format).

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. CHECKPOINTS  — edit these paths (name -> HF model dir)
# ---------------------------------------------------------------------------
declare -A MODELS=(
  ["base"]="/content/ckpt/hf_seer_124m"
  ["sft"]="/content/ckpt/sft_out/checkpoint-1500"
  ["dpo"]="/content/ckpt/dpo_results"
)

# ---------------------------------------------------------------------------
# 2. KNOBS (override via env: DEVICE=cpu bash run_evals.sh)
# ---------------------------------------------------------------------------
OUT_ROOT="${OUT_ROOT:-results/evals}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
BATCH="${BATCH:-auto}"
RUN_MMLU="${RUN_MMLU:-0}"
INSTALL="${INSTALL:-0}"

CAP_TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai,truthfulqa_mc2"
IF_TASKS="ifeval"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 3. DEPS (optional)
# ---------------------------------------------------------------------------
if [[ "$INSTALL" == "1" ]]; then
  echo ">>> installing eval deps"
  pip install -q "lm-eval[ifeval]" matplotlib numpy
fi

mkdir -p "$OUT_ROOT"

# ---------------------------------------------------------------------------
# 4. RUN
# ---------------------------------------------------------------------------
for name in "${!MODELS[@]}"; do
  path="${MODELS[$name]}"
  if [[ ! -e "$path" ]]; then
    echo "!!! skipping '$name' — path not found: $path"
    continue
  fi
  echo ""
  echo "==================== $name : $path ===================="
  margs="pretrained=${path},dtype=${DTYPE}"

  echo ">>> [$name] capability control (0-shot)"
  lm_eval --model hf --model_args "$margs" \
    --tasks "$CAP_TASKS" \
    --batch_size "$BATCH" --device "$DEVICE" \
    --output_path "$OUT_ROOT/$name/capability"

  echo ">>> [$name] instruction following (ifeval)"
  lm_eval --model hf --model_args "$margs" \
    --tasks "$IF_TASKS" \
    --batch_size "$BATCH" --device "$DEVICE" \
    --output_path "$OUT_ROOT/$name/ifeval"

  if [[ "$RUN_MMLU" == "1" ]]; then
    echo ">>> [$name] MMLU (5-shot, slow)"
    lm_eval --model hf --model_args "$margs" \
      --tasks mmlu --num_fewshot 5 \
      --batch_size "$BATCH" --device "$DEVICE" \
      --output_path "$OUT_ROOT/$name/mmlu"
  fi
done

# ---------------------------------------------------------------------------
# 5. AGGREGATE -> CSV + FIGURE
# ---------------------------------------------------------------------------
echo ""
echo ">>> aggregating results -> $OUT_ROOT/eval_summary.csv + figure"
python "$SCRIPT_DIR/aggregate_evals.py" --root "$OUT_ROOT" --out "$OUT_ROOT"

echo ""
echo "DONE. Download the '$OUT_ROOT' folder:"
echo "  - eval_summary.csv      (the before/after table)"
echo "  - eval_comparison.png/.pdf  (grouped-bar figure)"
