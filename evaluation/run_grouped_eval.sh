#!/bin/bash
# ============================================================================
# One-click Grouped Evaluation Script for TFQ / MCQ / OSQ
# Runs batch inference + accuracy verification for all three tasks
# ============================================================================
#
# Usage:
#   bash run_grouped_eval.sh [MODEL_PATH] [EVAL_SET]
#
# Arguments:
#   MODEL_PATH  - Checkpoint dir or full huggingface path.
#                 If a checkpoint dir (without /global_step_*/actor/huggingface),
#                 the script auto-detects the latest step's huggingface path.
#   EVAL_SET    - "main" or "other" (default: main)
#
# Environment variables (all optional):
#   MODEL_NAME     - Override the auto-extracted short model name
#   RESULT_NAME    - Override the result filename prefix (default: MODEL_NAME)
#   TFQ_RESULT     - Full path override for TFQ result file
#   MCQ_RESULT     - Full path override for MCQ result file
#   OSQ_RESULT     - Full path override for OSQ result file
#
# Examples:
#   bash evaluation/run_grouped_eval.sh /path/to/checkpoint_dir
#   bash evaluation/run_grouped_eval.sh /path/to/checkpoint_dir other
#   MODEL_NAME=my_model bash evaluation/run_grouped_eval.sh /path/to/checkpoint_dir
#
# ============================================================================
# ONE-CLICK Evaluation Commands for DiVA-GRPO v2 models:
# ============================================================================
# 1) graduated_ga0.3_soft_p2.0_wu0.3:
#    bash evaluation/run_grouped_eval.sh \
#      /mnt/shared-storage-user/zhangchenhao/work/MetaphorStar/qwen2_5_vl_7b_MetaphorQA_diva_grpo_graduated_gd0.7_ca1.0_dafalse_adfalse_ga0.3_soft_p2.0_wu0.3_20260302_180237
#
# 2) none_ga0.3_soft_p2.0:
#    bash evaluation/run_grouped_eval.sh \
#      /mnt/shared-storage-user/zhangchenhao/work/MetaphorStar/qwen2_5_vl_7b_MetaphorQA_diva_grpo_none_ca0.0_dafalse_adfalse_ga0.3_soft_p2.0_20260302_174609
#
# 3) none_ga0.5_soft_p3.0_wu0.2:
#    bash evaluation/run_grouped_eval.sh \
#      /mnt/shared-storage-user/zhangchenhao/work/MetaphorStar/qwen2_5_vl_7b_MetaphorQA_diva_grpo_none_ca0.0_dafalse_adfalse_ga0.5_soft_p3.0_wu0.2_20260302_183318
# ============================================================================

set -e

MODEL_PATH="${1:-/mnt/shared-storage-user/zhangchenhao/work/MetaphorStar/qwen2_5_vl_7b_MetaphorQA_diva_grpo_graduated_gd0.7_ca1.0_dafalse_adfalse_ga0.3_soft_p2.0_wu0.3_20260302_180237}"
EVAL_SET="${2:-main}"

WORK_DIR="/mnt/shared-storage-user/zhangchenhao/work/MetaphorStar"
EVAL_DIR="${WORK_DIR}/evaluation"
EVAL_DATA_DIR="${EVAL_DIR}/II-Bench/eval"
IMAGE_DIR="${EVAL_DIR}"

# ============================================================================
# Auto-resolve MODEL_PATH: if not ending with /huggingface, find it
# ============================================================================
if [[ ! "$MODEL_PATH" =~ /huggingface$ ]]; then
    # Strip trailing slash
    MODEL_PATH="${MODEL_PATH%/}"
    # Find the latest global_step_* directory
    LATEST_STEP=$(ls -d "${MODEL_PATH}"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
    if [ -n "$LATEST_STEP" ] && [ -d "${LATEST_STEP}/actor/huggingface" ]; then
        HF_PATH="${LATEST_STEP}/actor/huggingface"
        CKPT_DIR="$MODEL_PATH"
    else
        echo "ERROR: Cannot find huggingface model under: $MODEL_PATH"
        echo "Expected structure: MODEL_PATH/global_step_*/actor/huggingface"
        exit 1
    fi
else
    HF_PATH="$MODEL_PATH"
    # Walk up to checkpoint root: .../global_step_X/actor/huggingface -> ...
    CKPT_DIR=$(dirname $(dirname $(dirname "$MODEL_PATH")))
fi

STEP_NAME=$(basename $(dirname $(dirname "$HF_PATH")))  # e.g. global_step_18

# ============================================================================
# Extract MODEL_NAME: use checkpoint directory basename + step
# Can be overridden via MODEL_NAME env var
# ============================================================================
if [ -z "${MODEL_NAME:-}" ]; then
    CKPT_BASENAME=$(basename "$CKPT_DIR")
    MODEL_NAME="${CKPT_BASENAME}_${STEP_NAME}"
fi

# RESULT_NAME can override the filename prefix (default = MODEL_NAME)
RESULT_NAME="${RESULT_NAME:-${MODEL_NAME}}"

echo "============================================================"
echo "  Grouped Evaluation Pipeline"
echo "============================================================"
echo "  Checkpoint: $CKPT_DIR"
echo "  HF Model:   $HF_PATH"
echo "  Step:        $STEP_NAME"
echo "  Model Name:  $MODEL_NAME"
echo "  Result Name: $RESULT_NAME"
echo "  Eval Set:    $EVAL_SET"
echo "============================================================"

if [ "$EVAL_SET" = "other" ]; then
    SUFFIX="_other"
else
    SUFFIX=""
fi

cd "$WORK_DIR"

# ======================== Step 1: TFQ ========================
echo ""
echo "==============================="
echo "  [1/3] TFQ Grouped Evaluation"
echo "==============================="

TFQ_DATASET="${EVAL_DATA_DIR}/TFQ_II-Bench_eval${SUFFIX}_grouped.json"
TFQ_OUTPUT_DIR="${EVAL_DIR}/II-Bench/TFQ_grouped${SUFFIX}"
# TFQ_RESULT can be overridden via env var; default uses RESULT_NAME
TFQ_RESULT="${TFQ_RESULT:-${TFQ_OUTPUT_DIR}/${RESULT_NAME}_eval_result.json}"

mkdir -p "$TFQ_OUTPUT_DIR"

if [ ! -f "$TFQ_DATASET" ]; then
    echo "ERROR: TFQ grouped dataset not found: $TFQ_DATASET"
    exit 1
fi

echo "  Result: $TFQ_RESULT"

python3 "${EVAL_DIR}/qwen_eval_batch_grouped.py" \
    --model_path "$HF_PATH" \
    --dataset_path "$TFQ_DATASET" \
    --output_path "$TFQ_RESULT" \
    --image_dir "$IMAGE_DIR" \
    --batch_size 4 \
    --save_interval 2 \
    --task_type TFQ

python3 "${EVAL_DIR}/eval_grouped.py" \
    --task TFQ \
    --result_path "$TFQ_RESULT"


# ======================== Step 2: MCQ ========================
echo ""
echo "==============================="
echo "  [2/3] MCQ Evaluation"
echo "==============================="

MCQ_DATASET="${EVAL_DATA_DIR}/MCQ_II-Bench_eval${SUFFIX}.json"
MCQ_OUTPUT_DIR="${EVAL_DIR}/II-Bench/MCQ_grouped${SUFFIX}"
# MCQ_RESULT can be overridden via env var; default uses RESULT_NAME
MCQ_RESULT="${MCQ_RESULT:-${MCQ_OUTPUT_DIR}/${RESULT_NAME}_eval_result.json}"

mkdir -p "$MCQ_OUTPUT_DIR"

if [ ! -f "$MCQ_DATASET" ]; then
    echo "WARNING: MCQ dataset not found. Skipping."
else
    echo "  Result: $MCQ_RESULT"

    python3 "${EVAL_DIR}/qwen_eval_batch_grouped.py" \
        --model_path "$HF_PATH" \
        --dataset_path "$MCQ_DATASET" \
        --output_path "$MCQ_RESULT" \
        --image_dir "$IMAGE_DIR" \
        --batch_size 16 \
        --save_interval 5 \
        --task_type MCQ

    python3 "${EVAL_DIR}/eval_grouped.py" \
        --task MCQ \
        --result_path "$MCQ_RESULT"
fi


# ======================== Step 3: OSQ ========================
echo ""
echo "==============================="
echo "  [3/3] OSQ Evaluation"
echo "==============================="

OSQ_DATASET="${EVAL_DATA_DIR}/OSQ_II-Bench_eval${SUFFIX}.json"
OSQ_OUTPUT_DIR="${EVAL_DIR}/II-Bench/OSQ_grouped${SUFFIX}"
# OSQ_RESULT can be overridden via env var; default uses RESULT_NAME
OSQ_RESULT="${OSQ_RESULT:-${OSQ_OUTPUT_DIR}/${RESULT_NAME}_eval_result.json}"

mkdir -p "$OSQ_OUTPUT_DIR"

if [ ! -f "$OSQ_DATASET" ]; then
    echo "WARNING: OSQ dataset not found. Skipping."
else
    echo "  Result: $OSQ_RESULT"

    python3 "${EVAL_DIR}/qwen_eval_batch_grouped.py" \
        --model_path "$HF_PATH" \
        --dataset_path "$OSQ_DATASET" \
        --output_path "$OSQ_RESULT" \
        --image_dir "$IMAGE_DIR" \
        --batch_size 16 \
        --save_interval 2 \
        --task_type OSQ

    python3 "${EVAL_DIR}/eval_grouped.py" \
        --task OSQ \
        --result_path "$OSQ_RESULT"
fi


# ======================== Summary ========================
echo ""
echo "============================================================"
echo "  All Evaluations Complete!"
echo "============================================================"
echo "  Model:  $MODEL_NAME"
echo "  TFQ: ${TFQ_RESULT}"
[ -f "$MCQ_RESULT" ] && echo "  MCQ: ${MCQ_RESULT}"
[ -f "$OSQ_RESULT" ] && echo "  OSQ: ${OSQ_RESULT}"
echo "============================================================"
