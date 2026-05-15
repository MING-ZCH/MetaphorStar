#!/bin/bash
# =====================================================================================
# DiVA-GRPO v3 One-Click Evaluation Script
# Runs both GROUPED and INDIVIDUAL evaluation for TFQ / MCQ / OSQ
# =====================================================================================
#
# This script evaluates a v3 model checkpoint using BOTH evaluation formats:
#   1) Grouped: TFQ/MCQ/OSQ with grouped prompts (one image + all SQs)
#   2) Individual: TFQ/MCQ/OSQ with individual prompts (one question at a time)
#
# Usage:
#   bash evaluation/run_v3_eval.sh [MODEL_PATH] [EVAL_SET] [EVAL_MODE]
#
# Arguments:
#   MODEL_PATH  - Checkpoint dir or huggingface path (auto-resolves latest step)
#   EVAL_SET    - "main" or "other" (default: main)
#   EVAL_MODE   - "both", "grouped", or "individual" (default: both)
#
# Environment variables (all optional):
#   MODEL_NAME     - Override auto-extracted model name
#   RESULT_NAME    - Override result filename prefix
#
# Examples:
#   # Full eval (grouped + individual) for Phase 1 checkpoint
#   bash evaluation/run_v3_eval.sh /path/to/diva_grpo_v3_phase1_TIMESTAMP
#
#   # Full eval for Phase 2 checkpoint
#   bash evaluation/run_v3_eval.sh /path/to/diva_grpo_v3_phase2_TIMESTAMP
#
#   # Only individual eval
#   bash evaluation/run_v3_eval.sh /path/to/checkpoint main individual
#
#   # Only grouped eval (same as run_grouped_eval.sh)
#   bash evaluation/run_v3_eval.sh /path/to/checkpoint main grouped
# =====================================================================================

set -e

MODEL_PATH="${1:-}"
EVAL_SET="${2:-main}"
EVAL_MODE="${3:-both}"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: bash evaluation/run_v3_eval.sh MODEL_PATH [EVAL_SET] [EVAL_MODE]"
    echo "  EVAL_SET:  main | other (default: main)"
    echo "  EVAL_MODE: both | grouped | individual (default: both)"
    exit 1
fi

WORK_DIR="/mnt/shared-storage-user/work/MetaphorStar"
EVAL_DIR="${WORK_DIR}/evaluation"
EVAL_DATA_DIR="${EVAL_DIR}/II-Bench/eval"
IMAGE_DIR="${EVAL_DIR}"

# =====================================================================================
# Auto-resolve MODEL_PATH to huggingface directory
# =====================================================================================
if [[ ! "$MODEL_PATH" =~ /huggingface$ ]]; then
    MODEL_PATH="${MODEL_PATH%/}"
    LATEST_STEP=$(ls -d "${MODEL_PATH}"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
    if [ -n "$LATEST_STEP" ] && [ -d "${LATEST_STEP}/actor/huggingface" ]; then
        HF_PATH="${LATEST_STEP}/actor/huggingface"
        CKPT_DIR="$MODEL_PATH"
    else
        echo "ERROR: Cannot find huggingface model under: $MODEL_PATH"
        echo "Expected: MODEL_PATH/global_step_*/actor/huggingface"
        exit 1
    fi
else
    HF_PATH="$MODEL_PATH"
    CKPT_DIR=$(dirname $(dirname $(dirname "$MODEL_PATH")))
fi

STEP_NAME=$(basename $(dirname $(dirname "$HF_PATH")))

# Model name for result files
if [ -z "${MODEL_NAME:-}" ]; then
    CKPT_BASENAME=$(basename "$CKPT_DIR")
    MODEL_NAME="${CKPT_BASENAME}_${STEP_NAME}"
fi
RESULT_NAME="${RESULT_NAME:-${MODEL_NAME}}"

echo "============================================================"
echo "  DiVA-GRPO v3 Evaluation Pipeline"
echo "============================================================"
echo "  Checkpoint:  $CKPT_DIR"
echo "  HF Model:    $HF_PATH"
echo "  Step:         $STEP_NAME"
echo "  Model Name:   $MODEL_NAME"
echo "  Eval Set:     $EVAL_SET"
echo "  Eval Mode:    $EVAL_MODE"
echo "============================================================"

if [ "$EVAL_SET" = "other" ]; then
    SUFFIX="_other"
else
    SUFFIX=""
fi

cd "$WORK_DIR"

# =====================================================================================
# GROUPED EVALUATION (uses qwen_eval_batch_grouped.py + eval_grouped.py)
# Prompt format: one image + all SQs → numbered T/F list
# Generation: do_sample=False (greedy)
# =====================================================================================
run_grouped_eval() {
    echo ""
    echo "########################################"
    echo "  GROUPED EVALUATION"
    echo "########################################"

    # --- TFQ Grouped ---
    echo ""
    echo "  [G-1/3] TFQ Grouped"
    TFQ_G_DATASET="${EVAL_DATA_DIR}/TFQ_II-Bench_eval${SUFFIX}_grouped.json"
    TFQ_G_OUTPUT_DIR="${EVAL_DIR}/II-Bench/TFQ_grouped${SUFFIX}"
    TFQ_G_RESULT="${TFQ_G_OUTPUT_DIR}/${RESULT_NAME}_eval_result.json"
    mkdir -p "$TFQ_G_OUTPUT_DIR"

    if [ -f "$TFQ_G_DATASET" ]; then
        echo "  → $TFQ_G_RESULT"
        python3 "${EVAL_DIR}/qwen_eval_batch_grouped.py" \
            --model_path "$HF_PATH" \
            --dataset_path "$TFQ_G_DATASET" \
            --output_path "$TFQ_G_RESULT" \
            --image_dir "$IMAGE_DIR" \
            --batch_size 16 \
            --save_interval 4 \
            --task_type TFQ

        python3 "${EVAL_DIR}/eval_grouped.py" \
            --task TFQ \
            --result_path "$TFQ_G_RESULT"
    else
        echo "  SKIP: TFQ grouped dataset not found: $TFQ_G_DATASET"
    fi

    # --- MCQ Grouped ---
    echo ""
    echo "  [G-2/3] MCQ Grouped"
    MCQ_G_DATASET="${EVAL_DATA_DIR}/MCQ_II-Bench_eval${SUFFIX}.json"
    MCQ_G_OUTPUT_DIR="${EVAL_DIR}/II-Bench/MCQ_grouped${SUFFIX}"
    MCQ_G_RESULT="${MCQ_G_OUTPUT_DIR}/${RESULT_NAME}_eval_result.json"
    mkdir -p "$MCQ_G_OUTPUT_DIR"

    if [ -f "$MCQ_G_DATASET" ]; then
        echo "  → $MCQ_G_RESULT"
        python3 "${EVAL_DIR}/qwen_eval_batch_grouped.py" \
            --model_path "$HF_PATH" \
            --dataset_path "$MCQ_G_DATASET" \
            --output_path "$MCQ_G_RESULT" \
            --image_dir "$IMAGE_DIR" \
            --batch_size 16 \
            --save_interval 5 \
            --task_type MCQ

        python3 "${EVAL_DIR}/eval_grouped.py" \
            --task MCQ \
            --result_path "$MCQ_G_RESULT"
    else
        echo "  SKIP: MCQ dataset not found"
    fi

    # --- OSQ Grouped ---
    echo ""
    echo "  [G-3/3] OSQ Grouped"
    OSQ_G_DATASET="${EVAL_DATA_DIR}/OSQ_II-Bench_eval${SUFFIX}.json"
    OSQ_G_OUTPUT_DIR="${EVAL_DIR}/II-Bench/OSQ_grouped${SUFFIX}"
    OSQ_G_RESULT="${OSQ_G_OUTPUT_DIR}/${RESULT_NAME}_eval_result.json"
    mkdir -p "$OSQ_G_OUTPUT_DIR"

    if [ -f "$OSQ_G_DATASET" ]; then
        echo "  → $OSQ_G_RESULT"
        python3 "${EVAL_DIR}/qwen_eval_batch_grouped.py" \
            --model_path "$HF_PATH" \
            --dataset_path "$OSQ_G_DATASET" \
            --output_path "$OSQ_G_RESULT" \
            --image_dir "$IMAGE_DIR" \
            --batch_size 16 \
            --save_interval 2 \
            --task_type OSQ

        python3 "${EVAL_DIR}/eval_grouped.py" \
            --task OSQ \
            --result_path "$OSQ_G_RESULT"
    else
        echo "  SKIP: OSQ dataset not found"
    fi
}

# =====================================================================================
# INDIVIDUAL EVALUATION (uses qwen_eval_batch_cli.py + eval_cli.py)
# Prompt format: one image + one question → single T/F or ABCD answer
# Generation: do_sample=True, temperature=0.5, top_p=0.9
# =====================================================================================
run_individual_eval() {
    echo ""
    echo "########################################"
    echo "  INDIVIDUAL EVALUATION"
    echo "########################################"

    # --- TFQ Individual ---
    echo ""
    echo "  [I-1/3] TFQ Individual"
    TFQ_I_DATASET="${EVAL_DATA_DIR}/TFQ_II-Bench_eval${SUFFIX}.json"
    TFQ_I_OUTPUT_DIR="${EVAL_DIR}/II-Bench/TFQ${SUFFIX}"
    TFQ_I_RESULT="${TFQ_I_OUTPUT_DIR}/${RESULT_NAME}_eval_result.json"
    mkdir -p "$TFQ_I_OUTPUT_DIR"

    if [ -f "$TFQ_I_DATASET" ]; then
        echo "  → $TFQ_I_RESULT"
        python3 "${EVAL_DIR}/qwen_eval_batch_cli.py" \
            --model_path "$HF_PATH" \
            --dataset_path "$TFQ_I_DATASET" \
            --output_path "$TFQ_I_RESULT" \
            --image_dir "$IMAGE_DIR" \
            --batch_size 16 \
            --save_interval 5

        python3 "${EVAL_DIR}/eval_cli.py" \
            --task TFQ \
            --result_path "$TFQ_I_RESULT"
    else
        echo "  SKIP: TFQ individual dataset not found: $TFQ_I_DATASET"
    fi

    # --- MCQ Individual ---
    echo ""
    echo "  [I-2/3] MCQ Individual"
    MCQ_I_DATASET="${EVAL_DATA_DIR}/MCQ_II-Bench_eval${SUFFIX}.json"
    MCQ_I_OUTPUT_DIR="${EVAL_DIR}/II-Bench/MCQ${SUFFIX}"
    MCQ_I_RESULT="${MCQ_I_OUTPUT_DIR}/${RESULT_NAME}_eval_result.json"
    mkdir -p "$MCQ_I_OUTPUT_DIR"

    if [ -f "$MCQ_I_DATASET" ]; then
        echo "  → $MCQ_I_RESULT"
        python3 "${EVAL_DIR}/qwen_eval_batch_cli.py" \
            --model_path "$HF_PATH" \
            --dataset_path "$MCQ_I_DATASET" \
            --output_path "$MCQ_I_RESULT" \
            --image_dir "$IMAGE_DIR" \
            --batch_size 16 \
            --save_interval 5

        python3 "${EVAL_DIR}/eval_cli.py" \
            --task MCQ \
            --result_path "$MCQ_I_RESULT"
    else
        echo "  SKIP: MCQ individual dataset not found"
    fi

    # --- OSQ Individual ---
    echo ""
    echo "  [I-3/3] OSQ Individual"
    OSQ_I_DATASET="${EVAL_DATA_DIR}/OSQ_II-Bench_eval${SUFFIX}.json"
    OSQ_I_OUTPUT_DIR="${EVAL_DIR}/II-Bench/OSQ${SUFFIX}"
    OSQ_I_RESULT="${OSQ_I_OUTPUT_DIR}/${RESULT_NAME}_eval_result.json"
    mkdir -p "$OSQ_I_OUTPUT_DIR"

    if [ -f "$OSQ_I_DATASET" ]; then
        echo "  → $OSQ_I_RESULT"
        python3 "${EVAL_DIR}/qwen_eval_batch_cli.py" \
            --model_path "$HF_PATH" \
            --dataset_path "$OSQ_I_DATASET" \
            --output_path "$OSQ_I_RESULT" \
            --image_dir "$IMAGE_DIR" \
            --batch_size 16 \
            --save_interval 2

        echo "  NOTE: OSQ individual scoring — check results manually or use custom scorer"
    else
        echo "  SKIP: OSQ individual dataset not found"
    fi
}

# =====================================================================================
# Main Execution
# =====================================================================================
case "${EVAL_MODE}" in
    grouped)
        run_grouped_eval
        ;;
    individual)
        run_individual_eval
        ;;
    both)
        run_grouped_eval
        run_individual_eval
        ;;
    *)
        echo "ERROR: EVAL_MODE must be 'both', 'grouped', or 'individual'. Got: ${EVAL_MODE}"
        exit 1
        ;;
esac

# =====================================================================================
# Summary
# =====================================================================================
echo ""
echo "============================================================"
echo "  All Evaluations Complete! (${EVAL_MODE})"
echo "============================================================"
echo "  Model:  $MODEL_NAME"
echo ""

# Collect and display all results
echo "  Results:"
for f in $(find "${EVAL_DIR}/II-Bench" -name "${RESULT_NAME}*evaluation*" -type f 2>/dev/null | sort); do
    echo "    $(basename $(dirname $f))/$(basename $f)"
done
echo "============================================================"
