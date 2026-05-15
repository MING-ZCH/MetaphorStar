#!/bin/bash
# ============================================================================
# Merge + Evaluate Script for Reverse Curriculum Checkpoints
# Usage: bash merge_and_eval_reverse_curriculum.sh [step_12|step_6|both|merge_only]
# Default: both
# ============================================================================
set -euo pipefail

MODE="${1:-both}"
BASE_DIR="/mnt/shared-storage-user/work/MetaphorStar"
EASYR1_DIR="/mnt/shared-storage-user/work/EasyR1"
CKPT_DIR="${BASE_DIR}/reverse_curriculum_sq_grpo_qwenvl2_5_7Bb_e3_n32_lr1.0e-6_kl0.01_eb0.005_20260306_214457"
LOG_DIR="${BASE_DIR}/logs/merge"
EVAL_DIR="${BASE_DIR}/evaluation"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  Reverse Curriculum: Merge & Evaluate"
echo "  Mode: ${MODE}"
echo "  Checkpoints: ${CKPT_DIR}"
echo "============================================================"

merge_checkpoint() {
    local STEP=$1
    local ACTOR_DIR="${CKPT_DIR}/global_step_${STEP}/actor"
    local HF_DIR="${ACTOR_DIR}/huggingface"
    echo ""
    echo "[MERGE] Step ${STEP}..."
    if ls "${HF_DIR}/"*.safetensors 1>/dev/null 2>&1; then
        echo "  Already merged: ${HF_DIR}"
        return 0
    fi
    if [ ! -f "${ACTOR_DIR}/model_world_size_2_rank_0.pt" ]; then
        echo "  ERROR: Source not found: ${ACTOR_DIR}/model_world_size_2_rank_0.pt"
        return 1
    fi
    python3 "${EASYR1_DIR}/scripts/model_merger.py" \
        --local_dir "${ACTOR_DIR}" \
        2>&1 | tee "${LOG_DIR}/merge_step_${STEP}.log"
    if ls "${HF_DIR}/"*.safetensors 1>/dev/null 2>&1; then
        echo "  Merge OK: $(du -sh "${HF_DIR}" | cut -f1)"
    else
        echo "  Merge FAILED"; return 1
    fi
}

eval_checkpoint() {
    local STEP=$1
    local HF_DIR="${CKPT_DIR}/global_step_${STEP}/actor/huggingface"
    local MODEL_NAME="reverse_curriculum_sq_grpo_qwenvl2_5_7Bb_e3_n32_lr1.0e-6_kl0.01_eb0.005_20260306_214457_step${STEP}"
    echo ""
    echo "============================================================"
    echo "  [EVAL] Step ${STEP} - Model: ${HF_DIR}"
    echo "============================================================"
    if [ ! -f "${HF_DIR}/config.json" ] || ! ls "${HF_DIR}/"*.safetensors 1>/dev/null 2>&1; then
        echo "  ERROR: Model not ready. Run merge first."; return 1
    fi
    cd "${BASE_DIR}"

    # 1. TFQ Grouped (PRIMARY target >=86%)
    echo "[1/5] TFQ Grouped..."
    mkdir -p "${EVAL_DIR}/II-Bench/TFQ_grouped"
    python3 "${EVAL_DIR}/qwen_eval_batch_grouped.py" \
        --model_path "${HF_DIR}" \
        --dataset_path "${EVAL_DIR}/II-Bench/eval/TFQ_II-Bench_eval_grouped.json" \
        --output_path "${EVAL_DIR}/II-Bench/TFQ_grouped/${MODEL_NAME}_result.json" \
        --image_dir "${EVAL_DIR}" --batch_size 16 --task_type TFQ 2>&1 | tail -3
    python3 "${EVAL_DIR}/eval_grouped.py" --task TFQ \
        --result_path "${EVAL_DIR}/II-Bench/TFQ_grouped/${MODEL_NAME}_result.json" \
        2>&1 | tee "${EVAL_DIR}/II-Bench/TFQ_grouped/${MODEL_NAME}_evaluation.txt"

    # 2. TFQ Individual
    echo "[2/5] TFQ Individual..."
    mkdir -p "${EVAL_DIR}/II-Bench/TFQ"
    python3 "${EVAL_DIR}/qwen_eval_batch_cli.py" \
        --model_path "${HF_DIR}" \
        --dataset_path "${EVAL_DIR}/II-Bench/eval/TFQ_II-Bench_eval.json" \
        --output_path "${EVAL_DIR}/II-Bench/TFQ/${MODEL_NAME}_result.json" \
        --image_dir "${EVAL_DIR}" --batch_size 16 --save_interval 2 2>&1 | tail -3
    python3 -c "
import sys, os
sys.path.insert(0, '${EVAL_DIR}')
os.chdir('${BASE_DIR}')
from eval_TFQ import evaluate_accuracy
r = evaluate_accuracy('${EVAL_DIR}/II-Bench/TFQ/${MODEL_NAME}_result.json')
with open('${EVAL_DIR}/II-Bench/TFQ/${MODEL_NAME}_evaluation.txt', 'w') as f:
    f.write(f'Per-Q Accuracy: {r[\"Accuracy\"]:.4f} ({r[\"Correct\"]}/{r[\"Total\"]})\n')
    if 'Group_Accuracy' in r: f.write(f'Group Accuracy: {r[\"Group_Accuracy\"]:.4f}\n')
print(f'  TFQ Individual: {r[\"Accuracy\"]:.2%}')
"

    # 3. MCQ Grouped
    echo "[3/5] MCQ Grouped..."
    mkdir -p "${EVAL_DIR}/II-Bench/MCQ_grouped"
    python3 "${EVAL_DIR}/qwen_eval_batch_grouped.py" \
        --model_path "${HF_DIR}" \
        --dataset_path "${EVAL_DIR}/II-Bench/eval/MCQ_II-Bench_eval.json" \
        --output_path "${EVAL_DIR}/II-Bench/MCQ_grouped/${MODEL_NAME}_result.json" \
        --image_dir "${EVAL_DIR}" --batch_size 16 --task_type MCQ 2>&1 | tail -3
    python3 "${EVAL_DIR}/eval_grouped.py" --task MCQ \
        --result_path "${EVAL_DIR}/II-Bench/MCQ_grouped/${MODEL_NAME}_result.json" \
        2>&1 | tee "${EVAL_DIR}/II-Bench/MCQ_grouped/${MODEL_NAME}_evaluation.txt"


    # # 5. OSQ Grouped
    # echo "[5/5] OSQ Grouped..."
    # mkdir -p "${EVAL_DIR}/II-Bench/OSQ_grouped"
    # if [ -f "${EVAL_DIR}/II-Bench/eval/OSQ_II-Bench_eval.json" ]; then
    #     python3 "${EVAL_DIR}/qwen_eval_batch_grouped.py" \
    #         --model_path "${HF_DIR}" \
    #         --dataset_path "${EVAL_DIR}/II-Bench/eval/OSQ_II-Bench_eval.json" \
    #         --output_path "${EVAL_DIR}/II-Bench/OSQ_grouped/${MODEL_NAME}_result.json" \
    #         --image_dir "${EVAL_DIR}" --batch_size 8 --task_type OSQ 2>&1 | tail -3
    #     python3 "${EVAL_DIR}/eval_grouped.py" --task OSQ \
    #         --result_path "${EVAL_DIR}/II-Bench/OSQ_grouped/${MODEL_NAME}_result.json" \
    #         2>&1 | tee "${EVAL_DIR}/II-Bench/OSQ_grouped/${MODEL_NAME}_evaluation.txt"
    # else
    #     echo "  OSQ eval dataset not found, skipping..."
    # fi

    # Summary
    echo ""
    echo "============================================================"
    echo "  RESULTS SUMMARY - Step ${STEP}"
    echo "============================================================"
    for dir in TFQ_grouped TFQ MCQ_grouped MCQ OSQ_grouped; do
        local F="${EVAL_DIR}/II-Bench/${dir}/${MODEL_NAME}_evaluation.txt"
        if [ -f "${F}" ]; then
            echo "  [${dir}]:"
            cat "${F}" | sed 's/^/    /'
        fi
    done
    echo "============================================================"
}

case "${MODE}" in
    step_188)   merge_checkpoint 44 eval_checkpoint 44 ;;
    step_6)  merge_checkpoint 88 eval_checkpoint 88 ;;
    both)
        merge_checkpoint 129
        merge_checkpoint 88
        merge_checkpoint 44
        eval_checkpoint 129
        eval_checkpoint 88
        eval_checkpoint 44
        ;;
    merge_only)
        merge_checkpoint 12
        merge_checkpoint 6
        echo "Merge done. Run eval with: bash $0 step_12 or step_6"
        ;;
    *)
        echo "Usage: bash $0 [step_12|step_6|both|merge_only]"
        exit 1
        ;;
esac

echo ""
echo "All tasks completed!"
