#!/bin/bash
# ============================================================================
# Evaluation Script for Reverse Curriculum Model
# ============================================================================
# Runs all 5 evaluation tasks:
#   1. TFQ grouped  (primary target: >=86% group accuracy)
#   2. TFQ individual (target: >=70% group accuracy, >=95% per-Q)
#   3. MCQ grouped
#   4. MCQ individual (target: >=74%)
#   5. OSQ grouped
#
# Usage:
#   bash examples/eval_reverse_curriculum.sh <model_path>
#
# Example:
#   bash examples/eval_reverse_curriculum.sh \
#     /mnt/shared-storage-user/zhangchenhao/work/MetaphorStar/reverse_curriculum_sq_grpo_e3_.../global_step_21/actor/huggingface
# ============================================================================
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <model_path>"
    echo "  model_path: Path to HuggingFace checkpoint (e.g., .../global_step_XX/actor/huggingface)"
    exit 1
fi

MODEL_PATH="$1"
BASE_DIR="/mnt/shared-storage-user/zhangchenhao/work/MetaphorStar"
EVAL_DIR="${BASE_DIR}/evaluation"

# Extract a short name for output files
# Try to extract experiment_name from path like .../reverse_curriculum_.../global_step_XX/actor/huggingface
STEP_DIR=$(basename "$(dirname "$(dirname "${MODEL_PATH}")")")
EXP_DIR=$(basename "$(dirname "$(dirname "$(dirname "${MODEL_PATH}")")")")
MODEL_NAME="${EXP_DIR}_${STEP_DIR}"
if [ -z "${MODEL_NAME}" ] || [ "${MODEL_NAME}" = "_" ]; then
    MODEL_NAME="reverse_curriculum_$(date +%Y%m%d_%H%M%S)"
fi

echo "============================================================"
echo "  Reverse Curriculum Evaluation"
echo "============================================================"
echo "  Model:  ${MODEL_PATH}"
echo "  Name:   ${MODEL_NAME}"
echo "============================================================"

# Verify model exists
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    exit 1
fi

cd "${BASE_DIR}"

# ============================================================================
# 1. TFQ Grouped Evaluation (PRIMARY metric)
# ============================================================================
echo ""
echo "[1/5] TFQ Grouped Evaluation..."
TFQ_GROUP_OUTPUT="${EVAL_DIR}/II-Bench/TFQ_grouped/${MODEL_NAME}_result.json"
mkdir -p "$(dirname "${TFQ_GROUP_OUTPUT}")"

python3 "${EVAL_DIR}/qwen_eval_batch_grouped.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${EVAL_DIR}/II-Bench/eval/TFQ_II-Bench_eval_grouped.json" \
    --output_path "${TFQ_GROUP_OUTPUT}" \
    --image_dir "${EVAL_DIR}" \
    --batch_size 8 \
    --task_type TFQ

echo "Scoring TFQ Grouped..."
python3 "${EVAL_DIR}/eval_grouped.py" \
    --task TFQ \
    --result_path "${TFQ_GROUP_OUTPUT}"

# ============================================================================
# 2. TFQ Individual Evaluation
# ============================================================================
echo ""
echo "[2/5] TFQ Individual Evaluation..."
TFQ_INDIV_OUTPUT="${EVAL_DIR}/II-Bench/TFQ/${MODEL_NAME}_result.json"
mkdir -p "$(dirname "${TFQ_INDIV_OUTPUT}")"

python3 "${EVAL_DIR}/qwen_eval_batch_cli.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${EVAL_DIR}/II-Bench/eval/TFQ_II-Bench_eval.json" \
    --output_path "${TFQ_INDIV_OUTPUT}" \
    --image_dir "${EVAL_DIR}" \
    --batch_size 16 \
    --save_interval 5

echo "Scoring TFQ Individual..."
python3 -c "
import sys, os
sys.path.insert(0, '${EVAL_DIR}')
os.chdir('${BASE_DIR}')
from eval_TFQ import evaluate_accuracy
results = evaluate_accuracy('${TFQ_INDIV_OUTPUT}')
print(f'Individual pass@1: {results[\"Accuracy\"]:.2%} ({results[\"Correct\"]}/{results[\"Total\"]})')
print(f'Group-level: {results[\"Group_Accuracy\"]:.2%} ({results[\"Group_Correct\"]}/{results[\"Total_Groups\"]})')
output_txt = '${TFQ_INDIV_OUTPUT}'.replace('.json', '_evaluation.txt')
with open(output_txt, 'w') as f:
    f.write(f'Individual pass@1 accuracy: {results[\"Accuracy\"]:.2%} ({results[\"Correct\"]}/{results[\"Total\"]})\n')
    f.write(f'Group-level accuracy: {results[\"Group_Accuracy\"]:.2%} ({results[\"Group_Correct\"]}/{results[\"Total_Groups\"]})\n')
print(f'Saved to: {output_txt}')
"

# ============================================================================
# 3. MCQ Grouped Evaluation
# ============================================================================
echo ""
echo "[3/5] MCQ Grouped Evaluation..."
MCQ_GROUP_OUTPUT="${EVAL_DIR}/II-Bench/MCQ_grouped/${MODEL_NAME}_result.json"
mkdir -p "$(dirname "${MCQ_GROUP_OUTPUT}")"

python3 "${EVAL_DIR}/qwen_eval_batch_grouped.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${EVAL_DIR}/II-Bench/eval/TFQ_II-Bench_eval_grouped.json" \
    --output_path "${MCQ_GROUP_OUTPUT}" \
    --image_dir "${EVAL_DIR}" \
    --batch_size 8 \
    --task_type MCQ

echo "Scoring MCQ Grouped..."
python3 "${EVAL_DIR}/eval_grouped.py" \
    --task MCQ \
    --result_path "${MCQ_GROUP_OUTPUT}"

# ============================================================================
# 4. MCQ Individual Evaluation
# ============================================================================
echo ""
echo "[4/5] MCQ Individual Evaluation..."
MCQ_INDIV_OUTPUT="${EVAL_DIR}/II-Bench/MCQ/${MODEL_NAME}_result.json"
mkdir -p "$(dirname "${MCQ_INDIV_OUTPUT}")"

python3 "${EVAL_DIR}/qwen_eval_batch_cli.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${EVAL_DIR}/II-Bench/eval/MCQ_II-Bench_eval.json" \
    --output_path "${MCQ_INDIV_OUTPUT}" \
    --image_dir "${EVAL_DIR}" \
    --batch_size 16 \
    --save_interval 2

echo "Scoring MCQ Individual..."
python3 -c "
import sys, os
sys.path.insert(0, '${EVAL_DIR}')
os.chdir('${BASE_DIR}')
from eval_MCQ import evaluate_accuracy
results = evaluate_accuracy('${MCQ_INDIV_OUTPUT}')
print(f'MCQ accuracy: {results[\"Accuracy\"]:.2%} ({results[\"Correct\"]}/{results[\"Total\"]})')
output_txt = '${MCQ_INDIV_OUTPUT}'.replace('.json', '_evaluation.txt')
with open(output_txt, 'w') as f:
    f.write(f'MCQ accuracy: {results[\"Accuracy\"]:.2%} ({results[\"Correct\"]}/{results[\"Total\"]})\n')
print(f'Saved to: {output_txt}')
"

# ============================================================================
# 5. OSQ Grouped Evaluation
# ============================================================================
echo ""
echo "[5/5] OSQ Grouped Evaluation..."
OSQ_GROUP_OUTPUT="${EVAL_DIR}/II-Bench/OSQ_grouped/${MODEL_NAME}_result.json"
mkdir -p "$(dirname "${OSQ_GROUP_OUTPUT}")"

python3 "${EVAL_DIR}/qwen_eval_batch_grouped.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${EVAL_DIR}/II-Bench/eval/TFQ_II-Bench_eval_grouped.json" \
    --output_path "${OSQ_GROUP_OUTPUT}" \
    --image_dir "${EVAL_DIR}" \
    --batch_size 8 \
    --task_type OSQ

echo "Scoring OSQ Grouped..."
python3 "${EVAL_DIR}/eval_grouped.py" \
    --task OSQ \
    --result_path "${OSQ_GROUP_OUTPUT}"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================================"
echo "  Evaluation Complete: ${MODEL_NAME}"
echo "============================================================"
echo ""
echo "Result files:"
echo "  TFQ Grouped:    ${TFQ_GROUP_OUTPUT/.json/_evaluation.txt}"
echo "  TFQ Individual: ${TFQ_INDIV_OUTPUT/.json/_evaluation.txt}"
echo "  MCQ Grouped:    ${MCQ_GROUP_OUTPUT/.json/_evaluation.txt}"
echo "  MCQ Individual: ${MCQ_INDIV_OUTPUT/.json/_evaluation.txt}"
echo "  OSQ Grouped:    ${OSQ_GROUP_OUTPUT/.json/_evaluation.txt}"
echo ""
echo "Target Comparison:"
echo "  TFQ Grouped per-group:     target >= 86% (SQ-GRPO v0 baseline)"
echo "  TFQ Individual per-group:  target >= 70% (Reasoning GRPO baseline)"
echo "  TFQ Individual per-Q:      target >= 95.53% (Reasoning GRPO baseline)"
echo "  MCQ Individual:            target >= 74% (Reasoning GRPO baseline)"
echo "============================================================"
