#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=QwenVL/Qwen2.5-VL-3B-Instruct # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=MING-ZCH/TFQ-Data-Lite@train \
    data.val_files=MING-ZCH/TFQ-Bench-Lite@test \
    data.format_prompt=./examples/format_prompt/TFQ-GRPO_format.jinja \
    data.max_prompt_length=5500 \
    data.max_response_length=2048 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/TFQ-GRPO_reward.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_3b_TFQ_Data_Lite_reasoning_grpo \
    trainer.n_gpus_per_node=2