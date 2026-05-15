#!/usr/bin/env python3
"""
CLI wrapper for qwen_eval_batch.py (individual evaluation).
Enables calling batch inference with command-line arguments instead of hardcoded paths.

Usage:
    python3 evaluation/qwen_eval_batch_cli.py \
        --model_path reverse_curriculum_sq_grpo_3B_e1_n64_lr5.0e-7_kl0.01_eb0.005_20260305_053953/global_step_6/actor/huggingface \
        --dataset_path evaluation/II-Bench/eval/TFQ_II-Bench_eval.json \
        --output_path evaluation/II-Bench/TFQ/result.json \
        --image_dir evaluation \
        --batch_size 16 \
        --save_interval 2
"""

import os
import sys
import argparse
import torch
import time
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import gc
from concurrent.futures import ThreadPoolExecutor

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import json

# Import core functions from qwen_eval_batch
sys.path.insert(0, os.path.dirname(__file__))
from qwen_eval_batch import (
    load_model,
    preload_images,
    process_batch,
    generate_batch_response,
    cleanup_memory,
)


def main():
    parser = argparse.ArgumentParser(description="Individual batch evaluation for Qwen2.5-VL")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HuggingFace model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to eval dataset JSON")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save results JSON")
    parser.add_argument("--image_dir", type=str, default="evaluation", help="Base directory for images")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--save_interval", type=int, default=2, help="Save results every N batches")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load model
    processor, model = load_model(args.model_path)

    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"Error: dataset should be a list, got: {type(data)}")
        return

    # Check for existing results (resume support)
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded existing results: {len(results)} items")
    else:
        results = []

    processed_ids = {item['Question_ID'] for item in results}

    # Filter unprocessed data
    unprocessed_data = [item for item in data if item['id'] not in processed_ids]
    print(f"Pending: {len(unprocessed_data)} items")

    if not unprocessed_data:
        print("All data already processed!")
        return

    # Preload images
    image_paths = [os.path.join(args.image_dir, item['url']) for item in unprocessed_data]
    preloaded_images = preload_images(image_paths)

    # Prepare batch items
    batch_items = []
    for item in unprocessed_data:
        image_path = os.path.join(args.image_dir, item['url'])
        if image_path in preloaded_images:
            batch_items.append({
                'id': item['id'],
                'image_path': image_path,
                'image_file': item['url'],
                'question': item['eval_question'],  # Use eval_question for prompt
                'ground_truth': item['correct_answer']
            })

    print(f"Prepared {len(batch_items)} items for inference")

    # Batch processing loop
    total_batches = (len(batch_items) + args.batch_size - 1) // args.batch_size
    processed_count = 0

    for batch_idx in tqdm(range(total_batches), desc="Batches", colour="green"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(batch_items))
        batch = batch_items[start_idx:end_idx]

        inputs = process_batch(batch, processor, preloaded_images)
        if inputs is None:
            continue

        responses = generate_batch_response(model, inputs, processor)
        if responses is None:
            continue

        for i, response in enumerate(responses):
            if i < len(batch):
                item = batch[i]
                results.append({
                    'Question_ID': item['id'],
                    'Image_URL': item['image_file'],
                    'Question': item['question'],
                    'Ground_Truth': item['ground_truth'],
                    'Model_Output': response
                })
                processed_count += 1

        # Periodic save
        if (batch_idx + 1) % args.save_interval == 0:
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  Saved {len(results)} results")

        cleanup_memory()

    # Final save
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nComplete! {processed_count} new items processed. Total: {len(results)}")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
