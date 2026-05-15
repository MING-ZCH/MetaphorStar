"""
Grouped evaluation batch inference for SQ-GRPO models.
Each sample = one image + all its sub-questions (TFQ/MCQ/OSQ) bundled together.
This matches the SQ-GRPO training paradigm where the model sees all questions for one image at once.

Usage:
    python evaluation/qwen_eval_batch_grouped.py \
        --model_path /mnt/shared-storage-user/zhangchenhao/work/MetaphorStar/reverse_curriculum_sq_grpo_3B_e1_n64_lr5.0e-7_kl0.01_eb0.005_20260305_053953/global_step_6/actor/huggingface \
        --dataset_path evaluation/II-Bench/eval/TFQ_II-Bench_eval_grouped.json \
        --output_path evaluation/II-Bench/TFQ_grouped/qwen2_5_vl_7b_MetaphorQA_sq_grpo_result.json \
        --image_dir evaluation \
        --batch_size 8 \
        --task_type TFQ
"""

import os
from sympy import false
import torch
import time
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import gc
from concurrent.futures import ThreadPoolExecutor

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import json


def load_model(model_path):
    """Load model with multi-GPU support."""
    print(f"Loading model: {model_path}")
    
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    
    special_tokens = ["<think>", "</think>", "<answer>", "</answer>"]
    print("\nVerifying special tokens:")
    for token in special_tokens:
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        exists = token_id != processor.tokenizer.unk_token_id
        print(f"  {token}: {'OK (ID: ' + str(token_id) + ')' if exists else 'NOT FOUND'}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).eval()
    
    print(f"Model device map: {model.hf_device_map}")
    return processor, model


def preload_images(image_paths, max_workers=4):
    """Parallel image preloading."""
    def load_single_image(image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            return image_path, image
        except Exception as e:
            return image_path, None
    
    images = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(load_single_image, image_paths),
            total=len(image_paths),
            desc="Preloading images",
            colour="green"
        ))
    
    success_count = sum(1 for _, img in results if img is not None)
    for path, image in results:
        if image is not None:
            images[path] = image
    
    print(f"Images loaded: {success_count}/{len(image_paths)}")
    return images


def process_batch(batch_data, processor, preloaded_images):
    """Process a batch of grouped samples."""
    texts = []
    images = []
    valid_items = []
    
    for item in batch_data:
        image_path = item['image_path']
        question = item['question']
        
        if image_path not in preloaded_images:
            continue
        
        image = preloaded_images[image_path]
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        texts.append(text)
        images.append(image)
        valid_items.append(item)
    
    if not texts:
        return None, []
    
    try:
        if processor.tokenizer.padding_side != "left":
            processor.tokenizer.padding_side = "left"
        
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        return inputs, valid_items
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return None, []


def generate_batch_response(model, inputs, processor, max_new_tokens=8192):
    """Batch generation."""
    if inputs is None:
        return []
    
    try:
        with torch.no_grad():
            if hasattr(inputs, 'input_ids'):
                input_ids = inputs.input_ids
            elif 'input_ids' in inputs:
                input_ids = inputs['input_ids']
            else:
                return []
            
            pad_token_id = processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = processor.tokenizer.eos_token_id
            
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # temperature=0.5,
                # top_p=0.9,
                pad_token_id=pad_token_id,
                use_cache=True
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            
            responses = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return responses
    except Exception as e:
        print(f"Generation failed: {e}")
        return []


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Grouped evaluation batch inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to grouped eval JSON")
    parser.add_argument("--output_path", type=str, required=True, help="Output result JSON path")
    parser.add_argument("--image_dir", type=str, default="evaluation", help="Base dir for image paths")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (smaller for grouped since prompts are longer)")
    parser.add_argument("--save_interval", type=int, default=2, help="Save every N batches")
    parser.add_argument("--task_type", type=str, default="TFQ", choices=["TFQ", "MCQ", "OSQ"], help="Task type")
    args = parser.parse_args()
    
    os.chdir('/mnt/shared-storage-user/work/MetaphorStar')
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load model
    processor, model = load_model(args.model_path)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Total groups: {len(data)}")
    
    # Check existing results
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded existing results: {len(results)}")
    else:
        results = []
    
    processed_ids = {item['Group_ID'] for item in results}
    
    # Filter unprocessed
    unprocessed_data = [item for item in data if str(item['id']) not in processed_ids]
    print(f"Unprocessed groups: {len(unprocessed_data)}")
    
    if not unprocessed_data:
        print("All groups already processed!")
        return
    
    # Preload images
    image_paths = list(set(os.path.join(args.image_dir, item['url']) for item in unprocessed_data))
    preloaded_images = preload_images(image_paths)
    
    # Prepare batch items
    batch_items = []
    for item in unprocessed_data:
        image_path = os.path.join(args.image_dir, item['url'])
        if image_path in preloaded_images:
            batch_items.append({
                'id': str(item['id']),
                'image_path': image_path,
                'image_file': item['url'],
                'question': item['eval_question'],
                'ground_truth': item['correct_answer'],
                'num_questions': item.get('num_questions', 1),
                'original_ids': item.get('original_ids', []),
            })
    
    print(f"Batch items ready: {len(batch_items)}")
    
    # Main batch processing loop
    batch_size = args.batch_size
    total_batches = (len(batch_items) + batch_size - 1) // batch_size
    total_time = 0
    success_batches = 0
    
    pbar = tqdm(total=len(batch_items), desc="Overall progress", colour="red")
    
    for batch_idx in range(0, len(batch_items), batch_size):
        batch_data = batch_items[batch_idx:batch_idx + batch_size]
        current_batch = batch_idx // batch_size + 1
        
        if not batch_data:
            continue
        
        try:
            inputs, valid_items = process_batch(batch_data, processor, preloaded_images)
            
            if inputs is None:
                pbar.update(len(batch_data))
                continue
            
            start_time = time.time()
            responses = generate_batch_response(model, inputs, processor)
            batch_time = time.time() - start_time
            total_time += batch_time
            
            if not responses:
                pbar.update(len(batch_data))
                continue
            
            success_batches += 1
            
            min_len = min(len(responses), len(valid_items))
            for item, response in zip(valid_items[:min_len], responses[:min_len]):
                result = {
                    'Group_ID': item['id'],
                    'Image_URL': item['image_file'],
                    'Question': item['question'],
                    'Ground_Truth': item['ground_truth'],
                    'Model_Output': response,
                    'Num_Questions': item['num_questions'],
                    'Original_IDs': item['original_ids'],
                }
                results.append(result)
            
            pbar.update(len(batch_data))
            
            if current_batch % args.save_interval == 0 or batch_idx + batch_size >= len(batch_items):
                with open(args.output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                tqdm.write(f"Saved {len(results)} results")
            
            cleanup_memory()
            
        except Exception as e:
            tqdm.write(f"Error in batch {current_batch}: {e}")
            pbar.update(len(batch_data))
            cleanup_memory()
            continue
    
    pbar.close()
    
    # Final save
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Grouped evaluation complete!")
    print(f"Total groups processed: {len(results)}")
    print(f"Successful batches: {success_batches}/{total_batches}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Results saved to: {args.output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
