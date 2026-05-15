import os
import json
import torch
import time
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import gc
from concurrent.futures import ThreadPoolExecutor

# 添加此行使CUDA错误立即报告
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def load_model(model_path):
    """加载本地模型并优化多卡推理"""
    print(f"正在加载模型：{model_path}")
    
    # 添加模型加载进度提示
    with tqdm(total=2, desc="模型加载进度", unit="步骤") as pbar:
        # 加载processor
        processor = AutoProcessor.from_pretrained(model_path)
        processor.tokenizer.padding_side = "left"  # 确保左填充

        # 验证特定添加的tokens
        special_tokens = ["<think>", "</think>", "<answer>", "</answer>"]
        print("\n验证添加的特殊标记:")
        for token in special_tokens:
            token_id = processor.tokenizer.convert_tokens_to_ids(token)
            exists = token_id != processor.tokenizer.unk_token_id
            print(f"  • {token}: {'✓ 已加载 (ID: ' + str(token_id) + ')' if exists else '✗ 未找到'}")
        
        pbar.update(1)
        pbar.set_description("Processor加载完成")
        
        # 加载模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16, # 3b use float32, other models can use float16
            device_map="auto", # auto
            attn_implementation="flash_attention_2", # 3b uses eager, other models can use flash_attention_2
            trust_remote_code=True
        ).eval()
        pbar.update(1)
        pbar.set_description("模型加载完成")
    
    print(f"模型分布在设备：{model.hf_device_map}")
    return processor, model

def preload_images(image_paths, max_workers=4):
    """并行预加载图像"""
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
            desc="预加载图像",
            unit="张",
            colour="green"
        ))
    
    # 统计成功加载的图像
    success_count = 0
    for path, image in results:
        if image is not None:
            images[path] = image
            success_count += 1
    
    print(f"图像预加载完成：成功 {success_count}/{len(image_paths)} 张")
    return images

def process_batch(batch_data, processor, preloaded_images):
    """批量处理图像和问题"""
    texts = []
    images = []
    
    # 确保batch_data是列表
    if not isinstance(batch_data, (list, tuple)):
        print(f"警告：batch_data不是可迭代对象，类型：{type(batch_data)}")
        return None
    
    # 添加批内处理进度条
    for item in tqdm(batch_data, desc="处理批内样本", leave=False, colour="blue"):
        try:
            image_path = item['image_path']
            question = item['question']
            
            # 获取预加载的图像
            if image_path not in preloaded_images:
                continue
                
            image = preloaded_images[image_path]
            
            # # 图像验证 - 限制尺寸避免CUDA错误
            # max_size = 1024
            # if max(image.size) > max_size:
            #     # 等比例缩放
            #     ratio = max_size / max(image.size)
            #     new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            #     image = image.resize(new_size, Image.LANCZOS)
            #     print(f"已调整图像大小: {image_path} -> {new_size}")
            
            # # 问题文本验证 - 限制长度
            # if len(question) > 4096:
            #     question = question[:4096]
            #     print(f"截断过长问题文本: {len(question)} -> 4096")

            # 构建多模态输入
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
            
        except Exception as e:
            continue
    
    if not texts:
        return None
    
    # 批量处理
    try:
        # 确保padding_side正确
        if processor.tokenizer.padding_side != "left":
            processor.tokenizer.padding_side = "left"
        
        inputs = processor(
            text=texts, 
            images=images, 
            return_tensors="pt", 
            padding=True
        )
        
        # 确保数据在正确的设备上
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # device = "cpu"  # 强制使用CPU，避免CUDA错误
        
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) if hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
        
        return inputs
        
    except Exception as e:
        print(f"批处理失败: {e}")
        return None

def generate_batch_response(model, inputs, processor, max_new_tokens=8192):
    """批量生成模型回答"""
    if inputs is None:
        return []
    
    try:
        with torch.no_grad():
            # 获取input_ids，支持BatchFeature和字典两种格式
            if hasattr(inputs, 'input_ids'):
                input_ids = inputs.input_ids
            elif 'input_ids' in inputs:
                input_ids = inputs['input_ids']
            else:
                return []
            
            # 确保pad_token_id正确
            pad_token_id = processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = processor.tokenizer.eos_token_id
            
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,                
                # temperature=0.01,
                top_p=0.9,
                # top_p=0.001,
                pad_token_id=pad_token_id,
                use_cache=True
            )
            
            # 确保generated_ids和input_ids都是tensor且形状正确
            if not hasattr(generated_ids, 'shape') or not hasattr(input_ids, 'shape'):
                return []
            
            # 检查batch维度是否匹配
            if generated_ids.shape[0] != input_ids.shape[0]:
                return []
            
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
        print(f"批量生成失败: {e}")
        return []

def cleanup_memory():
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    """⭐⭐⭐主函数：加载数据、调用模型、保存结果⭐⭐⭐"""
    # 配置参数
    # model_path = "/fs-computility/niuyazhe/shared/chenhaozhang/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_II-Bench_full_reasoning_grpo/global_step_212/actor/huggingface"
    # model_path = "/fs-computility/niuyazhe/shared/chenhaozhang/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_II-Bench_grpo/global_step_75/actor/huggingface"    
    # model_path = "/fs-computility/niuyazhe/shared/VLM/QwenVL/Qwen2.5-VL-7B-Instruct"
    model_path = "/mnt/shared-storage-user/work/MetaphorStar/qwen2_5_vl_7b_MetaphorQA_sq_grpo/global_step_18/actor/huggingface"
    dataset_path = 'evaluation/II-Bench/eval/OSQ_II-Bench_eval.json'
    output_path = 'evaluation/II-Bench/OSQ/qwen2_5_vl_7b_II-Bench_sq_grpo_small_eval_result.json'
    image_dir = 'evaluation'
    
    # 批处理配置
    batch_size = 16
    save_interval = 2 # 5 for TFQ, 2 for MCQ and OSQ
    
    # 创建结果目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 加载模型
    processor, model = load_model(model_path)
    
    # 加载数据集
    print(f"加载数据集：{dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"错误：数据集格式异常，应为列表，实际为：{type(data)}")
        return
    
    # 检查结果文件是否存在
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"已加载现有结果：{len(results)}条")
    else:
        results = []
    
    # 获取已处理的问题ID
    processed_ids = {item['Question_ID'] for item in results}
    
    # 过滤未处理的数据，增加进度条
    print("正在过滤已处理的数据...")
    unprocessed_data = []
    for item in tqdm(data, desc="过滤数据", colour="cyan"):
        if item['id'] not in processed_ids:
            unprocessed_data.append(item)
    
    print(f"待处理数据：{len(unprocessed_data)}条")
    
    if not unprocessed_data:
        print("所有数据已处理完成！")
        return
    
    # 预加载所有图像
    image_paths = [os.path.join(image_dir, item['url']) for item in unprocessed_data]
    preloaded_images = preload_images(image_paths)
    
    # 准备批处理数据，增加进度条
    print("准备批处理数据...")
    batch_items = []
    for item in tqdm(unprocessed_data, desc="准备数据", colour="yellow"):
        image_path = os.path.join(image_dir, item['url'])
        if image_path in preloaded_images:
            batch_items.append({
                'id': item['id'],
                'image_path': image_path,
                'image_file': item['url'],
                'question': item['eval_question'], # 评测：eval_question
                'ground_truth': item['correct_answer']
            })
    
    print(f"准备批处理的数据：{len(batch_items)}条")
    
    # 主要的批量处理循环，增加详细进度条
    total_batches = (len(batch_items) + batch_size - 1) // batch_size
    processed_count = 0
    
    # 创建主进度条
    main_pbar = tqdm(
        total=len(batch_items),
        desc="总体进度",
        unit="Item",
        colour="red",
        position=0
    )
    
    # 创建批次进度条
    batch_pbar = tqdm(
        total=total_batches,
        desc="批次进度",
        unit="Batch",
        colour="magenta",
        position=1
    )
    
    # 统计信息
    success_batches = 0
    failed_batches = 0
    total_time = 0
    
    for batch_idx in range(0, len(batch_items), batch_size):
        try:
            batch_data = batch_items[batch_idx:batch_idx + batch_size]
            current_batch = batch_idx // batch_size + 1
            
            # 更新批次进度条描述
            batch_pbar.set_description(f"Batch {current_batch}/{total_batches}")
            
            if not batch_data:
                batch_pbar.update(1)
                continue
            
            # 批量处理输入
            inputs = process_batch(batch_data, processor, preloaded_images)
            
            if inputs is None:
                failed_batches += 1
                batch_pbar.update(1)
                main_pbar.update(len(batch_data))
                continue
            
            # 批量生成回答
            start_time = time.time()
            responses = generate_batch_response(model, inputs, processor)
            batch_time = time.time() - start_time
            total_time += batch_time
            
            if not responses:
                failed_batches += 1
                batch_pbar.update(1)
                main_pbar.update(len(batch_data))
                continue
            
            success_batches += 1
            
            # 确保responses和batch_data长度匹配
            if len(responses) != len(batch_data):
                min_len = min(len(responses), len(batch_data))
                responses = responses[:min_len]
                batch_data = batch_data[:min_len]
            
            # 处理结果
            for item, response in zip(batch_data, responses):
                result = {
                    'Question_ID': item['id'],
                    'Image_URL': item['image_file'],
                    'Question': item['question'],
                    'Ground_Truth': item['ground_truth'],
                    'Model_Output': response
                }
                results.append(result)
            
            processed_count += len(batch_data)
            
            # 更新进度条
            main_pbar.update(len(batch_data))
            batch_pbar.update(1)
            
            # 更新进度条后缀信息
            avg_time = total_time / success_batches if success_batches > 0 else 0
            main_pbar.set_postfix({
                'ss': success_batches,
                'wr': failed_batches,
                '平均耗时': f"{avg_time:.2f}s"
            })
            
            # 定期保存结果
            if current_batch % save_interval == 0 or batch_idx + batch_size >= len(batch_items):
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                tqdm.write(f"已保存 {len(results)} 条结果")
            
            # 清理内存
            cleanup_memory()
            
        except Exception as e:
            failed_batches += 1
            tqdm.write(f"处理批次 {current_batch} 时出错：{e}")
            batch_pbar.update(1)
            main_pbar.update(len(batch_data))
            cleanup_memory()
            continue
    
    # 关闭进度条
    main_pbar.close()
    batch_pbar.close()
    
    # 最终保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 显示最终统计信息
    print(f"\n{'='*50}")
    print(f"评测完成！")
    print(f"总计处理样本：{processed_count}")
    print(f"成功批次：{success_batches}")
    print(f"失败批次：{failed_batches}")
    print(f"总耗时：{total_time:.2f}秒")
    print(f"平均每样本耗时：{total_time/processed_count:.2f}s" if processed_count > 0 else "无效样本")
    print(f"结果已保存到：{output_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
