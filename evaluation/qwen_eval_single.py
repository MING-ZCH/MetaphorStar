import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 添加此行使CUDA错误立即报告
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def load_model(model_path):
    """加载本地模型并优化推理"""
    print(f"正在加载模型：{model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    # 加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16, # 3b use float32, other models can use float16
        device_map="auto", # auto
        attn_implementation="flash_attention_2", # 3b uses eager, other models can use flash_attention_2
        trust_remote_code=True
    ).eval()
    print("模型加载完成！")
    return processor, model

def process_image_question(image_path, question, processor):
    """处理图像和问题，生成模型输入"""
    # 构建多模态输入
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question}
        ]
    }]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 读取图像并处理
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=text, images=image, return_tensors="pt")
    
    return inputs.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_response(model, inputs, processor, max_new_tokens=4096):  # 降低最大token数
    """生成模型回答"""
    try:
        with torch.no_grad():
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,
                top_p=0.9
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            return processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
    except Exception as e:
        print(f"生成回答时出错: {e}")
        # 返回一个错误提示而不是中断整个处理
        return f"[生成错误: {str(e)}]"

def main():
    """⭐⭐⭐主函数：加载数据、调用模型、保存结果⭐⭐⭐"""
    # 配置参数
    model_path = "/mnt/shared-storage-user/zhangchenhao/work/MetaphorStar/qwen2.5_vl-7b_epoch5_batch32_resize_vocab_prompt/full/sft"
    dataset_path = 'evaluation/II-Bench/eval/OSQ_II-Bench_eval.json'
    output_path = 'evaluation/II-Bench/OSQ/qwen2_5_vl_7b_sft_eval_result.json'
    image_dir = 'evaluation'
    
    # 创建结果目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 加载模型
    processor, model = load_model(model_path)
    
    # 加载数据集
    print(f"加载数据集：{dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查结果文件是否存在
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"已加载现有结果：{len(results)}条")
    else:
        results = []
    
    # 获取已处理的问题ID
    processed_ids = {item['Question_ID'] for item in results}
    
    # 处理每个问题
    for idx, item in enumerate(tqdm(data, desc="图片识别中")):
        try:
            id = item['id']
            
            # 跳过已处理的问题
            if id in processed_ids:
                print(f"跳过已处理问题：{id}")
                continue
                
            question = item['eval_question']
            print('question: ', question)
            
            # 获取图像路径
            image_file = item['url']
            image_path = os.path.join(image_dir, image_file)
            
            ground_truth = item['correct_answer']
            print('ground truth: ', ground_truth)
            
            # 处理图像和问题，生成模型输入
            inputs = process_image_question(image_path, question, processor)
            
            # 调用模型生成回答
            model_response = generate_response(model, inputs, processor)
            print('answer: ', model_response)
            print('----------------------------------')
            
            # 组织结果
            result = {
                'Question_ID': id,
                'Image_URL': image_file,
                'Question': question,
                'Ground_Truth': ground_truth,
                'Model_Output': model_response
            }
            
            # 将结果添加到列表
            results.append(result)
            
            # 将更新后的结果写入JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"处理图片 {id if 'id' in locals() else '未知'} 时出错：{e}")
            continue

if __name__ == "__main__":
    main()