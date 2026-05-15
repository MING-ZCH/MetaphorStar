# MCQ 评测代码
import json
import re
import os
from collections import defaultdict

def extract_final_answer(model_output):
    """
    提取模型输出中的最终答案
    支持两种格式：<answer>...</answer> 和 字母
    """
    # 首先尝试匹配 <answer>...</answer> 格式
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = re.findall(answer_pattern, model_output, re.DOTALL)
    
    if answer_matches:
        # 返回最后一个匹配的结果，因为最终答案通常在末尾
        return answer_matches[-1].strip()
    
    en_pattern = r'Answer:\s*([A-Za-z])'
    match = re.search(en_pattern, model_output)
    if match:
        return match.group(1)
    
    # 如果都没有找到，返回None
    return None

def evaluate_accuracy(data_path):
    """
    评估模型输出的正确率
    
    参数:
    data_path: JSON文件路径
    
    返回:
    准确率和详细结果
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    correct = 0
    empty_answers = 0  # 统计空答案数量
    results = []
    
    # 按组统计
    groups = defaultdict(list)
    
    for idx, item in enumerate(data):
        model_output = item["Model_Output"]
        ground_truth = item["Ground_Truth"]
        # question_id = item["Question_ID"] # exp. "2_0"
        question_id = item.get("Question_ID", item.get("question_id", f"question_{idx}"))
        question = item["Question"]
        image_url = item.get("Image_URL", "")
        
        extracted_answer = extract_final_answer(model_output)
        
        # 统计空答案
        if extracted_answer is None or extracted_answer == "":
            empty_answers += 1
        
        is_correct = False
        if extracted_answer is not None and extracted_answer != "":
            # 这里可以根据需要添加更复杂的匹配逻辑
            # 例如忽略大小写、空格等
            is_correct = extracted_answer.lower() == ground_truth.lower()
        
        if is_correct:
            correct += 1
        
        result_item = {
            "Index": idx,
            "Question_ID": question_id,
            "Question": question,
            "Extracted_Answer": extracted_answer,
            "Ground_Truth": ground_truth,
            "Is_Correct": is_correct,
            "Image_URL": image_url
        }
        
        results.append(result_item)
        
        # 按Image_URL分组
        groups[image_url].append(result_item)
    
    accuracy = correct / total if total > 0 else 0
    
    # 计算组级别的正确率
    group_correct = 0
    total_groups = len(groups)
    group_stats = []
    
    for image_url, group_items in groups.items():
        # 检查该组所有case是否都正确
        all_correct = all(item["Is_Correct"] for item in group_items)
        if all_correct:
            group_correct += 1
        
        # 统计该组的详细信息
        group_total = len(group_items)
        group_correct_count = sum(1 for item in group_items if item["Is_Correct"])
        group_empty_count = sum(1 for item in group_items if item["Extracted_Answer"] in [None, ""])
        
        group_stats.append({
            "Image_URL": image_url,
            "Total_Cases": group_total,
            "Correct_Cases": group_correct_count,
            "Empty_Cases": group_empty_count,
            "All_Correct": all_correct,
            "Group_Accuracy": group_correct_count / group_total if group_total > 0 else 0
        })
    
    group_accuracy = group_correct / total_groups if total_groups > 0 else 0
    
    return {
        "Accuracy": accuracy,
        "Correct": correct,
        "Total": total,
        "Empty_Answers": empty_answers,
        "Group_Accuracy": group_accuracy,
        "Group_Correct": group_correct,
        "Total_Groups": total_groups,
        "Group_Stats": group_stats,
        "Detailed_Results": results
    }

def main():
    # ⭐⭐⭐执行评测并打印结果⭐⭐⭐
    os.chdir('/mnt/shared-storage-user/zhangchenhao/work/MetaphorStar')
    file_path = 'evaluation/II-Bench/MCQ/other/qwen25_vl_7b_II-Bench_reasoning_grpo_format03_result.json'
    results = evaluate_accuracy(file_path)

    print(f"Individual pass@1 accuracy: {results['Accuracy']:.2%} ({results['Correct']}/{results['Total']})")
    print(f"空答案数量: {results['Empty_Answers']}/{results['Total']}")
    print(f"错误答案数量: {results['Total'] - results['Correct'] - results['Empty_Answers']}/{results['Total']}")

    # 打印每个题目的评测结果，不正确的题目显示详细信息
    print("\n题目评测结果:")
    for result in results['Detailed_Results']:
        question_id = result.get('Question_ID', f"问题{result['Index']+1}") 
        if result['Is_Correct'] == False:
            if result['Extracted_Answer'] is None or result['Extracted_Answer'] == "":
                print(f"{question_id}: ❌ (空答案)")
            else:
                print(f"{question_id}: ❌")
            print(f"  问题: {result['Question'][:100]}..." if len(result['Question']) > 100 else f"  问题: {result['Question']}")
            print(f"  提取的答案: {result['Extracted_Answer']}")
            print(f"  真实答案: {result['Ground_Truth']}")
            print()

    # 统计错误数量和空答案数量
    incorrect_samples = [r for r in results['Detailed_Results'] if not r['Is_Correct']]
    empty_answer_samples = [r for r in results['Detailed_Results'] if r['Extracted_Answer'] is None or r['Extracted_Answer'] == ""]

    print(f"\n错误题目总数: {len(incorrect_samples)}/{results['Total']}")
    print(f"其中空答案题目: {len(empty_answer_samples)}/{results['Total']}")
    print(f"有答案但错误的题目: {len(incorrect_samples) - len(empty_answer_samples)}/{results['Total']}")

    # 将结果保存到txt文件
    output_txt_path = file_path.replace('.json', '_evaluation.txt')
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"模型评估结果\n")
        f.write(f"============\n\n")
        f.write(f"Individual pass@1 accuracy: {results['Accuracy']:.2%} ({results['Correct']}/{results['Total']})\n")
        f.write(f"空答案数量: {results['Empty_Answers']}/{results['Total']} ({results['Empty_Answers']/results['Total']:.2%})\n")
        f.write(f"错误答案数量: {results['Total'] - results['Correct'] - results['Empty_Answers']}/{results['Total']}\n\n")

        
        f.write("\n题目评测结果:\n")
        for result in results['Detailed_Results']:
            question_id = result.get('Question_ID', f"问题{result['Index']+1}")
            if result['Is_Correct'] == False:
                if result['Extracted_Answer'] is None or result['Extracted_Answer'] == "":
                    f.write(f"{question_id}: 错误 (空答案)\n")
                else:
                    f.write(f"{question_id}: 错误\n")
                f.write(f"  问题: {result['Question'][:100]}...\n" if len(result['Question']) > 100 else f"  问题: {result['Question']}\n")
                f.write(f"  提取的答案: {result['Extracted_Answer']}\n")
                f.write(f"  真实答案: {result['Ground_Truth']}\n\n")
        
        f.write(f"\n错误题目总数: {len(incorrect_samples)}/{results['Total']}\n")
        f.write(f"其中空答案题目: {len(empty_answer_samples)}/{results['Total']}\n")
        f.write(f"有答案但错误的题目: {len(incorrect_samples) - len(empty_answer_samples)}/{results['Total']}\n")

    print(f"\n评测结果已保存至 {output_txt_path}")

if __name__ == "__main__":
    main()    