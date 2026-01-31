# Evaluation Code
import json
import re
import os
from collections import defaultdict

def extract_final_answer(model_output):
    """
    Extract the final answer from the model output
    Supports two formats: <answer>...</answer> and T/F
    """
    # First try to match <answer>...</answer> format
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = re.findall(answer_pattern, model_output, re.DOTALL)
    
    if answer_matches:
        # Return the last matched result, as the final answer is usually at the end
        return answer_matches[-1].strip()
    
    en_pattern = r'Answer:\s*([TF])'
    match = re.search(en_pattern, model_output)
    if match:
        return match.group(1)
    
    # If <answer>...</answer> format not found, try to match separate T or F
    tf_pattern = r'\b(T|F)\b'
    tf_matches = re.findall(tf_pattern, model_output)
    
    if tf_matches:
        # Return the last matched result
        return tf_matches[-1].strip()
    
    # If none found, return None
    return None

def evaluate_accuracy(data_path):
    """
    Evaluate the accuracy of model output
    
    Args:
    data_path: Path to JSON file
    
    Returns:
    Accuracy and detailed results
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    correct = 0
    empty_answers = 0  # Count empty answers
    results = []
    
    # Group by Image_URL
    groups = defaultdict(list)
    
    for idx, item in enumerate(data):
        model_output = item["Model_Output"]
        ground_truth = item["Ground_Truth"]
        question_id = item["Question_ID"]
        question = item["Question"]
        image_url = item.get("Image_URL", "")
        
        extracted_answer = extract_final_answer(model_output)
        
        # Count empty answers
        if extracted_answer is None or extracted_answer == "":
            empty_answers += 1
        
        is_correct = False
        if extracted_answer is not None and extracted_answer != "":
            # Here you can add more complex matching logic if needed
            # e.g., ignore case, spaces, etc.
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
        
        # Group by Image_URL
        groups[image_url].append(result_item)
    
    accuracy = correct / total if total > 0 else 0
    
    # Calculate group-level accuracy
    group_correct = 0
    total_groups = len(groups)
    group_stats = []
    
    for image_url, group_items in groups.items():
        # Check if all cases in the group are correct
        all_correct = all(item["Is_Correct"] for item in group_items)
        if all_correct:
            group_correct += 1
        
        # Calculate group detailed statistics
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
    # ⭐⭐⭐ Run evaluation and print results ⭐⭐⭐
    os.chdir('evaluation')
    file_path = 'evaluation/TFQ/eval_result.json'
    results = evaluate_accuracy(file_path)

    print(f"Group-level accuracy: {results['Group_Accuracy']:.2%} ({results['Group_Correct']}/{results['Total_Groups']})")
    print(f"Number of empty answers: {results['Empty_Answers']}/{results['Total']}")
    print(f"Number of incorrect answers: {results['Total'] - results['Correct'] - results['Empty_Answers']}/{results['Total']}")

    # Print group level statistics
    print(f"\nGroup Level Statistics:")
    print(f"Total groups: {results['Total_Groups']}")
    print(f"Fully correct groups: {results['Group_Correct']}")
    print(f"Partially correct groups: {sum(1 for g in results['Group_Stats'] if not g['All_Correct'] and g['Correct_Cases'] > 0)}")
    print(f"Fully incorrect groups: {sum(1 for g in results['Group_Stats'] if g['Correct_Cases'] == 0)}")

    # Print evaluation results for each question, show detailed info for incorrect ones
    print("\nQuestion Evaluation Results:")
    for result in results['Detailed_Results']:
        question_id = result.get('Question_ID', f"Question {result['Index']+1}") 
        if result['Is_Correct'] == False:
            if result['Extracted_Answer'] is None or result['Extracted_Answer'] == "":
                print(f"{question_id}: ❌ (Empty Answer)")
            else:
                print(f"{question_id}: ❌")
            print(f"  Question: {result['Question'][:100]}..." if len(result['Question']) > 100 else f"  Question: {result['Question']}")
            print(f"  Extracted Answer: {result['Extracted_Answer']}")
            print(f"  Ground Truth: {result['Ground_Truth']}")
            print()

    # Count incorrect samples and empty answers
    incorrect_samples = [r for r in results['Detailed_Results'] if not r['Is_Correct']]
    empty_answer_samples = [r for r in results['Detailed_Results'] if r['Extracted_Answer'] is None or r['Extracted_Answer'] == ""]

    print(f"\nTotal incorrect questions: {len(incorrect_samples)}/{results['Total']}")
    print(f"Of which empty answers: {len(empty_answer_samples)}/{results['Total']}")
    print(f"Questions with incorrect answers: {len(incorrect_samples) - len(empty_answer_samples)}/{results['Total']}")

    # Save results to txt file
    output_txt_path = file_path.replace('.json', '_evaluation.txt')
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Model Evaluation Results\n")
        f.write(f"========================\n\n")
        f.write(f"Group-level accuracy (all cases in group must be correct): {results['Group_Accuracy']:.2%} ({results['Group_Correct']}/{results['Total_Groups']})\n")
        f.write(f"Number of empty answers: {results['Empty_Answers']}/{results['Total']} ({results['Empty_Answers']/results['Total']:.2%})\n")
        f.write(f"Number of incorrect answers: {results['Total'] - results['Correct'] - results['Empty_Answers']}/{results['Total']}\n\n")

        f.write(f"Group Level Statistics:\n")
        f.write(f"Total groups: {results['Total_Groups']}\n")
        f.write(f"Fully correct groups: {results['Group_Correct']}\n")
        f.write(f"Partially correct groups: {sum(1 for g in results['Group_Stats'] if not g['All_Correct'] and g['Correct_Cases'] > 0)}\n")
        f.write(f"Fully incorrect groups: {sum(1 for g in results['Group_Stats'] if g['Correct_Cases'] == 0)}\n\n")
        
        f.write("Group Detailed Statistics:\n")
        for i, group_stat in enumerate(results['Group_Stats']):
            f.write(f"Group {i+1}: {group_stat['Correct_Cases']}/{group_stat['Total_Cases']} correct")
            if group_stat['All_Correct']:
                f.write(" ✓")
            f.write(f" (Empty: {group_stat['Empty_Cases']})\n")
        
        f.write("\nQuestion Evaluation Results:\n")
        for result in results['Detailed_Results']:
            question_id = result.get('Question_ID', f"Question {result['Index']+1}")
            if result['Is_Correct'] == False:
                if result['Extracted_Answer'] is None or result['Extracted_Answer'] == "":
                    f.write(f"{question_id}: Error (Empty Answer)\n")
                else:
                    f.write(f"{question_id}: Error\n")
                f.write(f"  Question: {result['Question'][:100]}...\n" if len(result['Question']) > 100 else f"  Question: {result['Question']}\n")
                f.write(f"  Extracted Answer: {result['Extracted_Answer']}\n")
                f.write(f"  Ground Truth: {result['Ground_Truth']}\n\n")
        
        f.write(f"\nTotal incorrect questions: {len(incorrect_samples)}/{results['Total']}\n")
        f.write(f"Of which empty answers: {len(empty_answer_samples)}/{results['Total']}\n")
        f.write(f"Questions with incorrect answers: {len(incorrect_samples) - len(empty_answer_samples)}/{results['Total']}\n")

    print(f"\nEvaluation results saved to {output_txt_path}")

if __name__ == "__main__":
    main()    