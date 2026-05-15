"""
Grouped TFQ/MCQ/OSQ evaluation accuracy verification.
Evaluates model outputs from grouped evaluation (one image = one group of questions).
Computes both individual pass@1 accuracy AND group-level accuracy (all correct in group).

Usage:
    python evaluation/eval_grouped.py --task TFQ --result_path evaluation/II-Bench/TFQ_grouped/qwen2_5_vl_7b_MetaphorQA_sq_grpo_result.json
    python evaluation/eval_grouped.py --task MCQ --result_path path/to/result.json
    python evaluation/eval_grouped.py --task OSQ --result_path path/to/result.json
"""

import json
import re
import os
import argparse
from collections import defaultdict


# ============================================================================
# TFQ: Grouped True/False Question Evaluation
# ============================================================================

def parse_tfq_model_answers(model_output, num_expected=0):
    """Parse numbered T/F answers from model output's <answer> block.
    
    Matching strategy (ordered by priority):
    1. Find LAST <answer>...</answer> block, extract numbered T/F
    2. Fallback: search full output for numbered T/F patterns
    3. Fallback for single-question: if num_expected==1, find any standalone T/F
    
    Returns: dict {question_number: 'T'/'F'} 
    """
    answers = {}
    
    # Strategy 1: Find LAST <answer>...</answer> block
    answer_matches = re.findall(r"<answer>(.*?)</answer>", model_output, re.DOTALL)
    if answer_matches:
        answer_text = answer_matches[-1].strip()
        pattern = re.compile(r"(\d+)\s*[.):]\s*([TFtf])\b")
        for match in pattern.finditer(answer_text):
            idx = int(match.group(1))
            ans = match.group(2).upper()
            answers[idx] = ans
        
        # If <answer> found but no numbered patterns, extract first T/F as answer for Q1
        if not answers and num_expected <= 1:
            tf_match = re.search(r"\b([TFtf])\b", answer_text)
            if tf_match:
                answers[1] = tf_match.group(1).upper()
        
        if answers:
            return answers
    
    # Strategy 2: Fallback - search full output for numbered T/F patterns
    pattern = re.compile(r"(\d+)\s*[.):]\s*([TFtf])\b")
    for match in pattern.finditer(model_output):
        idx = int(match.group(1))
        ans = match.group(2).upper()
        if idx not in answers:
            answers[idx] = ans
    
    if answers:
        return answers
    
    # Strategy 3: Single question fallback
    if num_expected <= 1:
        final_match = re.search(r"(?:Final\s+answer|Answer)\s*[:]\s*([TFtf])\b", model_output, re.IGNORECASE)
        if final_match:
            answers[1] = final_match.group(1).upper()
        else:
            tf_matches = re.findall(r"\b([TFtf])\b", model_output)
            if tf_matches:
                answers[1] = tf_matches[-1].upper()
    
    return answers


def parse_tfq_ground_truth(ground_truth):
    """Parse ground truth: '1.T,2.F,3.T,...'"""
    answers = {}
    pattern = re.compile(r"(\d+)\s*[.):]\s*([TFtf])")
    matches = list(pattern.finditer(ground_truth))
    if matches:
        for match in matches:
            idx = int(match.group(1))
            ans = match.group(2).upper()
            answers[idx] = ans
    else:
        parts = [p.strip().upper() for p in ground_truth.split(",")]
        for i, part in enumerate(parts, 1):
            if part in ("T", "F"):
                answers[i] = part
    return answers


def evaluate_tfq_grouped(data):
    """Evaluate grouped TFQ results."""
    total_questions = 0
    total_correct = 0
    total_groups = len(data)
    groups_all_correct = 0
    empty_groups = 0
    
    group_stats = []
    
    for item in data:
        group_id = item.get('Group_ID', item.get('id', '?'))
        model_output = item.get('Model_Output', '')
        ground_truth = item.get('Ground_Truth', item.get('correct_answer', ''))
        image_url = item.get('Image_URL', item.get('url', ''))
        
        model_answers = parse_tfq_model_answers(model_output, num_expected=total_questions or len(parse_tfq_ground_truth(ground_truth)))
        gt_answers = parse_tfq_ground_truth(ground_truth)
        
        if not gt_answers:
            empty_groups += 1
            continue
        
        group_correct = 0
        group_total = len(gt_answers)
        per_q_results = []
        
        for q_idx in sorted(gt_answers.keys()):
            gt_ans = gt_answers[q_idx]
            model_ans = model_answers.get(q_idx, None)
            is_correct = model_ans is not None and model_ans == gt_ans
            
            if is_correct:
                group_correct += 1
                total_correct += 1
            total_questions += 1
            
            per_q_results.append({
                'question_num': q_idx,
                'model_answer': model_ans,
                'ground_truth': gt_ans,
                'is_correct': is_correct,
            })
        
        all_correct = (group_correct == group_total)
        if all_correct:
            groups_all_correct += 1
        
        group_stats.append({
            'Group_ID': group_id,
            'Image_URL': image_url,
            'Total_Questions': group_total,
            'Correct_Questions': group_correct,
            'All_Correct': all_correct,
            'Group_Accuracy': group_correct / group_total if group_total > 0 else 0,
            'Per_Question_Results': per_q_results,
        })
    
    individual_accuracy = total_correct / total_questions if total_questions > 0 else 0
    group_accuracy = groups_all_correct / total_groups if total_groups > 0 else 0
    
    return {
        'Individual_Accuracy': individual_accuracy,
        'Individual_Correct': total_correct,
        'Individual_Total': total_questions,
        'Group_Accuracy': group_accuracy,
        'Groups_All_Correct': groups_all_correct,
        'Total_Groups': total_groups,
        'Empty_Groups': empty_groups,
        'Group_Stats': group_stats,
    }


# ============================================================================
# MCQ: Multiple Choice Question Evaluation
# ============================================================================

def extract_mcq_answer(model_output):
    """Extract MCQ answer (single letter) from model output."""
    answer_match = re.findall(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
    if answer_match:
        text = answer_match[-1].strip()
        letter_match = re.match(r'([A-Fa-f])', text)
        if letter_match:
            return letter_match.group(1).upper()
        return text.strip()
    match = re.search(r'Answer:\s*([A-Fa-f])', model_output)
    if match:
        return match.group(1).upper()
    return None


def evaluate_mcq(data):
    """Evaluate MCQ results."""
    total = len(data)
    correct = 0
    empty = 0
    results = []
    
    for item in data:
        item_id = item.get('Group_ID', item.get('Question_ID', item.get('id', '?')))
        model_output = item.get('Model_Output', '')
        ground_truth = item.get('Ground_Truth', item.get('correct_answer', ''))
        
        extracted = extract_mcq_answer(model_output)
        
        if extracted is None or extracted == '':
            empty += 1
            is_correct = False
        else:
            is_correct = extracted.upper() == ground_truth.upper()
        
        if is_correct:
            correct += 1
        
        results.append({
            'ID': item_id,
            'Extracted': extracted,
            'Ground_Truth': ground_truth,
            'Is_Correct': is_correct,
        })
    
    return {
        'Individual_Accuracy': correct / total if total > 0 else 0,
        'Individual_Correct': correct,
        'Individual_Total': total,
        'Group_Accuracy': correct / total if total > 0 else 0,
        'Groups_All_Correct': correct,
        'Total_Groups': total,
        'Empty_Answers': empty,
        'Detailed_Results': results,
    }


# ============================================================================
# OSQ: Open-ended Short Question Evaluation
# ============================================================================

def extract_osq_answer(model_output):
    """Extract OSQ answer from model output."""
    answer_match = re.findall(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
    if answer_match:
        return answer_match[-1].strip()
    return model_output.strip()[:500]


def evaluate_osq(data):
    """Evaluate OSQ results (open-ended, needs LLM judge)."""
    total = len(data)
    results = []
    
    for item in data:
        item_id = item.get('Group_ID', item.get('Question_ID', item.get('id', '?')))
        model_output = item.get('Model_Output', '')
        ground_truth = item.get('Ground_Truth', item.get('correct_answer', ''))
        
        extracted = extract_osq_answer(model_output)
        
        results.append({
            'ID': item_id,
            'Extracted_Answer': extracted[:200],
            'Ground_Truth': ground_truth[:200],
        })
    
    return {
        'Individual_Accuracy': None,
        'Individual_Total': total,
        'Total_Groups': total,
        'Note': 'OSQ requires manual or LLM-based evaluation for accuracy',
        'Detailed_Results': results,
    }


# ============================================================================
# Output formatting
# ============================================================================

def print_and_save_results(results, task_type, output_txt_path):
    """Print and save evaluation results."""
    lines = []
    
    def log(s=""):
        print(s)
        lines.append(s)
    
    log(f"{'='*60}")
    log(f"Grouped Evaluation Results - {task_type}")
    log(f"{'='*60}")
    
    if task_type == "TFQ":
        log(f"\nIndividual pass@1 accuracy: {results['Individual_Accuracy']:.2%} ({results['Individual_Correct']}/{results['Individual_Total']})")
        log(f"Group-level accuracy (all correct): {results['Group_Accuracy']:.2%} ({results['Groups_All_Correct']}/{results['Total_Groups']})")
        
        log(f"\nGroup Statistics:")
        log(f"  Total groups: {results['Total_Groups']}")
        log(f"  Groups all correct: {results['Groups_All_Correct']}")
        partially_correct = sum(1 for g in results['Group_Stats'] if not g['All_Correct'] and g['Correct_Questions'] > 0)
        fully_wrong = sum(1 for g in results['Group_Stats'] if g['Correct_Questions'] == 0)
        log(f"  Groups partially correct: {partially_correct}")
        log(f"  Groups fully wrong: {fully_wrong}")
        
        log(f"\nPer-Group Breakdown:")
        for i, gs in enumerate(results['Group_Stats']):
            status = " ALL_CORRECT" if gs['All_Correct'] else ""
            log(f"  Group {gs['Group_ID']}: {gs['Correct_Questions']}/{gs['Total_Questions']} correct{status}")
        
        log(f"\nWrong Answers Detail:")
        for gs in results['Group_Stats']:
            if not gs['All_Correct']:
                for pq in gs['Per_Question_Results']:
                    if not pq['is_correct']:
                        log(f"  Group {gs['Group_ID']} Q{pq['question_num']}: model={pq['model_answer']}, gt={pq['ground_truth']}")
    
    elif task_type == "MCQ":
        log(f"\nIndividual pass@1 accuracy: {results['Individual_Accuracy']:.2%} ({results['Individual_Correct']}/{results['Individual_Total']})")
        log(f"Empty answers: {results.get('Empty_Answers', 0)}/{results['Individual_Total']}")
        
        log(f"\nWrong Answers:")
        for r in results['Detailed_Results']:
            if not r['Is_Correct']:
                log(f"  ID {r['ID']}: model={r['Extracted']}, gt={r['Ground_Truth']}")
    
    elif task_type == "OSQ":
        log(f"\n{results['Note']}")
        log(f"Total samples: {results['Individual_Total']}")
        
        log(f"\nSample Outputs (first 5):")
        for r in results['Detailed_Results'][:5]:
            log(f"  ID {r['ID']}:")
            log(f"    Model: {r['Extracted_Answer'][:100]}...")
            log(f"    GT: {r['Ground_Truth'][:100]}...")
    
    log(f"\n{'='*60}")
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nResults saved to: {output_txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Grouped evaluation accuracy verification")
    parser.add_argument("--task", type=str, required=True, choices=["TFQ", "MCQ", "OSQ"])
    parser.add_argument("--result_path", type=str, required=True, help="Path to result JSON")
    args = parser.parse_args()
    
    os.chdir('/mnt/shared-storage-user/work/MetaphorStar')
    
    print(f"Loading results: {args.result_path}")
    with open(args.result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items")
    
    if args.task == "TFQ":
        results = evaluate_tfq_grouped(data)
    elif args.task == "MCQ":
        results = evaluate_mcq(data)
    elif args.task == "OSQ":
        results = evaluate_osq(data)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    output_txt = args.result_path.replace('.json', '_evaluation.txt')
    print_and_save_results(results, args.task, output_txt)
    
    output_detail = args.result_path.replace('.json', '_evaluation_detail.json')
    with open(output_detail, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"Detailed results saved to: {output_detail}")


if __name__ == "__main__":
    main()
