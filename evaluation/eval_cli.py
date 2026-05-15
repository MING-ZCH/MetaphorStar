#!/usr/bin/env python3
"""
CLI wrapper for eval_TFQ.py and eval_MCQ.py scoring.
Usage:
    python3 evaluation/eval_cli.py --task TFQ --result_path path/to/result.json
    python3 evaluation/eval_cli.py --task MCQ --result_path path/to/result.json
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Score individual eval results")
    parser.add_argument("--task", type=str, required=True, choices=["TFQ", "MCQ"], help="Task type")
    parser.add_argument("--result_path", type=str, required=True, help="Path to result JSON")
    args = parser.parse_args()

    os.chdir('/mnt/shared-storage-user/zhangchenhao/work/MetaphorStar')

    if args.task == "TFQ":
        from eval_TFQ import evaluate_accuracy
        results = evaluate_accuracy(args.result_path)
        print(f"\n{'='*60}")
        print(f"  TFQ Individual Evaluation Results")
        print(f"{'='*60}")
        print(f"  Per-Question Accuracy: {results['Accuracy']:.2%} ({results['Correct']}/{results['Total']})")
        print(f"  Per-Group Accuracy:    {results['Group_Accuracy']:.2%} ({results['Group_Correct']}/{results['Total_Groups']})")
        print(f"  Empty Answers:         {results['Empty_Answers']}/{results['Total']}")
        print(f"{'='*60}")

    elif args.task == "MCQ":
        from eval_MCQ import evaluate_accuracy
        results = evaluate_accuracy(args.result_path)
        print(f"\n{'='*60}")
        print(f"  MCQ Individual Evaluation Results")
        print(f"{'='*60}")
        print(f"  Accuracy: {results['Accuracy']:.2%} ({results['Correct']}/{results['Total']})")
        print(f"  Empty Answers: {results['Empty_Answers']}/{results['Total']}")
        print(f"{'='*60}")

    # Save evaluation text
    output_txt = args.result_path.replace('.json', '_evaluation.txt')
    with open(output_txt, 'w', encoding='utf-8') as f:
        for key, value in results.items():
            if key not in ['Detailed_Results', 'Group_Stats']:
                f.write(f"{key}: {value}\n")
    print(f"  Evaluation saved to: {output_txt}")


if __name__ == "__main__":
    main()
