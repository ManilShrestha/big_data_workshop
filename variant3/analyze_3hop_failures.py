#!/usr/bin/env python3
"""
Analyze 3-hop evaluation failures to identify root causes.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

def load_results(result_file):
    """Load evaluation results."""
    with open(result_file, 'r') as f:
        return json.load(f)

def analyze_failures(results):
    """Analyze failed questions to identify patterns."""

    failures = []
    success_cases = []

    for item in results['results']:
        question = item['question_text']
        predicted = set(item['predicted_answers'])
        ground_truth = set(item['ground_truth_answers'])

        is_correct = item['is_correct']

        if not is_correct:
            failures.append({
                'question': question,
                'predicted': predicted,
                'ground_truth': ground_truth,
                'reasoning_path': item.get('reasoning_path', ''),
                'relations_used': item.get('relations_used', []),
                'nodes_expanded': item.get('nodes_expanded', 0),
                'metadata': item.get('metadata', {}),
            })
        else:
            success_cases.append({
                'question': question,
                'predicted': predicted,
                'ground_truth': ground_truth,
                'reasoning_path': item.get('reasoning_path', ''),
                'relations_used': item.get('relations_used', []),
                'nodes_expanded': item.get('nodes_expanded', 0),
            })

    return failures, success_cases

def categorize_failures(failures):
    """Categorize failures by type."""

    categories = {
        'no_answer': [],      # No paths found
        'wrong_entity': [],   # Found paths but wrong final entity
        'partial_match': [],  # Found some correct answers but missed others
        'hallucination': [],  # Predicted answers not in ground truth
    }

    for failure in failures:
        pred = failure['predicted']
        gt = failure['ground_truth']

        if len(pred) == 0:
            categories['no_answer'].append(failure)
        elif len(pred & gt) == 0:
            categories['wrong_entity'].append(failure)
        elif len(pred & gt) > 0 and len(pred & gt) < len(gt):
            categories['partial_match'].append(failure)
        else:
            categories['hallucination'].append(failure)

    return categories

def analyze_path_quality(failures):
    """Analyze the quality of paths found."""

    path_stats = {
        'nodes_expanded': [],
        'num_relations_used': [],
        'has_reasoning_path': 0,
    }

    for failure in failures:
        path_stats['nodes_expanded'].append(failure.get('nodes_expanded', 0))
        path_stats['num_relations_used'].append(len(failure.get('relations_used', [])))
        if failure.get('reasoning_path'):
            path_stats['has_reasoning_path'] += 1

    return path_stats

def print_analysis(failures, success_cases, categories, path_stats):
    """Print detailed analysis."""

    print("=" * 80)
    print("3-HOP FAILURE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal questions: {len(failures) + len(success_cases)}")
    print(f"Failures: {len(failures)} ({100*len(failures)/(len(failures)+len(success_cases)):.1f}%)")
    print(f"Success: {len(success_cases)} ({100*len(success_cases)/(len(failures)+len(success_cases)):.1f}%)")

    print("\n" + "=" * 80)
    print("FAILURE CATEGORIES")
    print("=" * 80)
    for category, cases in categories.items():
        print(f"\n{category.upper().replace('_', ' ')}: {len(cases)} cases")

    print("\n" + "=" * 80)
    print("SEARCH STATISTICS (Failures)")
    print("=" * 80)
    if path_stats['nodes_expanded']:
        print(f"Avg nodes expanded: {sum(path_stats['nodes_expanded'])/len(path_stats['nodes_expanded']):.2f}")
        print(f"Min nodes expanded: {min(path_stats['nodes_expanded'])}")
        print(f"Max nodes expanded: {max(path_stats['nodes_expanded'])}")
        print(f"Cases with 0 nodes: {sum(1 for x in path_stats['nodes_expanded'] if x == 0)}")

    if path_stats['num_relations_used']:
        print(f"\nAvg relations used: {sum(path_stats['num_relations_used'])/len(path_stats['num_relations_used']):.2f}")

    print(f"\nCases with reasoning path: {path_stats['has_reasoning_path']}/{len(failures)}")

    # Compare with success cases
    print("\n" + "=" * 80)
    print("SEARCH STATISTICS (Success cases - for comparison)")
    print("=" * 80)
    success_nodes = [case.get('nodes_expanded', 0) for case in success_cases]
    success_relations = [len(case.get('relations_used', [])) for case in success_cases]
    success_reasoning = sum(1 for case in success_cases if case.get('reasoning_path'))

    if success_nodes:
        print(f"Avg nodes expanded: {sum(success_nodes)/len(success_nodes):.2f}")
        print(f"Min nodes expanded: {min(success_nodes)}")
        print(f"Max nodes expanded: {max(success_nodes)}")

    if success_relations:
        print(f"\nAvg relations used: {sum(success_relations)/len(success_relations):.2f}")

    print(f"\nCases with reasoning path: {success_reasoning}/{len(success_cases)}")

    # Sample failures
    print("\n" + "=" * 80)
    print("SAMPLE FAILURES (5 per category)")
    print("=" * 80)

    for category, cases in categories.items():
        if cases:
            print(f"\n{category.upper().replace('_', ' ')}:")
            for i, case in enumerate(cases[:5]):
                print(f"\n  [{i+1}] Question: {case['question']}")
                print(f"      Ground truth: {case['ground_truth']}")
                print(f"      Predicted: {case['predicted']}")
                print(f"      Nodes expanded: {case.get('nodes_expanded', 0)}")
                print(f"      Relations used: {case.get('relations_used', [])}")

                # Show reasoning path if available
                reasoning_path = case.get('reasoning_path', '')
                if reasoning_path:
                    print(f"      Reasoning: {reasoning_path[:200]}{'...' if len(reasoning_path) > 200 else ''}")
                else:
                    print(f"      No reasoning path found")

def main():
    # Find most recent 3-hop result file
    results_dir = Path(__file__).parent.parent / 'results'
    result_files = sorted(results_dir.glob('variant3_edge_scorer_3-hop*.json'))

    if not result_files:
        print("No 3-hop result files found!")
        return

    result_file = result_files[-1]
    print(f"Analyzing: {result_file}\n")

    results = load_results(result_file)
    failures, success_cases = analyze_failures(results)
    categories = categorize_failures(failures)
    path_stats = analyze_path_quality(failures)

    print_analysis(failures, success_cases, categories, path_stats)

if __name__ == '__main__':
    main()
