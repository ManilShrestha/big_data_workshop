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
        question = item['question']
        predicted = set(item['predicted_answers'])
        ground_truth = set(item['ground_truth'])

        is_correct = item['correct']

        if not is_correct:
            failures.append({
                'question': question,
                'predicted': predicted,
                'ground_truth': ground_truth,
                'paths': item.get('paths', []),
                'search_trace': item.get('search_trace', {}),
                'reasoning': item.get('reasoning', '')
            })
        else:
            success_cases.append({
                'question': question,
                'predicted': predicted,
                'ground_truth': ground_truth,
                'paths': item.get('paths', []),
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
        'num_paths': [],
        'avg_path_scores': [],
        'hop_scores_by_position': [[], [], []],  # hop 1, hop 2, hop 3
    }

    for failure in failures:
        paths = failure.get('paths', [])
        path_stats['num_paths'].append(len(paths))

        if paths:
            for path in paths:
                if 'avg_score' in path:
                    path_stats['avg_path_scores'].append(path['avg_score'])

                # Collect hop scores
                for hop_idx, hop in enumerate(path.get('hops', [])):
                    if hop_idx < 3 and 'score' in hop:
                        path_stats['hop_scores_by_position'][hop_idx].append(hop['score'])

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
    print("PATH STATISTICS (Failures)")
    print("=" * 80)
    if path_stats['num_paths']:
        print(f"Avg paths found: {sum(path_stats['num_paths'])/len(path_stats['num_paths']):.2f}")
        print(f"Cases with 0 paths: {sum(1 for x in path_stats['num_paths'] if x == 0)}")

    if path_stats['avg_path_scores']:
        print(f"Avg path score: {sum(path_stats['avg_path_scores'])/len(path_stats['avg_path_scores']):.4f}")
        print(f"Min path score: {min(path_stats['avg_path_scores']):.4f}")
        print(f"Max path score: {max(path_stats['avg_path_scores']):.4f}")

    print("\nHop scores by position:")
    for hop_idx, scores in enumerate(path_stats['hop_scores_by_position']):
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"  Hop {hop_idx+1}: {avg_score:.4f} (n={len(scores)})")

    # Compare with success cases
    print("\n" + "=" * 80)
    print("SUCCESS CASE PATH STATISTICS (for comparison)")
    print("=" * 80)
    success_path_count = []
    success_scores = []
    success_hop_scores = [[], [], []]

    for case in success_cases:
        paths = case.get('paths', [])
        success_path_count.append(len(paths))

        if paths:
            for path in paths:
                if 'avg_score' in path:
                    success_scores.append(path['avg_score'])

                for hop_idx, hop in enumerate(path.get('hops', [])):
                    if hop_idx < 3 and 'score' in hop:
                        success_hop_scores[hop_idx].append(hop['score'])

    if success_path_count:
        print(f"Avg paths found: {sum(success_path_count)/len(success_path_count):.2f}")
    if success_scores:
        print(f"Avg path score: {sum(success_scores)/len(success_scores):.4f}")

    print("\nHop scores by position:")
    for hop_idx, scores in enumerate(success_hop_scores):
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"  Hop {hop_idx+1}: {avg_score:.4f} (n={len(scores)})")

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

                # Show top path if available
                paths = case.get('paths', [])
                if paths:
                    top_path = paths[0]
                    print(f"      Top path score: {top_path.get('avg_score', 'N/A')}")
                    print(f"      Path: ", end="")
                    for hop_idx, hop in enumerate(top_path.get('hops', [])):
                        if hop_idx > 0:
                            print(" -> ", end="")
                        print(f"{hop.get('entity', '?')} --[{hop.get('relation', '?')}]-->", end="")
                    if top_path.get('hops'):
                        last_hop = top_path['hops'][-1]
                        print(f" {last_hop.get('target', '?')}")
                else:
                    print(f"      No paths found")

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
