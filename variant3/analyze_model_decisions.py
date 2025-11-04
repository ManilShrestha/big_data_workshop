#!/usr/bin/env python3
"""
Analyze Variant 3 EdgeScorer decisions to understand:
1. Why some questions get F1=1.0 (perfect)
2. Why some questions get F1=0.0 (complete failure)
3. What relations are chosen and why
4. Text similarity vs model scores
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qa_system.config import Config


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def load_text_embeddings():
    """Load text embedding cache"""
    cache_path = Config.EMBEDDINGS_DIR / "variant3_text_embeddings_cache.pkl"
    print(f"Loading text embeddings from {cache_path}...")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def analyze_results_file(results_path: str, text_embeddings: dict):
    """Analyze a results JSON file"""

    print(f"\n{'='*80}")
    print(f"Analyzing: {Path(results_path).name}")
    print(f"{'='*80}\n")

    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)

    results = data['results']
    metrics = data['metrics']

    print(f"Total questions: {len(results)}")
    print(f"Overall metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Micro F1: {metrics['micro_f1_score']:.4f}")
    print(f"  Micro Recall: {metrics['micro_recall']:.4f}")
    print(f"  Micro Precision: {metrics['micro_precision']:.4f}\n")

    # Categorize by F1 score
    perfect = []  # F1 = 1.0
    failures = []  # F1 = 0.0
    low_f1 = []  # 0 < F1 < 0.5
    partial = []  # 0.5 <= F1 < 1.0

    for result in results:
        f1 = result['f1_score']
        if f1 == 1.0:
            perfect.append(result)
        elif f1 == 0.0:
            failures.append(result)
        elif 0 < f1 < 0.5:
            low_f1.append(result)
        else:
            partial.append(result)

    print(f"Performance breakdown:")
    print(f"  Perfect (F1=1.0): {len(perfect)} ({len(perfect)/len(results)*100:.1f}%)")
    print(f"  Failures (F1=0.0): {len(failures)} ({len(failures)/len(results)*100:.1f}%)")
    print(f"  Low F1 (0<F1<0.5): {len(low_f1)} ({len(low_f1)/len(results)*100:.1f}%)")
    print(f"  Partial (0.5≤F1<1): {len(partial)} ({len(partial)/len(results)*100:.1f}%)")

    # Analyze perfect cases
    print(f"\n{'='*80}")
    print(f"PERFECT CASES (F1=1.0) - Sample Analysis")
    print(f"{'='*80}\n")

    analyze_cases(perfect[:5], text_embeddings, "PERFECT")

    # Analyze failure cases
    print(f"\n{'='*80}")
    print(f"FAILURE CASES (F1=0.0) - Sample Analysis")
    print(f"{'='*80}\n")

    analyze_cases(failures[:5], text_embeddings, "FAILURE")

    # Analyze low F1 cases
    print(f"\n{'='*80}")
    print(f"LOW F1 CASES (0<F1<0.5) - Sample Analysis")
    print(f"{'='*80}\n")

    analyze_cases(low_f1[:5], text_embeddings, "LOW_F1")

    # Relation usage statistics
    print(f"\n{'='*80}")
    print(f"RELATION USAGE STATISTICS")
    print(f"{'='*80}\n")

    analyze_relation_usage(perfect, failures, low_f1, partial)

    # Text similarity analysis
    print(f"\n{'='*80}")
    print(f"TEXT SIMILARITY ANALYSIS")
    print(f"{'='*80}\n")

    analyze_text_similarity(perfect[:10], failures[:10], text_embeddings)


def analyze_cases(cases, text_embeddings, case_type):
    """Analyze individual cases in detail"""

    for i, case in enumerate(cases, 1):
        print(f"\n{'-'*80}")
        print(f"{case_type} Case #{i}")
        print(f"{'-'*80}")

        question = case['question_text']
        predicted = case['predicted_answers']
        ground_truth = case['ground_truth_answers']
        relations_used = case['relations_used']

        print(f"Question: {question}")
        print(f"Relations used: {' -> '.join(relations_used)}")
        print(f"Nodes expanded: {case['nodes_expanded']}")
        print(f"Search time: {case['search_time_ms']:.2f} ms")
        print(f"\nResults:")
        print(f"  Predicted: {len(predicted)} answers")
        print(f"  Ground truth: {len(ground_truth)} answers")
        print(f"  Correct: {case['num_correct']}")
        print(f"  Incorrect: {case['num_incorrect']}")
        print(f"  Missed: {case['num_missed']}")
        print(f"  Precision: {case['precision']:.4f}")
        print(f"  Recall: {case['recall']:.4f}")
        print(f"  F1: {case['f1_score']:.4f}")

        # Show sample answers
        if len(predicted) > 0:
            print(f"\nSample predicted answers (first 5):")
            for ans in predicted[:5]:
                in_gt = "✓" if ans in ground_truth else "✗"
                print(f"    {in_gt} {ans}")

        if len(ground_truth) > 0 and case_type in ["FAILURE", "LOW_F1"]:
            print(f"\nSample missed ground truth answers (first 5):")
            missed = [ans for ans in ground_truth if ans not in predicted]
            for ans in missed[:5]:
                print(f"    ✗ {ans}")

        # Analyze text similarity for relations
        if question in text_embeddings and len(relations_used) > 0:
            print(f"\nText similarity between question and relations:")
            q_emb = text_embeddings[question]

            for relation in relations_used:
                if relation in text_embeddings:
                    rel_emb = text_embeddings[relation]
                    sim = cosine_similarity(q_emb, rel_emb)
                    # Normalize to [0, 1]
                    sim_normalized = (sim + 1.0) / 2.0
                    print(f"    {relation}: {sim_normalized:.4f} (raw: {sim:.4f})")
                else:
                    print(f"    {relation}: NOT IN CACHE")


def analyze_relation_usage(perfect, failures, low_f1, partial):
    """Analyze which relations lead to success vs failure"""

    def get_relation_stats(cases):
        relation_counter = Counter()
        relation_lengths = []
        for case in cases:
            relations = case['relations_used']
            relation_lengths.append(len(relations))
            for rel in relations:
                relation_counter[rel] += 1
        return relation_counter, relation_lengths

    perfect_rels, perfect_lens = get_relation_stats(perfect)
    failure_rels, failure_lens = get_relation_stats(failures)
    low_f1_rels, low_f1_lens = get_relation_stats(low_f1)
    partial_rels, partial_lens = get_relation_stats(partial)

    print(f"Relation path lengths:")
    if perfect_lens:
        print(f"  Perfect: avg={np.mean(perfect_lens):.2f}, min={min(perfect_lens)}, max={max(perfect_lens)}")
    if failure_lens:
        print(f"  Failures: avg={np.mean(failure_lens):.2f}, min={min(failure_lens)}, max={max(failure_lens)}")
    if low_f1_lens:
        print(f"  Low F1: avg={np.mean(low_f1_lens):.2f}, min={min(low_f1_lens)}, max={max(low_f1_lens)}")
    if partial_lens:
        print(f"  Partial: avg={np.mean(partial_lens):.2f}, min={min(partial_lens)}, max={max(partial_lens)}")

    print(f"\nMost common relations in PERFECT cases:")
    for rel, count in perfect_rels.most_common(10):
        pct = count / len(perfect) * 100 if perfect else 0
        print(f"  {rel:<30} {count:>4} ({pct:.1f}%)")

    print(f"\nMost common relations in FAILURE cases:")
    for rel, count in failure_rels.most_common(10):
        pct = count / len(failures) * 100 if failures else 0
        print(f"  {rel:<30} {count:>4} ({pct:.1f}%)")

    print(f"\nMost common relations in LOW F1 cases:")
    for rel, count in low_f1_rels.most_common(10):
        pct = count / len(low_f1) * 100 if low_f1 else 0
        print(f"  {rel:<30} {count:>4} ({pct:.1f}%)")

    # Find relations that appear more in failures than perfect
    print(f"\nRelations over-represented in FAILURES:")
    failure_rate = {}
    for rel in set(list(failure_rels.keys()) + list(perfect_rels.keys())):
        fail_count = failure_rels.get(rel, 0)
        perfect_count = perfect_rels.get(rel, 0)
        total = fail_count + perfect_count
        if total >= 5:  # Only consider relations that appear at least 5 times
            failure_rate[rel] = fail_count / total if total > 0 else 0

    for rel, rate in sorted(failure_rate.items(), key=lambda x: x[1], reverse=True)[:10]:
        if rate > 0.5:  # More than 50% failure rate
            fail_count = failure_rels.get(rel, 0)
            perfect_count = perfect_rels.get(rel, 0)
            print(f"  {rel:<30} {rate*100:.1f}% failure ({fail_count} fail, {perfect_count} perfect)")


def analyze_text_similarity(perfect_cases, failure_cases, text_embeddings):
    """Compare text similarity patterns in perfect vs failure cases"""

    def compute_avg_similarity(cases):
        similarities = []
        for case in cases:
            question = case['question_text']
            relations = case['relations_used']

            if question not in text_embeddings:
                continue

            q_emb = text_embeddings[question]

            for rel in relations:
                if rel in text_embeddings:
                    rel_emb = text_embeddings[rel]
                    sim = cosine_similarity(q_emb, rel_emb)
                    # Normalize to [0, 1]
                    sim_normalized = (sim + 1.0) / 2.0
                    similarities.append(sim_normalized)

        return similarities

    perfect_sims = compute_avg_similarity(perfect_cases)
    failure_sims = compute_avg_similarity(failure_cases)

    if perfect_sims:
        print(f"Perfect cases - Question-Relation text similarity:")
        print(f"  Mean: {np.mean(perfect_sims):.4f}")
        print(f"  Median: {np.median(perfect_sims):.4f}")
        print(f"  Std: {np.std(perfect_sims):.4f}")
        print(f"  Min: {np.min(perfect_sims):.4f}")
        print(f"  Max: {np.max(perfect_sims):.4f}")

    if failure_sims:
        print(f"\nFailure cases - Question-Relation text similarity:")
        print(f"  Mean: {np.mean(failure_sims):.4f}")
        print(f"  Median: {np.median(failure_sims):.4f}")
        print(f"  Std: {np.std(failure_sims):.4f}")
        print(f"  Min: {np.min(failure_sims):.4f}")
        print(f"  Max: {np.max(failure_sims):.4f}")

    if perfect_sims and failure_sims:
        print(f"\nComparison:")
        diff = np.mean(perfect_sims) - np.mean(failure_sims)
        print(f"  Mean difference: {diff:+.4f}")
        if diff > 0:
            print(f"  → Perfect cases have HIGHER text similarity on average")
        else:
            print(f"  → Failure cases have HIGHER text similarity on average")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Variant 3 EdgeScorer decisions')
    parser.add_argument(
        'results_file',
        type=str,
        help='Path to results JSON file'
    )

    args = parser.parse_args()

    # Load text embeddings
    text_embeddings = load_text_embeddings()
    print(f"Loaded {len(text_embeddings):,} text embeddings\n")

    # Analyze results
    analyze_results_file(args.results_file, text_embeddings)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()