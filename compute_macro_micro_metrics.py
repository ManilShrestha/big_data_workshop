#!/usr/bin/env python3
"""
Compute macro and micro precision/recall/F1 metrics from result files.
For files that don't have these metrics, compute them from individual results.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Define all result files with their metadata
RESULT_FILES = {
    "Zero-shot GPT-4o-mini": {
        "1-hop": "variant0_llm_baseline_1-hop-test_batch.json",
        "2-hop": "variant0_llm_baseline_batch_2-hop-test_full_20251022_014041.json",
        "3-hop": "variant0_llm_baseline_batch_3-hop-test_full_20251022_014041.json",
    },
    "Zero-shot Qwen3-30B": {
        "1-hop": "variant0_qwen_baseline_1-hop-test_full_20251025_170954.json",
        "2-hop": "variant0_qwen_baseline_2-hop-test_full_20251025_170954.json",
        "3-hop": "variant0_qwen_baseline_3-hop-test_full_20251025_170954.json",
    },
    "Neural Edge Scoring Greedy": {
        "1-hop": "variant3a_greedy_1-hop_1-hop_20251101_031524.json",
        "2-hop": "variant3a_greedy_2-hop_2-hop_20251101_044154.json",
        "3-hop": "variant3a_greedy_3-hop_3-hop_20251101_044412.json",
    },
    "Neural Edge Scoring Whole Path": {
        "1-hop": "variant3_edge_scorer_1-hop_1-hop_20251031_152838.json",
        "2-hop": "variant3_edge_scorer_2-hop_2-hop_20251031_161024.json",
        "3-hop": "variant3_edge_scorer_3-hop_3-hop_20251030_214640.json",
    },
    "LLM Planning GPT-4o-mini": {
        "1-hop": "variant5_openai_gpt5mini_transe_1-hop-test.json",
        "2-hop": "variant5_openai_gpt5mini_transe_2-hop-test.json",
        "3-hop": "variant5_openai_gpt5mini_transe_3-hop-test.json",
    },
    "LLM Planning Qwen3-30B": {
        "1-hop": "variant5_qwen3-30B_guided_1-hop-test_full_20251025_231521.json",
        "2-hop": "variant5_qwen3-30B_guided_2-hop-test_full_20251026_020834.json",
        "3-hop": "variant5_qwen3-30B_guided_3-hop-test_full_20251026_084204.json",
    },
    "LLM Planning Qwen3-4B": {
        "1-hop": "variant5_qwen3-4B_guided_1-hop-test_full_20251101_221012.json",
        "2-hop": "variant5_qwen3-4B_guided_2-hop-test_full_20251101_225949.json",
        "3-hop": "variant5_qwen_guided_3-hop-test_full_20251102_181204.json",
    },
    "LoRA Finetuned Qwen3-4B": {
        "1-hop": "variant5_qwen4B_LoRA_guided_1-hop-test_full_20251103_014022.json",
        "2-hop": "variant7_qwen4B_LoRA_guided_2-hop-test_full_20251103_014022.json",
        "3-hop": "variant7_qwen4B_LoRA_guided_3-hop-test_full_20251103_014022.json",
    },
}

def compute_macro_micro_from_results(results: List[Dict]) -> Dict:
    """
    Compute macro and micro precision/recall/F1 from individual results.

    Macro: Average of per-question metrics
    Micro: Computed from total correct/incorrect/missed across all questions
    """
    if not results:
        return {}

    # Collect per-question metrics for macro averaging
    precisions = []
    recalls = []
    f1_scores = []

    # Collect totals for micro metrics
    total_tp = 0  # True positives (correct answers)
    total_fp = 0  # False positives (incorrect answers)
    total_fn = 0  # False negatives (missed answers)

    for result in results:
        # Get per-question metrics
        precisions.append(result.get('precision', 0.0))
        recalls.append(result.get('recall', 0.0))
        f1_scores.append(result.get('f1_score', 0.0))

        # Accumulate for micro metrics
        total_tp += result.get('num_correct', 0)
        total_fp += result.get('num_incorrect', 0)
        total_fn += result.get('num_missed', 0)

    # Compute macro metrics (average across questions)
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)

    # Compute micro metrics (from totals)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) \
               if (micro_precision + micro_recall) > 0 else 0.0

    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
    }

def load_and_compute_metrics(filepath: Path) -> Optional[Dict]:
    """
    Load a result file and extract or compute macro/micro metrics.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        metrics = data.get('metrics', {})

        # Check if metrics already exist
        if 'macro_avg_precision' in metrics:
            # Already has macro/micro metrics
            return {
                'macro_precision': metrics.get('macro_avg_precision', 0.0),
                'macro_recall': metrics.get('macro_avg_recall', 0.0),
                'macro_f1': metrics.get('macro_avg_f1_score', 0.0),
                'micro_precision': metrics.get('micro_precision', 0.0),
                'micro_recall': metrics.get('micro_recall', 0.0),
                'micro_f1': metrics.get('micro_f1_score', 0.0),
                'accuracy': metrics.get('accuracy', 0.0),
                'total_questions': metrics.get('total_questions', 0),
            }
        else:
            # Need to compute from individual results
            results = data.get('results', [])
            computed = compute_macro_micro_from_results(results)
            computed['accuracy'] = metrics.get('accuracy', 0.0)
            computed['total_questions'] = metrics.get('total_questions', 0)
            return computed

    except FileNotFoundError:
        print(f"Warning: File not found - {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {filepath}")
        return None
    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}")
        return None

def create_summary_table():
    """Create a summary table with all precision/recall/F1 metrics."""
    results_dir = Path('/home/ms5267/big_data_workshop/results')

    print("Computing macro/micro metrics from result files...")
    print("=" * 100)

    all_data = []

    for method_name, hop_files in RESULT_FILES.items():
        print(f"\n{method_name}")
        print("-" * 100)

        for hop_type, filename in hop_files.items():
            filepath = results_dir / filename
            metrics = load_and_compute_metrics(filepath)

            if metrics:
                row = {
                    'Method': method_name,
                    'Hop': hop_type,
                    'Total Questions': metrics['total_questions'],
                    'Accuracy': metrics['accuracy'],
                    'Macro Precision': metrics['macro_precision'],
                    'Macro Recall': metrics['macro_recall'],
                    'Macro F1': metrics['macro_f1'],
                    'Micro Precision': metrics['micro_precision'],
                    'Micro Recall': metrics['micro_recall'],
                    'Micro F1': metrics['micro_f1'],
                }

                all_data.append(row)

                # Print formatted output
                print(f"  {hop_type:6s} | "
                      f"N={metrics['total_questions']:5d} | "
                      f"Acc={metrics['accuracy']:.4f} | "
                      f"Macro: P={metrics['macro_precision']:.4f} R={metrics['macro_recall']:.4f} F1={metrics['macro_f1']:.4f} | "
                      f"Micro: P={metrics['micro_precision']:.4f} R={metrics['micro_recall']:.4f} F1={metrics['micro_f1']:.4f}")
            else:
                print(f"  {hop_type:6s} | MISSING")

    # Create DataFrame and save
    import pandas as pd
    df = pd.DataFrame(all_data)

    # Save comprehensive table
    output_file = results_dir / 'precision_recall_f1_summary.csv'
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\n{'=' * 100}")
    print(f"Saved summary to: {output_file}")

    # Create pivot tables
    print("\nCreating pivot tables...")

    pivot_metrics = [
        ('Macro Precision', 'macro_precision'),
        ('Macro Recall', 'macro_recall'),
        ('Macro F1', 'macro_f1'),
        ('Micro Precision', 'micro_precision'),
        ('Micro Recall', 'micro_recall'),
        ('Micro F1', 'micro_f1'),
    ]

    for metric_name, col_name in pivot_metrics:
        pivot = df.pivot(index='Method', columns='Hop', values=col_name.replace('_', ' ').title())
        pivot = pivot[['1-hop', '2-hop', '3-hop']]

        safe_name = col_name
        output_file = results_dir / f'pivot_{safe_name}.csv'
        pivot.to_csv(output_file, float_format='%.6f')
        print(f"  - Saved {metric_name}")

    # Print summary pivot tables
    print("\n" + "=" * 100)
    print("MACRO F1 SCORES")
    print("=" * 100)
    pivot_f1 = df.pivot(index='Method', columns='Hop', values='Macro F1')
    pivot_f1 = pivot_f1[['1-hop', '2-hop', '3-hop']]
    print(pivot_f1.to_string(float_format=lambda x: f'{x:.4f}'))

    print("\n" + "=" * 100)
    print("MICRO F1 SCORES")
    print("=" * 100)
    pivot_micro_f1 = df.pivot(index='Method', columns='Hop', values='Micro F1')
    pivot_micro_f1 = pivot_micro_f1[['1-hop', '2-hop', '3-hop']]
    print(pivot_micro_f1.to_string(float_format=lambda x: f'{x:.4f}'))

if __name__ == "__main__":
    create_summary_table()
