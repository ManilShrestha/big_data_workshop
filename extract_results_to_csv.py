#!/usr/bin/env python3
"""
Extract metrics from all result JSON files and create a comprehensive CSV.
"""

import json
import pandas as pd
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

def load_json_metrics(filepath: Path) -> Optional[Dict]:
    """Load metrics from a JSON result file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data.get('metrics', {})
    except FileNotFoundError:
        print(f"Warning: File not found - {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {filepath}")
        return None
    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}")
        return None

def extract_key_metrics(metrics: Dict) -> Dict:
    """Extract the most important metrics from the full metrics dict."""
    if not metrics:
        return {}

    return {
        'Total Questions': metrics.get('total_questions', 0),
        'Accuracy': metrics.get('accuracy', 0.0),
        'Success Rate': metrics.get('success_rate', 0.0),

        # Precision/Recall/F1
        'Macro Avg Precision': metrics.get('macro_avg_precision', 0.0),
        'Macro Avg Recall': metrics.get('macro_avg_recall', 0.0),
        'Macro Avg F1': metrics.get('macro_avg_f1_score', 0.0),
        'Micro Precision': metrics.get('micro_precision', 0.0),
        'Micro Recall': metrics.get('micro_recall', 0.0),
        'Micro F1': metrics.get('micro_f1_score', 0.0),

        # Answer statistics
        'Avg Correct per Q': metrics.get('avg_correct_per_question', 0.0),
        'Avg Incorrect per Q': metrics.get('avg_incorrect_per_question', 0.0),
        'Avg Missed per Q': metrics.get('avg_missed_per_question', 0.0),

        # Search efficiency
        'Avg Nodes Expanded': metrics.get('avg_nodes_expanded', 0.0),
        'Median Nodes Expanded': metrics.get('median_nodes_expanded', 0.0),

        # Time performance
        'Avg Time per Query (s)': metrics.get('avg_time_per_query_sec', 0.0),
        'Queries per Second': metrics.get('queries_per_second', 0.0),

        # Cost
        'Total Cost (USD)': metrics.get('total_cost_usd', 0.0),
        'Cost per Query (USD)': metrics.get('cost_per_query_usd', 0.0),
    }

def create_results_dataframe(results_dir: Path) -> pd.DataFrame:
    """Create a comprehensive DataFrame from all result files."""
    rows = []

    for method_name, hop_files in RESULT_FILES.items():
        for hop_type, filename in hop_files.items():
            filepath = results_dir / filename
            metrics = load_json_metrics(filepath)

            if metrics:
                row = {
                    'Method': method_name,
                    'Hop Type': hop_type,
                    'Filename': filename,
                }
                row.update(extract_key_metrics(metrics))
                rows.append(row)
            else:
                # Add empty row to show missing data
                rows.append({
                    'Method': method_name,
                    'Hop Type': hop_type,
                    'Filename': filename + ' (MISSING)',
                })

    return pd.DataFrame(rows)

def create_pivot_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create pivot tables for key metrics."""
    pivot_tables = {}

    # Key metrics to pivot
    metrics_to_pivot = [
        'Accuracy',
        'Macro Avg F1',
        'Micro F1',
        'Avg Nodes Expanded',
        'Avg Time per Query (s)',
        'Total Cost (USD)',
    ]

    for metric in metrics_to_pivot:
        if metric in df.columns:
            pivot = df.pivot(index='Method', columns='Hop Type', values=metric)
            # Reorder columns
            pivot = pivot[['1-hop', '2-hop', '3-hop']]
            pivot_tables[metric] = pivot

    return pivot_tables

def main():
    """Main execution function."""
    results_dir = Path('/home/ms5267/big_data_workshop/results')

    print("Extracting metrics from result files...")
    df = create_results_dataframe(results_dir)

    # Save comprehensive CSV
    output_csv = results_dir / 'all_results_comprehensive.csv'
    df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"\nSaved comprehensive results to: {output_csv}")

    # Create and save pivot tables
    print("\nCreating pivot tables...")
    pivot_tables = create_pivot_tables(df)

    # Save each pivot table
    for metric_name, pivot_df in pivot_tables.items():
        safe_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        output_file = results_dir / f'pivot_{safe_name}.csv'
        pivot_df.to_csv(output_file, float_format='%.6f')
        print(f"  - Saved {metric_name} pivot to: {output_file}")

    # Create a single combined pivot table file
    combined_output = results_dir / 'all_pivots_combined.csv'
    with open(combined_output, 'w') as f:
        for metric_name, pivot_df in pivot_tables.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"{metric_name}\n")
            f.write(f"{'='*80}\n")
            pivot_df.to_csv(f, float_format='%.6f')
            f.write("\n")

    print(f"\nSaved combined pivot tables to: {combined_output}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print("\nAccuracy by Method and Hop Type:")
    if 'Accuracy' in pivot_tables:
        print(pivot_tables['Accuracy'].to_string(float_format=lambda x: f'{x:.4f}'))

    print("\nMacro F1 Score by Method and Hop Type:")
    if 'Macro Avg F1' in pivot_tables:
        print(pivot_tables['Macro Avg F1'].to_string(float_format=lambda x: f'{x:.4f}'))

    print("\nAverage Nodes Expanded by Method and Hop Type:")
    if 'Avg Nodes Expanded' in pivot_tables:
        print(pivot_tables['Avg Nodes Expanded'].to_string(float_format=lambda x: f'{x:.2f}'))

    print("\n" + "="*80)
    print(f"Total methods evaluated: {len(RESULT_FILES)}")
    print(f"Total result files processed: {len(df)}")
    print(f"Missing files: {df['Filename'].str.contains('MISSING').sum()}")
    print("="*80)

if __name__ == "__main__":
    main()