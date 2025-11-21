#!/usr/bin/env python3
"""
Extract LoRA fine-tuning training data from Variant5 results.

This script converts GPT-5-mini's reasoning traces from Variant5 into
training data suitable for fine-tuning Qwen via LoRA.

Input: Variant5 result JSON files (e.g., variant5_openai_gpt5mini_transe_3-hop-test.json)
Output: JSONL files in instruction-tuning format for LoRA training

Each training example contains:
- Input: Question + available relations + hop count
- Output: GPT-5-mini's reasoning + relation sequence plan
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


# Available relations in MetaQA
METAQA_RELATIONS = [
    'directed_by',
    'starred_actors',
    'written_by',
    'in_language',
    'has_genre',
    'release_year',
    'has_tags',
    'has_imdb_rating',
    'has_imdb_votes',
]


def build_system_prompt() -> str:
    """Build the system prompt for the instruction-tuning format"""
    return """You are analyzing a knowledge graph question to determine which relations to traverse at each hop.

Your task is to select which relation(s) are most relevant at EACH hop to answer the question.

Think step-by-step about the logical path needed to answer the question."""


def build_user_prompt(question: str, hop_count: int) -> str:
    """Build the user prompt with question and available relations"""
    relations_list = "\n".join([f"- {rel}" for rel in METAQA_RELATIONS])

    return f"""Available relations in the knowledge graph:
{relations_list}

Question: "{question}"

This is a {hop_count}-hop question. You need to select which relation(s) might be relevant at EACH hop.

For each hop, select 1-2 most relevant relations from the list above. Think about the logical path needed to answer the question.

Respond in JSON format:
{{
  "reasoning": "brief explanation of the path",
  "hops": [
    ["relation1"],
    ["relation2"],
    ...
  ]
}}

Provide your analysis:"""


def build_assistant_response(reasoning: str, hops: List[List[str]]) -> str:
    """Build the assistant response (GPT-5-mini's output)"""
    response = {
        "reasoning": reasoning,
        "hops": hops
    }
    return json.dumps(response, indent=2)


def extract_training_examples(
    variant5_results: Dict[str, Any],
    filter_correct_only: bool = True,
    filter_failed_llm: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract training examples from Variant5 results

    Args:
        variant5_results: Loaded Variant5 results JSON
        filter_correct_only: Only include correctly answered questions
        filter_failed_llm: Exclude questions where LLM planning failed

    Returns:
        List of training examples in instruction-tuning format
    """
    training_examples = []

    results = variant5_results.get('results', [])

    for result in results:
        # Extract fields
        question_text = result.get('question_text', '')
        is_correct = result.get('is_correct', False)
        metadata = result.get('metadata', {})

        # Get LLM reasoning and planned hops
        llm_reasoning = metadata.get('llm_reasoning', '')
        llm_planned_hops = metadata.get('llm_planned_hops', [])

        # Infer hop count from planned hops
        hop_count = len(llm_planned_hops) if llm_planned_hops else 0

        # Apply filters
        if filter_correct_only and not is_correct:
            continue

        if filter_failed_llm and (llm_reasoning == "FAILED" or not llm_planned_hops):
            continue

        # Build instruction-tuning example
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(question_text, hop_count)
        assistant_response = build_assistant_response(llm_reasoning, llm_planned_hops)

        # Format for LoRA training (messages format)
        training_example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ],
            # Add metadata for debugging/analysis
            "metadata": {
                "question_id": result.get('question_id'),
                "hop_count": hop_count,
                "is_correct": is_correct,
                "precision": result.get('precision'),
                "recall": result.get('recall'),
                "f1_score": result.get('f1_score')
            }
        }

        training_examples.append(training_example)

    return training_examples


def save_training_data(
    training_examples: List[Dict[str, Any]],
    output_path: str,
    format: str = 'jsonl'
):
    """
    Save training examples to file

    Args:
        training_examples: List of training examples
        output_path: Output file path
        format: 'jsonl' or 'json'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'jsonl':
        # JSONL format (one example per line)
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in training_examples:
                f.write(json.dumps(example) + '\n')
    else:
        # JSON format (single array)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_examples, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(training_examples)} training examples to {output_path}")


def analyze_dataset(training_examples: List[Dict[str, Any]]):
    """Print statistics about the training dataset"""
    if not training_examples:
        print("  No training examples found!")
        return

    # Count by hop
    hop_counts = {}
    for ex in training_examples:
        hop_count = ex['metadata']['hop_count']
        hop_counts[hop_count] = hop_counts.get(hop_count, 0) + 1

    # Accuracy stats
    total = len(training_examples)
    correct = sum(1 for ex in training_examples if ex['metadata']['is_correct'])

    print(f"\n  Dataset Statistics:")
    print(f"  Total examples: {total}")
    print(f"  Correct answers: {correct} ({correct/total*100:.1f}%)")
    print(f"\n  Examples by hop count:")
    for hop in sorted(hop_counts.keys()):
        count = hop_counts[hop]
        print(f"    {hop}-hop: {count} ({count/total*100:.1f}%)")

    # Average metrics
    avg_precision = sum(ex['metadata']['precision'] for ex in training_examples) / total
    avg_recall = sum(ex['metadata']['recall'] for ex in training_examples) / total
    avg_f1 = sum(ex['metadata']['f1_score'] for ex in training_examples) / total

    print(f"\n  Average metrics:")
    print(f"    Precision: {avg_precision:.3f}")
    print(f"    Recall: {avg_recall:.3f}")
    print(f"    F1 Score: {avg_f1:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract LoRA training data from Variant5 results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from single file (correct answers only)
  python extract_lora_training_data.py \\
    --input results/variant5_openai_gpt5mini_transe_3-hop-test.json \\
    --output variant7/training_data/3hop_train.jsonl

  # Extract from multiple files (all hop counts)
  python extract_lora_training_data.py \\
    --input results/variant5_*-hop-test.json \\
    --output variant7/training_data/all_hops_train.jsonl

  # Include all examples (even incorrect ones)
  python extract_lora_training_data.py \\
    --input results/variant5_*.json \\
    --output variant7/training_data/all_examples.jsonl \\
    --no-filter-correct

  # Save as JSON instead of JSONL
  python extract_lora_training_data.py \\
    --input results/variant5_*-hop-test.json \\
    --output variant7/training_data/train.json \\
    --format json

Note:
  - By default, only correctly answered questions are included
  - Failed LLM planning attempts (reasoning="FAILED") are excluded
  - Output format: JSONL (one example per line) for easy streaming
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        required=True,
        help='Input Variant5 result JSON file(s) (supports glob patterns)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output training data file (.jsonl or .json)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['jsonl', 'json'],
        default='jsonl',
        help='Output format (default: jsonl)'
    )
    parser.add_argument(
        '--no-filter-correct',
        action='store_true',
        help='Include incorrect answers (default: correct only)'
    )
    parser.add_argument(
        '--no-filter-failed',
        action='store_true',
        help='Include failed LLM planning attempts (default: exclude)'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print(" VARIANT7: Extract LoRA Training Data from Variant5 Results")
    print("="*80)
    print(f"  Input files: {len(args.input)}")
    print(f"  Output: {args.output}")
    print(f"  Format: {args.format}")
    print(f"  Filter correct only: {not args.no_filter_correct}")
    print(f"  Filter failed LLM: {not args.no_filter_failed}")
    print("="*80 + "\n")

    # Process all input files
    all_training_examples = []

    for input_file in args.input:
        print(f"Processing: {input_file}")

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                variant5_results = json.load(f)

            # Extract training examples
            training_examples = extract_training_examples(
                variant5_results,
                filter_correct_only=not args.no_filter_correct,
                filter_failed_llm=not args.no_filter_failed
            )

            print(f"  Extracted {len(training_examples)} examples")
            all_training_examples.extend(training_examples)

        except Exception as e:
            print(f"  Error processing {input_file}: {e}")
            continue

    if not all_training_examples:
        print("\nNo training examples extracted. Exiting.")
        return

    # Print dataset statistics
    analyze_dataset(all_training_examples)

    # Save training data
    print(f"\nSaving training data...")
    save_training_data(all_training_examples, args.output, format=args.format)

    print("\n" + "="*80)
    print(" EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nTraining data saved to: {args.output}")
    print(f"Total examples: {len(all_training_examples)}")
    print("\nNext steps:")
    print("  1. Review the training data format")
    print("  2. Set up LoRA fine-tuning configuration")
    print("  3. Train Qwen model using this data")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
