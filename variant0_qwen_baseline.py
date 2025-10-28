#!/usr/bin/env python3
"""
Variant 0B: Qwen 0-Shot Direct QA (No Graph Traversal)

This is a comparison variant to Variant 0A (GPT-4o baseline):
- Entity Linking: None (no graph used)
- Relation Ranking: None (no graph used)
- Search: None (direct LLM answer)
- Model: Qwen 30B (self-hosted)

Expected performance vs. Variant 0A:
- Lower accuracy (Qwen 30B < GPT-4o quality)
- Zero cost (self-hosted vs. paid API)
- More JSON parsing failures (less instruction-following)
- More hallucinations (weaker reasoning)

This variant demonstrates:
1. Impact of model quality on accuracy
2. Cost-accuracy tradeoff
3. Robustness requirements for weaker models
4. Motivation for both better models AND better methods
"""

import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from qa_system.config import Config
from qa_system.llm_qa.qwen_qa import QwenLLMQA
from qa_system.utils.loader import load_qa_dataset
from qa_system.core.search_result import SearchResult


def fuzzy_match(pred: str, truth: str, threshold: float = 0.8) -> bool:
    """
    Check if two strings match with fuzzy matching (80% character overlap).

    Args:
        pred: Predicted answer (normalized)
        truth: Ground truth answer (normalized)
        threshold: Minimum character overlap ratio (default 0.8)

    Returns:
        True if strings match above threshold
    """
    from difflib import SequenceMatcher

    # Case insensitive comparison
    pred = pred.lower().strip()
    truth = truth.lower().strip()

    # Exact match
    if pred == truth:
        return True

    # Check if one contains the other (handles "The Matrix (1999)" vs "The Matrix")
    if pred in truth or truth in pred:
        # Calculate overlap ratio
        shorter = min(len(pred), len(truth))
        longer = max(len(pred), len(truth))
        if shorter / longer >= threshold:
            return True

    # Use SequenceMatcher for character-level similarity
    ratio = SequenceMatcher(None, pred, truth).ratio()
    return ratio >= threshold


def evaluate_qwen_qa(
    llm_qa,
    questions,
    dataset_name: str,
    output_path: str,
    batch_size: int = None,
    max_workers: int = None,
    verbose: bool = True,
    debug: bool = False
):
    """
    Evaluate Qwen LLM QA on a dataset

    Args:
        llm_qa: QwenLLMQA instance
        questions: List of Question objects
        dataset_name: Name of dataset (for logging)
        output_path: Path to save results JSON
        verbose: Print progress

    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print(f"\n[Evaluating] {dataset_name} dataset ({len(questions)} questions)...")
        print("-" * 80)

    eval_start_time = time.time()

    # Extract question texts
    question_texts = [q.text for q in questions]

    # Answer all questions using parallel calls
    if verbose:
        print("\n[QwenLLM] Answering questions with Qwen 30B (parallel calls)...")

    batch_results = llm_qa.answer_batch(
        question_texts,
        batch_size=batch_size,
        max_workers=max_workers,
        verbose=verbose,
        debug=debug
    )

    # Process results
    results = []
    total_correct = 0
    total_successful = 0
    total_cost = 0.0

    # Metrics accumulators
    total_correct_answers = 0  # True positives
    total_incorrect_answers = 0  # False positives (hallucinations)
    total_missed_answers = 0  # False negatives

    # Answer quality categories
    count_perfect = 0  # All correct, no hallucinations
    count_complete_with_hallucinations = 0  # All ground truth + hallucinations
    count_partial = 0  # Some correct, some missed
    count_hallucination_only = 0  # All wrong
    count_failed = 0  # No answer

    # JSON parsing failure tracking
    json_parse_failures = 0

    for i, (question, (predicted_answers, cost)) in enumerate(zip(questions, batch_results)):
        # Normalize answers for comparison
        predicted_normalized = [ans.lower().strip() for ans in predicted_answers]
        ground_truth_normalized = [ans.lower().strip() for ans in question.ground_truth_answers]

        # Use fuzzy matching to match predicted vs ground truth
        correct_answers = []
        incorrect_answers = []
        matched_ground_truth = set()

        for pred in predicted_normalized:
            matched = False
            for j, truth in enumerate(ground_truth_normalized):
                if j not in matched_ground_truth and fuzzy_match(pred, truth, threshold=0.8):
                    correct_answers.append(pred)
                    matched_ground_truth.add(j)
                    matched = True
                    break
            if not matched:
                incorrect_answers.append(pred)

        # Find missed answers (ground truth not matched)
        missed_answers = [truth for j, truth in enumerate(ground_truth_normalized) if j not in matched_ground_truth]

        num_correct = len(correct_answers)
        num_incorrect = len(incorrect_answers)
        num_missed = len(missed_answers)

        # Multiple correctness levels to capture nuanced behavior
        is_perfect = num_correct > 0 and num_missed == 0 and num_incorrect == 0  # All correct, no hallucinations
        is_correct_complete = num_correct > 0 and num_missed == 0  # All ground truth found (may have hallucinations)
        is_partial = num_correct > 0 and num_missed > 0  # Some correct, some missed
        is_hallucination_only = num_correct == 0 and num_incorrect > 0  # All hallucinations
        is_failed = len(predicted_answers) == 0  # No answer provided

        # Track JSON parse failures (empty answers likely due to parsing issues)
        if is_failed:
            json_parse_failures += 1

        # Legacy metric (strict: complete coverage required)
        is_correct = is_correct_complete
        is_successful = len(predicted_answers) > 0

        if is_correct:
            total_correct += 1
        if is_successful:
            total_successful += 1

        # Count answer quality categories
        if is_perfect:
            count_perfect += 1
            answer_category = "perfect"
        elif is_correct_complete and num_incorrect > 0:
            count_complete_with_hallucinations += 1
            answer_category = "complete_with_hallucinations"
        elif is_partial:
            count_partial += 1
            answer_category = "partial"
        elif is_hallucination_only:
            count_hallucination_only += 1
            answer_category = "hallucination_only"
        elif is_failed:
            count_failed += 1
            answer_category = "failed"
        else:
            answer_category = "unknown"

        total_correct_answers += num_correct
        total_incorrect_answers += num_incorrect
        total_missed_answers += num_missed
        total_cost += cost

        # Create SearchResult for compatibility with existing infrastructure
        result = SearchResult(
            question_id=question.question_id,
            question_text=question.text,
            predicted_answers=predicted_answers,
            ground_truth_answers=question.ground_truth_answers,
            nodes_expanded=0,  # No graph traversal
            search_time_ms=0.0,  # Cost is in LLM inference
            success=is_successful,
            reasoning_path=[],  # No reasoning path
            relations_used=[],  # No relations
            metadata={
                'cost_usd': cost,
                'correct_answers': list(correct_answers),
                'incorrect_answers': list(incorrect_answers),
                'missed_answers': list(missed_answers),
                'is_correct': is_correct,
                'is_perfect': is_perfect,
                'is_partial': is_partial,
                'is_hallucination_only': is_hallucination_only,
                'answer_category': answer_category,
                'hallucination_rate': num_incorrect / len(predicted_answers) if predicted_answers else 0.0
            }
        )

        results.append(result.to_dict())

        # Print progress
        if verbose and (i + 1) % 100 == 0:
            print(f"  [Progress] {i+1}/{len(questions)} questions processed...")

    # Calculate aggregate metrics
    import statistics

    accuracy = total_correct / len(questions) if questions else 0.0
    success_rate = total_successful / len(questions) if questions else 0.0
    cost_per_query = total_cost / len(questions) if questions else 0.0

    # Micro-averaging: Precision, Recall, F1 (aggregate all predictions first)
    micro_precision = total_correct_answers / (total_correct_answers + total_incorrect_answers) if (total_correct_answers + total_incorrect_answers) > 0 else 0.0
    micro_recall = total_correct_answers / (total_correct_answers + total_missed_answers) if (total_correct_answers + total_missed_answers) > 0 else 0.0
    micro_f1_score = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Macro-averaging: Calculate precision/recall/F1 per question, then average
    precisions = []
    recalls = []
    f1_scores = []

    for result_dict in results:
        metadata = result_dict.get('metadata', {})
        num_correct = len(metadata.get('correct_answers', []))
        num_incorrect = len(metadata.get('incorrect_answers', []))
        num_missed = len(metadata.get('missed_answers', []))

        # Calculate per-question metrics
        q_precision = num_correct / (num_correct + num_incorrect) if (num_correct + num_incorrect) > 0 else 0.0
        q_recall = num_correct / (num_correct + num_missed) if (num_correct + num_missed) > 0 else 0.0
        q_f1 = 2 * q_precision * q_recall / (q_precision + q_recall) if (q_precision + q_recall) > 0 else 0.0

        precisions.append(q_precision)
        recalls.append(q_recall)
        f1_scores.append(q_f1)

    macro_avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    macro_median_precision = statistics.median(precisions) if precisions else 0.0
    macro_median_recall = statistics.median(recalls) if recalls else 0.0
    macro_median_f1_score = statistics.median(f1_scores) if f1_scores else 0.0

    eval_time = time.time() - eval_start_time

    metrics = {
        # Core accuracy metrics
        'total_questions': len(questions),
        'correct_answers': total_correct,
        'accuracy': accuracy,
        'successful_searches': total_successful,
        'success_rate': success_rate,

        # Macro-averaging: average P/R/F1 across all questions (treats all questions equally)
        'macro_avg_precision': macro_avg_precision,
        'macro_avg_recall': macro_avg_recall,
        'macro_avg_f1_score': macro_avg_f1_score,
        'macro_median_precision': macro_median_precision,
        'macro_median_recall': macro_median_recall,
        'macro_median_f1_score': macro_median_f1_score,

        # Micro-averaging: aggregate all predictions then calculate (weights by answer count)
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1_score': micro_f1_score,

        # Legacy names (micro-averaged, for backwards compatibility)
        'precision': micro_precision,
        'recall': micro_recall,
        'f1_score': micro_f1_score,

        # Answer quality metrics
        'avg_correct_per_question': total_correct_answers / len(questions) if questions else 0.0,
        'avg_incorrect_per_question': total_incorrect_answers / len(questions) if questions else 0.0,
        'avg_missed_per_question': total_missed_answers / len(questions) if questions else 0.0,
        'total_correct_answers': total_correct_answers,
        'total_incorrect_answers': total_incorrect_answers,
        'total_missed_answers': total_missed_answers,

        # Efficiency metrics
        'avg_nodes_expanded': 0.0,  # No graph traversal
        'min_nodes_expanded': 0,
        'max_nodes_expanded': 0,
        'median_nodes_expanded': 0.0,

        # Timing metrics
        'avg_search_time_ms': 0.0,  # No graph search
        'min_search_time_ms': 0.0,
        'max_search_time_ms': 0.0,
        'median_search_time_ms': 0.0,
        'total_search_time_sec': 0.0,

        # Total evaluation time (includes all overhead: LLM inference, processing)
        'total_eval_time_sec': eval_time,
        'total_eval_time_min': eval_time / 60,
        'avg_time_per_query_sec': eval_time / len(questions) if questions else 0.0,

        # Cost metrics
        'total_cost_usd': total_cost,
        'cost_per_query_usd': cost_per_query,

        # Throughput
        'queries_per_second': len(questions) / eval_time if eval_time > 0 else 0.0,
        'queries_per_minute': (len(questions) / eval_time) * 60 if eval_time > 0 else 0.0,

        # Answer quality breakdown
        'count_perfect': count_perfect,
        'count_complete_with_hallucinations': count_complete_with_hallucinations,
        'count_partial': count_partial,
        'count_hallucination_only': count_hallucination_only,
        'count_failed': count_failed,
        'pct_perfect': count_perfect / len(questions) * 100 if questions else 0.0,
        'pct_complete_with_hallucinations': count_complete_with_hallucinations / len(questions) * 100 if questions else 0.0,
        'pct_partial': count_partial / len(questions) * 100 if questions else 0.0,
        'pct_hallucination_only': count_hallucination_only / len(questions) * 100 if questions else 0.0,
        'pct_failed': count_failed / len(questions) * 100 if questions else 0.0,

        # Qwen-specific metrics
        'json_parse_failures': json_parse_failures,
        'json_parse_failure_rate': json_parse_failures / len(questions) * 100 if questions else 0.0
    }

    # Save results
    output = {
        'variant_name': 'variant0_qwen_baseline',
        'dataset_name': dataset_name,
        'model': llm_qa.model,
        'mode': 'direct',
        'status': 'completed',
        'metrics': metrics,
        'results': results
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    if verbose:
        print(f"\n[Results] {dataset_name}")
        print(f"  Accuracy:           {accuracy:.2%}")
        print(f"  Precision (micro):  {micro_precision:.2%}")
        print(f"  Recall (micro):     {micro_recall:.2%}")
        print(f"  F1 Score (micro):   {micro_f1_score:.2%}")
        print(f"  Correct answers:    {total_correct_answers}")
        print(f"  Incorrect answers:  {total_incorrect_answers} (hallucinations)")
        print(f"  Missed answers:     {total_missed_answers}")
        print(f"\n  Answer Quality Breakdown:")
        print(f"    Perfect (all correct, no hallucinations):     {count_perfect:2d} ({count_perfect/len(questions)*100:5.1f}%)")
        print(f"    Complete + Hallucinations (complete but +):   {count_complete_with_hallucinations:2d} ({count_complete_with_hallucinations/len(questions)*100:5.1f}%)")
        print(f"    Partial (some correct, some missed):          {count_partial:2d} ({count_partial/len(questions)*100:5.1f}%)")
        print(f"    Hallucination Only (all wrong):               {count_hallucination_only:2d} ({count_hallucination_only/len(questions)*100:5.1f}%)")
        print(f"    Failed (no answer):                            {count_failed:2d} ({count_failed/len(questions)*100:5.1f}%)")
        print(f"\n  Qwen-Specific:")
        print(f"    JSON parse failures: {json_parse_failures} ({json_parse_failures/len(questions)*100:.1f}%)")
        print(f"\n  Cost per query:     ${cost_per_query:.6f} (FREE - self-hosted)")
        print(f"  Total cost:         ${total_cost:.6f}")
        print(f"  Eval time:          {eval_time:.1f}s")
        print(f"  Queries/sec:        {metrics['queries_per_second']:.1f}")
        print(f"  Results saved to:   {output_path}")

    return output


def main():
    import argparse
    from datetime import datetime

    # =========================================================================
    # Parse command-line arguments
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='Variant 0B: Qwen 0-Shot Direct QA (No Graph Traversal)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on 10 questions to test
  python variant0_qwen_baseline.py --limit 10

  # Run on full 1-hop dataset
  python variant0_qwen_baseline.py --datasets 1-hop

  # Run on all datasets with custom batch settings
  python variant0_qwen_baseline.py --datasets 1-hop 2-hop 3-hop --batch-size 20 --max-workers 5

  # Run with higher parallelism for faster processing
  python variant0_qwen_baseline.py --datasets 1-hop --batch-size 100 --max-workers 20
        """
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum questions per dataset (default: None = full dataset)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        choices=['1-hop', '2-hop', '3-hop'],
        default=['1-hop'],
        help='Datasets to evaluate (default: 1-hop)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Batch size for parallel processing (default: {Config.OPENAI_BATCH_SIZE})'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help=f'Max parallel workers (default: {Config.OPENAI_MAX_WORKERS})'
    )

    args = parser.parse_args()

    # Generate timestamp for unique output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*80)
    print(" VARIANT 0B: Qwen 0-Shot Direct QA (No Graph Traversal)")
    print("="*80)
    print(f"  Model: Qwen 30B (self-hosted)")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  Limit per dataset: {args.limit if args.limit else 'Full dataset'}")
    print(f"  Timestamp: {timestamp}")
    print("="*80 + "\n")

    # =========================================================================
    # Initialize Qwen LLM QA
    # =========================================================================
    print(f"[1/4] Initializing Qwen LLM QA...")
    print("-" * 80)

    llm_qa = QwenLLMQA()
    print(f"  Model: {llm_qa.model}")
    print(f"  Endpoint: {llm_qa.base_url}")
    print(f"  Components initialized.\n")

    # =========================================================================
    # Evaluate on Datasets
    # =========================================================================
    dataset_map = {
        '1-hop': ("1-hop-test", Config.QA_1HOP_TEST, 1),
        '2-hop': ("2-hop-test", Config.QA_2HOP_TEST, 2),
        '3-hop': ("3-hop-test", Config.QA_3HOP_TEST, 3),
    }

    datasets = [
        (dataset_map[ds][0], dataset_map[ds][1], dataset_map[ds][2], args.limit)
        for ds in args.datasets
    ]

    all_results = {}

    for dataset_name, dataset_path, hop_count, limit in datasets:
        print(f"\n[2/4] Evaluating on {dataset_name} dataset...")
        print("-" * 80)

        # Load dataset
        questions = load_qa_dataset(dataset_path, hop_count=hop_count, limit=limit)

        # Evaluate
        limit_str = f"_limit{limit}" if limit else "_full"
        output_path = f"results/variant0_qwen_baseline_{dataset_name}{limit_str}_{timestamp}.json"

        # Enable debug mode when limit is set
        debug_mode = limit is not None

        evaluation = evaluate_qwen_qa(
            llm_qa=llm_qa,
            questions=questions,
            dataset_name=dataset_name,
            output_path=output_path,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            verbose=True,
            debug=debug_mode
        )

        all_results[dataset_name] = evaluation['metrics']

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n[3/4] Summary Across All Datasets")
    print("="*80)

    for dataset_name, metrics in all_results.items():
        print(f"\n{dataset_name.upper()} Results:")
        print(f"  Accuracy:           {metrics['accuracy']:.2%}")
        print(f"  Precision:          {metrics['precision']:.2%}")
        print(f"  Recall:             {metrics['recall']:.2%}")
        print(f"  F1 Score:           {metrics['f1_score']:.2%}")
        print(f"  Hallucinations:     {metrics['total_incorrect_answers']} answers")
        print(f"  JSON failures:      {metrics['json_parse_failures']} ({metrics['json_parse_failure_rate']:.1f}%)")
        print(f"  Answer Quality:")
        print(f"    Perfect:          {metrics['count_perfect']} ({metrics['pct_perfect']:.1f}%)")
        print(f"    Complete+Halluc:  {metrics['count_complete_with_hallucinations']} ({metrics['pct_complete_with_hallucinations']:.1f}%)")
        print(f"    Partial:          {metrics['count_partial']} ({metrics['pct_partial']:.1f}%)")
        print(f"    Halluc Only:      {metrics['count_hallucination_only']} ({metrics['pct_hallucination_only']:.1f}%)")
        print(f"  Cost per query:     ${metrics['cost_per_query_usd']:.6f} (FREE)")

    print("\n" + "="*80)
    print("  VARIANT 0B EVALUATION COMPLETE")
    print("="*80 + "\n")

    print("Key findings:")
    print("  - Qwen 30B can answer directly but with lower accuracy than GPT-4o")
    print("  - Cost: $0 (self-hosted)")
    print("  - More JSON parsing failures than GPT-4o (less instruction-following)")
    print("  - Demonstrates model quality vs. cost tradeoff")
    print("  - Compare with Variant 0A (GPT-4o) to see quality gap")


if __name__ == "__main__":
    main()
