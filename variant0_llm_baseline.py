#!/usr/bin/env python3
"""
Variant 0: LLM 0-Shot Direct QA (No Graph Traversal)

This is the BASELINE variant to show limitations of pure LLM approach:
- Entity Linking: None (no graph used)
- Relation Ranking: None (no graph used)
- Search: None (direct LLM answer)

Expected performance:
- Medium accuracy (may hallucinate, may miss obscure entities)
- High cost (~$0.0001-0.0002 per query for GPT-4o-mini)
- Fast inference (no graph traversal)
- Cannot guarantee answers are in KB

This variant demonstrates:
1. Baseline accuracy without structured KB
2. Cost of pure LLM approach
3. Hallucination issues (answers not in KB)
4. Motivation for graph-grounded methods (Variants 1-5)
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
from qa_system.llm_qa.direct_qa import DirectLLMQA
from qa_system.llm_qa.batch_qa import BatchLLMQA
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


def evaluate_direct_llm(
    llm_qa,  # DirectLLMQA or BatchLLMQA instance
    questions,
    dataset_name: str,
    output_path: str,
    mode: str = "direct",
    poll_interval: int = 30,
    dry_run: bool = False,
    batch_id: str = None,
    verbose: bool = True
):
    """
    Evaluate direct LLM QA on a dataset

    Args:
        llm_qa: DirectLLMQA or BatchLLMQA instance
        questions: List of Question objects
        dataset_name: Name of dataset (for logging)
        output_path: Path to save results JSON
        mode: "direct" for parallel API calls, "batch" for OpenAI Batch API
        poll_interval: Seconds between batch status checks (batch mode only)
        dry_run: If True, create batch but don't wait (batch mode only)
        batch_id: Resume waiting for existing batch ID (batch mode only)
        verbose: Print progress

    Returns:
        Dictionary with evaluation results (or batch_id if dry_run)
    """
    if verbose:
        print(f"\n[Evaluating] {dataset_name} dataset ({len(questions)} questions)...")
        print("-" * 80)

    eval_start_time = time.time()

    # Extract question texts
    question_texts = [q.text for q in questions]

    # Answer all questions using appropriate method
    if mode == "batch":
        if batch_id:
            # Resume existing batch
            if verbose:
                print(f"\n[BatchLLM] Resuming batch: {batch_id}...")
            batch_results = llm_qa.retrieve_batch_results(
                batch_id=batch_id,
                poll_interval=poll_interval,
                verbose=verbose
            )
        elif dry_run:
            # Create batch but don't wait
            if verbose:
                print("\n[BatchLLM] Creating batch (dry-run mode - will not wait for completion)...")
            created_batch_id = llm_qa.create_batch(
                questions=question_texts,
                batch_description=f"Variant0 - {dataset_name}",
                verbose=verbose
            )
            if verbose:
                print(f"\n{'='*80}")
                print(f"DRY-RUN MODE: Batch created successfully!")
                print(f"Batch ID: {created_batch_id}")
                print(f"{'='*80}")
                print(f"\nTo resume and wait for this batch, run:")
                print(f"  python variant0_llm_baseline.py --mode batch --batch-id {created_batch_id}")
                print(f"\nTo check batch status manually, use:")
                print(f"  python batch_manager.py --status {created_batch_id}")
                print(f"{'='*80}\n")
            return {"batch_id": created_batch_id, "status": "created", "dry_run": True}
        else:
            # Normal batch mode: create and wait
            if verbose:
                print("\n[BatchLLM] Answering questions with OpenAI Batch API (50% cheaper, async)...")
            batch_results = llm_qa.answer_batch_sync(
                question_texts,
                poll_interval=poll_interval,
                verbose=verbose
            )
    else:  # direct mode
        if verbose:
            print("\n[DirectLLM] Answering questions with LLM (parallel calls)...")
        batch_results = llm_qa.answer_batch(question_texts, verbose=verbose)

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
    accuracy = total_correct / len(questions) if questions else 0.0
    success_rate = total_successful / len(questions) if questions else 0.0
    cost_per_query = total_cost / len(questions) if questions else 0.0

    # Precision, Recall, F1
    precision = total_correct_answers / (total_correct_answers + total_incorrect_answers) if (total_correct_answers + total_incorrect_answers) > 0 else 0.0
    recall = total_correct_answers / (total_correct_answers + total_missed_answers) if (total_correct_answers + total_missed_answers) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    eval_time = time.time() - eval_start_time

    metrics = {
        'total_questions': len(questions),
        'correct_answers': total_correct,
        'accuracy': accuracy,
        'successful_searches': total_successful,
        'success_rate': success_rate,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_correct_per_question': total_correct_answers / len(questions) if questions else 0.0,
        'avg_incorrect_per_question': total_incorrect_answers / len(questions) if questions else 0.0,
        'avg_missed_per_question': total_missed_answers / len(questions) if questions else 0.0,
        'total_correct_answers': total_correct_answers,
        'total_incorrect_answers': total_incorrect_answers,
        'total_missed_answers': total_missed_answers,
        'avg_nodes_expanded': 0.0,  # No graph traversal
        'total_cost_usd': total_cost,
        'cost_per_query_usd': cost_per_query,
        'total_eval_time_sec': eval_time,
        'queries_per_second': len(questions) / eval_time if eval_time > 0 else 0.0,
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
        'pct_failed': count_failed / len(questions) * 100 if questions else 0.0
    }

    # Save results
    output = {
        'variant_name': 'variant0_llm_baseline',
        'dataset_name': dataset_name,
        'model': llm_qa.model,
        'mode': mode,
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
        print(f"  Precision:          {precision:.2%}")
        print(f"  Recall:             {recall:.2%}")
        print(f"  F1 Score:           {f1_score:.2%}")
        print(f"  Correct answers:    {total_correct_answers}")
        print(f"  Incorrect answers:  {total_incorrect_answers} (hallucinations)")
        print(f"  Missed answers:     {total_missed_answers}")
        print(f"\n  Answer Quality Breakdown:")
        print(f"    Perfect (all correct, no hallucinations):     {count_perfect:2d} ({count_perfect/len(questions)*100:5.1f}%)")
        print(f"    Complete + Hallucinations (complete but +):   {count_complete_with_hallucinations:2d} ({count_complete_with_hallucinations/len(questions)*100:5.1f}%)")
        print(f"    Partial (some correct, some missed):          {count_partial:2d} ({count_partial/len(questions)*100:5.1f}%)")
        print(f"    Hallucination Only (all wrong):               {count_hallucination_only:2d} ({count_hallucination_only/len(questions)*100:5.1f}%)")
        print(f"    Failed (no answer):                            {count_failed:2d} ({count_failed/len(questions)*100:5.1f}%)")
        print(f"\n  Cost per query:     ${cost_per_query:.6f}")
        print(f"  Total cost:         ${total_cost:.6f}")
        print(f"  Eval time:          {eval_time:.1f}s")
        print(f"  Queries/sec:        {metrics['queries_per_second']:.1f}")
        print(f"  Results saved to:   {output_path}")

    return output


def main():
    import argparse

    # =========================================================================
    # Parse command-line arguments
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='Variant 0: LLM 0-Shot Direct QA (No Graph Traversal)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct mode (parallel API calls, immediate results)
  python variant0_llm_baseline.py --mode direct --limit 10

  # Batch mode - DRY RUN (create batch, get ID, don't wait)
  python variant0_llm_baseline.py --mode batch --limit 5 --dry-run

  # Batch mode - RESUME (wait for existing batch to complete)
  python variant0_llm_baseline.py --mode batch --batch-id batch_abc123

  # Check batch status manually
  python batch_manager.py --status batch_abc123

  # List all recent batches
  python batch_manager.py --list
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['direct', 'batch'],
        default='direct',
        help='API mode: "direct" for parallel calls (faster), "batch" for async (50%% cheaper)'
    )
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=30,
        help='Seconds between batch status checks (batch mode only, default: 30)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10000,
        help='Maximum questions per dataset (default: 10000)'
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
        '--dry-run',
        action='store_true',
        help='(Batch mode only) Create batch and print ID, but do not wait for completion'
    )
    parser.add_argument(
        '--batch-id',
        type=str,
        help='(Batch mode only) Resume waiting for an existing batch by ID'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print(" VARIANT 0: LLM 0-Shot Direct QA (No Graph Traversal)")
    print("="*80 + "\n")

    # =========================================================================
    # Initialize LLM QA (Direct or Batch mode)
    # =========================================================================
    print(f"[1/4] Initializing LLM QA ({args.mode} mode)...")
    print("-" * 80)

    if args.mode == "batch":
        llm_qa = BatchLLMQA()
        print(f"  Mode: Batch API (50% cheaper, async processing)")
        print(f"  Poll interval: {args.poll_interval}s")
    else:
        llm_qa = DirectLLMQA()
        print(f"  Mode: Direct API (parallel calls, immediate results)")

    print(f"  Model: {llm_qa.model}")
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
        mode_suffix = f"_{args.mode}" if args.mode == "batch" else ""
        output_path = f"results/variant0_llm_baseline_{dataset_name}{mode_suffix}.json"
        evaluation = evaluate_direct_llm(
            llm_qa=llm_qa,
            questions=questions,
            dataset_name=dataset_name,
            output_path=output_path,
            mode=args.mode,
            poll_interval=args.poll_interval,
            dry_run=args.dry_run,
            batch_id=args.batch_id,
            verbose=True
        )

        # Handle dry-run mode
        if args.dry_run and evaluation.get('dry_run'):
            print(f"\n[Info] Batch created for {dataset_name}. Exiting (dry-run mode).")
            return

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
        print(f"  Answer Quality:")
        print(f"    Perfect:          {metrics['count_perfect']} ({metrics['pct_perfect']:.1f}%)")
        print(f"    Complete+Halluc:  {metrics['count_complete_with_hallucinations']} ({metrics['pct_complete_with_hallucinations']:.1f}%)")
        print(f"    Partial:          {metrics['count_partial']} ({metrics['pct_partial']:.1f}%)")
        print(f"    Halluc Only:      {metrics['count_hallucination_only']} ({metrics['pct_hallucination_only']:.1f}%)")
        print(f"  Cost per query:     ${metrics['cost_per_query_usd']:.6f}")

    print("\n" + "="*80)
    print("  VARIANT 0 EVALUATION COMPLETE")
    print("="*80 + "\n")

    print("Key findings:")
    print("  - LLM can answer directly but may hallucinate entities not in KB")
    print("  - Cost: ~$0.0001-0.0002 per query")
    print("  - No graph grounding = no guarantee answers are valid")
    print("  - This motivates graph-based methods (Variants 1-5)")


if __name__ == "__main__":
    main()
