#!/usr/bin/env python3
"""
Variant 5B: Qwen Relation Ranking + LLM-Guided BFS

This is a comparison variant to Variant 5A (OpenAI upper bound):
- Entity Linking: Exact matching (free)
- Relation Ranking: Keyword-based (simple heuristic)
- Search: Qwen-guided BFS with relation sequencing
- Model: Qwen 30B (self-hosted)

Expected performance vs. Variant 5A:
- Lower accuracy (Qwen 30B < GPT-4o planning quality)
- Zero cost (self-hosted vs. paid API)
- More planning failures (less reliable JSON output)
- Weaker relation sequencing (less reasoning capability)

This variant demonstrates:
1. Impact of model quality on guided search performance
2. Whether graph grounding can compensate for weaker models
3. Cost-accuracy tradeoff in complex multi-hop reasoning
4. Robustness requirements for self-hosted models
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from qa_system.config import Config
from qa_system.entity_linkers.exact_matcher import ExactMatcher
from qa_system.relation_rankers.qwen_ranker import QwenRelationRanker
from qa_system.search_algorithms.llm_guided_bfs import LLMGuidedBFS
from qa_system.utils.loader import (
    load_graph,
    load_node2id,
    load_qa_dataset
)
from qa_system.utils.evaluator import Evaluator

def main():
    import argparse
    from datetime import datetime

    # =========================================================================
    # Parse command-line arguments
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='Variant 5B: Qwen Relation Ranking + LLM-Guided BFS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on 1-hop dataset with 10 questions (test mode)
  python variant5_qwen_guided.py --datasets 1-hop --limit 10

  # Run on full 1-hop dataset
  python variant5_qwen_guided.py --datasets 1-hop

  # Run on all datasets with 100 questions each
  python variant5_qwen_guided.py --datasets 1-hop 2-hop 3-hop --limit 100

Note:
  - Uses self-hosted Qwen 30B model (no cost)
  - Direct mode only (no batch API for Qwen)
  - Parallel API calls via ThreadPoolExecutor
  - Compare with Variant 5A (OpenAI) to see quality gap
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
        '--top-k',
        type=int,
        default=Config.NUM_TOP_RELATIONS,
        help=f'Number of top relations to use (default: {Config.NUM_TOP_RELATIONS})'
    )

    args = parser.parse_args()

    # Generate timestamp for unique output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*80)
    print(" VARIANT 5B: Qwen Relation Ranking + LLM-Guided BFS")
    print("="*80)
    print(f"  Model: Qwen 30B (self-hosted)")
    print(f"  Mode: Direct (parallel API calls)")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  Limit per dataset: {args.limit if args.limit else 'Full dataset'}")
    print(f"  Top-k relations: {args.top_k}")
    print(f"  Timestamp: {timestamp}")
    print("="*80 + "\n")

    # =========================================================================
    # Load Resources
    # =========================================================================
    print("[1/4] Loading graph...")
    print("-" * 80)

    graph = load_graph(Config.GRAPH_PATH)
    node2id = load_node2id(Config.NODE2ID_PATH)

    print()

    # =========================================================================
    # Initialize Components
    # =========================================================================
    print("[2/4] Initializing components...")
    print("-" * 80)

    print("  Initializing ExactMatcher (entity linking)...")
    entity_linker = ExactMatcher(node2id)

    print("  Initializing QwenRelationRanker...")
    relation_ranker = QwenRelationRanker()

    print("  Initializing LLMGuidedBFS (search algorithm)...")
    search_algo = LLMGuidedBFS(graph=graph)

    # Always use LLM planning for Variant 5B (comparison with 5A)
    # Direct mode only (no batch API for Qwen)
    use_llm_planning = True  # Variant 5B uses Qwen planning
    batch_planning = True  # Use parallel calls via ThreadPoolExecutor

    evaluator = Evaluator(
        entity_linker=entity_linker,
        relation_ranker=relation_ranker,
        search_algo=search_algo,
        variant_name="variant5_qwen_guided",
        use_llm_planning=use_llm_planning,
        batch_planning=batch_planning,
        poll_interval=None  # Not used for Qwen
    )

    print("  Components initialized.")
    print("  LLM Planning: Enabled (Qwen 30B - parallel API calls)")
    print("  Cost: $0.00 (self-hosted model)\n")

    # =========================================================================
    # Evaluate on Datasets
    # =========================================================================
    dataset_map = {
        '1-hop': ("1-hop-test", Config.QA_1HOP_TEST, 1),
        '2-hop': ("2-hop-test", Config.QA_2HOP_TEST, 2),
        '3-hop': ("3-hop-test", Config.QA_3HOP_TEST, 3),
    }

    datasets = [
        (dataset_map[ds][0], dataset_map[ds][1], dataset_map[ds][2])
        for ds in args.datasets
    ]

    all_results = {}

    for dataset_name, dataset_path, hop_count in datasets:
        print(f"\n[3/4] Evaluating on {dataset_name} dataset...")
        print("-" * 80)

        # Load dataset
        questions = load_qa_dataset(dataset_path, hop_count=hop_count, limit=args.limit)

        # Set incremental save path for this dataset with timestamp
        limit_str = f"_limit{args.limit}" if args.limit else "_full"
        output_path = f"results/variant5_qwen_guided_{dataset_name}{limit_str}_{timestamp}.json"
        evaluator.incremental_save_path = output_path

        # Evaluate (results will be saved incrementally)
        evaluation = evaluator.evaluate(
            questions=questions,
            top_k_relations=args.top_k,
            verbose=True
        )

        # Save final results (marks as 'completed')
        evaluator.save_results(evaluation, output_path)

        all_results[dataset_name] = evaluation['metrics']

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n[4/4] Summary Across All Datasets")
    print("="*80)

    for dataset_name, metrics in all_results.items():
        print(f"\n{dataset_name.upper()} Results:")
        print(f"  Accuracy:           {metrics['accuracy']:.2%}")
        print(f"  Precision (micro):  {metrics['micro_precision']:.2%}")
        print(f"  Recall (micro):     {metrics['micro_recall']:.2%}")
        print(f"  F1 Score (micro):   {metrics['micro_f1_score']:.2%}")
        print(f"  Avg nodes expanded: {metrics['avg_nodes_expanded']:.1f}")
        print(f"  Avg search time:    {metrics['avg_search_time_ms']:.1f} ms")
        print(f"  Cost per query:     ${metrics['cost_per_query_usd']:.6f} (FREE)")

    print("\n" + "="*80)
    print("  VARIANT 5B EVALUATION COMPLETE")
    print("="*80 + "\n")

    print("\nKey Features:")
    print("  - Entity linking: ExactMatcher (free)")
    print("  - Relation ranking: Keyword-based heuristic (simple)")
    print("  - LLM planning: Qwen 30B with robust JSON parsing")
    print("  - Search: LLM-guided BFS with relation sequencing")
    print("\nComparison points:")
    print("  - Compare accuracy with Variant 5A (OpenAI) to see model quality gap")
    print("  - Cost: $0 vs. ~$0.01 per query for OpenAI")
    print("  - Robustness: Track JSON parsing failures")

if __name__ == "__main__":
    main()
