#!/usr/bin/env python3
"""
Variant 5: OpenAI Relation Ranking + LLM-Guided BFS

This is the UPPER BOUND variant:
- Entity Linking: Exact matching (free)
- Relation Ranking: OpenAI embeddings (intelligent, costs ~$0.00001/query)
- Search: LLM-guided BFS with relation sequencing (intelligent path planning)

Expected performance:
- Highest accuracy (upper bound)
- Low cost (~$0.01 for 1000 queries)
- Best for 2-hop and 3-hop questions
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
from qa_system.relation_rankers.openai_ranker import OpenAIRelationRanker
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
        description='Variant 5: OpenAI Relation Ranking + LLM-Guided BFS (UPPER BOUND)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on 1-hop dataset with 10 questions (direct mode)
  python variant5_openai_guided.py --datasets 1-hop --limit 10

  # Run on full 1-hop dataset (no limit)
  python variant5_openai_guided.py --datasets 1-hop

  # Run on all datasets with 100 questions each (batch mode)
  python variant5_openai_guided.py --datasets 1-hop 2-hop 3-hop --limit 100 --mode batch

  # Run on full 2-hop dataset with batched LLM planning
  python variant5_openai_guided.py --datasets 2-hop --mode batch

Note:
  - Direct mode: LLM planning using parallel API calls (ThreadPoolExecutor)
    * Immediate results, no waiting
    * Regular API pricing

  - Batch mode: LLM planning using OpenAI Batch API (async queue)
    * 50% cheaper than direct mode
    * Requires polling/waiting (typically minutes to hours)
    * Recommended for large datasets (100+ questions)
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['direct', 'batch'],
        default='direct',
        help='Processing mode: "direct" for parallel API calls, "batch" for async Batch API (50%% cheaper)'
    )
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=60,
        help='Seconds between batch status checks (batch mode only, default: 60)'
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
    print(" VARIANT 5: OpenAI Relation Ranking + LLM-Guided BFS (UPPER BOUND)")
    print("="*80)
    print(f"  Mode: {args.mode}")
    if args.mode == 'batch':
        print(f"  Poll interval: {args.poll_interval}s")
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

    print("  Initializing OpenAIRelationRanker...")
    relation_ranker = OpenAIRelationRanker()

    print("  Initializing LLMGuidedBFS (search algorithm)...")
    search_algo = LLMGuidedBFS(graph=graph)

    # Always use LLM planning for Variant 5 (it's the upper bound)
    # Mode controls whether planning calls are batched or sequential:
    # - Batch mode: All questions batched per hop-count (cheaper, faster)
    # - Direct mode: Sequential LLM planning (one question at a time)
    use_llm_planning = True  # Variant 5 always uses GPT-4o planning
    batch_planning = (args.mode == 'batch')

    evaluator = Evaluator(
        entity_linker=entity_linker,
        relation_ranker=relation_ranker,
        search_algo=search_algo,
        variant_name=f"variant5_openai_guided_{args.mode}",
        use_llm_planning=use_llm_planning,
        batch_planning=batch_planning
    )

    print("  Components initialized.")
    if args.mode == 'batch':
        print("  LLM Planning: Enabled (BATCH mode - OpenAI Batch API, 50% cheaper)")
        print(f"  Polling: Every {args.poll_interval}s until complete\n")
    else:
        print("  LLM Planning: Enabled (DIRECT mode - parallel API calls, immediate results)\n")

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
        output_path = f"results/variant5_openai_guided_{args.mode}_{dataset_name}{limit_str}_{timestamp}.json"
        evaluator.incremental_save_path = output_path

        # Evaluate (results will be saved incrementally)
        # NOTE: Batching is automatic - Evaluator batches all LLM planning calls
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
        print(f"  Cost per query:     ${metrics['cost_per_query_usd']:.6f}")

    print("\n" + "="*80)
    print("  VARIANT 5 EVALUATION COMPLETE")
    print("="*80 + "\n")

    print("\nKey Features:")
    print("  - Entity linking: ExactMatcher (free)")
    print("  - Relation ranking: OpenAI embeddings + GPT-4o planning")
    print("  - LLM planning: Batched automatically (all questions per hop-count)")
    print("  - Search: LLM-guided BFS with intelligent path planning")

if __name__ == "__main__":
    main()
