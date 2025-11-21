#!/usr/bin/env python3
"""
Variant 3: EdgeScorer-Guided BFS (Learned Neural Approach)

This variant uses a trained neural EdgeScorer model to guide search:
- Entity Linking: Exact matching (free)
- Edge Scoring: Trained EdgeScorerDual model (context-aware, learned from data)
- Search: Beam search guided by edge scores

Expected performance:
- Competitive accuracy with Variant 5 (target: within 5%)
- Very fast (local GPU inference, no API latency)
- Very cheap (only question embedding, ~$0.00002/query)
- Best for cost-sensitive deployments
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qa_system.config import Config
from qa_system.entity_linkers.exact_matcher import ExactMatcher
from qa_system.utils.loader import (
    load_graph,
    load_node2id,
    load_qa_dataset
)
from qa_system.utils.evaluator import Evaluator
from variant3.eval_qa.edge_scorer_ranker import EdgeScorerRelationRanker
from variant3.eval_qa.edge_scorer_bfs import EdgeScorerBFS


def main():
    import argparse
    from datetime import datetime

    # =========================================================================
    # Parse command-line arguments
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='Variant 3: EdgeScorer-Guided BFS (Learned Neural Approach)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on 10 questions from 1-hop
  python variant3/eval_qa/variant3_qa_evaluator.py --datasets 1-hop --limit 10

  # Run on full 1-hop dataset
  python variant3/eval_qa/variant3_qa_evaluator.py --datasets 1-hop

  # Run on all datasets with 100 questions each
  python variant3/eval_qa/variant3_qa_evaluator.py --datasets 1-hop 2-hop 3-hop --limit 100

  # Full evaluation with custom beam width
  python variant3/eval_qa/variant3_qa_evaluator.py --datasets 1-hop 2-hop 3-hop --beam-width 10

  # Use CPU instead of GPU
  python variant3/eval_qa/variant3_qa_evaluator.py --datasets 1-hop --device cpu

Note:
  - EdgeScorer uses local GPU inference (very fast)
  - Only costs ~$0.00002 per question (for question embedding)
  - Beam width controls exploration vs speed tradeoff
  - Score threshold filters low-confidence edges
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
        choices=['1-hop', '2-hop', '3-hop', '3-hop-train'],
        default=['1-hop'],
        help='Datasets to evaluate (default: 1-hop)'
    )
    parser.add_argument(
        '--top-k-relations',
        type=int,
        default=3,
        help='Number of top relations to explore per hop (default: 3)'
    )
    parser.add_argument(
        '--max-nodes-per-relation',
        type=int,
        default=30,
        help='Maximum nodes to expand per relation (default: 30)'
    )
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.0,
        help='Minimum edge score threshold (default: 0.0, range: 0-1)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device for model inference (default: auto)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model checkpoint (default: models/variant3_edge_scorer_dual_best.pt)'
    )
    parser.add_argument(
        '--enable-hybrid',
        action='store_true',
        default=False,
        help='Enable hybrid text+model scoring (default: False, backward compatible)'
    )
    parser.add_argument(
        '--hybrid-alphas',
        type=str,
        default='1:0.7,2:0.5,3:0.3',
        help='Hop-specific alpha weights for hybrid scoring (format: "1:0.7,2:0.5,3:0.3")'
    )

    args = parser.parse_args()

    # Generate timestamp for unique output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse hybrid alphas if hybrid scoring is enabled
    hybrid_alphas = None
    if args.enable_hybrid:
        try:
            hybrid_alphas = {
                int(k): float(v)
                for k, v in (pair.split(':') for pair in args.hybrid_alphas.split(','))
            }
        except Exception as e:
            print(f"\n ERROR: Invalid hybrid-alphas format: {args.hybrid_alphas}")
            print(f"   Expected format: '1:0.7,2:0.5,3:0.3'")
            print(f"   Error: {e}")
            sys.exit(1)

    # Set model path
    if args.model_path is None:
        model_path = Config.BASE_DIR / "models" / "variant3_edge_scorer_dual_best.pt"
    else:
        model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"\n ERROR: Model checkpoint not found: {model_path}")
        print(f"\nPlease train the model first using:")
        print(f"  python variant3/variant3_train_edge_scorer_dual.py")
        sys.exit(1)

    print("\n" + "="*80)
    print(" VARIANT 3: EdgeScorer-Guided BFS (Global Path Selection)")
    print("="*80)
    print(f"  Model: {model_path.name}")
    print(f"  Device: {args.device}")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  Limit per dataset: {args.limit if args.limit else 'Full dataset'}")
    print(f"  Top-K relations: {args.top_k_relations}")
    print(f"  Max nodes/relation: {args.max_nodes_per_relation}")
    print(f"  Score threshold: {args.score_threshold}")
    print(f"  Hybrid scoring: {'ENABLED' if args.enable_hybrid else 'DISABLED'}")
    if args.enable_hybrid:
        print(f"  Hybrid alphas: {hybrid_alphas}")
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

    print("  Initializing EdgeScorerRelationRanker...")
    device = None if args.device == 'auto' else args.device
    edge_scorer = EdgeScorerRelationRanker(
        model_checkpoint_path=str(model_path),
        device=device
    )

    print("  Initializing EdgeScorerBFS (search algorithm)...")
    search_algo = EdgeScorerBFS(
        graph=graph,
        edge_scorer=edge_scorer,
        top_k_relations=args.top_k_relations,
        max_nodes_per_relation=args.max_nodes_per_relation,
        score_threshold=args.score_threshold,
        enable_hybrid=args.enable_hybrid,
        hybrid_alphas=hybrid_alphas
    )

    print("\n  Component initialization complete!")
    print()

    # =========================================================================
    # Run Evaluation on Each Dataset
    # =========================================================================
    print("[3/4] Running evaluations...")
    print("-" * 80)
    print()

    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*80}")
        print(f" Evaluating on {dataset_name} dataset")
        print(f"{'='*80}\n")

        # Load dataset
        print(f"  Loading {dataset_name} dataset...")
        hop_count = int(dataset_name.split('-')[0])
        if dataset_name == '1-hop':
            questions = load_qa_dataset(Config.QA_1HOP_TEST, hop_count=hop_count)
        elif dataset_name == '2-hop':
            questions = load_qa_dataset(Config.QA_2HOP_TEST, hop_count=hop_count)
        elif dataset_name == '3-hop':
            questions = load_qa_dataset(Config.QA_3HOP_TEST, hop_count=hop_count)
        elif dataset_name == '3-hop-train':
            questions = load_qa_dataset(Config.QA_3HOP_TRAIN, hop_count=hop_count)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Limit questions if specified
        if args.limit:
            questions = questions[:args.limit]
            print(f"  Limited to {len(questions)} questions")

        print(f"  Questions to evaluate: {len(questions)}\n")

        # Create evaluator
        incremental_path = Config.BASE_DIR / "results" / f"variant3_edge_scorer_{dataset_name}_{timestamp}.json"
        evaluator = Evaluator(
            entity_linker=entity_linker,
            relation_ranker=edge_scorer,  # For cost tracking
            search_algo=search_algo,
            variant_name=f"variant3_edge_scorer_{dataset_name}",
            use_llm_planning=False,  # EdgeScorer doesn't use LLM planning
            incremental_save_path=str(incremental_path)
        )

        # Run evaluation
        results = evaluator.evaluate(
            questions=questions,
            top_k_relations=None,  # Not used by EdgeScorer
            dataset_name=dataset_name
        )

        all_results[dataset_name] = results

        # Print summary for this dataset
        print(f"\n{'='*80}")
        print(f" {dataset_name} Results Summary")
        print(f"{'='*80}")
        print(f"  Results: {results}")
        print(f"{'='*80}\n")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*80)
    print(" FINAL SUMMARY - Variant 3")
    print("="*80)

    for dataset_name, results in all_results.items():
        metrics = results['metrics']
        print(f"\n{dataset_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1: {metrics['micro_f1_score']:.4f}")
        print(f"  Avg Cost: ${metrics['cost_per_query_usd']:.6f}")
        print(f"  Avg Time: {metrics['avg_search_time_ms']:.2f} ms")

    # Overall statistics
    total_questions = sum(r['metrics']['total_questions'] for r in all_results.values())
    total_cost = sum(r['metrics']['total_cost_usd'] for r in all_results.values())
    avg_accuracy = sum(r['metrics']['accuracy'] * r['metrics']['total_questions'] for r in all_results.values()) / total_questions
    avg_f1 = sum(r['metrics']['micro_f1_score'] * r['metrics']['total_questions'] for r in all_results.values()) / total_questions

    print(f"\nOverall:")
    print(f"  Total Questions: {total_questions}")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Weighted Avg Accuracy: {avg_accuracy:.4f}")
    print(f"  Weighted Avg F1: {avg_f1:.4f}")

    print("\n" + "="*80)
    print(" Evaluation complete!")
    print("="*80)

    # Print results file locations
    print(f"\nResults saved to:")
    for dataset_name in all_results.keys():
        result_file = Config.BASE_DIR / "results" / f"variant3_edge_scorer_{dataset_name}_{timestamp}.json"
        print(f"  - {result_file}")

    print()


if __name__ == "__main__":
    main()
