#!/usr/bin/env python3
"""
Debug Variant 5 on 2-hop and 3-hop questions
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set OpenAI key from environment if needed
if 'OPENAI_API_KEY' not in os.environ:
    # Try to load from .env file manually
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

from qa_system.config import Config
from qa_system.entity_linkers.exact_matcher import ExactMatcher
from qa_system.relation_rankers.openai_ranker import OpenAIRelationRanker
from qa_system.search_algorithms.backward_search_transe import BackwardSearchTransE
from qa_system.utils.loader import (
    load_graph,
    load_node2id,
    load_transe_embeddings,
    load_qa_dataset
)

def test_hop_type(hop_type, dataset_path, max_hops, num_questions=5):
    """Test a specific hop type with detailed output"""
    print("\n" + "="*80)
    print(f" DEBUGGING {hop_type.upper()} QUESTIONS (first {num_questions})")
    print("="*80 + "\n")

    # Load resources
    print("[1/4] Loading resources...")
    graph = load_graph(Config.GRAPH_PATH)
    node2id = load_node2id(Config.NODE2ID_PATH)
    entity_emb, relation_emb, relation2id = load_transe_embeddings(
        Config.TRANSE_ENTITY_PATH,
        Config.TRANSE_RELATION_PATH,
        Config.TRANSE_METADATA_PATH
    )

    # Initialize components
    print("\n[2/4] Initializing components...")
    entity_linker = ExactMatcher(node2id)
    relation_ranker = OpenAIRelationRanker()
    search_algo = BackwardSearchTransE(
        graph=graph,
        entity_embeddings=entity_emb,
        relation_embeddings=relation_emb,
        node2id=node2id,
        relation2id=relation2id
    )

    # Load questions
    print(f"\n[3/4] Loading {num_questions} {hop_type} questions...")
    questions = load_qa_dataset(dataset_path, hop_count=max_hops, limit=num_questions)

    # Test each question
    print(f"\n[4/4] Testing questions...")
    print("="*80)

    correct = 0
    for i, question in enumerate(questions):
        print(f"\n{'='*80}")
        print(f"QUESTION {i+1}/{len(questions)}")
        print(f"{'='*80}")
        print(f"Text: {question.text}")
        print(f"Ground truth: {question.ground_truth_answers}")
        print(f"Expected hops: {max_hops}")

        # Entity linking
        start_nodes = entity_linker.extract_and_link(question.text)
        print(f"\n[Entity Linking] Start nodes: {start_nodes}")

        if not start_nodes:
            print("  WARNING: No start nodes found!")
            continue

        # Relation ranking - try top-1 instead of all 9
        TOP_K = 1  # Using top-1 relation for better precision
        relations = relation_ranker.rank_relations(question.text, top_k=TOP_K)
        print(f"\n[Relation Ranking] Top {TOP_K} relation(s):")
        for rel, score in relations:
            print(f"  - {rel}: {score:.4f}")

        # Search
        print(f"\n[Search] Running backward search with max_hops={max_hops}...")
        result = search_algo.search(
            question=question,
            start_nodes=start_nodes,
            target_relations=relations,
            max_hops=max_hops
        )

        print(f"\n[Results]")
        print(f"  Predicted answers: {result.predicted_answers}")
        print(f"  Correct: {result.is_correct}")
        print(f"  Nodes expanded: {result.nodes_expanded}")
        print(f"  Search time: {result.search_time_ms:.2f} ms")

        if result.reasoning_path:
            print(f"\n[Reasoning Paths]")
            for path in result.reasoning_path[:5]:  # Show first 5 paths
                print(f"  {path}")

        if result.metadata:
            print(f"\n[Metadata] {result.metadata}")

        if result.is_correct:
            correct += 1
            print("\n  ✓ CORRECT!")
        else:
            print("\n  ✗ INCORRECT")

    print("\n" + "="*80)
    print(f"ACCURACY: {correct}/{len(questions)} = {correct/len(questions):.1%}")
    print("="*80 + "\n")

    return correct, len(questions)

def main():
    print("\n" + "="*80)
    print(" VARIANT 5 DEBUGGING - SUBSAMPLE TEST")
    print("="*80)

    results = {}

    # Test 2-hop
    correct_2, total_2 = test_hop_type(
        hop_type="2-hop",
        dataset_path=Config.QA_2HOP_TRAIN,
        max_hops=2,
        num_questions=5
    )
    results['2-hop'] = (correct_2, total_2)

    # Test 3-hop
    correct_3, total_3 = test_hop_type(
        hop_type="3-hop",
        dataset_path=Config.QA_3HOP_TRAIN,
        max_hops=3,
        num_questions=5
    )
    results['3-hop'] = (correct_3, total_3)

    # Summary
    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80)
    for hop_type, (correct, total) in results.items():
        acc = correct/total if total > 0 else 0
        print(f"{hop_type}: {correct}/{total} = {acc:.1%}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
