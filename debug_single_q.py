#!/usr/bin/env python3
"""
Debug a single question in detail
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load env manually
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

def main():
    print("\n" + "="*80)
    print(" DEBUGGING SINGLE QUESTION")
    print("="*80 + "\n")

    # Load resources
    print("Loading resources...")
    graph = load_graph(Config.GRAPH_PATH)
    node2id = load_node2id(Config.NODE2ID_PATH)
    entity_emb, relation_emb, relation2id = load_transe_embeddings(
        Config.TRANSE_ENTITY_PATH,
        Config.TRANSE_RELATION_PATH,
        Config.TRANSE_METADATA_PATH
    )

    # Initialize components
    print("Initializing components...")
    entity_linker = ExactMatcher(node2id)
    relation_ranker = OpenAIRelationRanker()
    search_algo = BackwardSearchTransE(
        graph=graph,
        entity_embeddings=entity_emb,
        relation_embeddings=relation_emb,
        node2id=node2id,
        relation2id=relation2id
    )

    # Load question
    questions = load_qa_dataset(Config.QA_2HOP_TRAIN, hop_count=2, limit=10)
    question = questions[1]  # "which movies have the same director of [Just Cause]"

    print(f"\nQuestion: {question.text}")
    print(f"Ground truth: {question.ground_truth_answers}")
    print()

    # Entity linking
    start_nodes = entity_linker.extract_and_link(question.text)
    print(f"Start nodes: {start_nodes}")

    # Relation ranking - only use top 1
    relations = relation_ranker.rank_relations(question.text, top_k=1)
    print(f"\nTop 1 relation:")
    for rel, score in relations:
        print(f"  - {rel}: {score:.4f}")

    # Search
    print(f"\nSearching with max_hops=2...")
    result = search_algo.search(
        question=question,
        start_nodes=start_nodes,
        target_relations=relations,
        max_hops=2
    )

    print(f"\nNodes expanded: {result.nodes_expanded}")
    print(f"Search time: {result.search_time_ms:.2f} ms")
    print(f"Total answers found: {len(result.predicted_answers)}")
    print(f"Correct answer in results: {set(question.ground_truth_answers) & set(result.predicted_answers)}")

    # Show some reasoning paths
    print(f"\nFirst 10 reasoning paths:")
    for i, path in enumerate(result.reasoning_path[:10]):
        print(f"  {i+1}. {path}")

    # Manually check the correct path
    print("\n" + "="*80)
    print("MANUAL PATH CHECK")
    print("="*80)

    start = "Just Cause"
    expected = "The Mambo Kings"

    print(f"\nLooking for path: {start} --> Director --> {expected}")

    # Step 1: Find director of Just Cause
    if start in graph:
        print(f"\nStep 1: What edges does '{start}' have?")
        for succ in list(graph.successors(start))[:10]:
            edge = graph[start][succ]
            rel = edge.get('relation', '')
            print(f"  {start} --{rel}--> {succ}")

            if rel == 'directed_by':
                director = succ
                print(f"\n  ✓ Director found: {director}")

                # Step 2: Find other movies by this director
                print(f"\nStep 2: What other movies did {director} direct?")
                for pred in list(graph.predecessors(director))[:10]:
                    edge = graph[pred][director]
                    rel = edge.get('relation', '')
                    if rel == 'directed_by':
                        print(f"  {pred} --{rel}--> {director}")
                        if pred == expected:
                            print(f"    ✓✓ FOUND THE ANSWER!")

if __name__ == "__main__":
    main()
