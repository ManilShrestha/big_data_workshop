"""
Visualize path exploration for a specific question.

Shows:
- The path taken (explored)
- Alternative paths NOT taken at each hop
- Edge scores for both explored and unexplored paths

This helps create diagrams showing why the model chose certain paths.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import pickle
import networkx as nx
from collections import defaultdict
from qa_system.config import Config
from qa_system.utils.loader import load_graph as load_graph_utils
from qa_system.entity_linkers.exact_matcher import ExactMatcher
from variant3.eval_qa.edge_scorer_ranker import EdgeScorerRelationRanker


def load_graph():
    """Load the knowledge graph"""
    print("[1/4] Loading knowledge graph...")
    graph_path = Config.GRAPH_PATH
    graph = load_graph_utils(str(graph_path))
    print(f"   Loaded graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


def load_model():
    """Load the trained EdgeScorer model"""
    print("[2/4] Loading EdgeScorer model...")
    model_path = Config.BASE_DIR / "models" / "variant3_edge_scorer_dual_best.pt"
    ranker = EdgeScorerRelationRanker(str(model_path))
    print(f"   Model loaded successfully")
    return ranker


def parse_path_string(path_str):
    """
    Parse path string like:
    "Inception --written_by--> Christopher Nolan --written_by--> Memento --has_genre--> Thriller"

    Returns list of (node, relation, next_node) tuples
    """
    parts = path_str.split(" --")

    hops = []
    current_node = parts[0]

    for i in range(1, len(parts)):
        # Format: "relation--> next_node"
        rel_and_node = parts[i].split("--> ")
        if len(rel_and_node) == 2:
            relation = rel_and_node[0]
            next_node = rel_and_node[1]
            hops.append((current_node, relation, next_node))
            current_node = next_node

    return hops


def get_all_edges_from_node(graph, node_id):
    """Get all outgoing and incoming edges from a node"""
    edges = []

    # Check if node exists
    if node_id not in graph:
        return edges

    # Outgoing edges
    for _, target, data in graph.out_edges(node_id, data=True):
        relation = data.get('relation', 'unknown')
        edges.append((node_id, relation, target))

    # Incoming edges (reversed)
    for source, _, data in graph.in_edges(node_id, data=True):
        relation = data.get('relation', 'unknown')
        edges.append((node_id, relation + "_reversed", source))

    return edges


def explore_question(question, path_taken_str, graph, ranker, max_alternatives=5):
    """
    Explore the graph for a question and show paths taken vs not taken.

    Args:
        question: Question text
        path_taken_str: String representation of the path taken
        graph: Knowledge graph
        ranker: EdgeScorer ranker
        max_alternatives: Max number of alternative paths to show per hop

    Returns:
        Dict with exploration data
    """
    print(f"\n[3/4] Analyzing question: {question}")
    print(f"   Path taken: {path_taken_str}")

    # Parse the path taken
    path_hops = parse_path_string(path_taken_str)
    print(f"   Number of hops: {len(path_hops)}")

    # Extract start node
    start_node = path_hops[0][0]
    question_hop_count = len(path_hops)

    exploration_data = {
        'question': question,
        'path_taken': path_taken_str,
        'start_node': start_node,
        'hop_count': question_hop_count,
        'hops': []
    }

    # Explore each hop
    for hop_idx, (current_node, taken_relation, taken_target) in enumerate(path_hops):
        print(f"\n   Hop {hop_idx + 1}/{question_hop_count}:")
        print(f"      Current node: {current_node}")
        print(f"      Taken: ({current_node}, {taken_relation}, {taken_target})")

        # Get all possible edges from current node
        all_edges = get_all_edges_from_node(graph, current_node)
        print(f"      Total edges available: {len(all_edges)}")

        if len(all_edges) == 0:
            print(f"      WARNING: No edges found from {current_node}")
            continue

        # Score all edges
        scores = ranker.score_edges_batch(
            question=question,
            edges=all_edges,
            hop=hop_idx,  # 0-indexed
            question_hop_count=question_hop_count
        )

        # Combine edges with scores
        scored_edges = list(zip(all_edges, scores))
        scored_edges.sort(key=lambda x: x[1], reverse=True)

        # Find the taken edge
        taken_edge = (current_node, taken_relation, taken_target)
        taken_score = None
        taken_rank = None

        for rank, ((node, rel, tgt), score) in enumerate(scored_edges, 1):
            if node == current_node and rel == taken_relation and tgt == taken_target:
                taken_score = score
                taken_rank = rank
                break

        if taken_score is not None:
            print(f"      Taken edge score: {taken_score:.4f} (rank {taken_rank}/{len(all_edges)})")
        else:
            print(f"      WARNING: Taken edge not found in scored edges!")

        # Get top alternatives (excluding the taken edge)
        alternatives = []
        for (node, rel, tgt), score in scored_edges:
            if (node, rel, tgt) != taken_edge:
                alternatives.append({
                    'node': node,
                    'relation': rel,
                    'target': tgt,
                    'score': score
                })
                if len(alternatives) >= max_alternatives:
                    break

        print(f"      Top {len(alternatives)} alternatives not taken:")
        for i, alt in enumerate(alternatives, 1):
            print(f"         {i}. ({alt['node']}, {alt['relation']}, {alt['target']}) - score: {alt['score']:.4f}")

        # Store hop data
        hop_data = {
            'hop_number': hop_idx + 1,
            'current_node': current_node,
            'taken_edge': {
                'node': current_node,
                'relation': taken_relation,
                'target': taken_target,
                'score': taken_score,
                'rank': taken_rank
            },
            'total_edges_available': len(all_edges),
            'alternatives_not_taken': alternatives,
            'top_5_edges': [
                {
                    'node': node,
                    'relation': rel,
                    'target': tgt,
                    'score': score,
                    'rank': rank
                }
                for rank, ((node, rel, tgt), score) in enumerate(scored_edges[:5], 1)
            ]
        }

        exploration_data['hops'].append(hop_data)

    return exploration_data


def main():
    # Load resources
    graph = load_graph()
    ranker = load_model()

    # Example question from the user's selection
    question = "what genres are the movies written by [Inception] writers"

    # One of the paths taken (from user's selection)
    # Note: The second hop uses written_by in REVERSE (from person to movie they wrote)
    # In the graph: Memento --written_by--> Christopher Nolan
    # In the path: Christopher Nolan --written_by_reversed--> Memento
    path_taken = "Inception --written_by--> Christopher Nolan --written_by_reversed--> Memento --has_genre--> Thriller"

    # Explore the question
    exploration_data = explore_question(
        question=question,
        path_taken_str=path_taken,
        graph=graph,
        ranker=ranker,
        max_alternatives=10  # Show top 10 alternatives per hop
    )

    # Save results
    print("\n[4/4] Saving exploration data...")
    output_file = Config.BASE_DIR / "results" / "path_exploration_visualization.json"
    with open(output_file, 'w') as f:
        json.dump(exploration_data, f, indent=2)

    print(f"   Saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("EXPLORATION SUMMARY")
    print("=" * 80)
    print(f"Question: {exploration_data['question']}")
    print(f"Path taken: {exploration_data['path_taken']}")
    print(f"\nHop-by-hop breakdown:")

    for hop in exploration_data['hops']:
        print(f"\n  Hop {hop['hop_number']}:")
        print(f"    From: {hop['current_node']}")
        print(f"    Took: ({hop['taken_edge']['relation']}, {hop['taken_edge']['target']})")
        print(f"    Score: {hop['taken_edge']['score']:.4f} (rank {hop['taken_edge']['rank']}/{hop['total_edges_available']})")
        print(f"    Alternatives ({len(hop['alternatives_not_taken'])} shown):")
        for i, alt in enumerate(hop['alternatives_not_taken'][:5], 1):
            print(f"      {i}. ({alt['relation']}, {alt['target']}) - {alt['score']:.4f}")

    print("\n" + "=" * 80)
    print(f"Full data saved to: {output_file}")
    print("=" * 80)

    # Create a simple text visualization
    print("\n" + "=" * 80)
    print("TEXT VISUALIZATION FOR DIAGRAM")
    print("=" * 80)

    for hop in exploration_data['hops']:
        print(f"\n{'=' * 80}")
        print(f"HOP {hop['hop_number']}: From '{hop['current_node']}'")
        print(f"{'=' * 80}")

        print(f"\n PATH TAKEN (Score: {hop['taken_edge']['score']:.4f}):")
        print(f"  └─ {hop['taken_edge']['relation']} → {hop['taken_edge']['target']}")

        print(f"\n PATHS NOT TAKEN (Top alternatives):")
        for i, alt in enumerate(hop['alternatives_not_taken'][:5], 1):
            print(f"  {i}. {alt['relation']} → {alt['target']} (Score: {alt['score']:.4f})")

        print(f"\n  Total edges available: {hop['total_edges_available']}")
        print(f"  Taken edge rank: {hop['taken_edge']['rank']}/{hop['total_edges_available']}")


if __name__ == "__main__":
    main()
