"""
Generate training data for Variant 3: EdgeScorer Neural Model

This script:
1. Loads questions from train/dev sets (1-hop, 2-hop, 3-hop)
2. For each question, runs BFS to find shortest path
3. Creates training samples:
   - For each node on path, create samples for ALL outgoing edges
   - Positive label (1) for edge on shortest path
   - Negative label (0) for all other edges
4. Saves training data with metadata for embedding caching

Output: training_data_variant3.pkl containing list of samples
Each sample: {
    'question_id': str,
    'question_text': str,
    'node_id': str,
    'edge_relation': str,
    'edge_target': str,
    'hop': int (0-indexed),
    'label': int (0 or 1)
}
"""

import pickle
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from tqdm import tqdm
import json

from qa_system.config import Config
from qa_system.utils.data_loader import load_questions, Question
from qa_system.graph.graph_loader import load_graph


def bfs_shortest_path(
    graph,
    start_nodes: Set[str],
    target_nodes: Set[str],
    max_hops: int = 3
) -> Optional[List[Tuple[str, str, str]]]:
    """
    Find shortest path using BFS.

    Args:
        graph: NetworkX graph
        start_nodes: Set of starting entity IDs
        target_nodes: Set of target entity IDs
        max_hops: Maximum path length

    Returns:
        List of (node_id, relation, target_id) tuples representing the path,
        or None if no path found
    """
    # BFS with path tracking
    queue = deque()
    for start in start_nodes:
        if start in graph:
            queue.append((start, []))  # (current_node, path_so_far)

    visited = set(start_nodes)

    while queue:
        current, path = queue.popleft()

        # Check if we reached target
        if current in target_nodes:
            return path

        # Don't explore beyond max hops
        if len(path) >= max_hops:
            continue

        # Explore neighbors
        if current in graph:
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    # Get edge data (relation)
                    edge_data = graph[current][neighbor]
                    relation = edge_data.get('relation', 'unknown')

                    # Add to path
                    new_path = path + [(current, relation, neighbor)]
                    queue.append((neighbor, new_path))

    return None


def create_training_samples(
    question: Question,
    graph,
    shortest_path: List[Tuple[str, str, str]]
) -> List[Dict]:
    """
    Create training samples from shortest path.

    For each node in path:
    - Create positive sample for correct edge
    - Create negative samples for all other outgoing edges

    Args:
        question: Question object
        graph: NetworkX graph
        shortest_path: List of (node_id, relation, target_id) tuples

    Returns:
        List of training samples
    """
    samples = []

    for hop, (node_id, correct_relation, correct_target) in enumerate(shortest_path):
        # Get all outgoing edges from current node
        if node_id not in graph:
            continue

        neighbors = graph[node_id]

        # Create samples for ALL outgoing edges
        for target_id in neighbors:
            edge_data = graph[node_id][target_id]
            relation = edge_data.get('relation', 'unknown')

            # Check if this edge is on the correct path
            is_correct = (relation == correct_relation and target_id == correct_target)

            samples.append({
                'question_id': question.question_id,
                'question_text': question.text,
                'node_id': node_id,
                'edge_relation': relation,
                'edge_target': target_id,
                'hop': hop,
                'label': 1 if is_correct else 0
            })

    return samples


def main():
    print("=" * 80)
    print("Variant 3: Generate Training Data for EdgeScorer")
    print("=" * 80)

    # Load graph
    print("\n[1/5] Loading knowledge graph...")
    graph = load_graph()
    print(f"   Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Load questions from train sets
    print("\n[2/5] Loading training questions...")
    all_questions = []

    for hop_type in ['1-hop', '2-hop', '3-hop']:
        if hop_type == '1-hop':
            qa_file = Config.QA_1HOP_TRAIN
        elif hop_type == '2-hop':
            qa_file = Config.QA_2HOP_TRAIN
        else:
            qa_file = Config.QA_3HOP_TRAIN

        questions = load_questions(qa_file)
        print(f"   Loaded {len(questions)} questions from {hop_type}")
        all_questions.extend(questions)

    print(f"   Total training questions: {len(all_questions)}")

    # Find shortest paths and create samples
    print("\n[3/5] Finding shortest paths and creating samples...")
    all_samples = []
    no_path_count = 0
    path_length_stats = {1: 0, 2: 0, 3: 0}

    for question in tqdm(all_questions, desc="Processing questions"):
        # Find shortest path
        path = bfs_shortest_path(
            graph,
            question.topic_entity,
            question.answer_entities,
            max_hops=3
        )

        if path is None:
            no_path_count += 1
            continue

        # Track path length
        path_length = len(path)
        if path_length in path_length_stats:
            path_length_stats[path_length] += 1

        # Create training samples
        samples = create_training_samples(question, graph, path)
        all_samples.extend(samples)

    print(f"   Questions with shortest path: {len(all_questions) - no_path_count}")
    print(f"   Questions without path: {no_path_count}")
    print(f"   Path length distribution:")
    for length, count in sorted(path_length_stats.items()):
        print(f"      {length}-hop: {count}")

    # Analyze sample statistics
    print("\n[4/5] Analyzing training samples...")
    positive_samples = [s for s in all_samples if s['label'] == 1]
    negative_samples = [s for s in all_samples if s['label'] == 0]

    print(f"   Total samples: {len(all_samples)}")
    print(f"   Positive samples: {len(positive_samples)}")
    print(f"   Negative samples: {len(negative_samples)}")
    print(f"   Class ratio (neg:pos): {len(negative_samples) / len(positive_samples):.2f}:1")

    # Count unique questions for embedding caching
    unique_questions = set(s['question_text'] for s in all_samples)
    print(f"   Unique questions: {len(unique_questions)}")

    # Save training data
    print("\n[5/5] Saving training data...")
    output_path = Config.BASE_DIR / "data" / "variant3_training_data.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(all_samples, f)

    print(f"   Training data saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Save metadata
    metadata = {
        'total_samples': len(all_samples),
        'positive_samples': len(positive_samples),
        'negative_samples': len(negative_samples),
        'class_ratio': len(negative_samples) / len(positive_samples),
        'unique_questions': len(unique_questions),
        'questions_with_path': len(all_questions) - no_path_count,
        'questions_without_path': no_path_count,
        'path_length_distribution': path_length_stats
    }

    metadata_path = Config.BASE_DIR / "data" / "variant3_training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   Metadata saved to: {metadata_path}")

    print("\n" + "=" * 80)
    print("Training data generation complete!")
    print("=" * 80)
    print(f"\nNext step: Run variant3_cache_embeddings.py to cache OpenAI embeddings")


if __name__ == "__main__":
    main()
