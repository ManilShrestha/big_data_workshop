"""
Generate training data for Variant 3: EdgeScorer Neural Model

This script:
1. Loads questions from train/dev sets (1-hop, 2-hop, 3-hop)
2. For each question, runs BFS to find shortest path
3. Creates training samples:
   - For each node on path, create samples for ALL edges (outgoing + incoming)
   - Positive label (1) for edges on shortest path
   - Negative label (0) for all other edges
4. Saves training data with metadata for embedding caching

Note: Graph is treated as non-directional (can traverse edges both ways)
      but relation names stay the same (no "reverse_" prefix)

Output: training_data_variant3.pkl containing list of samples
Each sample: {
    'question_id': str,
    'question_text': str,
    'node_id': str,
    'edge_relation': str,
    'edge_target': str,
    'hop': int (1, 2, or 3),
    'label': int (0 or 1)
}
"""

import pickle
import re
from collections import deque, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count
from functools import partial
import logging
from datetime import datetime

from qa_system.config import Config
from qa_system.utils.loader import load_qa_dataset, load_graph
from qa_system.core.question import Question


# Setup logging
def setup_logging():
    """Setup logging to both file and console."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"variant3_training_data_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return log_file


def build_entity_lookup(graph) -> Dict[str, str]:
    """Build case-insensitive entity lookup dictionary once."""
    lookup = {}
    for node in graph.nodes():
        lookup[node] = node  # Exact match
        lookup[node.lower()] = node  # Case-insensitive match
    return lookup


def build_edge_cache(graph) -> Tuple[Dict, Dict]:
    """Pre-cache all edges for faster access during sample creation."""
    outgoing_edges = defaultdict(list)
    incoming_edges = defaultdict(list)

    for node in graph.nodes():
        # Cache outgoing edges
        for _, target_id, _, edge_data in graph.out_edges(node, keys=True, data=True):
            relation = edge_data.get('relation', 'unknown')
            outgoing_edges[node].append((relation, target_id))

        # Cache incoming edges
        for source, _, _, edge_data in graph.in_edges(node, keys=True, data=True):
            relation = edge_data.get('relation', 'unknown')
            incoming_edges[node].append((relation, source))

    return dict(outgoing_edges), dict(incoming_edges)


def extract_topic_entities(question_text: str, entity_lookup: Dict[str, str]) -> Set[str]:
    """Extract topic entities from [brackets] in question text."""
    bracket_matches = re.findall(r'\[([^\]]+)\]', question_text)
    entities = set()

    for entity_str in bracket_matches:
        # Try exact match first, then case-insensitive
        if entity_str in entity_lookup:
            entities.add(entity_lookup[entity_str])
        elif entity_str.lower() in entity_lookup:
            entities.add(entity_lookup[entity_str.lower()])

    return entities


def extract_answer_entities(answer_list: List[str], entity_lookup: Dict[str, str]) -> Set[str]:
    """Extract answer entities that exist in the graph."""
    entities = set()

    for answer_str in answer_list:
        # Try exact match first, then case-insensitive
        if answer_str in entity_lookup:
            entities.add(entity_lookup[answer_str])
        elif answer_str.lower() in entity_lookup:
            entities.add(entity_lookup[answer_str.lower()])

    return entities


def bfs_exact_hop_path(
    graph,
    start_nodes: Set[str],
    target_nodes: Set[str],
    target_hops: int = 3
) -> Optional[List[Tuple[str, str, str]]]:
    """
    Find a path of EXACTLY target_hops length using BFS.

    This function finds paths of the exact hop count, even if shorter paths exist.
    Uses BFS to explore all paths level by level.

    Returns:
        List of (node_id, relation, target_id) tuples representing the path,
        or None if no path of exact length found
    """
    # Track paths explicitly: (current_node, hop_count, path_so_far)
    queue = deque()

    for start in start_nodes:
        if start in graph:
            queue.append((start, 0, []))

    # Track visited as (node, hop) tuples to allow revisiting at different hops
    visited = set()
    for start in start_nodes:
        visited.add((start, 0))

    while queue:
        current, hop, path = queue.popleft()

        # Check if we reached target at EXACT hop count
        if hop == target_hops and current in target_nodes:
            return path  # Found exact-hop path!

        # Don't explore beyond target hops
        if hop >= target_hops:
            continue

        # Explore in BOTH directions (graph is non-directional)
        # 1. Outgoing edges
        for _, neighbor, _, edge_data in graph.out_edges(current, keys=True, data=True):
            # Allow visiting same node at different hop counts
            if (neighbor, hop + 1) not in visited:
                visited.add((neighbor, hop + 1))
                relation = edge_data.get('relation', 'unknown')
                new_path = path + [(current, relation, neighbor)]
                queue.append((neighbor, hop + 1, new_path))

        # 2. Incoming edges (traverse backwards)
        for source, _, _, edge_data in graph.in_edges(current, keys=True, data=True):
            if (source, hop + 1) not in visited:
                visited.add((source, hop + 1))
                relation = edge_data.get('relation', 'unknown')
                new_path = path + [(current, relation, source)]
                queue.append((source, hop + 1, new_path))

    return None


def create_training_samples(
    question: Question,
    shortest_path: List[Tuple[str, str, str]],
    answer_entities: Set[str],
    outgoing_edges: Dict,
    incoming_edges: Dict
) -> List[Dict]:
    """
    Create training samples from shortest path using cached edges.

    For each node in path:
    - Create samples for ALL edges (outgoing + incoming)
    - Label as positive (1) if on the correct path OR leads to answer
    - Label as negative (0) otherwise

    Args:
        question: Question object
        shortest_path: List of (node_id, relation, target_id) tuples
        answer_entities: Set of answer entity IDs
        outgoing_edges: Pre-cached outgoing edges
        incoming_edges: Pre-cached incoming edges

    Returns:
        List of training samples
    """
    samples = []
    question_id = question.question_id
    question_text = question.text
    hop_count = question.hop_count

    for hop_idx, (node_id, correct_relation, correct_target) in enumerate(shortest_path):
        hop = hop_idx + 1  # Hops are 1-indexed

        # Create samples for ALL outgoing edges using cached data
        if node_id in outgoing_edges:
            is_final_hop = (hop == hop_count)
            for relation, target_id in outgoing_edges[node_id]:
                # Determine if this edge is correct
                if is_final_hop:
                    is_correct = (target_id in answer_entities)
                else:
                    is_correct = (relation == correct_relation and target_id == correct_target)

                samples.append({
                    'question_id': question_id,
                    'question_text': question_text,
                    'node_id': node_id,
                    'edge_relation': relation,
                    'edge_target': target_id,
                    'hop': hop,
                    'label': 1 if is_correct else 0
                })

        # Create samples for ALL incoming edges using cached data
        if node_id in incoming_edges:
            is_final_hop = (hop == hop_count)
            for relation, source in incoming_edges[node_id]:
                # Determine if this edge is correct
                if is_final_hop:
                    is_correct = (source in answer_entities)
                else:
                    is_correct = (relation == correct_relation and source == correct_target)

                samples.append({
                    'question_id': question_id,
                    'question_text': question_text,
                    'node_id': node_id,
                    'edge_relation': relation,
                    'edge_target': source,  # When traversing backwards, source becomes the target
                    'hop': hop,
                    'label': 1 if is_correct else 0
                })

    return samples


def process_question_batch(
    questions: List[Question],
    graph_path: str,
    entity_lookup: Dict[str, str],
    outgoing_edges: Dict,
    incoming_edges: Dict
) -> Tuple[List[Dict], int, Dict[int, int]]:
    """
    Process a batch of questions in parallel.

    Returns:
        Tuple of (samples, no_path_count, path_length_stats)
    """
    # Load graph in worker process (each process needs its own copy)
    graph = load_graph(graph_path)

    samples = []
    no_path_count = 0
    path_length_stats = {1: 0, 2: 0, 3: 0}

    for question in questions:
        # Extract topic and answer entities
        topic_entities = extract_topic_entities(question.text, entity_lookup)
        answer_entities = extract_answer_entities(question.ground_truth_answers, entity_lookup)

        # Skip if no entities found
        if not topic_entities or not answer_entities:
            no_path_count += 1
            continue

        # Find path of EXACT hop count (not shortest path!)
        # This allows using questions with shortcuts if they also have N-hop paths
        path = bfs_exact_hop_path(
            graph,
            topic_entities,
            answer_entities,
            target_hops=question.hop_count
        )

        if path is None:
            no_path_count += 1
            continue

        # Track path length (should always match question.hop_count now)
        path_length = len(path)
        if path_length in path_length_stats:
            path_length_stats[path_length] += 1

        # Create training samples
        question_samples = create_training_samples(
            question, path, answer_entities, outgoing_edges, incoming_edges
        )
        samples.extend(question_samples)

    return samples, no_path_count, path_length_stats


def main():
    # Setup logging
    log_file = setup_logging()

    logging.info("=" * 80)
    logging.info("Variant 3: Generate Training Data for EdgeScorer (OPTIMIZED)")
    logging.info("=" * 80)
    logging.info(f"Log file: {log_file}")

    # Load graph
    logging.info("\n[1/6] Loading knowledge graph...")
    graph = load_graph(str(Config.GRAPH_PATH))
    logging.info(f"   Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Build entity lookup and edge cache
    logging.info("\n[2/6] Building lookup indices...")
    logging.info("   Building entity lookup...")
    entity_lookup = build_entity_lookup(graph)
    logging.info(f"   Entity lookup built: {len(entity_lookup)} entries")

    logging.info("   Building edge cache...")
    outgoing_edges, incoming_edges = build_edge_cache(graph)
    logging.info(f"   Edge cache built: {len(outgoing_edges)} nodes with outgoing edges")

    # Load questions from train sets
    logging.info("\n[3/6] Loading training questions...")
    all_questions = []

    for hop_count in [1, 2, 3]:
        if hop_count == 1:
            qa_file = Config.QA_1HOP_TRAIN
        elif hop_count == 2:
            qa_file = Config.QA_2HOP_TRAIN
        else:
            qa_file = Config.QA_3HOP_TRAIN

        questions = load_qa_dataset(str(qa_file), hop_count=hop_count)
        logging.info(f"   Loaded {len(questions)} questions from {hop_count}-hop")
        all_questions.extend(questions)

    logging.info(f"   Total training questions: {len(all_questions)}")

    # Determine number of processes and batch size
    num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    batch_size = max(1, len(all_questions) // (num_processes * 4))  # 4 batches per process
    logging.info(f"\n[4/6] Processing questions in parallel...")
    logging.info(f"   Using {num_processes} processes")
    logging.info(f"   Batch size: {batch_size} questions per batch")

    # Split questions into batches
    batches = [all_questions[i:i + batch_size] for i in range(0, len(all_questions), batch_size)]
    logging.info(f"   Total batches: {len(batches)}")

    # Process batches in parallel
    all_samples = []
    no_path_count = 0
    path_length_stats = {1: 0, 2: 0, 3: 0}

    worker_func = partial(
        process_question_batch,
        graph_path=str(Config.GRAPH_PATH),
        entity_lookup=entity_lookup,
        outgoing_edges=outgoing_edges,
        incoming_edges=incoming_edges
    )

    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for progress tracking
        results = list(tqdm(
            pool.imap_unordered(worker_func, batches),
            total=len(batches),
            desc="Processing batches"
        ))

    # Aggregate results
    logging.info("   Aggregating results from all batches...")
    for samples, batch_no_path, batch_stats in results:
        all_samples.extend(samples)
        no_path_count += batch_no_path
        for length, count in batch_stats.items():
            if length in path_length_stats:
                path_length_stats[length] += count

    logging.info(f"   Questions with shortest path: {len(all_questions) - no_path_count}")
    logging.info(f"   Questions without path: {no_path_count}")
    logging.info(f"   Path length distribution:")
    for length, count in sorted(path_length_stats.items()):
        logging.info(f"      {length}-hop: {count}")

    # Analyze sample statistics
    logging.info("\n[5/6] Analyzing training samples...")
    positive_samples = [s for s in all_samples if s['label'] == 1]
    negative_samples = [s for s in all_samples if s['label'] == 0]

    logging.info(f"   Total samples: {len(all_samples)}")
    logging.info(f"   Positive samples: {len(positive_samples)}")
    logging.info(f"   Negative samples: {len(negative_samples)}")
    if len(positive_samples) > 0:
        logging.info(f"   Class ratio (neg:pos): {len(negative_samples) / len(positive_samples):.2f}:1")

    # Count unique questions for embedding caching
    unique_questions = set(s['question_text'] for s in all_samples)
    logging.info(f"   Unique questions: {len(unique_questions)}")

    # Save training data
    logging.info("\n[6/6] Saving training data...")
    output_path = Config.BASE_DIR / "data" / "variant3_training_data.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(all_samples, f)

    logging.info(f"   Training data saved to: {output_path}")
    logging.info(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Save metadata
    metadata = {
        'total_samples': len(all_samples),
        'positive_samples': len(positive_samples),
        'negative_samples': len(negative_samples),
        'class_ratio': len(negative_samples) / len(positive_samples) if len(positive_samples) > 0 else 0,
        'unique_questions': len(unique_questions),
        'questions_with_path': len(all_questions) - no_path_count,
        'questions_without_path': no_path_count,
        'path_length_distribution': path_length_stats
    }

    metadata_path = Config.BASE_DIR / "data" / "variant3_training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"   Metadata saved to: {metadata_path}")

    logging.info("\n" + "=" * 80)
    logging.info("Training data generation complete!")
    logging.info("=" * 80)
    logging.info(f"\nNext step: Run variant3_cache_embeddings.py to cache OpenAI embeddings")
    logging.info(f"\nLog file saved to: {log_file}")


if __name__ == "__main__":
    main()