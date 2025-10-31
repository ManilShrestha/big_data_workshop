"""
FAST diagnosis of 3-hop questions with shortcuts.

Uses batched BFS and precomputed adjacency for speed.
Only checks: shortest path length + does 3-hop path exist?
"""

import re
import json
import pickle
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Set, Tuple
from tqdm import tqdm
import numpy as np

from qa_system.config import Config
from qa_system.utils.loader import load_qa_dataset, load_graph
from qa_system.core.question import Question


def build_entity_lookup(graph) -> Dict[str, str]:
    """Build case-insensitive entity lookup."""
    lookup = {}
    for node in graph.nodes():
        lookup[node] = node
        lookup[node.lower()] = node
    return lookup


def build_adjacency_dict(graph) -> Dict[str, Set[str]]:
    """
    Precompute adjacency as dict for O(1) lookups.
    Combines outgoing and incoming edges (bidirectional).
    """
    print("  Building adjacency dict...")
    adj = defaultdict(set)

    for node in tqdm(graph.nodes(), desc="    Nodes"):
        # Outgoing
        for _, target in graph.out_edges(node):
            adj[node].add(target)
        # Incoming
        for source, _ in graph.in_edges(node):
            adj[node].add(source)

    return dict(adj)


def extract_topic_entities(question_text: str, entity_lookup: Dict[str, str]) -> Set[str]:
    """Extract topic entities from [brackets]."""
    bracket_matches = re.findall(r'\[([^\]]+)\]', question_text)
    entities = set()
    for entity_str in bracket_matches:
        if entity_str in entity_lookup:
            entities.add(entity_lookup[entity_str])
        elif entity_str.lower() in entity_lookup:
            entities.add(entity_lookup[entity_str.lower()])
    return entities


def extract_answer_entities(answers: List[str], entity_lookup: Dict[str, str]) -> Set[str]:
    """Extract answer entities."""
    entities = set()
    for answer in answers:
        if answer in entity_lookup:
            entities.add(entity_lookup[answer])
        elif answer.lower() in entity_lookup:
            entities.add(entity_lookup[answer.lower()])
    return entities


def check_paths(
    topic_entities: Set[str],
    answer_entities: Set[str],
    adj: Dict[str, Set[str]]
) -> Tuple[int, bool]:
    """
    Check shortest path and if 3-hop exists.

    Returns:
        (shortest_path_length, has_3hop_path)
        shortest_path_length is None if no path
    """
    if not topic_entities or not answer_entities:
        return None, False

    # Check if topic is already answer
    if topic_entities & answer_entities:
        return 0, False

    # BFS with hop tracking
    # frontier[hop] = set of nodes reachable at that hop
    frontier = [topic_entities, set(), set(), set()]  # hops 0, 1, 2, 3
    visited = set(topic_entities)

    shortest = None
    has_3hop = False

    for hop in range(3):
        if not frontier[hop]:
            break

        # Expand frontier[hop] -> frontier[hop+1]
        for node in frontier[hop]:
            if node not in adj:
                continue

            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier[hop + 1].add(neighbor)

                    # Check if neighbor is answer
                    if neighbor in answer_entities:
                        if shortest is None:
                            shortest = hop + 1
                        if hop + 1 == 3:
                            has_3hop = True

    return shortest, has_3hop


def main():
    print("=" * 80)
    print("FAST 3-Hop Diagnosis (Batched)")
    print("=" * 80)

    # Load graph
    print("\n[1/5] Loading knowledge graph...")
    graph = load_graph(str(Config.GRAPH_PATH))
    print(f"   Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

    # Build adjacency dict
    print("\n[2/5] Building adjacency dictionary...")
    adj = build_adjacency_dict(graph)
    print(f"   Adjacency built: {len(adj):,} nodes")

    # Build entity lookup
    print("\n[3/5] Building entity lookup...")
    entity_lookup = build_entity_lookup(graph)
    print(f"   Entity lookup: {len(entity_lookup):,} entries")

    # Load questions
    print("\n[4/5] Loading 3-hop training questions...")
    three_hop_path = Path("data/metaqa/3-hop/vanilla/qa_train.txt")
    questions = load_qa_dataset(str(three_hop_path), hop_count=3)
    print(f"   Loaded: {len(questions):,} questions")

    # Process all questions
    print("\n[5/5] Analyzing questions...")

    results = {
        'valid': [],
        'shortcut_1hop_no_3hop': [],
        'shortcut_1hop_has_3hop': [],
        'shortcut_2hop_no_3hop': [],
        'shortcut_2hop_has_3hop': [],
        'no_topic_entity': [],
        'no_answer_entity': [],
        'no_path': []
    }

    for question in tqdm(questions, desc="Processing"):
        topic_entities = extract_topic_entities(question.text, entity_lookup)
        answer_entities = extract_answer_entities(question.ground_truth_answers, entity_lookup)

        # Check entities
        if not topic_entities:
            results['no_topic_entity'].append(question.question_id)
            continue
        if not answer_entities:
            results['no_answer_entity'].append(question.question_id)
            continue

        # Check paths
        shortest, has_3hop = check_paths(topic_entities, answer_entities, adj)

        if shortest is None:
            results['no_path'].append(question.question_id)
        elif shortest == 3:
            results['valid'].append(question.question_id)
        elif shortest == 1:
            if has_3hop:
                results['shortcut_1hop_has_3hop'].append(question.question_id)
            else:
                results['shortcut_1hop_no_3hop'].append(question.question_id)
        elif shortest == 2:
            if has_3hop:
                results['shortcut_2hop_has_3hop'].append(question.question_id)
            else:
                results['shortcut_2hop_no_3hop'].append(question.question_id)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(questions)
    print(f"\nTotal 3-hop questions: {total:,}\n")

    print(f"{'Category':<30} {'Count':<10} {'%':<8}")
    print("-" * 80)

    for cat in ['valid', 'shortcut_1hop_has_3hop', 'shortcut_1hop_no_3hop',
                'shortcut_2hop_has_3hop', 'shortcut_2hop_no_3hop',
                'no_path', 'no_topic_entity', 'no_answer_entity']:
        count = len(results[cat])
        pct = 100 * count / total
        print(f"{cat:<30} {count:<10,} {pct:>6.2f}%")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    valid = len(results['valid'])
    shortcut_with_3hop = len(results['shortcut_1hop_has_3hop']) + len(results['shortcut_2hop_has_3hop'])
    shortcut_no_3hop = len(results['shortcut_1hop_no_3hop']) + len(results['shortcut_2hop_no_3hop'])

    print(f"\n1. Valid 3-hop only: {valid:,} ({100*valid/total:.1f}%)")
    print(f"   → These questions used for training")

    print(f"\n2. Has shortcut BUT also has 3-hop path: {shortcut_with_3hop:,} ({100*shortcut_with_3hop/total:.1f}%)")
    print(f"   → COULD be used for training on 3-hop paths!")
    print(f"   → Currently REJECTED by training script")

    print(f"\n3. Has shortcut and NO 3-hop path: {shortcut_no_3hop:,} ({100*shortcut_no_3hop/total:.1f}%)")
    print(f"   → MetaQA labeling is WRONG for these")
    print(f"   → Should be relabeled as 1-hop or 2-hop")

    potential_training = valid + shortcut_with_3hop
    print(f"\n**POTENTIAL 3-HOP TRAINING DATA: {potential_training:,} ({100*potential_training/total:.1f}%)**")
    print(f"   Current: {valid:,}")
    print(f"   If we use shortcuts with 3-hop: +{shortcut_with_3hop:,} ({100*shortcut_with_3hop/valid:.1f}% increase)")

    # Save results
    output_path = Path("results/3hop_diagnosis_fast.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'total': total,
                'valid': valid,
                'shortcut_with_3hop': shortcut_with_3hop,
                'shortcut_no_3hop': shortcut_no_3hop,
                'potential_training_data': potential_training
            },
            'counts': {cat: len(results[cat]) for cat in results}
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
