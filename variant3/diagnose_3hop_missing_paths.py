"""
Diagnose why 45% of 3-hop questions don't have valid 3-hop paths.

This script:
1. Loads all 3-hop training questions
2. Runs BFS to find shortest paths
3. Categorizes rejected questions by:
   - No path exists at all
   - Shorter path exists (1-hop or 2-hop shortcut)
   - Longer path exists (4+ hops)
   - Path exists but entities not found in graph
4. Samples examples from each category for manual inspection
"""

import re
import json
import pickle
from collections import deque, defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from tqdm import tqdm

from qa_system.config import Config
from qa_system.utils.loader import load_qa_dataset, load_graph
from qa_system.core.question import Question


def build_entity_lookup(graph) -> Dict[str, str]:
    """Build case-insensitive entity lookup dictionary."""
    lookup = {}
    for node in graph.nodes():
        lookup[node] = node  # Exact match
        lookup[node.lower()] = node  # Case-insensitive match
    return lookup


def extract_topic_entities(question_text: str, entity_lookup: Dict[str, str]) -> Set[str]:
    """Extract topic entities from [brackets] in question text."""
    bracket_matches = re.findall(r'\[([^\]]+)\]', question_text)
    entities = set()

    for entity_str in bracket_matches:
        # Try exact match first
        if entity_str in entity_lookup:
            entities.add(entity_lookup[entity_str])
        # Try case-insensitive
        elif entity_str.lower() in entity_lookup:
            entities.add(entity_lookup[entity_str.lower()])

    return entities


def extract_answer_entities(answers: List[str], entity_lookup: Dict[str, str]) -> Set[str]:
    """Extract answer entities from ground truth answers."""
    entities = set()

    for answer in answers:
        # Try exact match first
        if answer in entity_lookup:
            entities.add(entity_lookup[answer])
        # Try case-insensitive
        elif answer.lower() in entity_lookup:
            entities.add(entity_lookup[answer.lower()])

    return entities


def bfs_all_path_lengths(
    graph,
    topic_entities: Set[str],
    answer_entities: Set[str],
    max_hops: int = 5
) -> Set[int]:
    """
    Find ALL possible path lengths from topic entities to answer entities.

    Returns:
        Set of path lengths (e.g., {1, 3} means both 1-hop and 3-hop paths exist)
    """
    # Early exit if no entities
    if not topic_entities or not answer_entities:
        return set()

    # Check if any topic entity is already an answer
    if topic_entities & answer_entities:
        return {0}

    path_lengths = set()
    queue = deque()
    visited = {}  # node -> min hop count when first visited

    # Initialize with topic entities at hop 0
    for entity in topic_entities:
        queue.append((entity, 0))
        visited[entity] = 0

    while queue:
        current_node, hop_count = queue.popleft()

        # Skip if we've already explored from this node at this depth
        if hop_count > visited.get(current_node, float('inf')):
            continue

        # Check if path is too long
        if hop_count >= max_hops:
            continue

        # Explore neighbors (outgoing + incoming)
        next_hop = hop_count + 1

        # Pre-collect neighbors to avoid multiple graph calls
        neighbors = set()

        # Outgoing edges
        if current_node in graph:
            for _, target in graph.out_edges(current_node):
                neighbors.add(target)

        # Incoming edges
        for source, _ in graph.in_edges(current_node):
            neighbors.add(source)

        # Check neighbors
        for neighbor in neighbors:
            if neighbor in answer_entities:
                path_lengths.add(next_hop)

            # Continue exploring even if we found an answer
            # (to find all possible path lengths)
            if neighbor not in visited or visited[neighbor] > next_hop:
                visited[neighbor] = next_hop
                queue.append((neighbor, next_hop))

    return path_lengths


def categorize_question(
    question: Question,
    graph,
    entity_lookup: Dict[str, str]
) -> Dict:
    """
    Categorize why a question might be rejected.

    Returns:
        {
            'category': str,
            'shortest_path': int or None,
            'all_path_lengths': set of int,
            'has_3hop_path': bool,
            'topic_entities_found': int,
            'answer_entities_found': int
        }
    """
    # Extract entities
    topic_entities = extract_topic_entities(question.text, entity_lookup)
    answer_entities = extract_answer_entities(question.ground_truth_answers, entity_lookup)

    # Check if entities are in graph
    if not topic_entities:
        return {
            'category': 'no_topic_entity',
            'shortest_path': None,
            'all_path_lengths': set(),
            'has_3hop_path': False,
            'topic_entities_found': 0,
            'answer_entities_found': len(answer_entities)
        }

    if not answer_entities:
        return {
            'category': 'no_answer_entity',
            'shortest_path': None,
            'all_path_lengths': set(),
            'has_3hop_path': False,
            'topic_entities_found': len(topic_entities),
            'answer_entities_found': 0
        }

    # Find ALL path lengths
    all_path_lengths = bfs_all_path_lengths(graph, topic_entities, answer_entities, max_hops=5)

    if not all_path_lengths:
        return {
            'category': 'no_path',
            'shortest_path': None,
            'all_path_lengths': set(),
            'has_3hop_path': False,
            'topic_entities_found': len(topic_entities),
            'answer_entities_found': len(answer_entities)
        }

    shortest_path = min(all_path_lengths)
    has_3hop_path = 3 in all_path_lengths

    if shortest_path < question.hop_count:
        # Has shortcut - check if 3-hop also exists
        if has_3hop_path:
            category = f'shortcut_{shortest_path}hop_but_has_3hop'
        else:
            category = f'shortcut_{shortest_path}hop_no_3hop'
    elif shortest_path == question.hop_count:
        category = 'valid'
    else:
        category = 'longer_path'

    return {
        'category': category,
        'shortest_path': shortest_path,
        'all_path_lengths': all_path_lengths,
        'has_3hop_path': has_3hop_path,
        'topic_entities_found': len(topic_entities),
        'answer_entities_found': len(answer_entities)
    }


def main():
    print("=" * 80)
    print("Diagnosing 3-Hop Question Path Issues")
    print("=" * 80)

    # Load graph
    print("\n[1/4] Loading knowledge graph...")
    graph = load_graph(str(Config.GRAPH_PATH))
    print(f"   Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Build entity lookup
    print("\n[2/4] Building entity lookup...")
    entity_lookup = build_entity_lookup(graph)
    print(f"   Entity lookup: {len(entity_lookup)} entries")

    # Load 3-hop questions
    print("\n[3/4] Loading 3-hop training questions...")
    three_hop_path = Path("data/metaqa/3-hop/vanilla/qa_train.txt")
    questions = load_qa_dataset(str(three_hop_path), hop_count=3)
    print(f"   Loaded: {len(questions)} questions")

    # Analyze each question
    print("\n[4/4] Analyzing questions...")
    categories = defaultdict(list)

    for question in tqdm(questions, desc="Processing"):
        result = categorize_question(question, graph, entity_lookup)
        categories[result['category']].append({
            'question': question,
            'result': result
        })

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_questions = len(questions)
    print(f"\nTotal 3-hop questions: {total_questions:,}")
    print(f"\nBreakdown by category:")
    print(f"{'Category':<25} {'Count':<10} {'Percentage':<12} {'Description'}")
    print("-" * 80)

    # Determine category order dynamically
    category_order = sorted(categories.keys(), key=lambda x: (
        0 if x == 'valid' else
        1 if 'shortcut' in x and 'has_3hop' in x else
        2 if 'shortcut' in x and 'no_3hop' in x else
        3 if x == 'no_path' else
        4 if x == 'longer_path' else
        5
    ))

    for category in category_order:
        count = len(categories[category])
        if count > 0:
            pct = 100 * count / total_questions
            if category == 'valid':
                desc = "Valid 3-hop path exists"
            elif category.startswith('shortcut'):
                desc = f"Shorter path exists"
            elif category == 'no_path':
                desc = "No path exists"
            elif category == 'longer_path':
                desc = "Only 4+ hop paths exist"
            elif category == 'no_topic_entity':
                desc = "Topic entity not in graph"
            elif category == 'no_answer_entity':
                desc = "Answer entity not in graph"
            else:
                desc = ""

            print(f"{category:<25} {count:<10,} {pct:>6.2f}%      {desc}")

    # Show expected vs actual
    print(f"\n{'Expected valid questions:':<30} 62,021 (from metadata)")
    print(f"{'Actual valid questions:':<30} {len(categories['valid']):,}")
    print(f"{'Expected rejected questions:':<30} 52,175 (from metadata)")
    actual_rejected = total_questions - len(categories['valid'])
    print(f"{'Actual rejected questions:':<30} {actual_rejected:,}")

    # Sample examples from each category
    print("\n" + "=" * 80)
    print("SAMPLE EXAMPLES")
    print("=" * 80)

    for category in category_order:
        if len(categories[category]) > 0:
            print(f"\n{'='*80}")
            print(f"Category: {category.upper()}")
            print(f"Count: {len(categories[category]):,}")
            print(f"{'='*80}")

            # Show up to 3 examples
            for i, item in enumerate(categories[category][:3]):
                question = item['question']
                result = item['result']

                print(f"\nExample {i+1}:")
                print(f"  Question: {question.text}")
                print(f"  Answers: {', '.join(question.ground_truth_answers[:5])}")
                if len(question.ground_truth_answers) > 5:
                    print(f"           ... and {len(question.ground_truth_answers) - 5} more")
                print(f"  Topic entities found: {result['topic_entities_found']}")
                print(f"  Answer entities found: {result['answer_entities_found']}")

                if result['shortest_path'] is not None:
                    print(f"  Shortest path: {result['shortest_path']} hops")
                    print(f"  All path lengths: {sorted(result['all_path_lengths'])}")
                    print(f"  Has 3-hop path: {result['has_3hop_path']}")
                else:
                    print(f"  Shortest path: None")

            if len(categories[category]) > 3:
                print(f"\n  ... and {len(categories[category]) - 3:,} more examples")

    # Save detailed results
    output_path = Path("results/3hop_path_diagnosis.json")
    output_path.parent.mkdir(exist_ok=True)

    # Prepare data for JSON (convert to serializable format)
    json_data = {}
    for category, items in categories.items():
        json_data[category] = []
        for item in items[:100]:  # Save up to 100 examples per category
            question = item['question']
            result = item['result']
            json_data[category].append({
                'question_id': question.question_id,
                'question_text': question.text,
                'ground_truth_answers': question.ground_truth_answers,
                'topic_entities_found': result['topic_entities_found'],
                'answer_entities_found': result['answer_entities_found'],
                'shortest_path': result['shortest_path'],
                'all_path_lengths': sorted(result['all_path_lengths']),
                'has_3hop_path': result['has_3hop_path']
            })

    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'total_questions': total_questions,
                'valid_questions': len(categories['valid']),
                'rejected_questions': actual_rejected,
                'categories': {cat: len(items) for cat, items in categories.items()}
            },
            'examples': json_data
        }, f, indent=2)

    print(f"\n\nDetailed results saved to: {output_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
