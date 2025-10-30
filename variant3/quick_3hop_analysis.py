"""
QUICK analysis: Sample 3-hop questions to find examples fast.
"""
import re
from collections import deque
from typing import Set, List, Tuple, Optional
import random

from qa_system.config import Config
from qa_system.utils.loader import load_qa_dataset, load_graph


def extract_topic_entities(question_text: str, entity_lookup: dict) -> Set[str]:
    bracket_matches = re.findall(r'\[([^\]]+)\]', question_text)
    entities = set()
    for entity_str in bracket_matches:
        if entity_str in entity_lookup:
            entities.add(entity_lookup[entity_str])
        elif entity_str.lower() in entity_lookup:
            entities.add(entity_lookup[entity_str.lower()])
    return entities


def extract_answer_entities(answer_list: List[str], entity_lookup: dict) -> Set[str]:
    entities = set()
    for answer_str in answer_list:
        if answer_str in entity_lookup:
            entities.add(entity_lookup[answer_str])
        elif answer_str.lower() in entity_lookup:
            entities.add(entity_lookup[answer_str.lower()])
    return entities


def bfs_find_shortest(graph, start_nodes: Set[str], target_nodes: Set[str], max_hops: int = 3) -> Optional[List]:
    """Returns shortest path or None."""
    parent = {}
    queue = deque()

    for start in start_nodes:
        if start in graph:
            queue.append((start, 0))
            parent[start] = None

    visited_at_hop = {start: 0 for start in start_nodes}

    while queue:
        current, hop = queue.popleft()

        if current in target_nodes:
            # Reconstruct
            path = []
            node = current
            while parent[node] is not None:
                prev_node, relation, target_node = parent[node]
                path.append((prev_node, relation, target_node))
                node = prev_node
            return list(reversed(path))

        if hop >= max_hops:
            continue

        # Outgoing
        for _, neighbor, _, edge_data in graph.out_edges(current, keys=True, data=True):
            if neighbor not in visited_at_hop or visited_at_hop[neighbor] > hop + 1:
                visited_at_hop[neighbor] = hop + 1
                parent[neighbor] = (current, edge_data.get('relation', 'unknown'), neighbor)
                queue.append((neighbor, hop + 1))

        # Incoming
        for source, _, _, edge_data in graph.in_edges(current, keys=True, data=True):
            if source not in visited_at_hop or visited_at_hop[source] > hop + 1:
                visited_at_hop[source] = hop + 1
                parent[source] = (current, edge_data.get('relation', 'unknown'), source)
                queue.append((source, hop + 1))

    return None


def show_edges(graph, node: str, direction: str = "out", max_show: int = 10):
    """Show edges from a node."""
    if direction == "out":
        edges = list(graph.out_edges(node, keys=True, data=True))
        print(f"      Outgoing edges from '{node}': {len(edges)} total")
        for i, (_, neighbor, _, edge_data) in enumerate(edges[:max_show]):
            rel = edge_data.get('relation', 'unknown')
            print(f"        {i+1}. --[{rel}]--> {neighbor}")
        if len(edges) > max_show:
            print(f"        ... and {len(edges) - max_show} more")
        if len(edges) == 0:
            print(f"        (no outgoing edges)")
    else:
        edges = list(graph.in_edges(node, keys=True, data=True))
        print(f"      Incoming edges to '{node}': {len(edges)} total")
        for i, (source, _, _, edge_data) in enumerate(edges[:max_show]):
            rel = edge_data.get('relation', 'unknown')
            print(f"        {i+1}. {source} --[{rel}]-->")
        if len(edges) > max_show:
            print(f"        ... and {len(edges) - max_show} more")
        if len(edges) == 0:
            print(f"        (no incoming edges)")


def main():
    print("="*80)
    print("QUICK 3-HOP ANALYSIS (SAMPLED)")
    print("="*80)

    # Load graph
    print("\n[1] Loading graph...")
    graph = load_graph(str(Config.GRAPH_PATH))
    print(f"   {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

    entity_lookup = {}
    for node in graph.nodes():
        entity_lookup[node] = node
        entity_lookup[node.lower()] = node

    # Load 3-hop questions
    print("\n[2] Loading 3-hop questions...")
    questions = load_qa_dataset(str(Config.QA_3HOP_TRAIN), hop_count=3)
    print(f"   {len(questions):,} 3-hop questions")

    # Sample random questions
    print("\n[3] Sampling 1000 random questions...")
    sample = random.sample(questions, min(1000, len(questions)))

    found_1hop = None
    found_2hop = None
    found_no_path = None
    found_3hop = 0

    for i, q in enumerate(sample):
        if i % 100 == 0:
            print(f"   Checked {i}/1000...")

        topic_entities = extract_topic_entities(q.text, entity_lookup)
        answer_entities = extract_answer_entities(q.ground_truth_answers, entity_lookup)

        if not topic_entities or not answer_entities:
            continue

        path = bfs_find_shortest(graph, topic_entities, answer_entities, max_hops=3)

        if path is None:
            if found_no_path is None:
                found_no_path = (q, topic_entities, answer_entities, None)
        elif len(path) == 1:
            if found_1hop is None:
                found_1hop = (q, topic_entities, answer_entities, path)
        elif len(path) == 2:
            if found_2hop is None:
                found_2hop = (q, topic_entities, answer_entities, path)
        elif len(path) == 3:
            found_3hop += 1

        if found_1hop and found_2hop and found_no_path:
            break

    # Display results
    print(f"\n   Found 3-hop paths: {found_3hop} out of {i+1} checked")

    print(f"\n" + "="*80)
    print("EXAMPLE 1: 1-HOP SHORTCUT")
    print("="*80)

    if found_1hop:
        q, topics, answers, path = found_1hop
        print(f"\nQuestion: {q.text}")
        print(f"Expected hops: 3")
        print(f"Actual path length: 1")
        print(f"\nAnswers: {q.ground_truth_answers[:5]}")
        print(f"Topic entities: {list(topics)}")
        print(f"Answer entities: {list(answers)[:5]}")
        print(f"\nSHORTCUT PATH:")
        for node, rel, target in path:
            print(f"  {node} --[{rel}]--> {target}")
        print(f"\n⚠️  This question expects 3 hops, but a 1-hop shortcut exists!")
    else:
        print("\nNo 1-hop shortcut found in sample")

    print(f"\n" + "="*80)
    print("EXAMPLE 2: 2-HOP SHORTCUT")
    print("="*80)

    if found_2hop:
        q, topics, answers, path = found_2hop
        print(f"\nQuestion: {q.text}")
        print(f"Expected hops: 3")
        print(f"Actual path length: 2")
        print(f"\nAnswers: {q.ground_truth_answers[:5]}")
        print(f"Topic entities: {list(topics)}")
        print(f"Answer entities: {list(answers)[:5]}")
        print(f"\nSHORTCUT PATH:")
        for node, rel, target in path:
            print(f"  {node} --[{rel}]--> {target}")
        print(f"\n⚠️  This question expects 3 hops, but a 2-hop shortcut exists!")
    else:
        print("\nNo 2-hop shortcut found in sample")

    print(f"\n" + "="*80)
    print("EXAMPLE 3: NO PATH AT ALL (DISCONNECTED)")
    print("="*80)

    if found_no_path:
        q, topics, answers, _ = found_no_path
        print(f"\nQuestion: {q.text}")
        print(f"Expected hops: 3")
        print(f"Actual path: NONE FOUND")
        print(f"\nAnswers: {q.ground_truth_answers[:5]}")
        print(f"Topic entities: {list(topics)}")
        print(f"Answer entities: {list(answers)[:5]}")

        topic = list(topics)[0]
        answer = list(answers)[0]

        print(f"\n--- GRAPH EXPLORATION ---")
        print(f"\nTopic entity: '{topic}'")
        show_edges(graph, topic, "out", max_show=10)
        show_edges(graph, topic, "in", max_show=5)

        print(f"\nAnswer entity: '{answer}'")
        show_edges(graph, answer, "out", max_show=5)
        show_edges(graph, answer, "in", max_show=10)

        print(f"\n⚠️  NO PATH EXISTS within 3 hops!")
        print(f"   This means the graph is incomplete or the question/answer is incorrect.")
    else:
        print("\nNo disconnected examples found in sample")


if __name__ == "__main__":
    main()