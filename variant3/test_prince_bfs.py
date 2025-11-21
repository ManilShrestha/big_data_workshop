"""
Test the exact BFS behavior with the Prince example.
"""
import re
from collections import deque
from typing import Set, List, Tuple, Optional

from qa_system.config import Config
from qa_system.utils.loader import load_graph


def bfs_shortest_path_debug(
    graph,
    start_nodes: Set[str],
    target_nodes: Set[str],
    max_hops: int = 3
) -> Optional[List[Tuple[str, str, str]]]:
    """BFS with debug output."""
    parent = {}
    queue = deque()

    for start in start_nodes:
        if start in graph:
            queue.append((start, 0))
            parent[start] = None
            print(f"  Starting from: {start}")

    visited_at_hop = {}
    for start in start_nodes:
        visited_at_hop[start] = 0

    iterations = 0
    while queue:
        current, hop = queue.popleft()
        iterations += 1

        if iterations <= 20:  # Only show first 20 iterations
            print(f"\n  Iteration {iterations}: Exploring '{current}' at hop {hop}")

        # Check if we reached target
        if current in target_nodes:
            print(f"\n   FOUND TARGET: {current} at hop {hop}")
            # Reconstruct path
            path = []
            node = current
            while parent[node] is not None:
                prev_node, relation, target_node = parent[node]
                path.append((prev_node, relation, target_node))
                node = prev_node
            return list(reversed(path))

        # Don't explore beyond max hops
        if hop >= max_hops:
            if iterations <= 20:
                print(f"    Max hops reached, skipping")
            continue

        # Explore outgoing edges
        out_count = 0
        for _, neighbor, _, edge_data in graph.out_edges(current, keys=True, data=True):
            if neighbor not in visited_at_hop or visited_at_hop[neighbor] > hop + 1:
                visited_at_hop[neighbor] = hop + 1
                relation = edge_data.get('relation', 'unknown')
                parent[neighbor] = (current, relation, neighbor)
                queue.append((neighbor, hop + 1))

                if iterations <= 20 and out_count < 3:
                    print(f"    → Adding {neighbor} (via {relation}) at hop {hop + 1}")
                    out_count += 1

        # Explore incoming edges
        in_count = 0
        for source, _, _, edge_data in graph.in_edges(current, keys=True, data=True):
            if source not in visited_at_hop or visited_at_hop[source] > hop + 1:
                visited_at_hop[source] = hop + 1
                relation = edge_data.get('relation', 'unknown')
                parent[source] = (current, relation, source)
                queue.append((source, hop + 1))

                if iterations <= 20 and in_count < 3:
                    print(f"    ← Adding {source} (via {relation}) at hop {hop + 1}")
                    in_count += 1

    return None


def main():
    print("="*80)
    print("TEST: PRINCE BFS BEHAVIOR")
    print("="*80)

    # Load graph
    print("\n[1] Loading graph...")
    graph = load_graph(str(Config.GRAPH_PATH))

    # The question
    print("\n[2] Question:")
    print("   'who is listed as screenwriter of the films directed by")
    print("    the [Under the Cherry Moon] director'")

    print("\n[3] Expected answer: Prince")
    print("\n[4] Expected 3-hop path:")
    print("   1. Under the Cherry Moon --[directed_by]--> Prince")
    print("   2. Prince <--[directed_by]-- Graffiti Bridge")
    print("   3. Graffiti Bridge --[written_by]--> Prince")

    # Run BFS
    print("\n[5] Running BFS with debug output...")
    print("="*80)

    start_nodes = {"Under the Cherry Moon"}
    target_nodes = {"Prince"}

    path = bfs_shortest_path_debug(graph, start_nodes, target_nodes, max_hops=3)

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)

    if path:
        print(f"\n Path found with {len(path)} hops:")
        for i, (node, rel, target) in enumerate(path, 1):
            print(f"  {i}. {node} --[{rel}]--> {target}")

        if len(path) == 1:
            print(f"\n  Only 1-hop path found!")
            print(f"   This is the shortcut that causes the problem.")
        elif len(path) == 3:
            print(f"\n Correct 3-hop path found!")
    else:
        print("\n No path found")

    # Check if 3-hop path exists
    print("\n" + "="*80)
    print("ANALYSIS: Why didn't we find the 3-hop path?")
    print("="*80)

    print(f"\nThe BFS finds the 1-hop path first because:")
    print(f"  1. BFS explores in breadth-first order")
    print(f"  2. It discovers 'Prince' at hop 1 via 'directed_by'")
    print(f"  3. Once target is found, it immediately returns")
    print(f"  4. It never explores the 3-hop path!")

    print(f"\nThe problem:")
    print(f"  - BFS returns the FIRST path to the target")
    print(f"  - For multi-hop questions, we WANT the path that matches the hop count")
    print(f"  - But BFS doesn't know about the hop count requirement!")

    print(f"\nSolution options:")
    print(f"  1. Don't stop at first target - collect ALL paths up to 3 hops")
    print(f"  2. Filter by hop count (what you're currently doing)")
    print(f"  3. Exclude the target from early hops")


if __name__ == "__main__":
    main()