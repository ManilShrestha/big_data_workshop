"""
Test if the BFS fix actually works with the Prince example.
"""
from qa_system.config import Config
from qa_system.utils.loader import load_graph
import sys
sys.path.insert(0, '/home/ms5267/big_data_workshop')

# Import the updated BFS function
from variant3.variant3_create_training_data import bfs_shortest_path


def main():
    print("="*80)
    print("TEST: DOES BFS FIX WORK?")
    print("="*80)

    # Load graph
    print("\n[1] Loading graph...")
    graph = load_graph(str(Config.GRAPH_PATH))

    print("\n[2] Testing Prince example:")
    print("   Question: 'screenwriter of films directed by [Under the Cherry Moon] director'")
    print("   Expected: 3-hop path through Graffiti Bridge")
    print("   Problem: There's also a 1-hop shortcut")

    start_nodes = {"Under the Cherry Moon"}
    target_nodes = {"Prince"}

    print("\n[3] Running fixed BFS...")
    path = bfs_shortest_path(graph, start_nodes, target_nodes, max_hops=3)

    if path:
        print(f"\n    Path found with {len(path)} hops:")
        for i, (node, rel, target) in enumerate(path, 1):
            print(f"   {i}. {node} --[{rel}]--> {target}")

        if len(path) == 1:
            print(f"\n    STILL RETURNS 1-HOP SHORTCUT!")
            print(f"   The fix didn't work.")
        elif len(path) == 3:
            print(f"\n    SUCCESS! Found the 3-hop path!")
        else:
            print(f"\n     Found {len(path)}-hop path (unexpected)")
    else:
        print("\n    No path found!")

    # Test another case
    print(f"\n" + "="*80)
    print("DIAGNOSIS: Why is it still returning 1-hop?")
    print("="*80)

    print(f"""
The issue is that when we do:
    parent[neighbor] = (current, relation, neighbor)

This OVERWRITES the previous parent! So:
1. First we find: Under the Cherry Moon → Prince (hop 1)
   parent[Prince] = (Under the Cherry Moon, directed_by, Prince)

2. Later we try: Under the Cherry Moon → Prince → Graffiti Bridge → Prince (hop 3)
   But when we reach Prince at hop 3, parent[Prince] is STILL pointing to hop 1!

We can't reconstruct the 3-hop path because the parent pointer was overwritten.

The BFS visited_at_hop check prevents us from exploring Prince again at hop 3:
    if neighbor not in visited_at_hop or visited_at_hop[neighbor] > hop + 1:

Since Prince was visited at hop 1, and we're now at hop 2 trying to visit it at hop 3,
the condition (visited_at_hop[Prince]=1 > 3) is FALSE, so we DON'T add it to the queue!

SOLUTION: We need to track MULTIPLE parent paths, not just one.
""")


if __name__ == "__main__":
    main()