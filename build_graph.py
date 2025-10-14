"""
Build NetworkX graph from MetaQA knowledge base.
Task 2 from context.md - Build graph object
"""
import networkx as nx
import pickle
from pathlib import Path

def load_kb_to_graph(kb_path='data/metaqa/kb.txt'):
    """Load knowledge base triples into NetworkX DiGraph."""
    print(f"Loading knowledge base from {kb_path}...")

    triples = []
    with open(kb_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                s, r, o = parts
                triples.append((s, r, o))

    print(f"Loaded {len(triples):,} triples")

    # Create NetworkX directed graph
    print("Creating NetworkX graph...")
    G = nx.DiGraph()
    for s, r, o in triples:
        G.add_edge(s, o, relation=r)

    print(f"\nGraph Statistics:")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Density: {nx.density(G):.6f}")

    return G

def save_graph(G, output_path='data/metaqa/graph.pkl'):
    """Save graph to pickle file."""
    print(f"\nSaving graph to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)

    # Check file size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Saved graph ({size_mb:.1f} MB)")

if __name__ == '__main__':
    # Build graph
    G = load_kb_to_graph()

    # Save graph
    save_graph(G)

    print("\n✓ Task 2 complete: graph.pkl created")
