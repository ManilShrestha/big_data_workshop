"""
Visualize MetaQA knowledge graph.
Creates visualizations of the graph structure and statistics.
"""
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from collections import Counter
import numpy as np

def load_graph(graph_path='data/metaqa/graph.pkl'):
    """Load the saved graph."""
    print(f"Loading graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G

def visualize_relation_distribution(G, output_path='visualizations/relation_distribution.png'):
    """Plot distribution of relation types."""
    relations = [data['relation'] for _, _, data in G.edges(data=True)]
    rel_counts = Counter(relations)

    plt.figure(figsize=(12, 6))
    rels, counts = zip(*rel_counts.most_common())
    plt.bar(range(len(rels)), counts)
    plt.xticks(range(len(rels)), rels, rotation=45, ha='right')
    plt.xlabel('Relation Type')
    plt.ylabel('Count')
    plt.title('Distribution of Relation Types in MetaQA KB')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved relation distribution to {output_path}")
    plt.close()

def visualize_degree_distribution(G, output_path='visualizations/degree_distribution.png'):
    """Plot degree distribution."""
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Out-degree (movies pointing to entities)
    axes[0].hist(out_degrees, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Out-degree')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Out-degree Distribution')
    axes[0].set_yscale('log')

    # In-degree (entities being pointed to)
    axes[1].hist(in_degrees, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('In-degree')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('In-degree Distribution')
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved degree distribution to {output_path}")
    plt.close()

def visualize_subgraph(G, center_node='Kismet', max_depth=1, output_path='visualizations/example_subgraph.png'):
    """Visualize a small subgraph around a center node."""
    # Get neighborhood
    if center_node not in G:
        # Pick first movie node
        center_node = list(G.nodes())[0]

    # Get nodes within max_depth hops
    nodes = {center_node}
    for _ in range(max_depth):
        new_nodes = set()
        for node in nodes:
            new_nodes.update(G.successors(node))
        nodes.update(new_nodes)

    subgraph = G.subgraph(nodes)

    # Create layout
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)

    # Color nodes by type (center vs others)
    node_colors = ['red' if n == center_node else 'lightblue' for n in subgraph.nodes()]
    node_sizes = [3000 if n == center_node else 1000 for n in subgraph.nodes()]

    # Draw graph
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.3, arrows=True, arrowsize=10, width=1.5)

    # Draw labels
    nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight='bold')

    # Draw edge labels (relations)
    edge_labels = nx.get_edge_attributes(subgraph, 'relation')
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=7, font_color='red')

    plt.title(f'Subgraph around "{center_node}"', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved subgraph visualization to {output_path}")
    plt.close()

def print_graph_stats(G):
    """Print detailed graph statistics."""
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)

    print(f"\nBasic Stats:")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Density: {nx.density(G):.6f}")

    # Relation types
    relations = [data['relation'] for _, _, data in G.edges(data=True)]
    rel_counts = Counter(relations)
    print(f"\nRelation Types ({len(rel_counts)}):")
    for rel, count in rel_counts.most_common():
        print(f"  {rel}: {count:,}")

    # Degree stats
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    print(f"\nDegree Statistics:")
    print(f"  Avg in-degree: {np.mean(in_degrees):.2f} (max: {max(in_degrees)})")
    print(f"  Avg out-degree: {np.mean(out_degrees):.2f} (max: {max(out_degrees)})")

    # Find most connected nodes
    top_in = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:5]
    top_out = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\nTop 5 nodes by in-degree (most referenced):")
    for node, degree in top_in:
        print(f"  {node}: {degree}")

    print(f"\nTop 5 nodes by out-degree (most connections):")
    for node, degree in top_out:
        print(f"  {node}: {degree}")

if __name__ == '__main__':
    import os
    os.makedirs('visualizations', exist_ok=True)

    # Load graph
    G = load_graph()

    # Print statistics
    print_graph_stats(G)

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_relation_distribution(G)
    visualize_degree_distribution(G)
    visualize_subgraph(G)

    print("\n✓ Visualization complete!")
