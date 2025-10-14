"""
Export NetworkX graph subgraph to GEXF format for Gephi visualization
Same sampling strategy as interactive_graph.py - meaningful nodes with no missing links
"""
import pickle
import networkx as nx

# Load the graph
print("Loading graph from pickle...")
with open("data/metaqa/graph.pkl", "rb") as f:
    G = pickle.load(f)

print(f"Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Strategy: Start from highly connected nodes (movies) and expand to their neighborhoods
# This ensures we capture complete local structures without missing edges

# Find movies (nodes with outgoing edges for typical movie relations)
movie_relations = ['directed_by', 'written_by', 'starred_actors', 'release_year', 'has_genre', 'has_tags']
movies = []
for node in G.nodes():
    out_edges = G.out_edges(node, data=True)
    relations = [data['relation'] for _, _, data in out_edges]
    if any(rel in movie_relations for rel in relations):
        movies.append(node)
        if len(movies) >= 50:  # Sample 50 movies
            break

print(f"Selected {len(movies)} movies as seed nodes")

# For each movie, get ALL its connected nodes (complete ego network)
nodes_to_include = set(movies)
for movie in movies:
    # Add all predecessors and successors (1-hop neighborhood)
    nodes_to_include.update(G.predecessors(movie))
    nodes_to_include.update(G.successors(movie))

print(f"Total nodes after expanding neighborhoods: {len(nodes_to_include)}")

# Create subgraph with ALL edges between these nodes
subgraph = G.subgraph(nodes_to_include)

print(f"Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

# Export to GEXF format (best for Gephi)
output_file = "data/metaqa/graph_subgraph.gexf"
print(f"\nExporting to {output_file}...")
nx.write_gexf(subgraph, output_file)

print(f"\n✓ Export complete!")
print(f"\nTo use in Gephi:")
print(f"  1. Open Gephi")
print(f"  2. File → Open → Select '{output_file}'")
print(f"  3. The graph will load with all nodes, edges, and relationship attributes")
print(f"  This subgraph preserves ALL connections for included nodes - no missing links!")
