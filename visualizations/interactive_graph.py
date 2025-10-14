import pickle
import networkx as nx
from pyvis.network import Network

# Load graph from pickle
with open("/home/ms5267/big_data_workshop/data/metaqa/graph.pkl", "rb") as f:
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

# Create visualization
net = Network(height="750px", width="100%", directed=True, notebook=False)
net.from_nx(subgraph)

# Configure physics for better layout
net.toggle_physics(True)

# Save to HTML file
net.write_html("graph.html")
print(f"\n✓ Graph saved to graph.html")
print(f"  This subgraph preserves ALL connections for included nodes")
