#!/usr/bin/env python3
"""
Examine all edges for a specific film
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from qa_system.config import Config
from qa_system.utils.loader import load_graph

def examine_node(node_name):
    graph = load_graph(Config.GRAPH_PATH)

    if node_name not in graph:
        print(f"Node '{node_name}' not in graph!")
        return

    print(f"Examining node: {node_name}")
    print("="*80)

    # Outgoing edges
    print("\nOUTGOING edges (node --> X):")
    print("-"*80)
    out_count = 0
    for succ in graph.successors(node_name):
        edge = graph[node_name][succ]
        rel = edge.get('relation', '')
        print(f"  {node_name} --{rel}--> {succ}")
        out_count += 1
    if out_count == 0:
        print("  (none)")

    # Incoming edges
    print("\nINCOMING edges (X --> node):")
    print("-"*80)
    in_count = 0
    for pred in graph.predecessors(node_name):
        edge = graph[pred][node_name]
        rel = edge.get('relation', '')
        print(f"  {pred} --{rel}--> {node_name}")
        in_count += 1
    if in_count == 0:
        print("  (none)")

    print(f"\nTotal outgoing: {out_count}")
    print(f"Total incoming: {in_count}")

def main():
    print("\n" + "="*80)
    print(" EXAMINING SPECIFIC NODES")
    print("="*80 + "\n")

    # Examine the film
    examine_node("Hedgehog in the Fog")

    print("\n" + "="*80)

    # Also check Sergei Kozlov
    examine_node("Sergei Kozlov")

    print("\n" + "="*80)

    # Check Yuriy Norshteyn
    examine_node("Yuriy Norshteyn")

if __name__ == "__main__":
    main()
