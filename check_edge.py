#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from qa_system.config import Config
from qa_system.utils.loader import load_graph

graph = load_graph(Config.GRAPH_PATH)

# Check if there's an edge from Runner Runner to Just Cause
if graph.has_edge("Runner Runner", "Just Cause"):
    edge = graph["Runner Runner"]["Just Cause"]
    print(f"Edge exists: Runner Runner --{edge.get('relation', '???')}--> Just Cause")
else:
    print("No edge from Runner Runner to Just Cause")

if graph.has_edge("Just Cause", "Runner Runner"):
    edge = graph["Just Cause"]["Runner Runner"]
    print(f"Edge exists: Just Cause --{edge.get('relation', '???')}--> Runner Runner")
else:
    print("No edge from Just Cause to Runner Runner")

# Check what they have in common
print("\nJust Cause successors (first 20):")
for i, succ in enumerate(list(graph.successors("Just Cause"))[:20]):
    edge = graph["Just Cause"][succ]
    print(f"  Just Cause --{edge.get('relation', '')}--> {succ}")

print("\nRunner Runner successors (first 20):")
for i, succ in enumerate(list(graph.successors("Runner Runner"))[:20]):
    edge = graph["Runner Runner"][succ]
    print(f"  Runner Runner --{edge.get('relation', '')}--> {succ}")

    # Check if Just Cause also points to this
    if graph.has_edge("Just Cause", succ):
        jc_edge = graph["Just Cause"][succ]
        print(f"    ✓ Just Cause also points here: Just Cause --{jc_edge.get('relation', '')}--> {succ}")
