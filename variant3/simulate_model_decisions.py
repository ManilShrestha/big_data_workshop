#!/usr/bin/env python3
"""
Simulate Variant 3 EdgeScorer BFS decisions step-by-step.

For each question, shows:
1. Every BFS level (hop-by-hop)
2. All candidate edges from current frontier
3. Model scores and text similarities for each edge
4. Which edges were selected (top-K)
5. Why certain paths were taken or rejected
"""

import json
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Tuple, Dict
import sys
from collections import defaultdict
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qa_system.config import Config
from qa_system.utils.loader import load_graph
from qa_system.entity_linkers.exact_matcher import ExactMatcher
from variant3.eval_qa.edge_scorer_ranker import EdgeScorerRelationRanker


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class BFSSimulator:
    """Simulates BFS search with detailed logging"""

    def __init__(
        self,
        graph: nx.DiGraph,
        edge_scorer: EdgeScorerRelationRanker,
        text_embeddings: dict,
        top_k_relations: int = 3,
        max_nodes_per_relation: int = 30
    ):
        self.graph = graph
        self.edge_scorer = edge_scorer
        self.text_embeddings = text_embeddings
        self.top_k_relations = top_k_relations
        self.max_nodes_per_relation = max_nodes_per_relation

    def simulate_search(
        self,
        question: str,
        start_nodes: List[str],
        max_hops: int,
        ground_truth: List[str]
    ) -> Dict:
        """
        Simulate BFS search with detailed step-by-step logging.

        Returns a detailed trace of the search process.
        """
        trace = {
            'question': question,
            'start_nodes': start_nodes,
            'max_hops': max_hops,
            'ground_truth': ground_truth,
            'hops': []
        }

        # Initialize frontier
        initial_frontier = {}
        for start_node in start_nodes:
            if start_node in self.graph:
                initial_frontier[start_node] = {'path': [], 'score': 0.0}

        active_paths = [{
            'relations': [],
            'frontier': initial_frontier,
            'hop_scores': [],
        }]

        # Search hop-by-hop
        for current_hop in range(max_hops):
            hop_trace = self._simulate_hop(
                question=question,
                active_paths=active_paths,
                current_hop=current_hop,
                max_hops=max_hops,
                ground_truth=ground_truth
            )
            trace['hops'].append(hop_trace)

            # Update active paths
            active_paths = hop_trace['next_paths']

            if len(active_paths) == 0:
                break

        # Get final answers
        if len(active_paths) > 0:
            # Calculate aggregate score for each path
            # FIXED: Use multiplicative (chain) scoring instead of average
            for path in active_paths:
                if len(path['hop_scores']) > 0:
                    # Multiplicative: score = hop1 * hop2 * hop3
                    path['chain_score'] = 1.0
                    for hop_score in path['hop_scores']:
                        path['chain_score'] *= hop_score

                    # Also keep average for comparison
                    path['avg_score'] = sum(path['hop_scores']) / len(path['hop_scores'])
                else:
                    path['chain_score'] = 0.0
                    path['avg_score'] = 0.0

            # Select best path using chain score
            best_path = max(
                active_paths,
                key=lambda p: (p['chain_score'], p['hop_scores'][0] if len(p['hop_scores']) > 0 else 0.0)
            )

            trace['predicted_answers'] = list(best_path['frontier'].keys())
            trace['best_path_relations'] = best_path['relations']
            trace['best_path_chain_score'] = best_path['chain_score']
            trace['best_path_avg_score'] = best_path['avg_score']
        else:
            trace['predicted_answers'] = []
            trace['best_path_relations'] = []
            trace['best_path_chain_score'] = 0.0
            trace['best_path_avg_score'] = 0.0

        # Compute metrics
        predicted = set(trace['predicted_answers'])
        gt = set(ground_truth)
        correct = len(predicted & gt)

        trace['num_predicted'] = len(predicted)
        trace['num_correct'] = correct
        trace['num_incorrect'] = len(predicted - gt)
        trace['num_missed'] = len(gt - predicted)

        trace['precision'] = correct / len(predicted) if predicted else 0.0
        trace['recall'] = correct / len(gt) if gt else 0.0

        if trace['precision'] + trace['recall'] > 0:
            trace['f1'] = 2 * trace['precision'] * trace['recall'] / (trace['precision'] + trace['recall'])
        else:
            trace['f1'] = 0.0

        return trace

    def _simulate_hop(
        self,
        question: str,
        active_paths: List[Dict],
        current_hop: int,
        max_hops: int,
        ground_truth: List[str]
    ) -> Dict:
        """Simulate one hop of BFS"""

        hop_trace = {
            'hop_number': current_hop + 1,
            'num_active_paths': len(active_paths),
            'path_expansions': [],
            'next_paths': []
        }

        for path_idx, path_candidate in enumerate(active_paths):
            frontier_nodes = list(path_candidate['frontier'].keys())

            if len(frontier_nodes) == 0:
                continue

            # Collect ALL edges from this path's frontier
            all_edges = []  # [(source_node, relation, target)]
            edge_to_source = {}  # Track which source node each edge came from

            for node in frontier_nodes:
                # Outgoing edges
                for _, succ, key, data in self.graph.out_edges(node, keys=True, data=True):
                    relation = data.get('relation', 'unknown')
                    edge = (node, relation, succ)
                    all_edges.append(edge)
                    edge_to_source[edge] = node

                # Incoming edges (reversed)
                for pred, _, key, data in self.graph.in_edges(node, keys=True, data=True):
                    relation = data.get('relation', 'unknown')
                    relation_reversed = f"{relation}_reversed"
                    edge = (node, relation_reversed, pred)
                    all_edges.append(edge)
                    edge_to_source[edge] = node

            if len(all_edges) == 0:
                continue

            # Score all edges
            model_scores = self.edge_scorer.score_edges_batch(
                question=question,
                edges=all_edges,
                hop=current_hop,
                question_hop_count=max_hops
            )

            # Compute text similarities
            text_sims = self._compute_text_similarities(question, all_edges)

            # Create detailed edge info
            edge_details = []
            for (source, relation, target), model_score, text_sim in zip(all_edges, model_scores, text_sims):
                # Check if target is in ground truth
                in_gt = target in ground_truth

                edge_details.append({
                    'source': source,
                    'relation': relation,
                    'target': target,
                    'model_score': float(model_score),
                    'text_similarity': float(text_sim),
                    'in_ground_truth': in_gt
                })

            # Group by relation and rank
            relation_groups = defaultdict(list)
            for edge_info in edge_details:
                relation_groups[edge_info['relation']].append(edge_info)

            # Rank relations by best edge score
            relation_best_scores = {
                rel: max(edges_list, key=lambda x: x['model_score'])['model_score']
                for rel, edges_list in relation_groups.items()
            }

            top_relations = sorted(
                relation_best_scores.keys(),
                key=lambda r: relation_best_scores[r],
                reverse=True
            )[:self.top_k_relations]

            # Build expansion trace
            expansion_trace = {
                'path_index': path_idx,
                'previous_relations': path_candidate['relations'],
                'frontier_size': len(frontier_nodes),
                'frontier_nodes': frontier_nodes[:5],  # Show first 5
                'total_edges': len(all_edges),
                'relation_groups': {},
                'selected_relations': top_relations,
                'new_paths_created': 0
            }

            # For each relation group, show statistics
            for relation in relation_groups.keys():
                edges = relation_groups[relation]
                scores = [e['model_score'] for e in edges]
                text_sims = [e['text_similarity'] for e in edges]
                in_gt_count = sum(1 for e in edges if e['in_ground_truth'])

                expansion_trace['relation_groups'][relation] = {
                    'num_edges': len(edges),
                    'selected': relation in top_relations,
                    'best_score': float(max(scores)),
                    'avg_score': float(np.mean(scores)),
                    'avg_text_sim': float(np.mean(text_sims)),
                    'edges_to_ground_truth': in_gt_count,
                    'top_edges': sorted(edges, key=lambda x: x['model_score'], reverse=True)[:5]
                }

            hop_trace['path_expansions'].append(expansion_trace)

            # Expand selected relations
            for relation in top_relations:
                edges_for_relation = relation_groups[relation]

                # Sort by score and take top nodes
                edges_for_relation.sort(key=lambda x: x['model_score'], reverse=True)
                top_edges = edges_for_relation[:self.max_nodes_per_relation]

                # Build new frontier
                new_frontier = {}
                hop_scores = []

                for edge_info in top_edges:
                    target = edge_info['target']
                    score = edge_info['model_score']
                    source = edge_info['source']

                    # Get path to source
                    source_path = path_candidate['frontier'][source]['path']
                    new_path = source_path + [(relation, target, score)]

                    # De-duplicate: keep best score for each target
                    if target not in new_frontier or score > new_frontier[target]['score']:
                        new_frontier[target] = {
                            'path': new_path,
                            'score': score
                        }
                        hop_scores.append(score)

                if len(new_frontier) == 0:
                    continue

                # Calculate average score for this hop
                avg_hop_score = sum(hop_scores) / len(hop_scores)

                # Create new path candidate
                new_path_candidate = {
                    'relations': path_candidate['relations'] + [relation],
                    'frontier': new_frontier,
                    'hop_scores': path_candidate['hop_scores'] + [avg_hop_score],
                }

                hop_trace['next_paths'].append(new_path_candidate)
                expansion_trace['new_paths_created'] += 1

        return hop_trace

    def _compute_text_similarities(
        self,
        question: str,
        edges: List[Tuple[str, str, str]]
    ) -> List[float]:
        """Compute text similarity between question and each relation"""
        if question not in self.text_embeddings:
            return [0.0] * len(edges)

        q_emb = self.text_embeddings[question]

        similarities = []
        for source, relation, target in edges:
            if relation in self.text_embeddings:
                rel_emb = self.text_embeddings[relation]
                sim = cosine_similarity(q_emb, rel_emb)
                # Normalize to [0, 1]
                sim_normalized = (sim + 1.0) / 2.0
                similarities.append(sim_normalized)
            else:
                similarities.append(0.0)

        return similarities


def print_trace(trace: Dict):
    """Pretty print the simulation trace"""

    print(f"\n{'='*100}")
    print(f"SIMULATION TRACE")
    print(f"{'='*100}")

    print(f"\nQuestion: {trace['question']}")
    print(f"Start nodes: {trace['start_nodes']}")
    print(f"Max hops: {trace['max_hops']}")
    print(f"Ground truth answers: {len(trace['ground_truth'])} answers")
    print(f"  Sample: {trace['ground_truth'][:5]}")

    # Print each hop
    for hop_trace in trace['hops']:
        print(f"\n{'-'*100}")
        print(f"HOP {hop_trace['hop_number']}")
        print(f"{'-'*100}")

        print(f"Active paths: {hop_trace['num_active_paths']}")

        for expansion in hop_trace['path_expansions']:
            print(f"\n  Path #{expansion['path_index']}:")
            print(f"    Previous relations: {' -> '.join(expansion['previous_relations']) if expansion['previous_relations'] else 'START'}")
            print(f"    Frontier: {expansion['frontier_size']} nodes")
            print(f"    Total candidate edges: {expansion['total_edges']}")

            # Show relation groups
            print(f"\n    Relation Groups (Top 10 by best score):")
            print(f"    {'Relation':<35} {'Selected':<10} {'Edges':<8} {'Best':<8} {'Avg':<8} {'Text':<8} {'GT':<5}")
            print(f"    {'-'*95}")

            sorted_relations = sorted(
                expansion['relation_groups'].items(),
                key=lambda x: x[1]['best_score'],
                reverse=True
            )[:10]

            for relation, info in sorted_relations:
                selected = "✓ YES" if info['selected'] else "✗ no"
                print(f"    {relation:<35} {selected:<10} {info['num_edges']:<8} "
                      f"{info['best_score']:<8.4f} {info['avg_score']:<8.4f} "
                      f"{info['avg_text_sim']:<8.4f} {info['edges_to_ground_truth']:<5}")

            # Show top edges for selected relations
            print(f"\n    Selected Relations (Top-{len(expansion['selected_relations'])}):")
            for relation in expansion['selected_relations']:
                info = expansion['relation_groups'][relation]
                print(f"\n      {relation}:")
                print(f"        Top 3 edges:")
                print(f"        {'Target':<40} {'Model':<10} {'Text':<10} {'In GT':<8}")
                print(f"        {'-'*70}")

                for edge in info['top_edges'][:3]:
                    in_gt = "✓" if edge['in_ground_truth'] else "✗"
                    target_short = edge['target'][:37] + "..." if len(edge['target']) > 40 else edge['target']
                    print(f"        {target_short:<40} {edge['model_score']:<10.4f} "
                          f"{edge['text_similarity']:<10.4f} {in_gt:<8}")

            print(f"\n    New paths created: {expansion['new_paths_created']}")

    # Final results
    print(f"\n{'-'*100}")
    print(f"FINAL RESULTS")
    print(f"{'-'*100}")

    print(f"\nBest path: {' -> '.join(trace['best_path_relations'])}")
    print(f"Best path chain score: {trace.get('best_path_chain_score', 0.0):.4f}")
    print(f"Best path average score: {trace['best_path_avg_score']:.4f}")
    print(f"\nPredicted answers: {trace['num_predicted']}")
    print(f"  Sample: {trace['predicted_answers'][:5]}")

    print(f"\nMetrics:")
    print(f"  Correct: {trace['num_correct']}")
    print(f"  Incorrect: {trace['num_incorrect']}")
    print(f"  Missed: {trace['num_missed']}")
    print(f"  Precision: {trace['precision']:.4f}")
    print(f"  Recall: {trace['recall']:.4f}")
    print(f"  F1: {trace['f1']:.4f}")

    print(f"\n{'='*100}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Simulate Variant 3 EdgeScorer BFS decisions')
    parser.add_argument(
        'results_file',
        type=str,
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--num-perfect',
        type=int,
        default=10,
        help='Number of perfect cases (F1=1.0) to simulate'
    )
    parser.add_argument(
        '--num-failures',
        type=int,
        default=10,
        help='Number of failure cases (F1=0.0) to simulate'
    )
    parser.add_argument(
        '--num-low-f1',
        type=int,
        default=10,
        help='Number of low F1 cases (0<F1<0.5) to simulate'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Top-K relations to expand per hop'
    )
    parser.add_argument(
        '--max-nodes',
        type=int,
        default=30,
        help='Max nodes per relation'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for traces (JSON)'
    )

    args = parser.parse_args()

    print(f"{'='*100}")
    print(f"Variant 3 EdgeScorer BFS Simulation")
    print(f"{'='*100}\n")

    # Load results file
    print(f"Loading results from {args.results_file}...")
    with open(args.results_file, 'r') as f:
        results_data = json.load(f)

    results = results_data['results']

    # Categorize by F1
    perfect = [r for r in results if r['f1_score'] == 1.0]
    failures = [r for r in results if r['f1_score'] == 0.0]
    low_f1 = [r for r in results if 0 < r['f1_score'] < 0.5]
    partial = [r for r in results if 0.5 <= r['f1_score'] < 1.0]

    print(f"  Total questions: {len(results)}")
    print(f"  Perfect (F1=1.0): {len(perfect)}")
    print(f"  Failures (F1=0.0): {len(failures)}")
    print(f"  Low F1 (0<F1<0.5): {len(low_f1)}")
    print(f"  Partial (0.5≤F1<1): {len(partial)}")

    # Select cases to simulate
    cases_to_simulate = []
    cases_to_simulate.extend(perfect[:args.num_perfect])
    cases_to_simulate.extend(failures[:args.num_failures])
    cases_to_simulate.extend(low_f1[:args.num_low_f1])

    print(f"\nSimulating {len(cases_to_simulate)} cases...")
    print(f"  Perfect: {min(args.num_perfect, len(perfect))}")
    print(f"  Failures: {min(args.num_failures, len(failures))}")
    print(f"  Low F1: {min(args.num_low_f1, len(low_f1))}")

    # Load resources
    print(f"\n[1/4] Loading knowledge graph...")
    graph = load_graph(Config.GRAPH_PATH)
    print(f"  Nodes: {graph.number_of_nodes():,}")
    print(f"  Edges: {graph.number_of_edges():,}")

    print(f"\n[2/4] Loading text embeddings...")
    cache_path = Config.EMBEDDINGS_DIR / "variant3_text_embeddings_cache.pkl"
    with open(cache_path, 'rb') as f:
        text_embeddings = pickle.load(f)
    print(f"  Loaded {len(text_embeddings):,} embeddings")

    print(f"\n[3/4] Loading EdgeScorer model...")
    model_path = Config.BASE_DIR / "models" / "variant3_edge_scorer_dual_best.pt"
    edge_scorer = EdgeScorerRelationRanker(str(model_path))

    print(f"\n[4/4] Loading entity linker...")
    # Load the actual node2id.json that was used for training
    node2id_path = Config.EMBEDDINGS_DIR / "node2id.json"
    with open(node2id_path, 'r') as f:
        node2id = json.load(f)
    entity_linker = ExactMatcher(node2id)
    print(f"  Loaded {len(node2id):,} entity mappings")

    # Create simulator
    simulator = BFSSimulator(
        graph=graph,
        edge_scorer=edge_scorer,
        text_embeddings=text_embeddings,
        top_k_relations=args.top_k,
        max_nodes_per_relation=args.max_nodes
    )

    print(f"\n{'='*100}")
    print(f"Running simulations...")
    print(f"{'='*100}\n")

    all_traces = []

    for i, case in enumerate(cases_to_simulate, 1):
        print(f"\n{'#'*100}")
        print(f"CASE {i}/{len(cases_to_simulate)} - F1={case['f1_score']:.4f}")
        print(f"{'#'*100}")

        # Extract question info
        question_text = case['question_text']
        ground_truth = case['ground_truth_answers']

        # Entity linking - ExactMatcher.extract_and_link() extracts [brackets] and links
        start_nodes = entity_linker.extract_and_link(question_text)

        if not start_nodes:
            print(f"⚠ Could not link entity from question: {question_text}")
            continue

        # Determine hop count from question text
        if '2-hop' in args.results_file or 'movies directed by the director of' in question_text.lower():
            max_hops = 2
        elif '3-hop' in args.results_file or 'movies starred by' in question_text.lower():
            max_hops = 3
        else:
            max_hops = 1

        # Run simulation
        trace = simulator.simulate_search(
            question=question_text,
            start_nodes=start_nodes,
            max_hops=max_hops,
            ground_truth=ground_truth
        )

        # Print trace
        print_trace(trace)

        all_traces.append(trace)

    # Save traces if output specified
    if args.output:
        print(f"\nSaving traces to {args.output}...")
        with open(args.output, 'w') as f:
            json.dump(all_traces, f, indent=2)
        print(f"Saved {len(all_traces)} traces")

    print(f"\n{'='*100}")
    print(f"Simulation complete!")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
