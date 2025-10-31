"""
EdgeScorer-guided Beam Search for Variant 3.

Uses the trained EdgeScorerDual model to score and select edges during BFS.
"""

import networkx as nx
from typing import List, Tuple, Set
from collections import deque
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from qa_system.core.base_search import BaseSearch
from qa_system.core.search_result import SearchResult
from qa_system.core.question import Question
from variant3.eval_qa.edge_scorer_ranker import EdgeScorerRelationRanker


class EdgeScorerBFS(BaseSearch):
    """
    Beam search guided by EdgeScorerDual model.

    At each hop, scores all candidate edges from the current frontier
    using the trained EdgeScorer model, then expands only the top-k
    highest-scoring edges.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        edge_scorer: EdgeScorerRelationRanker,
        top_k_relations: int = 3,
        max_nodes_per_relation: int = 30,
        score_threshold: float = 0.0
    ):
        """
        Initialize EdgeScorer-guided BFS with global path selection.

        Args:
            graph: Knowledge graph
            edge_scorer: Trained EdgeScorerRelationRanker
            top_k_relations: Number of top relations to explore per hop (default: 3)
            max_nodes_per_relation: Maximum nodes to expand per relation (default: 30)
            score_threshold: Minimum score to consider an edge (default: 0.0)
        """
        super().__init__(graph)
        self.edge_scorer = edge_scorer
        self.top_k_relations = top_k_relations
        self.max_nodes_per_relation = max_nodes_per_relation
        self.score_threshold = score_threshold

    def search(
        self,
        question: Question,
        start_nodes: List[str],
        target_relations: List[Tuple[str, float]],  # Not used, kept for interface compatibility
        max_hops: int
    ) -> SearchResult:
        """
        Perform global path selection search guided by EdgeScorer.

        Explores top-K relations at each hop, then selects the best complete path.

        Args:
            question: Question object
            start_nodes: Starting entity nodes
            target_relations: Ignored (EdgeScorer doesn't need pre-ranked relations)
            max_hops: Maximum search depth

        Returns:
            SearchResult with predicted answers and paths
        """
        start_time = time.time()
        self.reset_metrics()

        question_text = question.text
        question_id = question.question_id

        # Initialize paths: each path tracks relation sequence and frontier nodes
        # PathCandidate = {
        #     'relations': List[str],         # Sequence of relations taken
        #     'frontier': Dict[str, dict],    # node -> {'path': [...], 'score': float}
        #     'hop_scores': List[float],      # Average score at each hop
        # }

        initial_frontier = {}
        for start_node in start_nodes:
            if start_node in self.graph:
                initial_frontier[start_node] = {'path': [], 'score': 0.0}

        active_paths = [{
            'relations': [],
            'frontier': initial_frontier,
            'hop_scores': [],
        }]

        # Expand hop-by-hop
        for current_hop in range(max_hops):
            next_hop = current_hop + 1
            next_paths = []

            for path_candidate in active_paths:
                frontier_nodes = list(path_candidate['frontier'].keys())

                if len(frontier_nodes) == 0:
                    continue

                # Collect ALL edges from this path's frontier
                all_edges = []  # [(source_node, relation, target)]

                for node in frontier_nodes:
                    self.nodes_expanded += 1

                    # Outgoing edges
                    for _, succ, key, data in self.graph.out_edges(node, keys=True, data=True):
                        relation = data.get('relation', 'unknown')
                        all_edges.append((node, relation, succ))

                    # Incoming edges (reversed)
                    for pred, _, key, data in self.graph.in_edges(node, keys=True, data=True):
                        relation = data.get('relation', 'unknown')
                        relation_reversed = f"{relation}_reversed"
                        all_edges.append((node, relation_reversed, pred))

                if len(all_edges) == 0:
                    continue

                # Score all edges
                scores = self.edge_scorer.score_edges_batch(
                    question=question_text,
                    edges=all_edges,
                    hop=current_hop
                )

                # Group edges by relation and find top-K relations
                relation_groups = {}  # relation -> [(edge, score), ...]

                for (source, relation, target), score in zip(all_edges, scores):
                    if score < self.score_threshold:
                        continue

                    if relation not in relation_groups:
                        relation_groups[relation] = []

                    relation_groups[relation].append({
                        'source': source,
                        'target': target,
                        'score': score
                    })

                if len(relation_groups) == 0:
                    continue

                # Rank relations by best edge score
                relation_best_scores = {
                    rel: max(edges_list, key=lambda x: x['score'])['score']
                    for rel, edges_list in relation_groups.items()
                }

                top_relations = sorted(
                    relation_best_scores.keys(),
                    key=lambda r: relation_best_scores[r],
                    reverse=True
                )[:self.top_k_relations]

                # Expand each top-K relation
                for relation in top_relations:
                    edges_for_relation = relation_groups[relation]

                    # Sort edges by score and take top max_nodes_per_relation
                    edges_for_relation.sort(key=lambda x: x['score'], reverse=True)
                    top_edges = edges_for_relation[:self.max_nodes_per_relation]

                    # Build new frontier
                    new_frontier = {}
                    hop_scores = []

                    for edge_info in top_edges:
                        target = edge_info['target']
                        score = edge_info['score']
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

                    next_paths.append(new_path_candidate)

            active_paths = next_paths

        # Select best path based on average score across all hops
        if len(active_paths) == 0:
            # No paths found
            end_time = time.time()
            search_time_ms = (end_time - start_time) * 1000

            return SearchResult(
                question_id=question_id,
                question_text=question_text,
                predicted_answers=[],
                ground_truth_answers=question.ground_truth_answers if hasattr(question, 'ground_truth_answers') else [],
                nodes_expanded=self.nodes_expanded,
                search_time_ms=search_time_ms,
                success=False,
                relations_used=[]
            )

        # Calculate aggregate score for each path
        for path in active_paths:
            if len(path['hop_scores']) > 0:
                path['avg_score'] = sum(path['hop_scores']) / len(path['hop_scores'])
            else:
                path['avg_score'] = 0.0

        # Select best path: highest average score, tie-break by hop 0 score
        best_path = max(
            active_paths,
            key=lambda p: (p['avg_score'], p['hop_scores'][0] if len(p['hop_scores']) > 0 else 0.0)
        )

        # Extract answers from best path's frontier
        predicted_answers = list(best_path['frontier'].keys())

        # Compile results
        end_time = time.time()
        search_time_ms = (end_time - start_time) * 1000

        ground_truth = question.ground_truth_answers if hasattr(question, 'ground_truth_answers') else []

        result = SearchResult(
            question_id=question_id,
            question_text=question_text,
            predicted_answers=predicted_answers,
            ground_truth_answers=ground_truth,
            nodes_expanded=self.nodes_expanded,
            search_time_ms=search_time_ms,
            success=len(predicted_answers) > 0,
            relations_used=best_path['relations']
        )

        return result


if __name__ == "__main__":
    """Test the search algorithm"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from qa_system.config import Config
    from qa_system.knowledge_graph.graph_loader import load_graph

    print("=" * 80)
    print("EdgeScorerBFS Test")
    print("=" * 80)

    # Load graph
    print("\n[1/3] Loading knowledge graph...")
    graph = load_graph(Config.GRAPH_PATH)
    print(f"   Nodes: {graph.number_of_nodes():,}")
    print(f"   Edges: {graph.number_of_edges():,}")

    # Load edge scorer
    print("\n[2/3] Loading EdgeScorer...")
    model_path = Config.BASE_DIR / "models" / "variant3_edge_scorer_dual_best.pt"
    edge_scorer = EdgeScorerRelationRanker(str(model_path))

    # Create search algorithm
    search = EdgeScorerBFS(
        graph=graph,
        edge_scorer=edge_scorer,
        top_k_relations=3,
        max_nodes_per_relation=30,
        score_threshold=0.0
    )

    print("\n[3/3] Testing search...")
    # Test question
    question = Question(
        id=1,
        text="What movies did the director of The Matrix direct?",
        answers=["The Matrix", "The Matrix Reloaded", "The Matrix Revolutions"]
    )

    start_nodes = ["The Matrix"]
    max_hops = 2

    print(f"   Question: {question.text}")
    print(f"   Start nodes: {start_nodes}")
    print(f"   Max hops: {max_hops}")

    # Search
    result = search.search(
        question=question,
        start_nodes=start_nodes,
        target_relations=[],  # Not used by EdgeScorer
        max_hops=max_hops
    )

    print(f"\n[Results]")
    print(f"   Predicted answers: {len(result.predicted_answers)}")
    print(f"   Ground truth: {len(result.ground_truth)}")
    print(f"   Nodes expanded: {result.nodes_expanded}")
    print(f"   Search time: {result.search_time_ms:.2f} ms")
    print(f"   Relations used: {len(result.relations_used)}")

    # Show predicted answers
    print(f"\n   Top 10 predicted answers:")
    for i, answer in enumerate(result.predicted_answers[:10], 1):
        in_gt = "✓" if answer in result.ground_truth else "✗"
        print(f"      {i}. {answer} {in_gt}")

    # Show sample paths
    if 'sample_paths' in result.metadata:
        print(f"\n   Sample paths:")
        for path_info in result.metadata['sample_paths'][:3]:
            print(f"      Answer: {path_info['answer']} (score: {path_info['score']:.4f})")
            for relation, target, score in path_info['path']:
                print(f"         --[{relation}]--> {target} (score: {score:.4f})")

    # Compute metrics
    correct = len(set(result.predicted_answers) & set(result.ground_truth))
    precision = correct / len(result.predicted_answers) if result.predicted_answers else 0
    recall = correct / len(result.ground_truth) if result.ground_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n   Metrics:")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall: {recall:.4f}")
    print(f"      F1: {f1:.4f}")

    print("\n" + "=" * 80)
    print("Test passed!")
    print("=" * 80)
