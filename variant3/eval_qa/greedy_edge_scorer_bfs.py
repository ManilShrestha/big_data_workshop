"""
Variant 3a: Greedy Best-First Search with EdgeScorer

Instead of maintaining multiple paths and aggregating scores,
this uses a greedy strategy:
1. At each hop, find the relation with the highest max score
2. Expand only that relation (top-K nodes)
3. Move forward with a single path

Much simpler, faster, and more interpretable than multi-path aggregation.
"""

import networkx as nx
from typing import List, Tuple, Set
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from qa_system.core.base_search import BaseSearch
from qa_system.core.search_result import SearchResult
from qa_system.core.question import Question
from variant3.eval_qa.edge_scorer_ranker import EdgeScorerRelationRanker


class GreedyEdgeScorerBFS(BaseSearch):
    """
    Greedy best-first search guided by EdgeScorer model.

    At each hop:
    - Score all candidate edges
    - Group by relation, find max score per relation
    - Pick the relation with highest max score
    - Expand top-K nodes from that relation only
    - Continue with single path (no aggregation needed!)
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        edge_scorer: EdgeScorerRelationRanker,
        top_k_relations: int = 1,  # Default 1 for pure greedy
        max_nodes_per_relation: int = 30,
        score_threshold: float = 0.0,
        enable_hybrid: bool = False,
        hybrid_alphas: dict = None
    ):
        """
        Initialize Greedy EdgeScorer-guided BFS.

        Args:
            graph: Knowledge graph
            edge_scorer: Trained EdgeScorerRelationRanker
            top_k_relations: Number of relations to try (default: 1 for pure greedy)
                           Set > 1 to try fallback relations if first fails
            max_nodes_per_relation: Maximum nodes to expand per relation
            score_threshold: Minimum score to consider an edge
            enable_hybrid: Enable hybrid text+model scoring
            hybrid_alphas: Hop-specific alpha weights for hybrid scoring
        """
        super().__init__(graph)

        # BACKWARD COMPATIBLE: Wrap edge scorer with hybrid if enabled
        if enable_hybrid:
            from .hybrid_edge_scorer import HybridEdgeScorer
            self.edge_scorer = HybridEdgeScorer(
                base_scorer=edge_scorer,
                enable_hybrid=True,
                static_alphas=hybrid_alphas,
                verbose=False
            )
            print(f"[GreedyEdgeScorerBFS] Hybrid scoring ENABLED with alphas: {hybrid_alphas or 'default'}")
        else:
            self.edge_scorer = edge_scorer

        self.top_k_relations = top_k_relations
        self.max_nodes_per_relation = max_nodes_per_relation
        self.score_threshold = score_threshold

    def search(
        self,
        question: Question,
        start_nodes: List[str],
        target_relations: List[Tuple[str, float]],  # Not used
        max_hops: int
    ) -> SearchResult:
        """
        Perform greedy best-first search guided by EdgeScorer.

        Args:
            question: Question object
            start_nodes: Starting entity nodes
            target_relations: Ignored (EdgeScorer doesn't need pre-ranked relations)
            max_hops: Maximum search depth

        Returns:
            SearchResult with predicted answers and path taken
        """
        start_time = time.time()
        self.reset_metrics()

        question_text = question.text
        question_id = question.question_id

        # Initialize: current frontier and path
        current_frontier = set()
        for start_node in start_nodes:
            if start_node in self.graph:
                current_frontier.add(start_node)

        if len(current_frontier) == 0:
            # No valid start nodes
            return self._empty_result(question, start_time)

        path_relations = []
        path_scores = []  # Track score at each hop for debugging

        # Greedy search: hop-by-hop
        for current_hop in range(max_hops):
            if len(current_frontier) == 0:
                # Stuck - no nodes to expand
                break

            # Collect ALL edges from current frontier
            all_edges = []
            for node in current_frontier:
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
                # No edges from frontier - stuck
                break

            # Score all edges
            scores = self.edge_scorer.score_edges_batch(
                question=question_text,
                edges=all_edges,
                hop=current_hop,
                question_hop_count=max_hops
            )

            # Group edges by relation and find max score per relation
            relation_max_scores = {}
            relation_edges = {}  # relation -> [(edge, score), ...]

            for edge, score in zip(all_edges, scores):
                if score < self.score_threshold:
                    continue

                source, relation, target = edge

                # Track max score for this relation
                if relation not in relation_max_scores:
                    relation_max_scores[relation] = score
                else:
                    relation_max_scores[relation] = max(relation_max_scores[relation], score)

                # Store edge with score
                if relation not in relation_edges:
                    relation_edges[relation] = []
                relation_edges[relation].append((edge, score))

            if len(relation_max_scores) == 0:
                # No valid relations (all below threshold)
                break

            # Sort relations by max score (descending)
            sorted_relations = sorted(
                relation_max_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Try top-K relations until we find one with valid nodes
            next_frontier = None
            chosen_relation = None
            chosen_max_score = None

            for relation, max_score in sorted_relations[:self.top_k_relations]:
                # Get all edges for this relation, sorted by score
                edges_for_relation = relation_edges[relation]
                edges_for_relation.sort(key=lambda x: x[1], reverse=True)

                # Take top-N nodes (de-duplicated by target)
                seen_targets = set()
                next_nodes = []

                for (source, rel, target), score in edges_for_relation:
                    if target not in seen_targets and len(next_nodes) < self.max_nodes_per_relation:
                        next_nodes.append(target)
                        seen_targets.add(target)

                if len(next_nodes) > 0:
                    # Success! Found a relation with valid expansion
                    next_frontier = set(next_nodes)
                    chosen_relation = relation
                    chosen_max_score = max_score
                    break

            if next_frontier is None or len(next_frontier) == 0:
                # All top-K relations led nowhere - stuck
                break

            # Move forward with chosen relation
            path_relations.append(chosen_relation)
            path_scores.append(chosen_max_score)
            current_frontier = next_frontier

        # Final answers are nodes in the frontier after max_hops
        predicted_answers = list(current_frontier)

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
            relations_used=path_relations
        )

        return result

    def _empty_result(self, question: Question, start_time: float) -> SearchResult:
        """Create empty result when search fails"""
        end_time = time.time()
        search_time_ms = (end_time - start_time) * 1000

        return SearchResult(
            question_id=question.question_id,
            question_text=question.text,
            predicted_answers=[],
            ground_truth_answers=question.ground_truth_answers if hasattr(question, 'ground_truth_answers') else [],
            nodes_expanded=self.nodes_expanded,
            search_time_ms=search_time_ms,
            success=False,
            relations_used=[]
        )


if __name__ == "__main__":
    """Test the greedy search algorithm"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from qa_system.config import Config
    from qa_system.utils.loader import load_graph

    print("=" * 80)
    print("GreedyEdgeScorerBFS Test")
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
    search = GreedyEdgeScorerBFS(
        graph=graph,
        edge_scorer=edge_scorer,
        top_k_relations=1,  # Pure greedy
        max_nodes_per_relation=30,
        score_threshold=0.0
    )

    print("\n[3/3] Testing search...")
    # Test question
    from qa_system.core.question import Question
    question = Question(
        question_id=1,
        text="What movies did the director of The Matrix direct?",
        ground_truth_answers=["The Matrix", "The Matrix Reloaded", "The Matrix Revolutions"]
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
        target_relations=[],
        max_hops=max_hops
    )

    print(f"\n[Results]")
    print(f"   Predicted answers: {len(result.predicted_answers)}")
    print(f"   Ground truth: {len(result.ground_truth_answers)}")
    print(f"   Nodes expanded: {result.nodes_expanded}")
    print(f"   Search time: {result.search_time_ms:.2f} ms")
    print(f"   Path taken: {' -> '.join(result.relations_used)}")

    # Show predicted answers
    print(f"\n   Top 10 predicted answers:")
    for i, answer in enumerate(result.predicted_answers[:10], 1):
        in_gt = "" if answer in result.ground_truth_answers else ""
        print(f"      {i}. {answer} {in_gt}")

    # Compute metrics
    correct = len(set(result.predicted_answers) & set(result.ground_truth_answers))
    precision = correct / len(result.predicted_answers) if result.predicted_answers else 0
    recall = correct / len(result.ground_truth_answers) if result.ground_truth_answers else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n   Metrics:")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall: {recall:.4f}")
    print(f"      F1: {f1:.4f}")

    print("\n" + "=" * 80)
    print("Test passed!")
    print("=" * 80)
