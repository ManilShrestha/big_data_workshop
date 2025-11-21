"""
Hybrid Edge Scorer for Variant 3 QA Evaluation.

Combines model-based edge scoring with text similarity between question and relations.
This addresses the limitation where the model sometimes confuses similar relations
(e.g., directed_by vs written_by vs starred_actors).

Hybrid Formula:
    hybrid_score = alpha * model_score + (1 - alpha) * text_similarity

Where alpha is a hop-dependent weight that can be:
- Static (configured per hop)
- Adaptive (based on model confidence)
- Learned (from validation data)

Phase 1: Static hop-dependent weights
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .edge_scorer_ranker import EdgeScorerRelationRanker


class HybridEdgeScorer:
    """
    Wrapper around EdgeScorerRelationRanker that adds text similarity component.

    Fully backward compatible: when enable_hybrid=False, behaves identically
    to the base scorer.
    """

    def __init__(
        self,
        base_scorer: EdgeScorerRelationRanker,
        enable_hybrid: bool = False,
        static_alphas: Optional[Dict[int, float]] = None,
        normalize_text_sim: bool = True,
        verbose: bool = False
    ):
        """
        Initialize Hybrid Edge Scorer.

        Args:
            base_scorer: The trained EdgeScorerRelationRanker model
            enable_hybrid: If False, acts as pass-through (backward compatible)
            static_alphas: Hop-specific alpha weights {hop: alpha}
                          e.g., {1: 0.7, 2: 0.5, 3: 0.3}
                          Higher alpha = trust model more
                          Lower alpha = trust text similarity more
            normalize_text_sim: If True, normalize text similarities to [0, 1]
            verbose: Print debugging information
        """
        self.base_scorer = base_scorer
        self.enable_hybrid = enable_hybrid
        self.verbose = verbose

        # Default alpha values (designed for 3-hop questions)
        # Hop 1: Trust model more for broad exploration
        # Hop 2: Balanced between model and text
        # Hop 3: Trust text similarity more for precise matching
        self.static_alphas = static_alphas or {
            1: 0.7,  # 70% model, 30% text similarity
            2: 0.7,  # 30% model, 70% text similarity
            3: 0.7   # 80% model, 20% text similarity
        }

        self.normalize_text_sim = normalize_text_sim

        # Statistics tracking
        self.stats = {
            'total_batches': 0,
            'total_edges': 0,
            'avg_model_score': 0.0,
            'avg_text_sim': 0.0,
            'avg_hybrid_score': 0.0,
            'per_hop_stats': {}
        }

        if self.verbose:
            status = "ENABLED" if self.enable_hybrid else "DISABLED"
            print(f"[HybridEdgeScorer] Initialized ({status})")
            if self.enable_hybrid:
                print(f"   Alpha weights: {self.static_alphas}")

    def score_edges_batch(
        self,
        question: str,
        edges: List[Tuple[str, str, str]],
        hop: int,
        question_hop_count: int = None
    ) -> List[float]:
        """
        Score a batch of edges using hybrid model + text similarity.

        Args:
            question: Question text
            edges: List of (current_node, relation, target_node) tuples
            hop: Current hop (0-indexed: 0, 1, 2)
            question_hop_count: Total hop count (1, 2, or 3)

        Returns:
            List of scores for each edge
        """
        if len(edges) == 0:
            return []

        # Get model scores (always computed)
        model_scores = self.base_scorer.score_edges_batch(
            question=question,
            edges=edges,
            hop=hop,
            question_hop_count=question_hop_count
        )

        # If hybrid disabled, return model scores directly (backward compatible)
        if not self.enable_hybrid:
            return model_scores

        # Compute text similarities
        text_sims = self._compute_text_similarities(question, edges)

        # Get alpha for this hop (1-indexed)
        hop_1indexed = hop + 1
        alpha = self.static_alphas.get(hop_1indexed, 0.5)

        # Combine scores
        hybrid_scores = [
            alpha * model_score + (1 - alpha) * text_sim
            for model_score, text_sim in zip(model_scores, text_sims)
        ]

        # Update statistics
        self._update_stats(hop_1indexed, model_scores, text_sims, hybrid_scores)

        if self.verbose and self.stats['total_batches'] % 100 == 0:
            self._print_stats()

        return hybrid_scores

    def _compute_text_similarities(
        self,
        question: str,
        edges: List[Tuple[str, str, str]]
    ) -> List[float]:
        """
        Compute cosine similarity between question and each edge's relation.

        Args:
            question: Question text
            edges: List of (current_node, relation, target_node) tuples

        Returns:
            List of cosine similarities [0, 1]
        """
        # Get question embedding (once for all edges)
        question_emb = self.base_scorer._get_text_embedding(question)

        similarities = []
        for current_node, relation, target_node in edges:
            # Get relation embedding
            relation_emb = self.base_scorer._get_text_embedding(relation)

            # Compute cosine similarity
            cos_sim = self._cosine_similarity(question_emb, relation_emb)

            # Normalize to [0, 1] range if requested
            # Cosine similarity is [-1, 1], we shift to [0, 1]
            if self.normalize_text_sim:
                cos_sim = (cos_sim + 1.0) / 2.0

            similarities.append(cos_sim)

        return similarities

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Returns:
            Similarity in [-1, 1] range
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _update_stats(
        self,
        hop: int,
        model_scores: List[float],
        text_sims: List[float],
        hybrid_scores: List[float]
    ):
        """Update running statistics"""
        batch_size = len(model_scores)

        self.stats['total_batches'] += 1
        self.stats['total_edges'] += batch_size

        # Update global averages
        n = self.stats['total_edges']
        self.stats['avg_model_score'] = (
            (self.stats['avg_model_score'] * (n - batch_size) + sum(model_scores))
            / n
        )
        self.stats['avg_text_sim'] = (
            (self.stats['avg_text_sim'] * (n - batch_size) + sum(text_sims))
            / n
        )
        self.stats['avg_hybrid_score'] = (
            (self.stats['avg_hybrid_score'] * (n - batch_size) + sum(hybrid_scores))
            / n
        )

        # Per-hop statistics
        if hop not in self.stats['per_hop_stats']:
            self.stats['per_hop_stats'][hop] = {
                'count': 0,
                'avg_model': 0.0,
                'avg_text': 0.0,
                'avg_hybrid': 0.0
            }

        hop_stats = self.stats['per_hop_stats'][hop]
        prev_count = hop_stats['count']
        new_count = prev_count + batch_size

        hop_stats['avg_model'] = (
            (hop_stats['avg_model'] * prev_count + sum(model_scores))
            / new_count
        )
        hop_stats['avg_text'] = (
            (hop_stats['avg_text'] * prev_count + sum(text_sims))
            / new_count
        )
        hop_stats['avg_hybrid'] = (
            (hop_stats['avg_hybrid'] * prev_count + sum(hybrid_scores))
            / new_count
        )
        hop_stats['count'] = new_count

    def _print_stats(self):
        """Print current statistics"""
        print(f"\n[HybridEdgeScorer] Statistics after {self.stats['total_batches']} batches:")
        print(f"   Total edges scored: {self.stats['total_edges']:,}")
        print(f"   Avg model score: {self.stats['avg_model_score']:.4f}")
        print(f"   Avg text similarity: {self.stats['avg_text_sim']:.4f}")
        print(f"   Avg hybrid score: {self.stats['avg_hybrid_score']:.4f}")

        print(f"\n   Per-hop breakdown:")
        for hop in sorted(self.stats['per_hop_stats'].keys()):
            hop_stats = self.stats['per_hop_stats'][hop]
            alpha = self.static_alphas.get(hop, 0.5)
            print(f"      Hop {hop} (α={alpha:.1f}): "
                  f"model={hop_stats['avg_model']:.3f}, "
                  f"text={hop_stats['avg_text']:.3f}, "
                  f"hybrid={hop_stats['avg_hybrid']:.3f} "
                  f"({hop_stats['count']:,} edges)")

    def get_statistics(self) -> Dict:
        """Get current statistics as dictionary"""
        return self.stats.copy()

    def print_final_stats(self):
        """Print final statistics summary"""
        print(f"\n{'='*80}")
        print(f"[HybridEdgeScorer] Final Statistics")
        print(f"{'='*80}")
        print(f"Total batches processed: {self.stats['total_batches']:,}")
        print(f"Total edges scored: {self.stats['total_edges']:,}")
        print(f"\nGlobal Averages:")
        print(f"   Model score:      {self.stats['avg_model_score']:.4f}")
        print(f"   Text similarity:  {self.stats['avg_text_sim']:.4f}")
        print(f"   Hybrid score:     {self.stats['avg_hybrid_score']:.4f}")

        print(f"\nPer-Hop Breakdown:")
        for hop in sorted(self.stats['per_hop_stats'].keys()):
            hop_stats = self.stats['per_hop_stats'][hop]
            alpha = self.static_alphas.get(hop, 0.5)
            print(f"\n   Hop {hop} (alpha = {alpha:.2f}):")
            print(f"      Edges scored:   {hop_stats['count']:,}")
            print(f"      Avg model:      {hop_stats['avg_model']:.4f}")
            print(f"      Avg text sim:   {hop_stats['avg_text']:.4f}")
            print(f"      Avg hybrid:     {hop_stats['avg_hybrid']:.4f}")
            print(f"      Weight ratio:   {alpha:.0%} model, {(1-alpha):.0%} text")

        print(f"{'='*80}\n")

    # Pass-through methods for interface compatibility

    def rank_relations(self, question: str, top_k: int = None):
        """Pass-through to base scorer"""
        return self.base_scorer.rank_relations(question, top_k)

    def get_cost(self) -> float:
        """Pass-through to base scorer"""
        return self.base_scorer.get_cost()

    def get_last_query_cost(self) -> float:
        """Pass-through to base scorer"""
        return self.base_scorer.get_last_query_cost()


if __name__ == "__main__":
    """Test the hybrid scorer"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from qa_system.config import Config

    print("=" * 80)
    print("HybridEdgeScorer Test")
    print("=" * 80)

    # Load base model
    model_path = Config.BASE_DIR / "models" / "variant3_edge_scorer_dual_best.pt"
    print(f"\nLoading base scorer from {model_path}...")
    base_scorer = EdgeScorerRelationRanker(str(model_path))

    # Create hybrid scorer (enabled)
    print(f"\nCreating hybrid scorer...")
    hybrid_scorer = HybridEdgeScorer(
        base_scorer=base_scorer,
        enable_hybrid=True,
        static_alphas={1: 0.5, 2: 0.5, 3: 0.5},
        verbose=True
    )

    # Test question
    question = "What movies were written by the writer of Little Odessa?"

    # Test edges
    test_edges = [
        ("Little Odessa", "written_by", "James Gray"),
        ("Little Odessa", "directed_by", "James Gray"),
        ("Little Odessa", "starred_actors", "Tim Roth"),
        ("Little Odessa", "has_genre", "Drama"),
    ]

    print(f"\n[Test] Scoring {len(test_edges)} edges for hop 0...")
    print(f"   Question: {question}")
    print(f"   Edges:")
    for node, rel, target in test_edges:
        print(f"      {node} --[{rel}]--> {target}")

    # Score with model only (disable hybrid)
    hybrid_scorer.enable_hybrid = False
    model_scores = hybrid_scorer.score_edges_batch(question, test_edges, hop=0, question_hop_count=3)

    # Score with hybrid
    hybrid_scorer.enable_hybrid = True
    hybrid_scores = hybrid_scorer.score_edges_batch(question, test_edges, hop=0, question_hop_count=3)

    print(f"\n[Results Comparison]")
    print(f"{'Relation':<20} {'Model Score':<12} {'Hybrid Score':<12} {'Delta':<10}")
    print(f"{'-'*60}")

    for (node, rel, target), model_score, hybrid_score in zip(test_edges, model_scores, hybrid_scores):
        delta = hybrid_score - model_score
        print(f"{rel:<20} {model_score:>11.4f} {hybrid_score:>11.4f} {delta:>+9.4f}")

    # Show top-ranked by each method
    print(f"\n[Rankings]")

    model_ranked = sorted(zip(test_edges, model_scores), key=lambda x: x[1], reverse=True)
    hybrid_ranked = sorted(zip(test_edges, hybrid_scores), key=lambda x: x[1], reverse=True)

    print(f"\n   Model-only ranking:")
    for i, ((node, rel, target), score) in enumerate(model_ranked, 1):
        print(f"      {i}. {rel:<20} ({score:.4f})")

    print(f"\n   Hybrid ranking:")
    for i, ((node, rel, target), score) in enumerate(hybrid_ranked, 1):
        print(f"      {i}. {rel:<20} ({score:.4f})")

    # Print statistics
    hybrid_scorer.print_final_stats()

    print(f"\n[Cost]")
    print(f"   Total cost: ${hybrid_scorer.get_cost():.6f}")

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)