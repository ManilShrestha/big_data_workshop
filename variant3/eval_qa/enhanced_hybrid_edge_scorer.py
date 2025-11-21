"""
Enhanced Hybrid Edge Scorer with Relation-Specific Text Similarity.

Extends the basic HybridEdgeScorer with relation-specific keyword matching
for relations like "directed_by" and "written_by". This addresses cases where
generic cosine similarity may not capture intent-specific keywords.

Key Enhancement:
- Uses keyword/alias matching for specific relations (directed_by, written_by)
- Computes TF-IDF-like scores based on keyword presence in question
- Combines with base hybrid scoring for improved relation selection

Example:
    Question: "What movies did the director of The Matrix direct?"
    Relation: "directed_by"
    Keywords: ["director", "directed", "direct", "directors"]
    → Boost score if keywords found in question
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Optional, Set

# Handle both relative and absolute imports
try:
    from .hybrid_edge_scorer import HybridEdgeScorer
    from .edge_scorer_ranker import EdgeScorerRelationRanker
except ImportError:
    from hybrid_edge_scorer import HybridEdgeScorer
    from edge_scorer_ranker import EdgeScorerRelationRanker


class EnhancedHybridEdgeScorer(HybridEdgeScorer):
    """
    Enhanced hybrid scorer with relation-specific keyword matching.

    This scorer adds an additional layer on top of the base hybrid scorer:
    1. Base model score (EdgeScorer neural network)
    2. Text similarity (cosine similarity between question and relation)
    3. Relation-specific keyword matching (NEW)

    Final score = alpha * model_score + beta * text_sim + gamma * keyword_score
    """

    # Relation-specific keyword aliases
    # Only written_by is monitored for keyword boost
    RELATION_KEYWORDS = {
        "written_by": {
            "keywords": ["writer", "written", "write", "wrote", "writers", "screenplay",
                        "screenwriter", "author", "authored", "penned"],
            "weight": 1.0
        },
    }

    def __init__(
        self,
        base_scorer: EdgeScorerRelationRanker,
        enable_hybrid: bool = False,
        enable_keyword_boost: bool = True,
        static_alphas: Optional[Dict[int, float]] = None,
        keyword_boost_multiplier: float = 2.0,
        normalize_text_sim: bool = True,
        verbose: bool = False
    ):
        """
        Initialize Enhanced Hybrid Edge Scorer.

        Args:
            base_scorer: The trained EdgeScorerRelationRanker model
            enable_hybrid: If False, acts as pass-through (backward compatible)
            enable_keyword_boost: Enable relation-specific keyword matching
            static_alphas: Hop-specific alpha weights for model vs text similarity
            keyword_boost_multiplier: Multiplier for scores when keywords match (e.g., 2.0 or 3.0)
                                     Score will be multiplied by this value when keywords detected
            normalize_text_sim: If True, normalize text similarities to [0, 1]
            verbose: Print debugging information
        """
        super().__init__(
            base_scorer=base_scorer,
            enable_hybrid=enable_hybrid,
            static_alphas=static_alphas,
            normalize_text_sim=normalize_text_sim,
            verbose=verbose
        )

        self.enable_keyword_boost = enable_keyword_boost
        self.keyword_boost_multiplier = keyword_boost_multiplier

        # Compile regex patterns for efficient matching
        self._compile_keyword_patterns()

        # Additional statistics for keyword matching
        self.keyword_stats = {
            'total_keyword_boosts': 0,
            'boosts_per_relation': {},
            'avg_keyword_score': 0.0
        }

        if self.verbose:
            print(f"[EnhancedHybridEdgeScorer] Keyword boost: {'ENABLED' if enable_keyword_boost else 'DISABLED'}")
            if enable_keyword_boost:
                print(f"   Keyword boost multiplier: {keyword_boost_multiplier}x")
                print(f"   Monitored relations: {list(self.RELATION_KEYWORDS.keys())}")

    def _compile_keyword_patterns(self):
        """Compile regex patterns for each relation's keywords."""
        self.keyword_patterns = {}

        for relation, config in self.RELATION_KEYWORDS.items():
            keywords = config["keywords"]
            # Create pattern that matches whole words (case-insensitive)
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.keyword_patterns[relation] = re.compile(pattern, re.IGNORECASE)

    def score_edges_batch(
        self,
        question: str,
        edges: List[Tuple[str, str, str]],
        hop: int,
        question_hop_count: int = None
    ) -> List[float]:
        """
        Score a batch of edges using enhanced hybrid scoring.

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

        # Get base hybrid scores (model + text similarity if enabled)
        base_scores = super().score_edges_batch(
            question=question,
            edges=edges,
            hop=hop,
            question_hop_count=question_hop_count
        )

        # If keyword boost disabled, return base scores
        if not self.enable_keyword_boost:
            return base_scores

        # Check which edges should get keyword boost (directed_by and written_by only)
        boosted_scores = []
        boost_applied_count = 0

        for i, (current_node, relation, target_node) in enumerate(edges):
            base_score = base_scores[i]

            # Only boost written_by relation
            if relation in ["written_by", "written_by_reversed"]:
                # Check if question contains relevant keywords
                base_relation = relation.replace("_reversed", "")
                if base_relation in self.keyword_patterns:
                    pattern = self.keyword_patterns[base_relation]
                    matches = pattern.findall(question)

                    if matches:
                        # Apply multiplicative boost
                        boosted_score = min(1.0, base_score * self.keyword_boost_multiplier)
                        boosted_scores.append(boosted_score)
                        boost_applied_count += 1

                        # Track statistics
                        self._record_boost(relation, base_score, boosted_score)
                    else:
                        boosted_scores.append(base_score)
                else:
                    boosted_scores.append(base_score)
            else:
                # Not a target relation, keep original score
                boosted_scores.append(base_score)

        return boosted_scores

    def _record_boost(self, relation: str, base_score: float, boosted_score: float):
        """Record a keyword boost application."""
        self.keyword_stats['total_keyword_boosts'] += 1

        if relation not in self.keyword_stats['boosts_per_relation']:
            self.keyword_stats['boosts_per_relation'][relation] = {
                'count': 0,
                'avg_base_score': 0.0,
                'avg_boosted_score': 0.0
            }

        rel_stats = self.keyword_stats['boosts_per_relation'][relation]
        prev_count = rel_stats['count']
        new_count = prev_count + 1

        rel_stats['avg_base_score'] = (
            (rel_stats['avg_base_score'] * prev_count + base_score) / new_count
        )
        rel_stats['avg_boosted_score'] = (
            (rel_stats['avg_boosted_score'] * prev_count + boosted_score) / new_count
        )
        rel_stats['count'] = new_count

    def print_final_stats(self):
        """Print final statistics including keyword matching."""
        # Call parent's print_final_stats first
        super().print_final_stats()

        # Print keyword statistics
        if self.enable_keyword_boost:
            print(f"\n{'='*80}")
            print(f"[EnhancedHybridEdgeScorer] Keyword Matching Statistics")
            print(f"{'='*80}")
            print(f"Total keyword boosts applied: {self.keyword_stats['total_keyword_boosts']:,}")
            print(f"Average keyword score: {self.keyword_stats['avg_keyword_score']:.4f}")

            if self.keyword_stats['boosts_per_relation']:
                print(f"\nBoosts per relation:")
                for relation in sorted(self.keyword_stats['boosts_per_relation'].keys()):
                    rel_stats = self.keyword_stats['boosts_per_relation'][relation]
                    print(f"   {relation:<25} Count: {rel_stats['count']:>6,}  "
                          f"Avg Score: {rel_stats['avg_score']:.4f}")

            print(f"{'='*80}\n")


if __name__ == "__main__":
    """Test the enhanced hybrid scorer"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from qa_system.config import Config

    print("=" * 80)
    print("EnhancedHybridEdgeScorer Test")
    print("=" * 80)

    # Load base model
    model_path = Config.BASE_DIR / "models" / "variant3_edge_scorer_dual_best.pt"
    print(f"\nLoading base scorer from {model_path}...")
    base_scorer = EdgeScorerRelationRanker(str(model_path))

    # Create enhanced hybrid scorer
    print(f"\nCreating enhanced hybrid scorer...")
    enhanced_scorer = EnhancedHybridEdgeScorer(
        base_scorer=base_scorer,
        enable_hybrid=True,
        enable_keyword_boost=True,
        static_alphas={1: 0.7, 2: 0.7, 3: 0.7},
        keyword_weight=0.3,
        verbose=True
    )

    # Test question with director keyword
    question = "What movies did the director of The Matrix direct?"

    # Test edges
    test_edges = [
        ("The Matrix", "directed_by", "Wachowski Brothers"),
        ("The Matrix", "written_by", "Wachowski Brothers"),
        ("The Matrix", "starred_actors", "Keanu Reeves"),
        ("The Matrix", "has_genre", "Sci-Fi"),
    ]

    print(f"\n[Test 1] Question with 'director' keyword")
    print(f"   Question: {question}")
    print(f"   Edges:")
    for node, rel, target in test_edges:
        print(f"      {node} --[{rel}]--> {target}")

    # Score with base hybrid (no keyword boost)
    enhanced_scorer.enable_keyword_boost = False
    base_scores = enhanced_scorer.score_edges_batch(question, test_edges, hop=0, question_hop_count=2)

    # Score with keyword boost
    enhanced_scorer.enable_keyword_boost = True
    enhanced_scores = enhanced_scorer.score_edges_batch(question, test_edges, hop=0, question_hop_count=2)

    print(f"\n[Results Comparison]")
    print(f"{'Relation':<20} {'Base Score':<12} {'Enhanced Score':<15} {'Delta':<10}")
    print(f"{'-'*65}")

    for (node, rel, target), base_score, enhanced_score in zip(test_edges, base_scores, enhanced_scores):
        delta = enhanced_score - base_score
        marker = " " if delta > 0.05 else ""
        print(f"{rel:<20} {base_score:>11.4f} {enhanced_score:>14.4f} {delta:>+9.4f}{marker}")

    # Test question with writer keyword
    question2 = "What movies were written by the writer of Little Odessa?"
    test_edges2 = [
        ("Little Odessa", "written_by", "James Gray"),
        ("Little Odessa", "directed_by", "James Gray"),
        ("Little Odessa", "starred_actors", "Tim Roth"),
        ("Little Odessa", "has_genre", "Drama"),
    ]

    print(f"\n[Test 2] Question with 'writer/written' keywords")
    print(f"   Question: {question2}")
    print(f"   Edges:")
    for node, rel, target in test_edges2:
        print(f"      {node} --[{rel}]--> {target}")

    # Score with keyword boost
    enhanced_scores2 = enhanced_scorer.score_edges_batch(question2, test_edges2, hop=0, question_hop_count=2)

    print(f"\n[Results]")
    print(f"{'Relation':<20} {'Enhanced Score':<15}")
    print(f"{'-'*40}")

    ranked = sorted(zip(test_edges2, enhanced_scores2), key=lambda x: x[1], reverse=True)
    for i, ((node, rel, target), score) in enumerate(ranked, 1):
        marker = " " if i == 1 else ""
        print(f"{i}. {rel:<20} {score:>14.4f}{marker}")

    # Print statistics
    enhanced_scorer.print_final_stats()

    print(f"\n[Cost]")
    print(f"   Total cost: ${enhanced_scorer.get_cost():.6f}")

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)
