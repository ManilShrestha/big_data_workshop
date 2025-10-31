"""
EdgeScorer-based relation ranker for Variant 3 QA evaluation.

Loads the trained EdgeScorerDual model and uses it to score edges
during knowledge graph search.
"""

import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from pathlib import Path
from openai import OpenAI

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from qa_system.config import Config
from qa_system.relation_rankers.openai_ranker import calculate_api_cost
from variant3.variant3_edge_scorer_dual import EdgeScorerDual


class EdgeScorerRelationRanker:
    """
    Use trained EdgeScorerDual model to score edges during search.

    Unlike traditional relation rankers that rank relations statically,
    this ranker scores edges based on the current search context:
    - Current node
    - Current hop
    - Target node

    This makes it context-aware and suitable for multi-hop reasoning.
    """

    def __init__(
        self,
        model_checkpoint_path: str,
        device: str = None,
        openai_api_key: str = None
    ):
        """
        Initialize EdgeScorer ranker.

        Args:
            model_checkpoint_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu' (auto-detect if None)
            openai_api_key: OpenAI API key for embedding new questions
        """
        print(f"[EdgeScorerRanker] Initializing...")

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"   Device: {self.device}")

        # OpenAI client for new questions
        self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for embedding new questions")
        self.openai_client = OpenAI(api_key=self.openai_api_key)

        # Cost tracking
        self.total_cost = 0.0
        self.last_query_cost = 0.0
        self.question_cache = {}  # Runtime cache for questions embedded during this session

        # Load model
        print(f"[EdgeScorerRanker] Loading model from {model_checkpoint_path}...")
        self.model = self._load_model(model_checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"   Model loaded successfully")

        # Load embeddings
        print(f"[EdgeScorerRanker] Loading embeddings...")
        self._load_embeddings()
        print(f"   Embeddings loaded successfully")

        # Zero vectors for missing embeddings
        self.zero_text = np.zeros(1536, dtype=np.float32)
        self.zero_graph = np.zeros(256, dtype=np.float32)

        print(f"[EdgeScorerRanker] Initialization complete!")

    def _load_model(self, checkpoint_path: str) -> EdgeScorerDual:
        """Load trained model from checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        # Load checkpoint (weights_only=False for compatibility with numpy types)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Create model with same architecture
        model = EdgeScorerDual(
            text_dim=1536,
            graph_dim=256,
            hidden_dim=512,
            max_hops=3,
            dropout=0.3
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def _load_embeddings(self):
        """Load all embedding caches"""
        emb_dir = Config.EMBEDDINGS_DIR

        # Text embeddings (unified cache)
        text_cache_path = emb_dir / "variant3_text_embeddings_cache.pkl"
        print(f"   Loading text embeddings from {text_cache_path.name}...")
        with open(text_cache_path, 'rb') as f:
            self.text_embeddings_cache = pickle.load(f)
        print(f"      {len(self.text_embeddings_cache):,} unique texts")

        # Graph embeddings (TransE)
        print(f"   Loading graph embeddings...")
        self.entity_graph_embeddings = np.load(Config.TRANSE_ENTITY_PATH)
        self.relation_graph_embeddings = np.load(Config.TRANSE_RELATION_PATH)
        print(f"      Entities: {self.entity_graph_embeddings.shape}")
        print(f"      Relations: {self.relation_graph_embeddings.shape}")

        # Entity and relation mappings
        node2id_path = emb_dir / "node2id.json"
        relation2id_path = emb_dir / "relation2id.json"

        with open(node2id_path, 'r') as f:
            self.entity2id = json.load(f)

        with open(relation2id_path, 'r') as f:
            self.relation2id = json.load(f)

        print(f"      Entity mappings: {len(self.entity2id):,}")
        print(f"      Relation mappings: {len(self.relation2id):,}")

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding (from cache or OpenAI API)"""
        # Check static cache first
        if text in self.text_embeddings_cache:
            return self.text_embeddings_cache[text]

        # Check runtime cache
        if text in self.question_cache:
            return self.question_cache[text]

        # Not in cache - need to embed via OpenAI API
        # This should only happen for new questions
        print(f"   [EdgeScorerRanker] Embedding new text via OpenAI API...")
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        embedding = np.array(response.data[0].embedding, dtype=np.float32)

        # Track cost
        cost = calculate_api_cost(response)
        self.total_cost += cost
        self.last_query_cost = cost

        # Cache for this session
        self.question_cache[text] = embedding

        return embedding

    def _get_entity_graph_embedding(self, entity_id: str) -> np.ndarray:
        """Get graph embedding for an entity"""
        if entity_id in self.entity2id:
            idx = self.entity2id[entity_id]
            return self.entity_graph_embeddings[idx]
        else:
            return self.zero_graph

    def _get_relation_graph_embedding(self, relation: str) -> np.ndarray:
        """Get graph embedding for a relation"""
        if relation in self.relation2id:
            idx = self.relation2id[relation]
            return self.relation_graph_embeddings[idx]
        else:
            return self.zero_graph

    def score_edges_batch(
        self,
        question: str,
        edges: List[Tuple[str, str, str]],
        hop: int
    ) -> List[float]:
        """
        Score a batch of edges using the trained model.

        Args:
            question: Question text
            edges: List of (current_node, relation, target_node) tuples
            hop: Current hop (0, 1, or 2)

        Returns:
            List of scores (probabilities 0-1) for each edge
        """
        if len(edges) == 0:
            return []

        # Get question embedding (once for all edges)
        question_text_emb = self._get_text_embedding(question)

        # Prepare batch tensors
        batch_size = len(edges)
        question_text_batch = []
        node_text_batch = []
        node_graph_batch = []
        edge_text_batch = []
        edge_graph_batch = []
        target_text_batch = []
        target_graph_batch = []
        hop_batch = []

        for current_node, relation, target_node in edges:
            # Question
            question_text_batch.append(question_text_emb)

            # Current node
            node_text = self._get_text_embedding(current_node) if current_node else self.zero_text
            node_graph = self._get_entity_graph_embedding(current_node) if current_node else self.zero_graph
            node_text_batch.append(node_text)
            node_graph_batch.append(node_graph)

            # Relation (edge)
            edge_text = self._get_text_embedding(relation) if relation else self.zero_text
            edge_graph = self._get_relation_graph_embedding(relation) if relation else self.zero_graph
            edge_text_batch.append(edge_text)
            edge_graph_batch.append(edge_graph)

            # Target node
            target_text = self._get_text_embedding(target_node) if target_node else self.zero_text
            target_graph = self._get_entity_graph_embedding(target_node) if target_node else self.zero_graph
            target_text_batch.append(target_text)
            target_graph_batch.append(target_graph)

            # Hop
            hop_batch.append(hop)

        # Convert to tensors
        question_text_tensor = torch.tensor(np.array(question_text_batch), dtype=torch.float32).to(self.device)
        node_text_tensor = torch.tensor(np.array(node_text_batch), dtype=torch.float32).to(self.device)
        node_graph_tensor = torch.tensor(np.array(node_graph_batch), dtype=torch.float32).to(self.device)
        edge_text_tensor = torch.tensor(np.array(edge_text_batch), dtype=torch.float32).to(self.device)
        edge_graph_tensor = torch.tensor(np.array(edge_graph_batch), dtype=torch.float32).to(self.device)
        target_text_tensor = torch.tensor(np.array(target_text_batch), dtype=torch.float32).to(self.device)
        target_graph_tensor = torch.tensor(np.array(target_graph_batch), dtype=torch.float32).to(self.device)
        hop_tensor = torch.tensor(hop_batch, dtype=torch.long).to(self.device)

        # Run inference (no gradients needed)
        with torch.no_grad():
            scores = self.model.predict(
                question_text_tensor,
                node_text_tensor,
                node_graph_tensor,
                edge_text_tensor,
                edge_graph_tensor,
                target_text_tensor,
                target_graph_tensor,
                hop_tensor
            )

        # Convert to list of floats
        scores_list = scores.squeeze(1).cpu().numpy().tolist()

        return scores_list

    def rank_relations(self, question: str, top_k: int = None):
        """
        Dummy method for interface compatibility with Evaluator.
        EdgeScorer doesn't rank relations in isolation - it needs full context.
        This method returns an empty list since EdgeScorer works differently.
        """
        return []

    def get_cost(self) -> float:
        """Get total accumulated cost"""
        return self.total_cost

    def get_last_query_cost(self) -> float:
        """Get cost of last query"""
        return self.last_query_cost


if __name__ == "__main__":
    """Test the ranker"""
    print("=" * 80)
    print("EdgeScorerRelationRanker Test")
    print("=" * 80)

    # Load model
    model_path = Config.BASE_DIR / "models" / "variant3_edge_scorer_dual_best.pt"
    ranker = EdgeScorerRelationRanker(str(model_path))

    # Test question
    question = "What movies did the director of The Matrix direct?"

    # Test edges
    test_edges = [
        ("The Matrix", "directed_by", "Wachowski Brothers"),
        ("The Matrix", "starred_actors", "Keanu Reeves"),
        ("The Matrix", "has_genre", "Sci-Fi"),
        ("The Matrix", "release_year", "1999"),
    ]

    print(f"\n[Test] Scoring {len(test_edges)} edges for hop 0...")
    print(f"   Question: {question}")
    print(f"   Edges:")
    for node, rel, target in test_edges:
        print(f"      {node} --[{rel}]--> {target}")

    # Score edges
    scores = ranker.score_edges_batch(question, test_edges, hop=0)

    print(f"\n[Results]")
    scored_edges = list(zip(test_edges, scores))
    scored_edges.sort(key=lambda x: x[1], reverse=True)

    for (node, rel, target), score in scored_edges:
        print(f"   {score:.4f}: {node} --[{rel}]--> {target}")

    print(f"\n[Cost]")
    print(f"   Total cost: ${ranker.get_cost():.6f}")

    print("\n" + "=" * 80)
    print("Test passed!")
    print("=" * 80)
