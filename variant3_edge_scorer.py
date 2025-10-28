"""
EdgeScorer: Neural network for scoring edges during graph traversal

Architecture (Gated Fusion):
1. Project all inputs to 512-dim:
   - question_embedding: 1536 → 512
   - node_embedding: 256 → 512
   - edge_embedding: 256 → 512
   - hop_number: scalar → 512 (learned embedding)

2. Gated Fusion:
   - Learn 4 attention gates from concatenated input
   - Weighted sum: fused = Σ gate_i * embedding_i
   - Gates sum to 1 (softmax), interpretable weights

3. Classification:
   - MLP: 512 → 256 → 1
   - Output: probability that edge is correct

Parameters: ~1.5M (lightweight for fast inference)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeScorer(nn.Module):
    """
    Neural network for scoring edges in knowledge graph traversal.

    Given:
    - question_embedding: OpenAI text embedding (1536-dim)
    - node_embedding: TransE entity embedding (256-dim)
    - edge_embedding: TransE relation embedding (256-dim)
    - hop_number: current hop (0, 1, or 2 for 1/2/3-hop questions)

    Output:
    - score: probability [0, 1] that this edge is correct
    """

    def __init__(
        self,
        question_dim: int = 1536,
        node_dim: int = 256,
        edge_dim: int = 256,
        hidden_dim: int = 512,
        max_hops: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Projection layers: map all inputs to same dimension
        self.question_proj = nn.Linear(question_dim, hidden_dim)
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # Hop embedding: learned representation for each hop number
        # hop 0 = 1st hop, hop 1 = 2nd hop, hop 2 = 3rd hop
        self.hop_embed = nn.Embedding(max_hops, hidden_dim)

        # Gated fusion: learn importance weights for each input
        # Input: concatenated projections [B, 4*512]
        # Output: 4 gates [B, 4] summing to 1 (softmax)
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
            nn.Softmax(dim=1)  # Gates sum to 1
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
            # No sigmoid here - use BCEWithLogitsLoss for numerical stability
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(
        self,
        question_emb: torch.Tensor,  # [B, 1536]
        node_emb: torch.Tensor,      # [B, 256]
        edge_emb: torch.Tensor,      # [B, 256]
        hop: torch.Tensor            # [B] - integers 0, 1, or 2
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            question_emb: Question text embeddings [batch_size, 1536]
            node_emb: Current node embeddings [batch_size, 256]
            edge_emb: Edge relation embeddings [batch_size, 256]
            hop: Hop numbers [batch_size] (0-indexed)

        Returns:
            logits: Edge scores [batch_size, 1] (pre-sigmoid)
        """
        # Project all to hidden_dim
        q = self.question_proj(question_emb)  # [B, 512]
        n = self.node_proj(node_emb)          # [B, 512]
        e = self.edge_proj(edge_emb)          # [B, 512]
        h = self.hop_embed(hop)               # [B, 512]

        # Concatenate for gate computation
        concat = torch.cat([q, n, e, h], dim=1)  # [B, 2048]

        # Compute gates: [B, 4] with softmax (sum to 1)
        gates = self.gate_network(concat)  # [B, 4]

        # Apply gated fusion: weighted sum
        # gates[:, 0:1] → [B, 1], q → [B, 512]
        # Broadcasting: [B, 1] * [B, 512] → [B, 512]
        fused = (gates[:, 0:1] * q +
                 gates[:, 1:2] * n +
                 gates[:, 2:3] * e +
                 gates[:, 3:4] * h)  # [B, 512]

        # Classify
        logits = self.classifier(fused)  # [B, 1]

        return logits

    def predict(
        self,
        question_emb: torch.Tensor,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        hop: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict probabilities (with sigmoid).

        Returns:
            probs: Edge probabilities [batch_size, 1]
        """
        logits = self.forward(question_emb, node_emb, edge_emb, hop)
        probs = torch.sigmoid(logits)
        return probs

    def get_gates(
        self,
        question_emb: torch.Tensor,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        hop: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention gates for interpretability.

        Returns:
            gates: [batch_size, 4] where gates[:, i] is weight for:
                0 = question, 1 = node, 2 = edge, 3 = hop
        """
        q = self.question_proj(question_emb)
        n = self.node_proj(node_emb)
        e = self.edge_proj(edge_emb)
        h = self.hop_embed(hop)

        concat = torch.cat([q, n, e, h], dim=1)
        gates = self.gate_network(concat)

        return gates


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("=" * 80)
    print("EdgeScorer Model Test")
    print("=" * 80)

    # Create model
    model = EdgeScorer(
        question_dim=1536,
        node_dim=256,
        edge_dim=256,
        hidden_dim=512,
        max_hops=3,
        dropout=0.3
    )

    print(f"\nModel architecture:")
    print(model)

    print(f"\nTotal parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    question_emb = torch.randn(batch_size, 1536)
    node_emb = torch.randn(batch_size, 256)
    edge_emb = torch.randn(batch_size, 256)
    hop = torch.tensor([0, 1, 2, 0])  # Mix of hops

    print(f"\n[Forward pass test]")
    print(f"Input shapes:")
    print(f"  question_emb: {question_emb.shape}")
    print(f"  node_emb: {node_emb.shape}")
    print(f"  edge_emb: {edge_emb.shape}")
    print(f"  hop: {hop.shape}")

    # Forward pass
    logits = model(question_emb, node_emb, edge_emb, hop)
    probs = model.predict(question_emb, node_emb, edge_emb, hop)
    gates = model.get_gates(question_emb, node_emb, edge_emb, hop)

    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  probs: {probs.shape}")
    print(f"  gates: {gates.shape}")

    print(f"\nSample outputs:")
    print(f"  logits: {logits[:2, 0].tolist()}")
    print(f"  probs: {probs[:2, 0].tolist()}")
    print(f"\n  gates (question, node, edge, hop):")
    for i in range(2):
        print(f"    Sample {i}: {gates[i].tolist()}")
        print(f"    Sum: {gates[i].sum().item():.4f} (should be 1.0)")

    print("\n" + "=" * 80)
    print("Model test passed!")
    print("=" * 80)
