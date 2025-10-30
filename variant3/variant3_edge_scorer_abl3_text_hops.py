"""
EdgeScorer Ablation 3: Text Embeddings + Hops (No Graph)

This ablation uses text embeddings and hop information but NO graph embeddings.
This tests whether the graph embeddings provide additional value beyond text+hop.

Architecture:
- Question: text_emb (1536)
- Node: text_emb (1536)
- Edge: text_emb (1536)
- Target: text_emb (1536)
- Hop: learned_emb (512)

No graph embeddings.
"""

import torch
import torch.nn as nn


class EdgeScorerTextHops(nn.Module):
    """
    EdgeScorer using text embeddings + hop information (no graph embeddings).

    Inputs:
    - question_text_emb: 1536-dim OpenAI embedding
    - node_text_emb: 1536-dim OpenAI embedding
    - edge_text_emb: 1536-dim OpenAI embedding
    - target_text_emb: 1536-dim OpenAI embedding
    - hop: 0, 1, or 2

    Output:
    - score: P(this edge+target leads to answer | question, current_node, hop)
    """

    def __init__(
        self,
        text_dim: int = 1536,
        hidden_dim: int = 512,
        max_hops: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project each text embedding to hidden dimension
        self.question_proj = nn.Linear(text_dim, hidden_dim)
        self.node_proj = nn.Linear(text_dim, hidden_dim)
        self.edge_proj = nn.Linear(text_dim, hidden_dim)
        self.target_proj = nn.Linear(text_dim, hidden_dim)

        # Hop embedding
        self.hop_embed = nn.Embedding(max_hops, hidden_dim)

        # Cross-component gated fusion
        # 5 components: question, node, edge, target, hop
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
            nn.Softmax(dim=1)  # 5 gates sum to 1
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(
        self,
        question_text_emb: torch.Tensor,     # [B, 1536]
        node_text_emb: torch.Tensor,         # [B, 1536]
        edge_text_emb: torch.Tensor,         # [B, 1536]
        target_text_emb: torch.Tensor,       # [B, 1536]
        hop: torch.Tensor                    # [B]
    ) -> torch.Tensor:
        """
        Forward pass with text embeddings and hop information.

        Returns:
            logits: [B, 1] (pre-sigmoid)
        """
        # Project all text components
        q = self.question_proj(question_text_emb)  # [B, 512]
        n = self.node_proj(node_text_emb)          # [B, 512]
        e = self.edge_proj(edge_text_emb)          # [B, 512]
        t = self.target_proj(target_text_emb)      # [B, 512]

        # Hop embedding
        h = self.hop_embed(hop)  # [B, 512]

        # Concatenate all components
        concat = torch.cat([q, n, e, t, h], dim=1)  # [B, 2560]

        # Compute cross-component gates
        gates = self.gate_network(concat)  # [B, 5]

        # Apply gated fusion
        fused = (gates[:, 0:1] * q +
                 gates[:, 1:2] * n +
                 gates[:, 2:3] * e +
                 gates[:, 3:4] * t +
                 gates[:, 4:5] * h)  # [B, 512]

        # Classify
        logits = self.classifier(fused)  # [B, 1]

        return logits

    def predict(
        self,
        question_text_emb: torch.Tensor,
        node_text_emb: torch.Tensor,
        edge_text_emb: torch.Tensor,
        target_text_emb: torch.Tensor,
        hop: torch.Tensor
    ) -> torch.Tensor:
        """Predict with sigmoid"""
        logits = self.forward(
            question_text_emb, node_text_emb,
            edge_text_emb, target_text_emb, hop
        )
        return torch.sigmoid(logits)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 80)
    print("EdgeScorerTextHops Model Test (Ablation 3)")
    print("=" * 80)

    # Create model
    model = EdgeScorerTextHops(
        text_dim=1536,
        hidden_dim=512,
        max_hops=3,
        dropout=0.3
    )

    print(f"\nModel architecture:")
    print(model)

    print(f"\nTotal parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    question_text = torch.randn(batch_size, 1536)
    node_text = torch.randn(batch_size, 1536)
    edge_text = torch.randn(batch_size, 1536)
    target_text = torch.randn(batch_size, 1536)
    hop = torch.tensor([0, 1, 2, 0])

    print(f"\n[Forward pass test]")
    print(f"Input shapes:")
    print(f"  question_text: {question_text.shape}")
    print(f"  node_text: {node_text.shape}")
    print(f"  edge_text: {edge_text.shape}")
    print(f"  target_text: {target_text.shape}")
    print(f"  hop: {hop.shape}")

    # Forward
    logits = model(question_text, node_text, edge_text, target_text, hop)
    probs = model.predict(question_text, node_text, edge_text, target_text, hop)

    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  probs: {probs.shape}")

    print(f"\nSample outputs:")
    print(f"  probs: {probs[:2, 0].tolist()}")

    print("\n" + "=" * 80)
    print("Model test passed!")
    print("=" * 80)
