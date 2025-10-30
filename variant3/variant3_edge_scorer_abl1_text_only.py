"""
EdgeScorer Ablation 1: Text Embeddings Only (No Hops, No Graph)

This is a simplified version that only uses text embeddings to verify that
the dual model is actually learning from graph embeddings and hop information.

Architecture:
- Question: text_emb (1536)
- Node: text_emb (1536)
- Edge: text_emb (1536)
- Target: text_emb (1536)

No hop information, no graph embeddings.
"""

import torch
import torch.nn as nn


class EdgeScorerTextOnly(nn.Module):
    """
    Simplified EdgeScorer using only text embeddings.

    Inputs:
    - question_text_emb: 1536-dim OpenAI embedding
    - node_text_emb: 1536-dim OpenAI embedding
    - edge_text_emb: 1536-dim OpenAI embedding
    - target_text_emb: 1536-dim OpenAI embedding

    Output:
    - score: P(this edge+target leads to answer | question, current_node)
    """

    def __init__(
        self,
        text_dim: int = 1536,
        hidden_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project each text embedding to hidden dimension
        self.question_proj = nn.Linear(text_dim, hidden_dim)
        self.node_proj = nn.Linear(text_dim, hidden_dim)
        self.edge_proj = nn.Linear(text_dim, hidden_dim)
        self.target_proj = nn.Linear(text_dim, hidden_dim)

        # Cross-component gated fusion
        # 4 components: question, node, edge, target
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
            nn.Softmax(dim=1)  # 4 gates sum to 1
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

    def forward(
        self,
        question_text_emb: torch.Tensor,     # [B, 1536]
        node_text_emb: torch.Tensor,         # [B, 1536]
        edge_text_emb: torch.Tensor,         # [B, 1536]
        target_text_emb: torch.Tensor        # [B, 1536]
    ) -> torch.Tensor:
        """
        Forward pass with text embeddings only.

        Returns:
            logits: [B, 1] (pre-sigmoid)
        """
        # Project all components
        q = self.question_proj(question_text_emb)  # [B, 512]
        n = self.node_proj(node_text_emb)          # [B, 512]
        e = self.edge_proj(edge_text_emb)          # [B, 512]
        t = self.target_proj(target_text_emb)      # [B, 512]

        # Concatenate all components
        concat = torch.cat([q, n, e, t], dim=1)  # [B, 2048]

        # Compute cross-component gates
        gates = self.gate_network(concat)  # [B, 4]

        # Apply gated fusion
        fused = (gates[:, 0:1] * q +
                 gates[:, 1:2] * n +
                 gates[:, 2:3] * e +
                 gates[:, 3:4] * t)  # [B, 512]

        # Classify
        logits = self.classifier(fused)  # [B, 1]

        return logits

    def predict(
        self,
        question_text_emb: torch.Tensor,
        node_text_emb: torch.Tensor,
        edge_text_emb: torch.Tensor,
        target_text_emb: torch.Tensor
    ) -> torch.Tensor:
        """Predict with sigmoid"""
        logits = self.forward(
            question_text_emb, node_text_emb,
            edge_text_emb, target_text_emb
        )
        return torch.sigmoid(logits)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 80)
    print("EdgeScorerTextOnly Model Test (Ablation 1)")
    print("=" * 80)

    # Create model
    model = EdgeScorerTextOnly(
        text_dim=1536,
        hidden_dim=512,
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

    print(f"\n[Forward pass test]")
    print(f"Input shapes:")
    print(f"  question_text: {question_text.shape}")
    print(f"  node_text: {node_text.shape}")
    print(f"  edge_text: {edge_text.shape}")
    print(f"  target_text: {target_text.shape}")

    # Forward
    logits = model(question_text, node_text, edge_text, target_text)
    probs = model.predict(question_text, node_text, edge_text, target_text)

    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  probs: {probs.shape}")

    print(f"\nSample outputs:")
    print(f"  probs: {probs[:2, 0].tolist()}")

    print("\n" + "=" * 80)
    print("Model test passed!")
    print("=" * 80)
