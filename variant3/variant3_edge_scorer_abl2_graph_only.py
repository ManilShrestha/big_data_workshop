"""
EdgeScorer Ablation 2: Graph Embeddings Only (No Text)

This ablation uses only graph (TransE) embeddings to verify that the dual
model benefits from text embeddings. No text, no hops.

Architecture:
- Question: Cannot use graph embedding (questions aren't in KG), so we use a learned embedding
- Node: graph_emb (256)
- Edge: graph_emb (256)
- Target: graph_emb (256)

No text embeddings, no hop information.
"""

import torch
import torch.nn as nn


class EdgeScorerGraphOnly(nn.Module):
    """
    Simplified EdgeScorer using only graph (TransE) embeddings.

    Inputs:
    - node_graph_emb: 256-dim TransE embedding
    - edge_graph_emb: 256-dim TransE embedding
    - target_graph_emb: 256-dim TransE embedding

    Note: Questions don't have graph embeddings, so we use a learned embedding
    that gets updated during training to represent "query context".

    Output:
    - score: P(this edge+target leads to answer | current_node)
    """

    def __init__(
        self,
        graph_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Learned query embedding (since questions aren't in the KG)
        # This represents the "question context" in the graph space
        self.query_embedding = nn.Parameter(torch.randn(1, graph_dim))

        # Project each graph embedding to hidden dimension
        self.query_proj = nn.Linear(graph_dim, hidden_dim)
        self.node_proj = nn.Linear(graph_dim, hidden_dim)
        self.edge_proj = nn.Linear(graph_dim, hidden_dim)
        self.target_proj = nn.Linear(graph_dim, hidden_dim)

        # Cross-component gated fusion
        # 4 components: query, node, edge, target
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

        # Initialize query embedding
        nn.init.normal_(self.query_embedding, mean=0, std=0.1)

    def forward(
        self,
        node_graph_emb: torch.Tensor,        # [B, 256]
        edge_graph_emb: torch.Tensor,        # [B, 256]
        target_graph_emb: torch.Tensor       # [B, 256]
    ) -> torch.Tensor:
        """
        Forward pass with graph embeddings only.

        Returns:
            logits: [B, 1] (pre-sigmoid)
        """
        batch_size = node_graph_emb.size(0)

        # Expand query embedding for batch
        query_emb = self.query_embedding.expand(batch_size, -1)  # [B, 256]

        # Project all components
        q = self.query_proj(query_emb)                 # [B, 512]
        n = self.node_proj(node_graph_emb)             # [B, 512]
        e = self.edge_proj(edge_graph_emb)             # [B, 512]
        t = self.target_proj(target_graph_emb)         # [B, 512]

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
        node_graph_emb: torch.Tensor,
        edge_graph_emb: torch.Tensor,
        target_graph_emb: torch.Tensor
    ) -> torch.Tensor:
        """Predict with sigmoid"""
        logits = self.forward(node_graph_emb, edge_graph_emb, target_graph_emb)
        return torch.sigmoid(logits)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 80)
    print("EdgeScorerGraphOnly Model Test (Ablation 2)")
    print("=" * 80)

    # Create model
    model = EdgeScorerGraphOnly(
        graph_dim=256,
        hidden_dim=512,
        dropout=0.3
    )

    print(f"\nModel architecture:")
    print(model)

    print(f"\nTotal parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    node_graph = torch.randn(batch_size, 256)
    edge_graph = torch.randn(batch_size, 256)
    target_graph = torch.randn(batch_size, 256)

    print(f"\n[Forward pass test]")
    print(f"Input shapes:")
    print(f"  node_graph: {node_graph.shape}")
    print(f"  edge_graph: {edge_graph.shape}")
    print(f"  target_graph: {target_graph.shape}")

    # Forward
    logits = model(node_graph, edge_graph, target_graph)
    probs = model.predict(node_graph, edge_graph, target_graph)

    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  probs: {probs.shape}")

    print(f"\nSample outputs:")
    print(f"  probs: {probs[:2, 0].tolist()}")

    print(f"\nLearned query embedding shape: {model.query_embedding.shape}")

    print("\n" + "=" * 80)
    print("Model test passed!")
    print("=" * 80)
