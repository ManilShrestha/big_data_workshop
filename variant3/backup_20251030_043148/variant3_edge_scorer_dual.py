"""
EdgeScorer with Dual Embeddings: Text (semantic) + Graph (structural)

Architecture:
1. Dual embeddings for all components:
   - Question: text_emb (1536)
   - Node: text_emb (1536) + graph_emb (256)
   - Edge: text_emb (1536) + graph_emb (256)
   - Target: text_emb (1536) + graph_emb (256)  <- NEW!
   - Hop: learned_emb (512)

2. Fusion per component:
   - For node/edge/target: concat [text; graph] then project
   - Learns to weight semantic vs structural per component

3. Gated Fusion across components:
   - 5 attention gates (question, node, edge, target, hop)
   - Weighted sum of all signals

4. Classification:
   - MLP: 512 → 256 → 128 → 1
   - Output: P(edge leads to answer)

Parameters: ~3.5M (still fast, more expressive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEmbeddingFusion(nn.Module):
    """Fuse text and graph embeddings for a single component"""

    def __init__(self, text_dim: int, graph_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()

        # Project each modality
        self.text_proj = nn.Linear(text_dim, output_dim // 2)
        self.graph_proj = nn.Linear(graph_dim, output_dim // 2)

        # Fusion gate: learn to weight text vs graph
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, 2),
            nn.Softmax(dim=1)  # [text_weight, graph_weight] sum to 1
        )

        # Final projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_emb: [B, text_dim]
            graph_emb: [B, graph_dim]
        Returns:
            fused: [B, output_dim]
        """
        # Project
        t = self.text_proj(text_emb)    # [B, output_dim/2]
        g = self.graph_proj(graph_emb)  # [B, output_dim/2]

        # Concatenate
        concat = torch.cat([t, g], dim=1)  # [B, output_dim]

        # Compute fusion gates
        gates = self.fusion_gate(concat)  # [B, 2]

        # Weighted combination (element-wise)
        # Expand gates: [B, 2, 1] and embeddings: [B, 1, output_dim/2]
        text_weighted = gates[:, 0:1] * t  # [B, output_dim/2]
        graph_weighted = gates[:, 1:2] * g  # [B, output_dim/2]

        # Concatenate weighted
        fused = torch.cat([text_weighted, graph_weighted], dim=1)  # [B, output_dim]

        # Final projection with residual
        out = self.output_proj(fused) + concat  # Residual connection
        out = self.dropout(out)

        return out


class EdgeScorerDual(nn.Module):
    """
    Enhanced EdgeScorer with dual embeddings (text + graph) for all components.

    Inputs:
    - question_text_emb: 1536-dim OpenAI embedding
    - node_text_emb: 1536-dim OpenAI embedding
    - node_graph_emb: 256-dim TransE embedding
    - edge_text_emb: 1536-dim OpenAI embedding
    - edge_graph_emb: 256-dim TransE embedding
    - target_text_emb: 1536-dim OpenAI embedding (candidate node)
    - target_graph_emb: 256-dim TransE embedding (candidate node)
    - hop: 0, 1, or 2

    Output:
    - score: P(this edge+target leads to answer | question, current_node, hop)
    """

    def __init__(
        self,
        text_dim: int = 1536,
        graph_dim: int = 256,
        hidden_dim: int = 512,
        max_hops: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Question: only text (no graph structure for question)
        self.question_proj = nn.Linear(text_dim, hidden_dim)

        # Dual fusion for node, edge, target
        self.node_fusion = DualEmbeddingFusion(text_dim, graph_dim, hidden_dim, dropout)
        self.edge_fusion = DualEmbeddingFusion(text_dim, graph_dim, hidden_dim, dropout)
        self.target_fusion = DualEmbeddingFusion(text_dim, graph_dim, hidden_dim, dropout)

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
        node_graph_emb: torch.Tensor,        # [B, 256]
        edge_text_emb: torch.Tensor,         # [B, 1536]
        edge_graph_emb: torch.Tensor,        # [B, 256]
        target_text_emb: torch.Tensor,       # [B, 1536]
        target_graph_emb: torch.Tensor,      # [B, 256]
        hop: torch.Tensor                    # [B]
    ) -> torch.Tensor:
        """
        Forward pass with dual embeddings.

        Returns:
            logits: [B, 1] (pre-sigmoid)
        """
        # Process question (text only)
        q = self.question_proj(question_text_emb)  # [B, 512]

        # Fuse dual embeddings for node, edge, target
        n = self.node_fusion(node_text_emb, node_graph_emb)      # [B, 512]
        e = self.edge_fusion(edge_text_emb, edge_graph_emb)      # [B, 512]
        t = self.target_fusion(target_text_emb, target_graph_emb)  # [B, 512]

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
        node_graph_emb: torch.Tensor,
        edge_text_emb: torch.Tensor,
        edge_graph_emb: torch.Tensor,
        target_text_emb: torch.Tensor,
        target_graph_emb: torch.Tensor,
        hop: torch.Tensor
    ) -> torch.Tensor:
        """Predict with sigmoid"""
        logits = self.forward(
            question_text_emb, node_text_emb, node_graph_emb,
            edge_text_emb, edge_graph_emb,
            target_text_emb, target_graph_emb, hop
        )
        return torch.sigmoid(logits)

    def get_gates(
        self,
        question_text_emb: torch.Tensor,
        node_text_emb: torch.Tensor,
        node_graph_emb: torch.Tensor,
        edge_text_emb: torch.Tensor,
        edge_graph_emb: torch.Tensor,
        target_text_emb: torch.Tensor,
        target_graph_emb: torch.Tensor,
        hop: torch.Tensor
    ) -> dict:
        """
        Get all gates for interpretability.

        Returns:
            {
                'component_gates': [B, 5],  # question, node, edge, target, hop
                'node_fusion_gates': [B, 2],  # text vs graph for node
                'edge_fusion_gates': [B, 2],  # text vs graph for edge
                'target_fusion_gates': [B, 2]  # text vs graph for target
            }
        """
        # Component embeddings
        q = self.question_proj(question_text_emb)
        n = self.node_fusion(node_text_emb, node_graph_emb)
        e = self.edge_fusion(edge_text_emb, edge_graph_emb)
        t = self.target_fusion(target_text_emb, target_graph_emb)
        h = self.hop_embed(hop)

        concat = torch.cat([q, n, e, t, h], dim=1)
        component_gates = self.gate_network(concat)

        # Get fusion gates (requires extracting from fusion modules)
        # For simplicity, re-compute
        node_concat = torch.cat([
            self.node_fusion.text_proj(node_text_emb),
            self.node_fusion.graph_proj(node_graph_emb)
        ], dim=1)
        node_fusion_gates = self.node_fusion.fusion_gate(node_concat)

        edge_concat = torch.cat([
            self.edge_fusion.text_proj(edge_text_emb),
            self.edge_fusion.graph_proj(edge_graph_emb)
        ], dim=1)
        edge_fusion_gates = self.edge_fusion.fusion_gate(edge_concat)

        target_concat = torch.cat([
            self.target_fusion.text_proj(target_text_emb),
            self.target_fusion.graph_proj(target_graph_emb)
        ], dim=1)
        target_fusion_gates = self.target_fusion.fusion_gate(target_concat)

        return {
            'component_gates': component_gates,  # [B, 5]
            'node_fusion_gates': node_fusion_gates,  # [B, 2]
            'edge_fusion_gates': edge_fusion_gates,  # [B, 2]
            'target_fusion_gates': target_fusion_gates  # [B, 2]
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 80)
    print("EdgeScorerDual Model Test")
    print("=" * 80)

    # Create model
    model = EdgeScorerDual(
        text_dim=1536,
        graph_dim=256,
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
    node_graph = torch.randn(batch_size, 256)
    edge_text = torch.randn(batch_size, 1536)
    edge_graph = torch.randn(batch_size, 256)
    target_text = torch.randn(batch_size, 1536)
    target_graph = torch.randn(batch_size, 256)
    hop = torch.tensor([0, 1, 2, 0])

    print(f"\n[Forward pass test]")
    print(f"Input shapes:")
    print(f"  question_text: {question_text.shape}")
    print(f"  node_text: {node_text.shape}, node_graph: {node_graph.shape}")
    print(f"  edge_text: {edge_text.shape}, edge_graph: {edge_graph.shape}")
    print(f"  target_text: {target_text.shape}, target_graph: {target_graph.shape}")
    print(f"  hop: {hop.shape}")

    # Forward
    logits = model(question_text, node_text, node_graph,
                   edge_text, edge_graph,
                   target_text, target_graph, hop)
    probs = model.predict(question_text, node_text, node_graph,
                          edge_text, edge_graph,
                          target_text, target_graph, hop)
    gates_dict = model.get_gates(question_text, node_text, node_graph,
                                  edge_text, edge_graph,
                                  target_text, target_graph, hop)

    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  probs: {probs.shape}")
    print(f"  component_gates: {gates_dict['component_gates'].shape}")

    print(f"\nSample outputs:")
    print(f"  probs: {probs[:2, 0].tolist()}")
    print(f"\n  Component gates (q, n, e, t, h):")
    for i in range(2):
        print(f"    Sample {i}: {gates_dict['component_gates'][i].tolist()}")
        print(f"    Sum: {gates_dict['component_gates'][i].sum().item():.4f}")

    print(f"\n  Node fusion gates (text, graph):")
    for i in range(2):
        print(f"    Sample {i}: {gates_dict['node_fusion_gates'][i].tolist()}")

    print("\n" + "=" * 80)
    print("Model test passed!")
    print("=" * 80)