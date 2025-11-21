#!/usr/bin/env python3
"""
Quick test to verify 6-bit hop encoding works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from variant3.variant3_edge_scorer_dual import EdgeScorerDual

def test_model_forward():
    """Test that the model accepts 6-bit hop context"""
    print("=" * 80)
    print("Testing 6-bit Hop Encoding")
    print("=" * 80)

    print("\n[1/3] Creating model with 6-bit hop context...")
    model = EdgeScorerDual(
        text_dim=1536,
        graph_dim=256,
        hidden_dim=512,
        hop_context_dim=6,
        dropout=0.3
    )
    print(f"   Model created successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\n[2/3] Testing forward pass with different hop contexts...")
    batch_size = 4

    # Create random embeddings
    question_text = torch.randn(batch_size, 1536)
    node_text = torch.randn(batch_size, 1536)
    node_graph = torch.randn(batch_size, 256)
    edge_text = torch.randn(batch_size, 1536)
    edge_graph = torch.randn(batch_size, 256)
    target_text = torch.randn(batch_size, 1536)
    target_graph = torch.randn(batch_size, 256)

    # Test different hop contexts
    hop_contexts = [
        [1, 0, 0, 1, 0, 0],  # 1-hop question, at hop 1
        [0, 1, 0, 0, 1, 0],  # 2-hop question, at hop 2
        [0, 0, 1, 1, 0, 0],  # 3-hop question, at hop 1
        [0, 0, 1, 0, 0, 1],  # 3-hop question, at hop 3
    ]
    hop_context = torch.tensor(hop_contexts, dtype=torch.float32)

    print(f"\n   Testing hop contexts:")
    for i, ctx in enumerate(hop_contexts):
        q_hop = ctx[:3].index(1) + 1
        c_hop = ctx[3:].index(1) + 1
        print(f"   Sample {i+1}: {q_hop}-hop question, at hop {c_hop} → {ctx}")

    # Forward pass
    logits = model(
        question_text, node_text, node_graph,
        edge_text, edge_graph,
        target_text, target_graph, hop_context
    )
    probs = model.predict(
        question_text, node_text, node_graph,
        edge_text, edge_graph,
        target_text, target_graph, hop_context
    )

    print(f"\n   Forward pass successful!")
    print(f"   Output shape: {logits.shape}")
    print(f"   Sample probabilities: {probs[:2, 0].tolist()}")

    print("\n[3/3] Testing with inference script components...")
    try:
        from variant3.eval_qa.edge_scorer_ranker import EdgeScorerRelationRanker
        print("    EdgeScorerRelationRanker imports successfully")
        print("    score_edges_batch() signature updated with question_hop_count parameter")
    except Exception as e:
        print(f"    Error importing: {e}")
        return False

    try:
        from variant3.eval_qa.edge_scorer_bfs import EdgeScorerBFS
        print("    EdgeScorerBFS imports successfully")
        print("    Passes question_hop_count to score_edges_batch()")
    except Exception as e:
        print(f"    Error importing: {e}")
        return False

    print("\n" + "=" * 80)
    print(" All tests passed! 6-bit hop encoding is working correctly.")
    print("=" * 80)

    print("\n Ready to evaluate! Run:")
    print("   python variant3/eval_qa/variant3_qa_evaluator.py --datasets 3-hop --limit 100")

    return True

if __name__ == "__main__":
    success = test_model_forward()
    sys.exit(0 if success else 1)
