# Graph Embeddings for MetaQA

This document describes the embedding generation system for the MetaQA knowledge graph.

## Overview

We've implemented **three embedding methods** that only require the graph structure (no QA training data):

1. **FastRP** - Fast Random Projection (fastest, unsupervised)
2. **Node2Vec** - Random walk-based embeddings (popular, flexible)
3. **TransE** - Knowledge Graph Embedding (best for reasoning, GPU-accelerated)

---

## Quick Start

### Train Individual Methods

```bash
# FastRP (fastest - ~5 seconds)
python train_embeddings.py --method fastrp

# Node2Vec with 200 walks and 20 epochs
python train_embeddings.py --method node2vec --walks 200 --n2v-epochs 20

# TransE with early stopping
python train_embeddings.py --method transe --epochs 100 --patience 10

# Train all methods
python train_embeddings.py --method all
```

### Node2Vec with Checkpoints (for ablation study)

```bash
python train_embeddings.py --method node2vec \
  --walks 200 \
  --checkpoints 10 50 100 200 \
  --n2v-epochs 20
```

This creates embeddings at 10, 50, 100, and 200 walks for comparison.

---

## Features

### ✅ **Modular Training**
- Train each method independently
- No need to retrain if one method fails
- Separate argparse for each configuration

### ✅ **Comprehensive Logging**
- **Datetime timestamps**: `[2025-10-14 11:17:17]`
- **Elapsed time**: `[1.90s]`
- **Saved to**: `logs/train_{method}_{timestamp}.log`

### ✅ **Early Stopping for TransE**
- Monitors validation metrics (hits@10)
- Stops if no improvement for `patience` epochs
- Saves training time and prevents overfitting

### ✅ **Checkpointing for Node2Vec**
- Save embeddings at different walk counts
- Perfect for ablation studies
- Compare quality vs training time tradeoffs

### ✅ **GPU Acceleration**
- TransE automatically uses NVIDIA A40
- Large batch sizes (2048) for faster training
- Proper loss convergence tracking

---

## All Parameters

```bash
python train_embeddings.py --help
```

### General
- `--method` : {fastrp, node2vec, transe, all} (required)
- `--dim` : Embedding dimension (default: 128)
- `--graph` : Path to graph pickle (default: data/metaqa/graph.pkl)
- `--output` : Output directory (default: embeddings)

### FastRP
- `--iterations` : Propagation iterations (default: 5)

### Node2Vec
- `--walks` : Number of walks per node (default: 200)
- `--walk-length` : Walk length (default: 30)
- `--n2v-epochs` : Word2Vec epochs (default: 10)
- `--workers` : Parallel workers (default: 8)
- `--checkpoints` : Walk counts for checkpoints (e.g., 10 50 100 200)

### TransE
- `--epochs` : Max training epochs (default: 100)
- `--batch-size` : Batch size (default: 2048)
- `--lr` : Learning rate (default: 0.01)
- `--patience` : Early stopping patience (default: 10)

---

## Output Files

After training, you'll have:

```
embeddings/
├── fastrp_embeddings.npy             # FastRP embeddings
├── fastrp_metadata.json              # FastRP config + timing
├── node2vec_embeddings_walks10.npy   # Node2Vec checkpoint 1
├── node2vec_embeddings_walks50.npy   # Node2Vec checkpoint 2
├── node2vec_embeddings_walks100.npy  # Node2Vec checkpoint 3
├── node2vec_embeddings_walks200.npy  # Node2Vec checkpoint 4
├── node2vec_metadata_walks*.json     # Metadata for each checkpoint
├── transe_embeddings.npy             # TransE embeddings
├── transe_metadata.json              # TransE config + metrics
└── node2id.json                      # Node to ID mapping (shared)

logs/
├── train_fastrp_YYYYMMDD_HHMMSS.log
├── train_node2vec_YYYYMMDD_HHMMSS.log
└── train_transe_YYYYMMDD_HHMMSS.log
```

---

## Metadata JSON Example

Each method saves detailed metadata:

### FastRP
```json
{
  "method": "FastRP",
  "embedding_dim": 128,
  "iterations": 5,
  "normalization_strength": 0.5,
  "num_nodes": 43234,
  "training_time_seconds": 4.77,
  "timestamp": "2025-10-14T11:17:17"
}
```

### Node2Vec
```json
{
  "method": "Node2Vec",
  "embedding_dim": 128,
  "walk_length": 30,
  "num_walks": 200,
  "p": 1.0,
  "q": 1.0,
  "workers": 8,
  "w2v_epochs": 20,
  "num_nodes": 43234,
  "training_time_seconds": 180.5,
  "checkpoint": true,
  "timestamp": "2025-10-14T11:20:00"
}
```

### TransE
```json
{
  "method": "TransE",
  "embedding_dim": 128,
  "num_epochs_requested": 100,
  "num_epochs_actual": 65,
  "early_stopped": true,
  "batch_size": 2048,
  "learning_rate": 0.01,
  "patience": 10,
  "num_nodes": 43234,
  "num_relations": 9,
  "training_time_seconds": 145.2,
  "device": "cuda",
  "evaluation_metrics": {
    "hits@1": 0.234,
    "hits@10": 0.567,
    "mean_rank": 150.5
  },
  "timestamp": "2025-10-14T11:25:00"
}
```

---

## Training Data Requirements

### ❓ Do we need training data for embeddings?

**Answer: NO!**

- ❌ **FastRP & Node2Vec**: Only need graph structure
- ❌ **TransE**: Only needs KB triples (the edges in graph.pkl)
- ✅ **All three methods**: Use only `graph.pkl`

The **1-hop/2-hop/3-hop vanilla folders** are for:
- ✅ Testing your A* algorithm
- ✅ Evaluating question answering accuracy
- ❌ NOT for training embeddings

---

## Performance Comparison

| Method    | Training Time | Quality | Use Case |
|-----------|--------------|---------|----------|
| FastRP    | ~5 seconds   | Good    | Baseline, fast iteration |
| Node2Vec  | ~3-5 minutes | Better  | General graph embeddings |
| TransE    | ~2-5 minutes | Best    | KG reasoning, relation-aware |

*Times on NVIDIA A40 GPU with 43K nodes, 124K edges*

---

## For Your Paper

### Ablation Study
Use Node2Vec checkpoints to show:
- How embedding quality improves with more walks
- Training time vs accuracy tradeoff
- Optimal configuration selection

### Methods Comparison
Compare all three methods on:
- A* search accuracy (1-hop, 2-hop, 3-hop)
- Nodes expanded
- Runtime
- Heuristic quality

### Early Stopping Analysis
For TransE:
- Show convergence curves
- Demonstrate overfitting prevention
- Report actual vs requested epochs

---

## Next Steps

1. ✅ Generate all embeddings
2. ⏭️ Implement A* search with embedding-based heuristic
3. ⏭️ Evaluate on 1-hop/2-hop/3-hop test sets
4. ⏭️ Create comparison plots
5. ⏭️ Write paper results section

---

## Troubleshooting

### Node2Vec is slow
- Reduce `--walks` (try 50 or 100)
- Reduce `--n2v-epochs` (try 5)
- Increase `--workers`

### TransE not improving
- Check GPU is being used (should say "Device: CUDA")
- Increase `--patience` for early stopping
- Try different `--lr` (0.001 or 0.1)

### Out of memory
- Reduce `--batch-size` for TransE
- Use smaller `--dim` (try 64)

---

## Questions?

Check logs in `logs/` directory for detailed training information including:
- Exact timestamps
- Loss curves
- Validation metrics
- Early stopping decisions
