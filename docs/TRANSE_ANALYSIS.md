# TransE Performance Analysis and Fixes

## Executive Summary

Your TransE embeddings **are actually working correctly**, but your validation script was using the wrong metrics (cosine similarity instead of L2 distance). This document explains the issue and provides fixes.

## The Problem

### What You Observed
- **FastRP**: Connected similarity = 0.563, ✅ PASS
- **Node2Vec**: Connected similarity = 0.696, ✅ PASS
- **TransE**: Connected similarity = 0.082, ⚠️ WEAK

### Root Cause

TransE embeddings are **fundamentally different** from FastRP/Node2Vec:

| Aspect | FastRP/Node2Vec | TransE |
|--------|----------------|---------|
| **Goal** | Similar embeddings for connected nodes | Translation: h + r ≈ t |
| **Metric** | **Cosine Similarity** | **L2/L1 Distance** |
| **Edge Types** | Ignored | Essential (requires relation embeddings) |
| **Evaluation** | Proximity-based | Link prediction with relations |

## Why Cosine Similarity Fails for TransE

### TransE Optimization
TransE learns embeddings such that:
```
h + r ≈ t
```

Where:
- `h` = head entity embedding
- `r` = relation embedding
- `t` = tail entity embedding

The model minimizes: `||h + r - t||₂` (L2 distance)

### Your Validation Script
```python
# This is what your validation does (WRONG for TransE):
sim = cosine_similarity(emb[h], emb[t])
```

This **ignores relations** and uses the **wrong metric**!

### Correct TransE Evaluation
```python
# Correct TransE evaluation:
score = -||h + r - t||₂  # Higher score = better
```

## Evidence TransE IS Working

From your [embeddings/transe_metadata.json](../embeddings/transe_metadata.json):

```json
{
  "evaluation_metrics": {
    "both": {
      "realistic": {
        "hits_at_10": 0.04992781520692974,  // ~5% accuracy at top-10
        "hits_at_5": 0.02863330125120308,   // ~3% accuracy at top-5
        "hits_at_3": 0.016201475777991657   // ~1.6% accuracy at top-3
      }
    }
  }
}
```

For a knowledge graph with **43,234 entities**, getting **5% Hits@10** is reasonable! Random chance would be ~0.02%.

## Fixes Applied

### 1. Updated Training Script ([train_embeddings.py](../train_embeddings.py))

**Added relation embedding extraction and saving:**

```python
# Extract relation embeddings (line 396-399)
relation_embeddings = result.model.relation_representations[0](
    indices=None
).detach().cpu().numpy()

# Updated return signature (line 438)
return embeddings, metadata, relation_embeddings, train_tf

# Updated save method to include relations (line 464-468)
if relation_embeddings is not None:
    rel_file = output_path / f"{method_name}_relation_embeddings{suffix}.npy"
    np.save(rel_file, relation_embeddings)
```

### 2. Added TransE-Specific Validation ([validate_embeddings.py](../validate_embeddings.py))

**New method `transe_link_prediction_check()` (line 503-590):**

```python
def transe_link_prediction_check(self, emb: np.ndarray, name: str):
    """TransE-specific validation using L2 distance."""

    # Load relation embeddings
    rel_emb = np.load(rel_emb_file)

    # For each edge (h, r, t):
    score_connected = -np.linalg.norm(h_emb + r_emb - t_emb)

    # Compare to random pairs
    score_random = -np.linalg.norm(h_emb + r_emb - random_t_emb)

    # Connected edges should have HIGHER scores
    return scores and verdict
```

## Recommendations

### 1. Retrain TransE with Updated Script

```bash
python train_embeddings.py --method transe --epochs 100
```

This will now save relation embeddings needed for proper validation.

### 2. Run Updated Validation

```bash
python validate_embeddings.py --embedding transe_embeddings.npy --method transe
```

You should see:
- ⚠️ Warning about cosine similarity (expected)
- ✅ TransE-specific link prediction check (new!)

### 3. Improve TransE Performance (Optional)

If you want better TransE results, try:

**A. Increase Training Epochs**
```bash
python train_embeddings.py --method transe --epochs 300
```

**B. Tune Hyperparameters**
```bash
python train_embeddings.py --method transe \
    --epochs 200 \
    --batch-size 4096 \
    --lr 0.001 \
    --dim 256
```

**C. Try Different TransE Variants**

In [train_embeddings.py](../train_embeddings.py:360), change:
```python
model='TransE',  # Try: 'TransH', 'TransR', 'DistMult', 'ComplEx'
```

### 4. Understanding the Results

**Low cosine similarity (0.082) is EXPECTED for TransE!**

- TransE doesn't optimize for node proximity
- It optimizes for relational patterns
- Evaluation must use L2 distance + relations

**Your TransE model achieved:**
- Hits@10: 5% (vs. 0.02% random)
- This is ~250x better than random!

## When to Use Each Method

| Use Case | Recommended Method |
|----------|-------------------|
| Node classification | FastRP, Node2Vec |
| Community detection | Node2Vec |
| Link prediction (typed) | **TransE** |
| QA with relations | **TransE** |
| Fast training | FastRP |
| Best for MetaQA | Node2Vec or TransE |

## Next Steps

1. ✅ **Retrain TransE** with updated script to save relations
2. ✅ **Re-validate** with TransE-specific metrics
3. **Compare** all three methods for your QA task
4. **Tune** hyperparameters based on downstream performance

## Key Takeaway

**TransE is NOT broken!** Your validation was using the wrong metric. After retraining with the updated script and using proper L2-based evaluation, you'll see TransE's true performance.

---

**Modified Files:**
- [train_embeddings.py](../train_embeddings.py:391-438) - Saves relation embeddings
- [validate_embeddings.py](../validate_embeddings.py:503-652) - Added TransE validation

**Generated:** 2025-10-14
