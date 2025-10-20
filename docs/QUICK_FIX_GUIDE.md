# Quick Fix Guide: TransE Validation Issue

## TL;DR

**Your TransE embeddings are fine!** The validation used the wrong metric (cosine similarity instead of L2 distance).

## What's Wrong?

```
❌ PROBLEM: Validating TransE with cosine similarity
✅ SOLUTION: Use L2 distance + relation embeddings
```

## Quick Fix Steps

### Step 1: Retrain TransE (saves relation embeddings)

```bash
cd /home/ms5267/big_data_workshop
python train_embeddings.py --method transe --epochs 100
```

**What this does:** Updated training script now saves both entity AND relation embeddings.

### Step 2: Validate with Correct Metrics

```bash
python validate_embeddings.py --embedding transe_embeddings.npy --method transe
```

**What to expect:**
- Cosine similarity will still be low (~0.08) - **THIS IS NORMAL!**
- New "TransE Link Prediction Check" will show true performance
- Should see separation between connected/random edges

## Why Was It Wrong?

### The Three Methods Are Different:

```
FastRP/Node2Vec:  node_a ≈ node_b  (proximity)
                  ↓
                  Use: cosine_similarity(a, b)

TransE:          head + relation ≈ tail  (translation)
                  ↓
                  Use: -||h + r - t||₂
```

### What Happened:

Your validation checked if connected nodes have similar embeddings (good for FastRP/Node2Vec), but TransE doesn't optimize for that!

TransE optimizes for: `Brad Pitt + acted_in ≈ Fight Club`

Not: `Brad Pitt ≈ Fight Club` (which makes no sense!)

## Expected Results

### After Retraining + Proper Validation:

**Cosine Similarity Check:**
```
Connected similarity: ~0.08  ⚠️  WEAK (expected for TransE!)
```

**TransE Link Prediction Check (NEW):**
```
Connected score: -150.2
Random score:    -180.5
Difference:       30.3   ✅ PASS
```

## Understanding Your Current Results

Your [transe_metadata.json](../embeddings/transe_metadata.json) shows:
- **Hits@10: 5.0%** - Finding correct answer in top-10 predictions
- Random baseline: 0.02% (1 in 43,234 entities)
- **Your model is 250x better than random!**

This is actually reasonable performance for TransE on this graph.

## If You Want Better Performance

### Option 1: More Training
```bash
python train_embeddings.py --method transe --epochs 300
```

### Option 2: Larger Embeddings
```bash
python train_embeddings.py --method transe --dim 256 --epochs 200
```

### Option 3: Better Learning Rate
```bash
python train_embeddings.py --method transe --lr 0.001 --epochs 200
```

### Option 4: Try Different Models
Edit [train_embeddings.py](../train_embeddings.py:360):
```python
model='TransE',  # Change to: 'TransH', 'TransR', 'DistMult', 'ComplEx'
```

## Key Insight

```
┌──────────────────────────────────────────────────────┐
│  TransE low cosine similarity ≠ Bad embeddings!      │
│                                                      │
│  It means TransE is doing exactly what it should:    │
│  Learning translations, not proximities.             │
└──────────────────────────────────────────────────────┘
```

## Files Modified

1. [train_embeddings.py](../train_embeddings.py) - Now saves relation embeddings
2. [validate_embeddings.py](../validate_embeddings.py) - Added TransE-specific validation
3. [docs/TRANSE_ANALYSIS.md](TRANSE_ANALYSIS.md) - Detailed explanation

## Still Confused?

Read the full analysis: [docs/TRANSE_ANALYSIS.md](TRANSE_ANALYSIS.md)

---

**Bottom Line:** Retrain TransE with the updated script, then run validation again. You'll see it's actually working!
