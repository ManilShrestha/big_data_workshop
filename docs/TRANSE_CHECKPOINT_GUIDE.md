# TransE Training with Checkpoints - Usage Guide

## Overview

I've added checkpoint support to TransE training, allowing you to save embeddings at multiple training stages (epochs) and compare performance over time.

## Quick Start

### Basic Training (No Checkpoints)

```bash
# Train TransE for 100 epochs (default)
python train_embeddings.py --method transe --epochs 100
```

### Training with Checkpoints

```bash
# Save checkpoints at epochs 50, 100, and 200
python train_embeddings.py --method transe --epochs 200 --checkpoints 50 100 200
```

This will save:
- `transe_embeddings_epochs50.npy` + `transe_relation_embeddings_epochs50.npy`
- `transe_embeddings_epochs100.npy` + `transe_relation_embeddings_epochs100.npy`
- `transe_embeddings_epochs200.npy` + `transe_relation_embeddings_epochs200.npy`

## Examples

### Example 1: Quick Evaluation at Multiple Epochs

```bash
# Train with checkpoints every 25 epochs up to 100
python train_embeddings.py --method transe \
    --epochs 100 \
    --checkpoints 25 50 75 100
```

**Output files:**
```
embeddings/
├── transe_embeddings_epochs25.npy
├── transe_relation_embeddings_epochs25.npy
├── transe_metadata_epochs25.json
├── transe_embeddings_epochs50.npy
├── transe_relation_embeddings_epochs50.npy
├── transe_metadata_epochs50.json
├── transe_embeddings_epochs75.npy
├── transe_relation_embeddings_epochs75.npy
├── transe_metadata_epochs75.json
├── transe_embeddings_epochs100.npy
├── transe_relation_embeddings_epochs100.npy
└── transe_metadata_epochs100.json
```

### Example 2: Long Training with Periodic Checkpoints

```bash
# Train for 300 epochs with checkpoints every 50 epochs
python train_embeddings.py --method transe \
    --epochs 300 \
    --checkpoints 50 100 150 200 250 300
```

### Example 3: Fine-tuning Exploration

```bash
# Check if model is converged or needs more training
python train_embeddings.py --method transe \
    --epochs 500 \
    --checkpoints 100 200 300 400 500 \
    --lr 0.001
```

### Example 4: Hyperparameter Tuning

```bash
# Larger embeddings with more training
python train_embeddings.py --method transe \
    --dim 256 \
    --epochs 200 \
    --checkpoints 50 100 150 200 \
    --lr 0.001 \
    --batch-size 4096
```

## How It Works

### Incremental Training

The checkpoint system trains **incrementally**:

1. Train from epoch 0 → checkpoint 1
2. Continue training from checkpoint 1 → checkpoint 2
3. Continue training from checkpoint 2 → checkpoint 3
4. ... and so on

This ensures:
- ✅ Consistent model evolution
- ✅ Same random seed across checkpoints
- ✅ GPU memory efficiency
- ✅ Faster than separate training runs

### What Gets Saved

For each checkpoint at epoch N, you get:

**Files:**
- `transe_embeddings_epochsN.npy` - Entity embeddings (43234 × 128)
- `transe_relation_embeddings_epochsN.npy` - Relation embeddings (9 × 128)
- `transe_metadata_epochsN.json` - Training metadata + evaluation metrics

**Metadata includes:**
```json
{
  "method": "TransE",
  "num_epochs": 50,
  "total_planned_epochs": 200,
  "training_time_seconds": 26.5,
  "checkpoint": true,
  "evaluation_metrics": {
    "both": {
      "realistic": {
        "hits_at_10": 0.0499,
        "hits_at_5": 0.0286,
        ...
      }
    }
  }
}
```

## Validation

### Validate All Checkpoints

```bash
# After training with checkpoints, validate all of them
python validate_embeddings.py --all
```

This will validate each checkpoint separately and generate comparison reports.

### Validate Specific Checkpoint

```bash
python validate_embeddings.py --embedding transe_embeddings_epochs100.npy --method transe
```

## Comparing Checkpoints

### Method 1: Check Metadata

```bash
# View Hits@10 progression
for f in embeddings/transe_metadata_epochs*.json; do
    echo "=== $f ==="
    jq '.num_epochs, .evaluation_metrics.both.realistic.hits_at_10' "$f"
done
```

### Method 2: Python Script

```python
import json
import glob
import matplotlib.pyplot as plt

# Load all checkpoint metadata
checkpoints = []
for file in sorted(glob.glob('embeddings/transe_metadata_epochs*.json')):
    with open(file) as f:
        meta = json.load(f)
        checkpoints.append({
            'epochs': meta['num_epochs'],
            'hits_at_10': meta['evaluation_metrics']['both']['realistic']['hits_at_10'],
            'time': meta['training_time_seconds']
        })

# Plot performance over training
epochs = [c['epochs'] for c in checkpoints]
hits = [c['hits_at_10'] for c in checkpoints]

plt.plot(epochs, hits, marker='o')
plt.xlabel('Training Epochs')
plt.ylabel('Hits@10')
plt.title('TransE Performance vs Training Epochs')
plt.grid(True)
plt.savefig('transe_convergence.png')
```

## Performance Expectations

### Training Time (GPU)

Based on your current setup (43K entities, 9 relations):

| Epochs | Approx Time | Checkpoints | Total Time |
|--------|-------------|-------------|------------|
| 100 | ~53s | None | ~53s |
| 100 | ~55s | 50, 100 | ~55s |
| 200 | ~100s | 50, 100, 150, 200 | ~105s |
| 500 | ~250s | every 100 | ~260s |

**Note:** Checkpoint overhead is minimal (~5-10%)

### Expected Hits@10 Progression

Typical learning curve for TransE on knowledge graphs:

```
Epochs    Hits@10    Status
------    -------    ------
25        ~2-3%      Early learning
50        ~4-5%      Steady improvement
100       ~5-6%      Baseline performance
200       ~6-7%      Diminishing returns
300+      ~7-8%      Possible overfitting
```

Your current result (100 epochs → 5% Hits@10) is on track!

## Advanced Options

### Full Command Reference

```bash
python train_embeddings.py --method transe \
    --epochs 300 \              # Maximum training epochs
    --checkpoints 50 100 200 \  # Save at these epochs
    --dim 128 \                 # Embedding dimension (64, 128, 256, 512)
    --batch-size 2048 \         # Batch size (higher = faster, more memory)
    --lr 0.01 \                 # Learning rate (0.001, 0.01, 0.1)
    --output embeddings         # Output directory
```

### Checkpoint Strategies

**Strategy 1: Early Stopping Detection**
```bash
--epochs 500 --checkpoints 50 100 150 200 250 300 350 400 450 500
# Review results, stop if plateau detected
```

**Strategy 2: Quick Baseline**
```bash
--epochs 100 --checkpoints 25 50 75 100
# Find if more training helps
```

**Strategy 3: Production Training**
```bash
--epochs 300 --checkpoints 100 200 300
# Final model + backup checkpoints
```

## Troubleshooting

### Issue: Training Too Slow

```bash
# Increase batch size (needs more GPU memory)
python train_embeddings.py --method transe --batch-size 4096

# Or reduce evaluation overhead
python train_embeddings.py --method transe --checkpoints 100  # Single checkpoint
```

### Issue: Poor Performance

```bash
# Try more training
python train_embeddings.py --method transe --epochs 300 --checkpoints 100 200 300

# Or tune hyperparameters
python train_embeddings.py --method transe --lr 0.001 --dim 256
```

### Issue: Out of Memory

```bash
# Reduce batch size
python train_embeddings.py --method transe --batch-size 1024

# Or reduce embedding dimension
python train_embeddings.py --method transe --dim 64
```

## Checkpoint vs No Checkpoint

### Without Checkpoints
```bash
python train_embeddings.py --method transe --epochs 100
```
- Saves only final model
- Faster (no intermediate evaluation)
- Good for production

### With Checkpoints
```bash
python train_embeddings.py --method transe --epochs 100 --checkpoints 25 50 75 100
```
- Saves multiple versions
- Track learning progress
- Find optimal stopping point
- Good for experimentation

## Next Steps

1. **Train with checkpoints:**
   ```bash
   python train_embeddings.py --method transe --epochs 200 --checkpoints 50 100 150 200
   ```

2. **Validate all checkpoints:**
   ```bash
   python validate_embeddings.py --all
   ```

3. **Compare results:**
   - Check `validation_results/` for each checkpoint
   - Review Hits@10 progression
   - Choose best checkpoint for your QA task

4. **Fine-tune if needed:**
   - If still improving at epoch 200 → train longer
   - If plateau early → try different hyperparameters

## See Also

- [TRANSE_ANALYSIS.md](TRANSE_ANALYSIS.md) - Why TransE uses different metrics
- [QUICK_FIX_GUIDE.md](QUICK_FIX_GUIDE.md) - Setting up TransE validation
- [EMBEDDING_COMPARISON.md](EMBEDDING_COMPARISON.md) - Compare all methods

---

**Key Takeaway:** Checkpoints let you track TransE learning over time and find the optimal stopping point without multiple training runs!
