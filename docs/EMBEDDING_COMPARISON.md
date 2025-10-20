# Embedding Methods Comparison: FastRP vs Node2Vec vs TransE

## Performance Summary (Current Results)

| Method | Connected Sim | Random Sim | Separation | Status |
|--------|--------------|------------|------------|---------|
| **FastRP** (5 iter) | 0.563 | 0.029 | **0.534** | ✅ PASS |
| **Node2Vec** (50 walks) | 0.696 | 0.161 | **0.536** | ✅ PASS |
| **TransE** | 0.082 | -0.002 | 0.084 | ⚠️ WRONG METRIC! |

## Why TransE Appears "Weak"

TransE is evaluated with the **wrong metric**. Cosine similarity is not meaningful for TransE embeddings.

### Correct Evaluation

After retraining with the updated script:

| Method | Evaluation Metric | What It Measures |
|--------|------------------|------------------|
| FastRP | Cosine similarity | Structural proximity |
| Node2Vec | Cosine similarity | Community structure |
| **TransE** | **-‖h+r-t‖₂** | **Relational translation** |

## Method Characteristics

### FastRP (Fast Random Projection)

**How it works:**
1. Initialize random embeddings
2. Iteratively average neighbor embeddings
3. L2 normalize after each iteration

**Strengths:**
- ⚡ **Very fast** (5 iterations in ~5 seconds)
- 📊 Good structural properties
- 🎯 Captures node degree information

**Weaknesses:**
- Ignores edge types/relations
- Less expressive than deep methods
- Fixed iteration count

**Best for:**
- Quick baseline
- Large graphs
- Node classification
- Degree-aware tasks

**Your results:**
- Training time: 4.85 seconds
- Connected/random separation: 0.534
- Mean norm: 1.0 (normalized)

---

### Node2Vec (Random Walk + Word2Vec)

**How it works:**
1. Generate random walks from each node
2. Treat walks as "sentences"
3. Train Word2Vec (Skip-gram) model

**Strengths:**
- 🏆 **Best similarity preservation**
- 🎨 Rich community structure
- 🔧 Tunable (p, q parameters)

**Weaknesses:**
- 🐌 Slow training (50 walks: 27 minutes)
- 💾 Memory intensive
- Ignores edge types

**Best for:**
- Node classification
- Community detection
- Link prediction (untyped)
- When you have time to train

**Your results:**
- Training time: 1618 seconds (~27 min)
- Connected/random separation: 0.536
- **Highest cosine similarity** (0.696)

---

### TransE (Translation Embeddings)

**How it works:**
1. Convert graph to triples (head, relation, tail)
2. Learn: h + r ≈ t
3. Optimize: min ‖h + r - t‖₂

**Strengths:**
- 🎯 **Handles typed relations**
- 🔗 Great for link prediction
- ⚡ GPU accelerated
- 📊 Relation embeddings for reasoning

**Weaknesses:**
- Requires relation types
- Cannot be evaluated with cosine similarity
- 1-to-N relations challenging

**Best for:**
- **Knowledge graph completion**
- **Multi-hop reasoning**
- **Typed link prediction**
- **QA with relations**

**Your results:**
- Training time: 52 seconds (GPU)
- Hits@10: 5.0% (250x better than random!)
- ⚠️ Cosine similarity: 0.082 (EXPECTED - wrong metric!)

## When to Use Each Method

### Scenario-Based Guide

| Task | Best Choice | Why |
|------|-------------|-----|
| **Quick baseline** | FastRP | Fast, good results |
| **Node classification** | Node2Vec | Best similarity |
| **Community detection** | Node2Vec | Preserves communities |
| **Link prediction (no types)** | Node2Vec | High similarity |
| **Link prediction (typed)** | **TransE** | Uses relations |
| **Multi-hop QA** | **TransE** | Relation reasoning |
| **Large graph (>1M nodes)** | FastRP | Scalable |
| **Limited compute** | FastRP | Fastest |

### For MetaQA Specifically

MetaQA has **typed relations** (acted_in, directed_by, etc.):

```
┌─────────────────────────────────────────────────┐
│ Recommendation for MetaQA:                      │
│                                                 │
│ 1st Choice: TransE (if you need relations)     │
│ 2nd Choice: Node2Vec (if relations ignored)    │
│ 3rd Choice: FastRP (for quick experiments)     │
└─────────────────────────────────────────────────┘
```

## Performance-Complexity Trade-off

```
High Performance
    ↑
    │         Node2Vec ●
    │                / \
    │               /   \
    │         TransE     \
    │            ●        \
    │           /          \
    │      FastRP           \
    │         ●              \
    │                         \
    └──────────────────────────────→ Training Time
    Fast                          Slow
```

## Detailed Metrics Comparison

### Training Efficiency

| Metric | FastRP | Node2Vec | TransE |
|--------|--------|----------|---------|
| **Time** | 5s | 1618s | 53s |
| **Relative** | 1x | 324x | 11x |
| **Device** | CPU | CPU | GPU |
| **Memory** | Low | High | Medium |

### Embedding Quality (Your Graph: 43,234 nodes)

| Metric | FastRP | Node2Vec | TransE |
|--------|--------|----------|---------|
| **Connected Sim** | 0.563 | 0.696 | N/A* |
| **Separation** | 0.534 | 0.536 | N/A* |
| **QA Accuracy** | ? | ? | 5% @10 |

*N/A = Cosine similarity not applicable for TransE

## Understanding TransE Metrics

### What "Hits@K" Means

For each test triple (h, r, t):
1. Compute score: -‖h + r - tail‖ for ALL tails
2. Rank all entities by score
3. Check if correct tail is in top-K

**Your TransE Results:**
- Hits@1: 0.016% (correct answer is #1)
- Hits@3: 1.6% (correct answer in top-3)
- Hits@5: 2.9% (correct answer in top-5)
- **Hits@10: 5.0%** (correct answer in top-10)

**Context:**
- Random baseline: 1/43,234 = 0.002%
- Your model: 5.0% = **250x better than random**
- For comparison, state-of-the-art on FB15k: 30-50% Hits@10

## Improving Each Method

### FastRP Improvements

```bash
# Try more iterations
python train_embeddings.py --method fastrp --iterations 10

# Try checkpoints to find optimal iteration
python train_embeddings.py --method fastrp \
    --iterations 10 --iteration-checkpoints 1 2 5 10
```

### Node2Vec Improvements

```bash
# More walks for better coverage
python train_embeddings.py --method node2vec --walks 200

# Tune p and q (currently both = 1.0)
# p > 1: local exploration (BFS-like)
# q > 1: global exploration (DFS-like)
```

### TransE Improvements

```bash
# More epochs
python train_embeddings.py --method transe --epochs 300

# Larger embeddings + better learning rate
python train_embeddings.py --method transe \
    --dim 256 --epochs 200 --lr 0.001

# Try related models (edit train_embeddings.py:360)
model='TransH'  # or 'TransR', 'DistMult', 'ComplEx'
```

## Validation Checklist

After retraining TransE with the updated script:

- [ ] Retrain TransE: `python train_embeddings.py --method transe`
- [ ] Check relation embeddings saved: `ls embeddings/*relation*`
- [ ] Run validation: `python validate_embeddings.py --all`
- [ ] Review TransE link prediction check (NEW)
- [ ] Compare all three methods for your QA task

## Key Takeaways

1. **FastRP**: Fast, good baseline, ignores relations
2. **Node2Vec**: Best similarity, slow, ignores relations
3. **TransE**: Uses relations, different evaluation, REQUIRED for typed link prediction

4. **Your TransE is NOT broken!** It just needs:
   - Relation embeddings saved ✅ (fixed)
   - Proper L2-based validation ✅ (added)

5. **For MetaQA**: TransE is likely your best choice IF you want to leverage the typed relations in the knowledge graph.

## Next Steps

1. Retrain TransE with updated [train_embeddings.py](../train_embeddings.py)
2. Run validation with updated [validate_embeddings.py](../validate_embeddings.py)
3. Compare downstream QA performance of all three methods
4. Choose the best method for your use case

---

**See also:**
- [TRANSE_ANALYSIS.md](TRANSE_ANALYSIS.md) - Detailed TransE explanation
- [QUICK_FIX_GUIDE.md](QUICK_FIX_GUIDE.md) - Step-by-step fix
