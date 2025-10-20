# Embedding Validation Guide

This guide explains how to validate your graph embeddings using the comprehensive validation suite.

## 📋 Overview

The validation script (`validate_embeddings.py`) performs:

### Universal Sanity Checks
- **Cosine Similarity Distribution**: Compares similarity between connected vs random node pairs
- **t-SNE/UMAP Visualization**: 2D projection to visualize embedding structure
- **Nearest Neighbors**: Checks semantic proximity of related entities
- **Path Distance Correlation**: Validates if embedding distance reflects graph distance

### Model-Specific Checks
- **FastRP**: Degree correlation (structural similarity)
- **Node2Vec**: Cluster analysis (community structure)
- **TransE**: (Currently uses universal checks; relation-specific checks require relation embeddings)

### Downstream Validation
- **QA Probe Test**: Tests if correct answers are in top-k nearest neighbors

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate your virtual environment
source venv/bin/activate

# Install required packages
pip install matplotlib seaborn scikit-learn umap-learn
```

### 2. Validate a Specific Embedding

```bash
# Validate FastRP with 5 iterations
python validate_embeddings.py --embedding fastrp_embeddings_iter5.npy

# Validate Node2Vec with 200 walks
python validate_embeddings.py --embedding node2vec_embeddings_walks200.npy --method node2vec

# Validate TransE
python validate_embeddings.py --embedding transe_embeddings.npy --method transe
```

### 3. Validate All Embeddings

```bash
# Run validation on all embeddings in the embeddings/ directory
python validate_embeddings.py --all
```

---

## 📊 Understanding the Results

### Output Files

The validation creates several outputs in `validation_results/`:

1. **Visualizations** (PNG files):
   - `{embedding}_tsne.png` - t-SNE 2D projection
   - `{embedding}_umap.png` - UMAP 2D projection (if available)
   - `{embedding}_degree_correlation.png` - Degree vs position (FastRP)
   - `{embedding}_path_correlation.png` - Embedding distance vs graph distance

2. **JSON Reports**:
   - `{embedding}_validation.json` - Detailed results for each embedding
   - `validation_summary.json` - Comparison of all embeddings

### Interpreting Metrics

#### ✅ Good Signs

| Check | Good Result | What It Means |
|-------|-------------|---------------|
| **Connected Similarity** | > 0.6 | Embeddings preserve graph topology |
| **Connected - Random** | > 0.2 | Clear separation between related/unrelated nodes |
| **Path Correlation** | > 0.5 | Embedding distance reflects graph distance |
| **QA Accuracy@10** | > 60% | Useful for question answering |
| **Intra > Inter** (Node2Vec) | Difference > 0.2 | Strong community structure |
| **Degree Correlation** (FastRP) | |r| > 0.3 | Structural similarity captured |

#### ⚠️ Warning Signs

- Connected similarity ≈ Random similarity → Embeddings don't capture structure
- Low path correlation < 0.3 → Not useful for path-based reasoning
- QA accuracy < 40% → Poor for retrieval tasks

---

## 📈 Example Output

```
========================================================================
📊 Cosine Similarity Check: fastrp_embeddings_iter5
========================================================================

   Connected pairs avg similarity: 0.7234
   Random pairs avg similarity:    0.2156
   Difference:                     0.5078

   ✅ PASS - Clear separation between connected and random

========================================================================
🛤️  Path Distance Correlation: fastrp_embeddings_iter5
========================================================================

   Valid paths found: 87
   Correlation (emb_dist vs path_length): 0.6523

   ✅ PASS - Strong correlation

========================================================================
❓ QA Probe Test: fastrp_embeddings_iter5
========================================================================

   Valid questions: 73
   Hits@10:        51 (69.9%)

   ✅ PASS - Good QA performance
```

---

## 🎯 Choosing the Best Embedding

The validation script automatically recommends embeddings for different tasks:

### For Question Answering (QA)
- Look for highest **QA Accuracy@10**
- Typically Node2Vec performs well (captures neighborhood similarity)

### For Path Finding / A* Search
- Look for highest **Path Correlation**
- FastRP or Node2Vec usually perform best

### For Relational Reasoning
- TransE is designed for this
- Check if relation patterns are captured (requires additional analysis)

---

## 🔧 Advanced Usage

### Custom Parameters

```bash
# Use different graph file
python validate_embeddings.py --embedding fastrp_embeddings_iter5.npy \
    --graph data/metaqa/graph.pkl

# Use different QA dataset
python validate_embeddings.py --all \
    --qa-path data/metaqa/2-hop/vanilla/qa_test.txt

# Custom output directory
python validate_embeddings.py --all \
    --output my_validation_results
```

### Comparing Different Iterations

To compare how increasing iterations affects FastRP:

```bash
python validate_embeddings.py --all

# Check validation_summary.json for:
# - fastrp_iter3
# - fastrp_iter5
# - fastrp_iter10
# - fastrp_iter20
```

---

## 🐛 Troubleshooting

### UMAP Not Available
- Optional dependency, script falls back to PCA
- Install with: `pip install umap-learn`

### Memory Issues
- Reduce sample size in the script
- Validate one embedding at a time instead of --all

### Missing QA Data
- QA probe will be skipped
- Other validations still run

---

## 📚 Validation Methodology

The validation is based on:

1. **Structural Validation**: Embeddings should preserve graph topology
   - Connected nodes should be closer than random pairs
   - Embedding distance should correlate with graph distance

2. **Semantic Validation**: Embeddings should capture meaning
   - Nearest neighbors should be semantically related
   - Clusters should represent communities

3. **Task Validation**: Embeddings should be useful downstream
   - QA retrieval accuracy
   - Path finding efficiency

Each embedding method has different strengths:
- **FastRP**: Fast, captures structural similarity (node roles)
- **Node2Vec**: Captures neighborhood/community structure  
- **TransE**: Captures relational semantics (h + r ≈ t)

---

## 📝 Summary Checklist

After running validation:

- [ ] Check cosine similarity: connected > random + 0.2?
- [ ] Examine t-SNE plots: clear clusters visible?
- [ ] Review nearest neighbors: semantically relevant?
- [ ] Check path correlation: > 0.5 for reasoning tasks?
- [ ] Verify QA accuracy: > 60% for retrieval tasks?
- [ ] Compare embeddings: which works best for your use case?

The `validation_summary.json` provides a complete comparison to help you choose the best embedding for your A* search and reasoning tasks.

