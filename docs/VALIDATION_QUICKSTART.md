# 🚀 Embedding Validation - Quick Start

## Installation

```bash
source venv/bin/activate
pip install -r validation_requirements.txt
```

## Usage

### Validate Single Embedding
```bash
python validate_embeddings.py --embedding fastrp_embeddings_iter5.npy
```

### Validate All Embeddings
```bash
python validate_embeddings.py --all
```

### Compare Specific Embeddings
```bash
# FastRP with different iterations
python validate_embeddings.py --embedding fastrp_embeddings_iter5.npy
python validate_embeddings.py --embedding fastrp_embeddings_iter10.npy

# Node2Vec with different walks
python validate_embeddings.py --embedding node2vec_embeddings_walks50.npy
python validate_embeddings.py --embedding node2vec_embeddings_walks200.npy
```

---

## 📊 Key Metrics Explained

### ✅ What "GOOD" Looks Like

| Metric | Good Value | Meaning |
|--------|-----------|---------|
| **Connected Similarity** | > 0.6 | Neighbors are close in embedding space |
| **Difference (Conn - Random)** | > 0.3 | Clear separation between related/unrelated |
| **QA Accuracy@10** | > 60% | Good for question answering |
| **Path Correlation** | > 0.5 | Useful for A* heuristic |
| **Intra > Inter Cluster** | Diff > 0.2 | Strong community structure (Node2Vec) |

### 📈 Example Results (Node2Vec walks50)

```
Connected pairs avg similarity: 0.6963  ✅
Random pairs avg similarity:    0.1586
Difference:                     0.5377  ✅ (> 0.3)

QA Hits@10:        81/100 (81.0%)      ✅ (> 60%)

Nearest Neighbors for "Brad Pitt":
  - brad pitt (0.93)
  - The Dark Side of the Sun (0.75)
  - Johnny Suede (0.75)
  - Movies he acted in! ✅
```

---

## 🎯 Choosing Best Embedding

### For A* Heuristic (Path Finding)
**→ Look for highest Path Correlation**
- FastRP iter5: Good structural similarity
- Node2Vec walks200: Best neighborhood capture

### For QA/Retrieval
**→ Look for highest QA Accuracy**
- Node2Vec: 81-84% (excellent!)
- FastRP: 73% (good)

### Check Results
```bash
cat validation_results/validation_summary.json
```

---

## 📁 Output Files

After running validation, check:

```
validation_results/
├── {embedding}_validation.json     # Detailed metrics
├── {embedding}_tsne.png            # t-SNE visualization
├── {embedding}_umap.png            # UMAP visualization
├── {embedding}_degree_correlation.png  # FastRP specific
└── validation_summary.json         # Compare ALL embeddings
```

---

## 🐛 Common Issues

### OpenBLAS Threading Error
**FIXED** - Script now automatically sets thread limits

### "Not enough valid paths" Warning
**Normal** - Some nodes are isolated, validation continues

### Memory Issues
**Solution** - Validate one at a time instead of `--all`

---

## 📊 Current Best Performers

Based on validation results:

**🥇 Node2Vec (walks50):**
- QA Accuracy: 81-84% ✅
- Connected Sim: 0.70 ✅
- Best for: Question Answering

**🥈 FastRP (iter5):**
- QA Accuracy: 73% ✅
- Connected Sim: 0.56 ✅
- Best for: Fast training, decent performance

**🥉 Node2Vec (walks200):**
- Likely even better (validate to confirm!)
- More walks = better neighborhood capture

---

## 💡 Quick Tips

1. **First time?** Run: `python validate_embeddings.py --all`

2. **Compare iterations?** Check `validation_summary.json`

3. **Need visualization?** Look at `*_tsne.png` and `*_umap.png`

4. **For A* search?** Use embedding with:
   - High connected similarity (> 0.6)
   - Good QA accuracy (> 60%)
   - Strong path correlation (if available)

5. **Fast decision?** Node2Vec walks50+ is your safest bet! 🎯

---

## 📚 Full Documentation

- **Detailed Guide:** `VALIDATION_README.md`
- **Validation Script:** `validate_embeddings.py`
- **Requirements:** `validation_requirements.txt`

---

## ✅ Validation Checklist

Run this after training embeddings:

- [ ] Install dependencies: `pip install -r validation_requirements.txt`
- [ ] Run validation: `python validate_embeddings.py --all`
- [ ] Check summary: `cat validation_results/validation_summary.json`
- [ ] Review visualizations: `ls validation_results/*.png`
- [ ] Choose best embedding based on your use case
- [ ] Use in A* search or QA system!

**Happy Validating! 🎉**

