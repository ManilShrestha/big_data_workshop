# MetaQA Knowledge Graph Reasoning - Project Context

## Project Overview

This is a research project for an IEEE paper on **LLM-Guided Knowledge Graph Reasoning using A* Search with Graph Embeddings**. The goal is to combine traditional graph traversal (A*) with modern LLMs and graph embeddings to answer multi-hop questions over knowledge graphs.

**Current Status:** Phase 1 Complete | Phase 2 Implementation Started
**Last Updated:** October 20, 2025

---

## Quick Start

### Key Commands

```bash
# Phase 1: Build graph & train embeddings
python build_graph.py
python train_embeddings.py --method all
python validate_embeddings.py --all

# Phase 2: Run QA ablation study (IN PROGRESS)
python run_ablation.py --variants 1 2 3      # BFS, FastRP, TransE (no OpenAI)
python run_ablation.py --variants 4 5        # With OpenAI entity linking
python run_ablation.py --all                 # All 5 variants
python analyze_results.py                    # Generate paper tables/plots
```

---

## Project Structure

```
big_data_workshop/
├── build_graph.py              # Phase 1: Build NetworkX graph from KB
├── train_embeddings.py         # Phase 1: Train FastRP/Node2Vec/TransE
├── validate_embeddings.py      # Phase 1: Validate embedding quality
│
├── qa_system/                  # Phase 2: QA implementation (NEW)
│   ├── entity_linkers/         # Entity extraction & linking
│   ├── search_algorithms/      # BFS, A* variants
│   ├── config.py               # Configuration
│   └── evaluator.py            # Metrics & evaluation
│
├── run_ablation.py             # Phase 2: Run 5-variant ablation study (NEW)
├── analyze_results.py          # Phase 2: Generate paper tables/plots (NEW)
│
├── data/metaqa/
│   ├── kb.txt                  # 134,741 KB triples
│   ├── graph.pkl               # NetworkX graph (43K nodes, 124K edges)
│   ├── 1-hop/vanilla/          # 96K QA pairs
│   ├── 2-hop/vanilla/          # 118K QA pairs
│   └── 3-hop/vanilla/          # 114K QA pairs
│
├── embeddings/
│   ├── fastrp_embeddings_iter*.npy
│   ├── node2vec_embeddings_walks*.npy
│   ├── transe_embeddings_epochs*.npy
│   ├── transe_relation_embeddings_epochs*.npy
│   ├── node2id.json            # Node-to-ID mapping
│   └── *_metadata.json         # Training metadata
│
├── results/                    # Phase 2: Ablation results (NEW)
├── logs/                       # Training logs
├── docs/                       # Detailed documentation
│
├── claude.md                   # Project overview (this file)
├── phase2_context.md           # Phase 2 detailed implementation plan (NEW)
├── context.md                  # Original 10-day roadmap
└── TRANSE_QUICK_REFERENCE.md   # TransE training guide
```

---

## Dataset Information

### MetaQA Knowledge Graph
- **Source:** https://github.com/yuyuz/MetaQA
- **KB Triples:** 134,741 (entity1|relation|entity2 format)
- **Nodes:** 43,234 unique entities
- **Edges:** 124,680 directed edges
- **Domain:** Movies (directors, actors, writers, release years, etc.)
- **Relations:** `directed_by`, `starred_actors`, `written_by`, `in_language`, etc.

### QA Datasets (For Evaluation - NOT Training)
- **1-hop questions:** 96,106 train examples
- **2-hop questions:** 118,980 train examples
- **3-hop questions:** 114,196 train examples
- **Format:** `question[TAB]answer1|answer2|...`

**IMPORTANT:** QA files are ONLY for evaluation. Embeddings are trained on the KB graph structure (unsupervised).

---

## Embedding Methods

### 1. FastRP (Fast Random Projection)
- **Type:** Unsupervised, structure-based
- **Speed:** ~4 seconds
- **Parameters:**
  - Embedding dim: 128
  - Iterations: 3-5 (optimal: 3-4)
  - Normalization: 0.5
- **Checkpoints:** `fastrp_embeddings_iter{1,2,3,4,5}.npy`
- **Best for:** Quick baseline, structural similarity

### 2. Node2Vec
- **Type:** Unsupervised, random walk + Word2Vec
- **Speed:** ~2-5 min per checkpoint
- **Parameters:**
  - Embedding dim: 128
  - Walk length: 30
  - Num walks: 10-200
  - p=1.0, q=1.0 (BFS/DFS bias)
  - Word2Vec epochs: 10
- **Checkpoints:** `node2vec_embeddings_walks{10,50,100,200}.npy`
- **Best for:** Community detection, node similarity

### 3. TransE
- **Type:** Knowledge graph embedding (entity + relation)
- **Speed:** ~15-30 min (GPU accelerated)
- **Parameters:**
  - Embedding dim: 128
  - Epochs: 50-300
  - Batch size: 2048
  - Learning rate: 0.01
  - Train/val split: 90/10
- **Outputs:**
  - Entity embeddings: `transe_embeddings_epochs{N}.npy` (43234 × 128)
  - Relation embeddings: `transe_relation_embeddings_epochs{N}.npy` (9 × 128)
- **Metrics:** Hits@10 (~5-8%), MRR
- **Best for:** Link prediction, relation-aware reasoning

**Key Insight:** TransE uses translation-based embeddings (h + r ≈ t), NOT cosine similarity. Low cosine similarity (0.08) is normal and expected!

---

## Implementation Phases

### Phase 1: Setup & Data Preparation ✅ COMPLETE

**Task 1:** Download MetaQA ✅
- Status: Complete - dataset in `data/metaqa/`

**Task 2:** Build graph object ✅
- Script: `build_graph.py`
- Output: `data/metaqa/graph.pkl` (8.2 MB)
- Graph: 43,234 nodes, 124,680 edges

**Task 3:** Compute embeddings ✅
- Script: `train_embeddings.py`
- Methods: FastRP, Node2Vec, TransE
- Features:
  - Modular training with argparse
  - Checkpoint support for ablation studies
  - GPU acceleration for TransE
  - Automatic logging to `logs/`
  - Metadata tracking (JSON)

### Phase 2: QA System & Ablation Study 🔄 IN PROGRESS

**Overview:** Implement 5-variant ablation study comparing different entity linking and search heuristics.

**See [phase2_context.md](phase2_context.md) for complete implementation details.**

#### 5 Ablation Variants

| # | Name | Entity Linking | Search Heuristic | Cost | Status |
|---|------|----------------|------------------|------|--------|
| 1 | BFS Baseline | Exact match | None (BFS) | $0 | TODO |
| 2 | FastRP A* | Exact match | FastRP cosine | $0 | TODO |
| **3** | **TransE A*** ⭐ | **Exact match** | **TransE distance** | **$0** | **TODO** |
| 4 | OpenAI+FastRP | OpenAI embeddings | FastRP cosine | $$$ | TODO |
| 5 | OpenAI+TransE | OpenAI embeddings | TransE distance | $$$ | TODO |

⭐ **Variant 3 is our main contribution** - demonstrating that relation-aware embeddings achieve near-optimal accuracy at zero inference cost.

#### Implementation Tasks

**Task 4:** Build entity linkers
- [x] Design architecture (see [phase2_context.md](phase2_context.md))
- [ ] Implement `ExactMatcher` (Variants 1-3)
- [ ] Implement `OpenAILinker` (Variants 4-5)

**Task 5:** Implement search algorithms
- [ ] BFS baseline (Variant 1)
- [ ] A* with FastRP heuristic (Variants 2, 4)
- [ ] A* with TransE heuristic (Variants 3, 5) - **Main method**

**Task 6:** Build evaluation framework
- [ ] Metrics tracking (accuracy, nodes expanded, cost)
- [ ] Question parsing & relation extraction
- [ ] Results logging & analysis

**Task 7:** Run ablation experiments
- [ ] Test on 1-hop questions (1K samples)
- [ ] Test on 2-hop questions (1K samples)
- [ ] Generate comparison tables for paper

### Phase 3: Paper Writing 🔄 TODO

**Task 8:** Create visualizations
- [ ] Accuracy comparison plots (all 5 variants)
- [ ] Cost-accuracy tradeoff analysis
- [ ] Sample reasoning paths
- [ ] System architecture diagram

**Task 9:** Write paper sections
- [ ] Abstract & Introduction
- [ ] Method (TransE-guided A*)
- [ ] Experiments & Ablations
- [ ] Results & Discussion
- [ ] Conclusion

### Phase 4: Final Submission 🔄 TODO

**Task 10:** Finalize & submit
- Polish paper based on results
- Prepare supplementary materials
- Submit to IEEE conference

---

## Key Technical Details

### Graph Construction
```python
# Format: entity1|relation|entity2
# Example: "The Dark Knight|directed_by|Christopher Nolan"
# Graph: NetworkX DiGraph with edge attribute 'relation'
```

### Node-to-ID Mapping
- All embeddings use consistent node ordering
- Mapping stored in `embeddings/node2id.json`
- Use `node2id[entity_name]` to get embedding index

### Training Pipeline
```
kb.txt → build_graph.py → graph.pkl → train_embeddings.py → embeddings/*.npy
                                                           ↓
                                    (90% train / 10% validation for TransE)
```

### Validation Metrics
- **FastRP/Node2Vec:** Cosine similarity for connected vs random pairs
- **TransE:** Link prediction score (h + r ≈ t), NOT cosine similarity
- **All methods:** Clustering coefficient, mean norm

---

## Important Notes & Gotchas

1. **Data Usage:**
   - ✅ `kb.txt` → Used for graph and embeddings
   - ❌ `qa_*.txt` → NOT used for embeddings, only for evaluation

2. **TransE Validation:**
   - DO NOT use cosine similarity (wrong metric!)
   - Use TransE-specific link prediction: score = ||h + r - t||
   - Low cosine similarity (0.08) is NORMAL

3. **GPU Acceleration:**
   - TransE automatically detects CUDA
   - Check logs for GPU name and memory
   - Batch size: Increase if you have more VRAM

4. **Checkpoints:**
   - Use `--checkpoints` flag to save intermediate models
   - Helps find optimal hyperparameters
   - Example: `--checkpoints 50 100 150 200`

5. **Ablation Studies:**
   - FastRP: Test iterations (1-5)
   - Node2Vec: Test walk counts (10, 50, 100, 200)
   - TransE: Test epochs (50, 100, 200, 300)

---

## Next Steps (Phase 2)

### Current Focus: 5-Variant Ablation Study

**See [phase2_context.md](phase2_context.md) for detailed implementation plan**

### Immediate TODO
1. ✅ Phase 1 complete: Embeddings trained and validated
2. ✅ Phase 2 architecture designed: 5-variant ablation study
3. 🔄 **NEXT:** Implement core infrastructure
   - Create `qa_system/` module structure
   - Implement `ExactMatcher` entity linker
   - Implement BFS baseline (Variant 1)
4. 🔄 Implement Variants 2-3 (FastRP A*, TransE A*)
5. 🔄 Add OpenAI integration (Variants 4-5)
6. 🔄 Run experiments & generate results

### Implementation Strategy
- **Priority 1:** Variants 1-3 (no OpenAI dependency) - prove TransE > FastRP
- **Priority 2:** Variants 4-5 (with OpenAI) - show cost scaling argument
- **Priority 3:** Analysis & visualization for paper

---

## References & Documentation

### Project Documentation
- **[phase2_context.md](phase2_context.md)** - **Phase 2 implementation plan (5-variant ablation study)** ⭐
- [context.md](context.md) - Original 10-day roadmap
- [TRANSE_QUICK_REFERENCE.md](TRANSE_QUICK_REFERENCE.md) - TransE training commands
- [docs/TRANSE_ANALYSIS.md](docs/TRANSE_ANALYSIS.md) - Why TransE validation was wrong
- [docs/TRANSE_CHECKPOINT_GUIDE.md](docs/TRANSE_CHECKPOINT_GUIDE.md) - Checkpoint training guide
- [docs/EMBEDDING_COMPARISON.md](docs/EMBEDDING_COMPARISON.md) - Compare all 3 methods
- [docs/VALIDATION_README.md](docs/VALIDATION_README.md) - Embedding validation guide

### External Resources
- **MetaQA:** https://github.com/yuyuz/MetaQA
- **PyKEEN (TransE):** https://pykeen.readthedocs.io/
- **Node2Vec:** https://arxiv.org/abs/1607.00653
- **TransE:** https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

---

## Research Goals

### Main Research Question
Can relation-aware graph embeddings (TransE) achieve competitive accuracy with expensive LLM-based methods while maintaining zero inference cost?

### Hypotheses
1. Graph embeddings provide better heuristics than uninformed search (BFS)
2. **TransE outperforms FastRP/Node2Vec for multi-hop reasoning** (relation-aware vs. structural)
3. Expensive LLM entity linking provides diminishing returns for structured KGs like MetaQA
4. **Our method (TransE A*) achieves 90%+ accuracy of OpenAI methods at 0% cost**

### Metrics to Report
- **Accuracy:** Correct answers / Total questions (1-hop, 2-hop)
- **Efficiency:** Avg nodes expanded per question
- **Runtime:** Avg milliseconds per question
- **Cost:** Per-query cost (training + inference)
- **Comparison:** All 5 variants on 1-hop and 2-hop datasets

---

## Development Workflow

### When making changes
1. Always log to `logs/` directory with timestamps
2. Save checkpoints for long-running tasks
3. Update metadata JSON files
4. Run validation after training embeddings
5. Keep track of hyperparameters in filenames

### Git workflow
- Main branch: `main`
- Ignore: venv/, data/, embeddings/, logs/, *.pkl (see .gitignore)
- Track: Python scripts, docs, requirements.txt

### Dependencies
- Python 3.10+
- NetworkX, NumPy, scikit-learn
- node2vec (optional, for Node2Vec)
- PyKEEN, PyTorch (optional, for TransE)
- See requirements files for complete list

---

## Common Commands

```bash
# Check embedding files
ls -lh embeddings/

# View training logs
cat logs/train_transe_*.log

# Quick validation
python validate_embeddings.py --all

# Compare TransE checkpoints
for f in embeddings/transe_metadata_epochs*.json; do
    echo "$f:"
    jq '.num_epochs, .evaluation_metrics.both.realistic.hits_at_10' "$f"
    echo
done

# Export graph for visualization
python export_for_gephi.py

# Git status
git status
```

---

## Summary

### Current Status (October 20, 2025)

**Phase 1: COMPLETE ✅**
- MetaQA dataset downloaded and processed
- Knowledge graph built (43K nodes, 124K edges)
- Three embedding methods trained: FastRP (2.9s), Node2Vec (5min), TransE (54s on GPU)
- All embeddings validated

**Phase 2: IN PROGRESS 🔄**
- Architecture designed: 5-variant ablation study
- Goal: Prove TransE-guided A* achieves 90%+ accuracy of expensive LLM methods at $0 cost
- Next: Implement entity linkers and search algorithms

**Phase 3: TODO**
- Paper writing and visualization

### Key Insight for Paper
**"Relation-aware embeddings (TransE) are the key to multi-hop reasoning, not expensive LLM entity linking. Our method achieves practical accuracy at zero marginal cost."**

---

**Current Focus:** Implementing Phase 2 QA system - see [phase2_context.md](phase2_context.md) for details
