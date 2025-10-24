# MetaQA Knowledge Graph Reasoning - Project Context

## Project Overview

This is a research project for an IEEE paper on **LLM-Guided Knowledge Graph Reasoning using A* Search with Graph Embeddings**. The goal is to combine traditional graph traversal (A*) with modern LLMs and graph embeddings to answer multi-hop questions over knowledge graphs.

**Current Status:** Phase 1 Complete | Phase 2 Partial (Variants 0, 5 Complete)
**Last Updated:** October 22, 2025

---

## Quick Start

### Key Commands

```bash
# Phase 1: Build graph & train embeddings
python build_graph.py
python train_embeddings.py --method all
python validate_embeddings.py --all

# Phase 2: Run QA ablation study (PARTIALLY IMPLEMENTED)
# Variant 0: LLM Direct QA (baseline)
python variant0_llm_baseline.py --mode batch --datasets 1-hop 2-hop --limit 1000

# Variant 5: LLM-Guided BFS (upper bound)
python variant5_openai_guided.py --mode batch --datasets 1-hop 2-hop --limit 1000

# TODO: Implement Variants 1-3 (graph-based methods)
# python variant1_bfs_baseline.py
# python variant2_fastrp_astar.py
# python variant3_transe_astar.py  # MAIN CONTRIBUTION

# Analyze results
python analyze_results.py  # Generate paper tables/plots (TODO)
```

---

## Project Structure

```
big_data_workshop/
├── build_graph.py              # Phase 1: Build NetworkX graph from KB
├── train_embeddings.py         # Phase 1: Train FastRP/Node2Vec/TransE
├── validate_embeddings.py      # Phase 1: Validate embedding quality
│
├── qa_system/                  # Phase 2: QA implementation (PARTIALLY IMPLEMENTED)
│   ├── core/                   # Base classes
│   │   ├── base_entity_linker.py
│   │   ├── base_relation_ranker.py
│   │   ├── base_search.py
│   │   ├── question.py
│   │   └── search_result.py
│   ├── entity_linkers/         # Entity extraction & linking
│   │   └── exact_matcher.py   # ✅ ExactMatcher (bracket extraction)
│   ├── relation_rankers/       # Relation selection
│   │   └── openai_ranker.py   # ✅ OpenAI embeddings + GPT-4o planning
│   ├── search_algorithms/      # Search algorithms
│   │   └── llm_guided_bfs.py  # ✅ LLM-guided BFS (Variant 5)
│   ├── llm_qa/                 # Direct LLM QA (Variant 0)
│   │   ├── direct_qa.py       # ✅ Parallel API calls
│   │   ├── batch_qa.py        # ✅ OpenAI Batch API (50% cheaper)
│   │   └── prompts.py
│   ├── utils/
│   │   ├── loader.py          # Data loading utilities
│   │   └── evaluator.py       # Evaluation framework
│   └── config.py              # Configuration
│
├── variant0_llm_baseline.py   # ✅ Variant 0: LLM Direct QA
├── variant5_openai_guided.py  # ✅ Variant 5: LLM-Guided BFS
│
├── results/                   # Phase 2: Ablation results
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

**Overview:** Implement 5-variant ablation study comparing different entity linking and search approaches.

**Current Implementation Status:**
- ✅ **Variant 0**: LLM Direct QA (baseline, no graph)
- ✅ **Variant 5**: OpenAI Relation Ranking + LLM-Guided BFS (upper bound)
- ❌ **Variants 1-3**: Graph-based search methods (TODO - main contribution)

#### 5 Ablation Variants (ACTUAL IMPLEMENTATION)

| # | Name | Entity Linking | Relation Selection | Search | Cost | Status |
|---|------|----------------|-------------------|--------|------|--------|
| **0** | **LLM Direct QA** | None (no graph) | None | None (direct answer) | $$$ | ✅ **DONE** |
| 1 | BFS Baseline | ExactMatcher | None (all relations) | BFS | $0 | ❌ TODO |
| 2 | FastRP A* | ExactMatcher | FastRP cosine | A* | $0 | ❌ TODO |
| **3** | **TransE A*** ⭐ | **ExactMatcher** | **TransE distance** | **A*** | **$0** | ❌ **TODO** |
| **5** | **LLM-Guided BFS** | **ExactMatcher** | **GPT-4o planning** | **BFS** | **$$** | ✅ **DONE** |

⭐ **Variant 3 is our main contribution** - demonstrating that relation-aware embeddings achieve near-optimal accuracy at zero inference cost.

**Note:** Variant 4 (OpenAI entity linking) is optional and may be skipped to focus on the core contribution.

#### Implementation Tasks

**Task 4:** Build entity linkers
- [x] Design architecture
- [x] Implement `ExactMatcher` (Variants 1-5) ✅
- [ ] (Optional) Implement `OpenAILinker` for Variant 4

**Task 5:** Implement search algorithms
- [x] LLM-guided BFS (Variant 5) ✅
- [ ] BFS baseline (Variant 1) ❌ TODO
- [ ] A* with FastRP heuristic (Variant 2) ❌ TODO
- [ ] A* with TransE heuristic (Variant 3) ❌ TODO - **Main method**

**Task 6:** Build evaluation framework
- [x] Metrics tracking (accuracy, nodes expanded, cost) ✅
- [x] Question parsing & relation extraction ✅
- [x] Results logging & analysis ✅
- [x] Batch processing support (OpenAI Batch API) ✅

**Task 7:** Run ablation experiments
- [x] Variant 0: LLM Direct QA ✅
  - 1-hop: 9,947 questions, **56.0% accuracy**, $5.50 cost
  - 2-hop: 14,872 questions, **24.7% accuracy**, $8.21 cost
- [x] Variant 5: LLM-Guided BFS ✅
  - 1-hop: 9,947 questions, **99.9% accuracy**, ~$1.00 cost
  - 2-hop: 14,872 questions, **99.9% accuracy**, ~$1.50 cost
  - 3-hop: 14,274 questions, **97.0% accuracy**, ~$2.00 cost
- [ ] Variant 1: BFS Baseline ❌ TODO
- [ ] Variant 2: FastRP A* ❌ TODO
- [ ] Variant 3: TransE A* ❌ TODO - **MAIN CONTRIBUTION**
- [ ] Generate comparison tables for paper ❌ TODO

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

### Current Focus: Complete Core Graph-Based Variants (1-3)

**Completed:**
- ✅ Phase 1: Embeddings trained and validated
- ✅ Variant 0: LLM Direct QA (baseline showing need for graph grounding)
- ✅ Variant 5: LLM-Guided BFS (upper bound with GPT-4o planning)
- ✅ Infrastructure: ExactMatcher, Evaluator, Batch processing

### Immediate TODO (Priority Order)
1. 🔄 **NEXT:** Implement Variant 1 (BFS Baseline)
   - Simple breadth-first search with no heuristic
   - Establishes lower bound for graph-based methods
   - Expected: ~60-70% accuracy on 1-hop, ~40-50% on 2-hop
2. 🔄 Implement Variant 3 (TransE A*) - **MAIN CONTRIBUTION**
   - A* search with TransE relation embeddings as heuristic
   - Expected: ~90-95% accuracy on 1-hop, ~80-85% on 2-hop
   - Proves relation-aware embeddings work at $0 inference cost
3. 🔄 Implement Variant 2 (FastRP A*)
   - A* search with FastRP node embeddings as heuristic
   - Proves TransE > FastRP for multi-hop reasoning
   - Expected: ~80-85% accuracy on 1-hop, ~60-70% on 2-hop
4. 🔄 Run full ablation study (1K samples per hop-count)
5. 🔄 Generate comparison tables and cost-accuracy plots
6. 🔄 Error analysis: Where does Variant 3 fail vs Variant 5?
7. 🔄 Write paper draft

### Implementation Strategy
- **Priority 1:** Variants 1-3 (graph-based, $0 inference) - Core contribution
- **Priority 2:** Analysis comparing Variant 3 vs Variant 5 (cost-accuracy tradeoff)
- **Priority 3:** Paper writing with visualizations
- **Optional:** Variant 4 (OpenAI entity linking + FastRP) - not critical for story

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

### Current Status (October 22, 2025)

**Phase 1: COMPLETE ✅**
- MetaQA dataset downloaded and processed
- Knowledge graph built (43K nodes, 124K edges)
- Three embedding methods trained: FastRP (2.9s), Node2Vec (5min), TransE (54s on GPU)
- All embeddings validated

**Phase 2: PARTIAL ✅**
- ✅ QA system infrastructure complete
- ✅ **Variant 0** (LLM Direct QA): 56% on 1-hop, 25% on 2-hop - shows hallucination problem
- ✅ **Variant 5** (LLM-Guided BFS): 99.9% on 1-hop/2-hop, 97% on 3-hop - upper bound
- ❌ **Variants 1-3** (graph-based methods): TODO - **MAIN CONTRIBUTION MISSING**

**Actual Results So Far:**

| Variant | 1-hop Acc | 2-hop Acc | 3-hop Acc | Cost/Query | Dataset Size |
|---------|-----------|-----------|-----------|------------|--------------|
| 0: LLM Direct | 56.0% | 24.7% | - | $0.00055 | 9,947 / 14,872 |
| 5: LLM-Guided | 99.9% | 99.9% | 97.0% | $0.0001 | 9,947 / 14,872 / 14,274 |

**Phase 3: TODO**
- Implement Variants 1-3 (BFS, FastRP A*, TransE A*)
- Run full ablation study
- Generate comparison plots
- Paper writing

### Key Insight for Paper
**"Relation-aware embeddings (TransE) can replace expensive LLM planning for multi-hop reasoning, achieving 85-95% accuracy at zero inference cost vs 99% at $0.0001/query."**

**Critical Gap:** Need to implement Variant 3 (TransE A*) to prove this hypothesis!

---

**Current Focus:** Implement Variants 1-3 to complete the ablation study
