---

## 📋 Project Progress Summary

**Last Updated:** October 22, 2025

**Current Status:** Phase 1 Complete ✅ | Phase 2 Partial ✅ | Variants 0 & 5 Complete

**Completed Work:**
- ✅ MetaQA dataset integrated (134K KB triples, 329K QA pairs)
- ✅ Knowledge graph built (43K nodes, 124K edges)
- ✅ Three embedding methods trained with ablation studies
- ✅ **Variant 0 (LLM Direct QA)**: 56% on 1-hop, 25% on 2-hop (~24K questions tested)
- ✅ **Variant 5 (LLM-Guided BFS)**: 99.9% on 1-hop/2-hop, 97% on 3-hop (~39K questions tested)
- ✅ QA system infrastructure: ExactMatcher, Evaluator, Batch processing

**Next Steps:**
- 🔄 Implement Variant 1 (BFS Baseline) - establish graph-based lower bound
- 🔄 Implement Variant 3 (TransE A*) - **MAIN CONTRIBUTION**
- 🔄 Implement Variant 2 (FastRP A*) - prove TransE superiority
- 🔄 Generate comparison plots and paper tables

**Key Files:**
- `data/metaqa/graph.pkl` - NetworkX graph (43,234 nodes, 124,680 edges)
- `embeddings/` - FastRP, Node2Vec, TransE embeddings (dim=128)
- `train_embeddings.py` - Modular training script
- `build_graph.py` - Graph construction from KB

---

## 🗓️ Overall Roadmap

| Phase                              | Status | Outcome                                |
| ---------------------------------- | ------ | -------------------------------------- |
| **Phase 1 – Setup & Data**         | ✅ COMPLETE | MetaQA ready, graph + embeddings built |
| **Phase 2 – Algorithm Core**       | 🔄 PARTIAL | Variants 0 & 5 complete, need 1-3 for core contribution |
| **Phase 3 – Experiments**          | 🔄 PARTIAL | Baselines tested, need full ablation study |
| **Phase 4 – Writing & Submission** | 🔄 TODO | Full IEEE paper draft + figures + refs |

---

## 🧩 Phase 1 — Setup & Data Preparation ✅

### **Task 1. Download and inspect MetaQA** ✅ COMPLETE

* **Goal:** get the smallest KG + QA split (1-hop & 2-hop).
* **Steps:**

  1. `git clone https://github.com/yuyuz/MetaQA`
  2. Explore `data/1hop/triples.txt` & `questions.txt`
* **Deliverable:** cleaned triples + QA CSV (`entity1,relation,entity2` and `question,answer`)
* **Status:** ✅ MetaQA dataset downloaded to `data/metaqa/`
  * `kb.txt` - 134,741 KG triples
  * `1-hop/vanilla/` - 96,106 train questions
  * `2-hop/vanilla/` - 118,980 train questions
  * `3-hop/vanilla/` - 114,196 train questions
* **Time:** 2 h

---

### **Task 2. Build graph object** ✅ COMPLETE

* **Goal:** load triples into `networkx.DiGraph()`.
* **Implementation:** `build_graph.py`
* **Deliverable:** serialized `graph.pkl` (8.2 MB)
* **Status:** ✅ Complete
  * Graph loaded from `kb.txt`
  * 43,234 nodes
  * 124,680 edges
  * Saved as `data/metaqa/graph.pkl`
* **Time:** 3 h

---

### **Task 3. Compute embeddings (FastRP, Node2Vec, TransE)** ✅ COMPLETE

* **Goal:** numerical vectors for all nodes.
* **Implementation:** `train_embeddings.py` - Modular training script with ablation support
* **Deliverables:** All embeddings saved to `embeddings/` directory
* **Status:** ✅ Complete with advanced ablation studies

  **FastRP embeddings:**
  * ✅ Multiple iteration checkpoints (3, 5, 10, 20 iterations)
  * Files: `fastrp_embeddings_iter{N}.npy`
  * Optimal: 3-4 iterations recommended

  **Node2Vec embeddings:**
  * ✅ Multiple walk checkpoints (10, 50, 100, 200 walks)
  * Files: `node2vec_embeddings_walks{N}.npy`
  * Training: 10 epochs per checkpoint

  **TransE embeddings:**
  * ✅ GPU-accelerated training (100 epochs)
  * File: `transe_embeddings.npy`
  * Trained on complete KB graph structure

  **Shared resources:**
  * `node2id.json` - Node to ID mapping (1.1 MB)
  * Metadata JSON files for each embedding configuration
  * Dimension: 128 for all embeddings

* **Training times:** FastRP ~4s, Node2Vec ~2-5min/checkpoint, TransE ~15-30min
* **Time:** 4 h + ablation experiments

---

## ⚙️ Phase 2 — Algorithm Implementation

### **Task 4. Implement entity linkers** ✅ COMPLETE

* **Goal:** Extract entities from questions
* **Deliverables:**
  * ✅ `ExactMatcher` - Extract bracketed entities and link to graph (free)
  * ✅ Base classes and infrastructure
* **Status:** Complete - used in Variants 1-5
* **Time:** 4 hours

---

### **Task 5. Implement search algorithms** 🔄 PARTIAL

* **Goal:** Implement different search strategies
* **Deliverables:**
  * ✅ `LLMGuidedBFS` - Pure BFS following GPT-4o planned relation sequence (Variant 5)
  * ❌ `BFSBaseline` - Blind BFS with no heuristic (Variant 1) - **TODO**
  * ❌ `FastRPAStarSearch` - A* with FastRP node embeddings (Variant 2) - **TODO**
  * ❌ `TransEAStarSearch` - A* with TransE relation embeddings (Variant 3) - **TODO** ⭐
* **Status:** Partial - only LLM-guided BFS complete
* **Time:** 2-3 days total (1-2 days remaining)

---

### **Task 6. Implement evaluation framework** ✅ COMPLETE

* **Goal:** Track metrics, handle batching, save results
* **Deliverables:**
  * ✅ `Evaluator` class with comprehensive metrics (accuracy, precision, recall, F1, cost)
  * ✅ Batch processing support (OpenAI Batch API for 50% cost savings)
  * ✅ Incremental result saving
  * ✅ Question loading utilities
* **Status:** Complete
* **Time:** 1 day

---

### **Task 7. Implement variant runners** 🔄 PARTIAL

* **Goal:** Standalone scripts for each variant
* **Deliverables:**
  * ✅ `variant0_llm_baseline.py` - LLM direct QA (no graph)
  * ✅ `variant5_openai_guided.py` - LLM-guided BFS with GPT-4o planning
  * ❌ `variant1_bfs_baseline.py` - BFS baseline - **TODO**
  * ❌ `variant2_fastrp_astar.py` - FastRP A* - **TODO**
  * ❌ `variant3_transe_astar.py` - TransE A* (MAIN CONTRIBUTION) - **TODO**
* **Status:** Partial - 2/5 complete
* **Time:** 1-2 days remaining

---

## 📊 Phase 3 — Experiments & Ablations

### **Task 8. Run experiments** 🔄 PARTIAL

* **Completed:**
  * ✅ Variant 0 (LLM Direct): Full 1-hop (9,947 qs) & 2-hop (14,872 qs) datasets
  * ✅ Variant 5 (LLM-Guided): Full 1-hop, 2-hop, 3-hop datasets (~39K total questions)
* **TODO:**
  * ❌ Variant 1 (BFS Baseline): 1K samples per hop
  * ❌ Variant 2 (FastRP A*): 1K samples per hop
  * ❌ Variant 3 (TransE A*): 1K samples per hop - **MAIN CONTRIBUTION**
* **Current Results:**

  | Variant | 1-hop | 2-hop | 3-hop | Cost/Query |
  |---------|-------|-------|-------|------------|
  | 0: LLM Direct | 56.0% | 24.7% | - | $0.00055 |
  | 5: LLM-Guided | 99.9% | 99.9% | 97.0% | $0.0001 |

* **Time:** 1-2 days for remaining variants

---

### **Task 9. Visualization** 🔄 TODO

* **Goal:** Generate plots + figures for paper.
* **Deliverables:**

  * ❌ Accuracy comparison plot (all 5 variants)
  * ❌ Cost-accuracy tradeoff plot
  * ❌ Nodes expanded comparison
  * ❌ Sample reasoning paths for each variant
  * ❌ System architecture diagram
* **Status:** Not started - waiting for Variants 1-3 results
* **Time:** 1 day

---

## 🖋️ Phase 4 — Writing & Submission

### **Task 10. Write paper draft**

* **Use:** IEEEtran LaTeX template (Overleaf).
* **Deliverables:**

  * Abstract & Intro (first day)
  * Method section with diagram (day 8)
  * Experiments (day 9)
  * Conclusion & references (day 10)
* **Time:** 3 days total writing + polish

---

### **Task 11. Finalize submission**

* **Goal:** Polish figures, BibTeX, run spell-check.
* **Deliverable:** `paper.pdf` ≤ 8 pages.
* **Time:** ½ day

---

## 🧠 Milestones Summary

| Day | Milestone                            | Status |
| --- | ------------------------------------ | ------ |
| 1-2 | Graph + embeddings ready             | ✅ COMPLETE |
| 3-5 | QA infrastructure + baselines        | ✅ PARTIAL (Variants 0, 5 done) |
| 6-7 | Core variants (1-3) implemented      | 🔄 IN PROGRESS (Next) |
| 8-9 | Full ablation + plots complete       | 🔄 TODO |
| 10  | Final paper PDF ready for submission | 🔄 TODO |

---

## 🔬 Important Notes on Embedding Training

### Data Usage Clarification

**Knowledge Graph (KB) vs QA datasets:**
- ✅ **kb.txt (134,741 triples)** → Used to build `graph.pkl` → Used for embedding training
- ❌ **QA files (1-hop/2-hop/3-hop)** → NOT used for embeddings → Used ONLY for evaluation

**Why this is correct:**
- TransE/FastRP/Node2Vec learn from **graph structure** (unsupervised)
- Embeddings capture entity relationships, not question-answer patterns
- QA files are for testing the A* reasoning algorithm (Phase 3)

**Embedding Training Pipeline:**
```
kb.txt → build_graph.py → graph.pkl → train_embeddings.py → embeddings/*.npy
                                                           ↓
                                    (90% train / 10% validation split for TransE)
```

### Ablation Studies Available

Run different configurations to compare performance:

```bash
# FastRP - test different iterations
python train_embeddings.py --method fastrp --iterations 5 --iteration-checkpoints 1 2 3 4 5

# Node2Vec - test different walk counts
python train_embeddings.py --method node2vec --walks 200 --checkpoints 10 50 100 200

# TransE - full KB training
python train_embeddings.py --method transe --epochs 100
```

---

## 🧩 Optional Enhancements (only if time left)

* Tune λ heuristic via grid search.
* Add path interpretability analysis (table of reasoning chains).
* Add GrailQA subset for extra benchmark.
* Compare different embedding dimensions (64, 128, 256).
* Test relation-aware heuristics using TransE relation embeddings.

---

## 🚀 Quick Reference Commands

### Phase 1 - Data & Embeddings (COMPLETE)

```bash
# Build graph from KB
python build_graph.py

# Train all embeddings
python train_embeddings.py --method all

# Train specific method with ablations
python train_embeddings.py --method fastrp --iterations 5 --iteration-checkpoints 1 2 3 4 5
python train_embeddings.py --method node2vec --walks 200 --checkpoints 10 50 100 200 --epochs 10
python train_embeddings.py --method transe --epochs 100 --batch-size 2048
```

### Phase 2 - Algorithm Implementation (PARTIAL)

```bash
# ✅ IMPLEMENTED:
# Variant 0: LLM Direct QA
python variant0_llm_baseline.py --mode batch --datasets 1-hop 2-hop --limit 1000

# Variant 5: LLM-Guided BFS
python variant5_openai_guided.py --mode batch --datasets 1-hop 2-hop --limit 1000

# ❌ TODO (implement these):
# Variant 1: BFS Baseline
# python variant1_bfs_baseline.py --datasets 1-hop 2-hop --limit 1000

# Variant 2: FastRP A*
# python variant2_fastrp_astar.py --datasets 1-hop 2-hop --limit 1000

# Variant 3: TransE A* (MAIN CONTRIBUTION)
# python variant3_transe_astar.py --datasets 1-hop 2-hop --limit 1000
```

### Phase 3 - Experiments (PARTIAL)

```bash
# View existing results
ls -lh results/*.json

# ❌ TODO:
# python analyze_results.py --variants 0 1 2 3 5
# python generate_plots.py --output paper_figures/
```

---

## 📊 Dataset Statistics

| File | Lines | Purpose |
|------|-------|---------|
| `kb.txt` | 134,741 | Complete knowledge graph triples |
| `1-hop/vanilla/qa_train.txt` | 96,106 | 1-hop training questions |
| `2-hop/vanilla/qa_train.txt` | 118,980 | 2-hop training questions |
| `3-hop/vanilla/qa_train.txt` | 114,196 | 3-hop training questions |
| **Graph** | 43,234 nodes | 124,680 edges (from kb.txt) |

**Graph Characteristics:**
- Average degree: ~2.88
- Directed graph with labeled relations
- Relations: `directed_by`, `starred_actors`, `written_by`, etc.

