---

## 📋 Project Progress Summary

**Last Updated:** October 14, 2025

**Current Status:** Phase 1 Complete ✅ | Ready for Phase 2 (A* Algorithm Implementation)

**Completed Work:**
- ✅ MetaQA dataset integrated (134K KB triples, 329K QA pairs)
- ✅ Knowledge graph built (43K nodes, 124K edges)
- ✅ Three embedding methods trained with ablation studies
- ✅ Modular training pipeline with checkpoint support

**Next Steps:**
- 🔄 Implement A* traversal with embedding heuristic (Task 4)
- 🔄 Add LLM planner for entity extraction (Task 5)
- 🔄 Build evaluation pipeline for QA datasets (Task 8)

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
| **Phase 2 – Algorithm Core**       | 🔄 TODO | Working A* traversal + LLM guidance    |
| **Phase 3 – Experiments**          | 🔄 TODO | Baseline + ablation results + plots    |
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

### **Task 4. Implement baseline A***

* **Goal:** A* traversal using embedding heuristic.
* **Steps:**

  * Define cost function `f(n)=g(n)+λ*(1-cos(E_n,E_q))`
  * Implement open/closed lists with `heapq`
  * Terminate when answer found or depth > max_hops
* **Deliverable:** `a_star_reasoner.py` that returns answer + path
* **Time:** 1 day

---

### **Task 5. Add LLM planner (front-end)**

* **Goal:** extract entities and relations from question.
* **Prompt:**

  ```
  Extract entities and relations:
  Q: "Who directed the movie that starred Brad Pitt?"
  A: entities=["Brad Pitt"], relations=["acted_in","directed_by"]
  ```
* **Use:** `openai.ChatCompletion` or `llama3` (local)
* **Deliverable:** `llm_planner.py` returns JSON
* **Time:** ½ day

---

### **Task 6. Integrate planner + A***

* **Goal:** seed A* with LLM entities.
* **Pipeline:**

  1. Get entities from LLM
  2. Start A* from those nodes
  3. Compare accuracy vs non-LLM baseline
* **Deliverable:** `llm_guided_astar.py`
* **Time:** 1 day

---

### **Task 7. Implement metrics + logging**

* **Goal:** track accuracy, nodes expanded, runtime.
* **Deliverable:** CSV log for each run
* **Time:** 3 h

---

## 📊 Phase 3 — Experiments & Ablations

### **Task 8. Run experiments**

* **Setups:**

  1. BFS baseline
  2. A* (FastRP only)
  3. LLM-seeded A*
* **Datasets:** 1-hop (1k Qs) & 2-hop (1k Qs)
* **Deliverables:** `results.csv`
* **Time:** 1 day

---

### **Task 9. Visualization**

* **Goal:** Generate plots + figures for paper.
* **Deliverables:**

  * Accuracy vs Hops
  * Expansions vs Runtime
  * Mermaid diagram of system
  * Sample traversal path visual
* **Time:** ½ day

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
| 3-5 | A* + LLM pipeline working            | 🔄 IN PROGRESS (Next) |
| 6-7 | Experiments + plots complete         | 🔄 TODO |
| 8-10| Final paper PDF ready for submission | 🔄 TODO |

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

### Phase 2 - Algorithm Implementation (TODO)

```bash
# To be implemented:
# python a_star_reasoner.py --embeddings embeddings/fastrp_embeddings_iter3.npy
# python llm_planner.py --question "Who directed Inception?"
# python llm_guided_astar.py --qa-file data/metaqa/1-hop/vanilla/qa_test.txt
```

### Phase 3 - Experiments (TODO)

```bash
# To be implemented:
# python run_experiments.py --methods bfs,astar,llm-astar --dataset 1-hop
# python generate_plots.py --results results.csv
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

