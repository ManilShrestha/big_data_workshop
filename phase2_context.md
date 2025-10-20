# Phase 2: QA System Implementation & Ablation Study

**Status:** In Progress
**Created:** October 20, 2025
**Goal:** Implement LLM-Guided A* Search with 5 ablation variants

---

## Executive Summary

This document details the implementation plan for Phase 2 of the MetaQA Knowledge Graph Reasoning project. We're building a modular QA system with 5 variants to demonstrate that **TransE-guided A* search achieves near-optimal accuracy at zero inference cost**, making it practical for large-scale deployment.

### Key Research Question
Can relation-aware graph embeddings (TransE) match the accuracy of expensive LLM-based methods while maintaining zero inference cost?

### Main Hypothesis
TransE-guided A* (Variant 3) will achieve 90%+ of OpenAI+TransE accuracy (Variant 5) at 0% of the cost.

---

## Ablation Study Design

### Comparison Matrix

| Variant | Entity Linking | A* Heuristic | Training Cost | Per-Query Cost | Purpose |
|---------|----------------|--------------|---------------|----------------|---------|
| **1. BFS Baseline** | Exact match | None (BFS) | $0 | $0 | Lower bound - uninformed search |
| **2. FastRP A*** | Exact match | FastRP cosine | $0 (2.9s) | $0 | Structural embeddings baseline |
| **3. TransE A*** ⭐ | Exact match | TransE distance | $0 (54s) | $0 | **Our main contribution** |
| **4. OpenAI+FastRP** | OpenAI embeddings | FastRP cosine | $0.02 | $0.0001 | Expensive linking, weak heuristic |
| **5. OpenAI+TransE** | OpenAI embeddings | TransE distance | $0.02 | $0.0001 | Upper bound - best possible |

⭐ **Variant 3** is our main method - demonstrating that relation-aware embeddings are the key, not expensive entity linking.

---

## Technical Architecture

### Project Structure

```
big_data_workshop/
├── qa_system/
│   ├── __init__.py
│   │
│   ├── entity_linkers/
│   │   ├── __init__.py
│   │   ├── base_linker.py           # Abstract base class
│   │   ├── exact_matcher.py         # Regex-based exact matching
│   │   └── openai_linker.py         # OpenAI text-embedding-3-small
│   │
│   ├── search_algorithms/
│   │   ├── __init__.py
│   │   ├── base_search.py           # Abstract base class
│   │   ├── bfs_search.py            # Variant 1: Breadth-first search
│   │   ├── astar_fastrp.py          # Variants 2, 4: A* + FastRP
│   │   └── astar_transe.py          # Variants 3, 5: A* + TransE
│   │
│   ├── config.py                     # Configuration & hyperparameters
│   ├── evaluator.py                  # Evaluation framework
│   └── utils.py                      # Helper functions
│
├── run_ablation.py                   # Main experiment runner
├── analyze_results.py                # Generate paper tables/plots
│
├── results/
│   ├── variant1_bfs_baseline.json
│   ├── variant2_fastrp_astar.json
│   ├── variant3_transe_astar.json
│   ├── variant4_openai_fastrp.json
│   ├── variant5_openai_transe.json
│   └── ablation_summary.csv
│
└── phase2_context.md                 # This file
```

---

## Component Specifications

### 1. Entity Linkers

#### 1.1 ExactMatcher (Variants 1, 2, 3)
**Purpose:** Extract entities from questions using bracket notation `[entity_name]`

**Example:**
```python
question = "what movies are about [ginger rogers]"
entities = exact_matcher.extract(question)
# Returns: ["Ginger Rogers", "ginger rogers"]

graph_nodes = exact_matcher.link_to_graph(entities, node2id)
# Returns: ["Ginger Rogers"] (exact match in graph)
```

**Implementation:**
```python
import re

class ExactMatcher:
    def __init__(self, node2id):
        self.node2id = node2id
        self.entities = set(node2id.keys())

    def extract_entities(self, question):
        """Extract entities from [brackets]"""
        matches = re.findall(r'\[([^\]]+)\]', question)
        return matches

    def link_to_graph(self, entity_strings):
        """Find exact matches in graph"""
        linked = []
        for s in entity_strings:
            # Try exact match (case-sensitive)
            if s in self.entities:
                linked.append(s)
            # Try capitalized version
            elif s.title() in self.entities:
                linked.append(s.title())
            # Try all caps
            elif s.upper() in self.entities:
                linked.append(s.upper())
        return linked
```

**Cost:** $0
**Accuracy:** High for MetaQA (most questions have exact entity names in brackets)

---

#### 1.2 OpenAILinker (Variants 4, 5)
**Purpose:** Fuzzy entity matching using OpenAI embeddings

**Example:**
```python
question = "movies about brad pit"  # Typo!
entities = openai_linker.extract(question)
# Uses GPT-4 to extract: ["brad pit"]

graph_nodes = openai_linker.link_to_graph(entities)
# Embeds "brad pit" → finds "Brad Pitt" (cosine sim = 0.98)
# Returns: ["Brad Pitt"]
```

**Implementation:**
```python
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class OpenAILinker:
    def __init__(self, node2id, openai_api_key):
        self.node2id = node2id
        self.client = openai.OpenAI(api_key=openai_api_key)

        # One-time: Embed all graph entities (cache this!)
        self.entity_embeddings = self._embed_all_entities()

    def _embed_all_entities(self):
        """Embed all 43K entities (one-time cost: $0.02)"""
        entities = list(self.node2id.keys())
        embeddings = {}

        # Batch embed (1536 dims per entity)
        for i in range(0, len(entities), 1000):
            batch = entities[i:i+1000]
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            for j, emb in enumerate(response.data):
                embeddings[batch[j]] = emb.embedding

        return embeddings

    def extract_entities(self, question):
        """Use GPT-4 to extract entity strings"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Extract entity names from this question: '{question}'\n"
                          f"Return ONLY a JSON array of entity strings."
            }]
        )
        # Parse: ["Brad Pitt", "Christopher Nolan"]
        return eval(response.choices[0].message.content)

    def link_to_graph(self, entity_strings, top_k=3):
        """Find top-k closest graph entities"""
        linked = []

        for s in entity_strings:
            # Embed query entity
            query_emb = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=[s]
            ).data[0].embedding

            # Compute cosine similarity to all graph entities
            similarities = {}
            for entity, emb in self.entity_embeddings.items():
                sim = cosine_similarity([query_emb], [emb])[0][0]
                similarities[entity] = sim

            # Return top-k matches
            top_matches = sorted(similarities.items(),
                               key=lambda x: x[1],
                               reverse=True)[:top_k]
            linked.extend([m[0] for m in top_matches if m[1] > 0.8])

        return linked
```

**Cost:**
- One-time: $0.02 (embed 43K entities)
- Per query: $0.0001 (1 GPT-4-mini call + 1-3 embeddings)

**Accuracy:** Handles typos, variations, but not always needed for MetaQA

---

### 2. Search Algorithms

#### 2.1 BFS Baseline (Variant 1)
**Purpose:** Uninformed search - shows value of ANY heuristic

**Pseudocode:**
```python
def bfs_search(graph, start_nodes, target_relation, max_depth=3):
    """
    Standard BFS with no heuristic

    Args:
        start_nodes: Entities extracted from question
        target_relation: e.g., "directed_by" for "who directed X?"
        max_depth: Max hops (1, 2, or 3)

    Returns:
        answer_nodes: List of entities
        path: Reasoning path
        stats: {nodes_expanded, depth, runtime}
    """
    queue = [(node, [node], 0) for node in start_nodes]
    visited = set()
    answers = []

    while queue:
        current, path, depth = queue.pop(0)

        if depth > max_depth:
            continue

        visited.add(current)

        # Check all neighbors
        for neighbor in graph.neighbors(current):
            edge_relation = graph[current][neighbor]['relation']

            # If we found the target relation, this is an answer
            if edge_relation == target_relation:
                answers.append(neighbor)

            # Continue searching
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor], depth + 1))

    return answers, path, stats
```

**Characteristics:**
- Explores ALL neighbors equally (no prioritization)
- High node expansion (inefficient)
- Complete (finds all answers if they exist)

---

#### 2.2 A* with FastRP (Variants 2, 4)
**Purpose:** Structural heuristic - nodes similar in graph structure

**Heuristic:**
```python
h(node) = 1 - cosine_similarity(E_node, E_goal)
```

Where:
- `E_node`: FastRP embedding of current node (128-dim)
- `E_goal`: FastRP embedding of goal/answer node
- Lower h(n) → higher priority (closer to goal)

**Problem:** Doesn't understand relations!
- `E_movie` is similar to `E_actor` (connected in graph)
- But can't distinguish `directed_by` from `starred_actors`

**Pseudocode:**
```python
import heapq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def astar_fastrp(graph, start_nodes, goal_nodes, embeddings, node2id):
    """
    A* search with FastRP cosine similarity heuristic

    Args:
        start_nodes: Source entities
        goal_nodes: Target entities (if known) or sample from relation
        embeddings: FastRP embeddings (43234 × 128)
        node2id: Entity name → embedding index
    """
    # Priority queue: (f_score, node, path, g_score)
    heap = []
    for start in start_nodes:
        f = heuristic_fastrp(start, goal_nodes[0], embeddings, node2id)
        heapq.heappush(heap, (f, start, [start], 0))

    visited = set()
    answers = []

    while heap:
        f, current, path, g = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)

        # Check if we reached a goal
        if current in goal_nodes:
            answers.append((current, path))
            continue

        # Expand neighbors
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                g_new = g + 1  # Each edge costs 1
                h = heuristic_fastrp(neighbor, goal_nodes[0],
                                    embeddings, node2id)
                f_new = g_new + h

                heapq.heappush(heap, (f_new, neighbor,
                                     path + [neighbor], g_new))

    return answers

def heuristic_fastrp(node, goal, embeddings, node2id):
    """h(n) = 1 - cosine_sim(node, goal)"""
    idx_n = node2id[node]
    idx_g = node2id[goal]

    emb_n = embeddings[idx_n].reshape(1, -1)
    emb_g = embeddings[idx_g].reshape(1, -1)

    sim = cosine_similarity(emb_n, emb_g)[0][0]
    return 1 - sim  # Lower is better
```

**Expected Performance:**
- Better than BFS (fewer nodes expanded)
- But not relation-aware (accuracy limited)

---

#### 2.3 A* with TransE (Variants 3, 5) ⭐ **MAIN METHOD**
**Purpose:** Relation-aware heuristic - understands `h + r ≈ t`

**Heuristic:**
```python
h(node) = ||E_node + E_relation - E_goal||₂
```

Where:
- `E_node`: TransE entity embedding (256-dim)
- `E_relation`: TransE relation embedding (256-dim)
- `E_goal`: TransE entity embedding of goal
- Lower distance → higher priority

**Why this works:**
- TransE learns: `E_head + E_relation ≈ E_tail`
- Example: `E_movie + E_directed_by ≈ E_director`
- A* prioritizes paths that minimize this translation error

**Pseudocode:**
```python
def astar_transe(graph, start_nodes, target_relation,
                 entity_embeddings, relation_embeddings,
                 node2id, relation2id):
    """
    A* search with TransE translation-based heuristic

    Args:
        start_nodes: Source entities from question
        target_relation: e.g., "directed_by"
        entity_embeddings: (43234 × 256)
        relation_embeddings: (9 × 256)
        relation2id: Relation name → embedding index
    """
    # Get relation embedding
    rel_idx = relation2id[target_relation]
    E_rel = relation_embeddings[rel_idx]

    # Priority queue
    heap = []
    for start in start_nodes:
        # Heuristic: distance to "ideal goal" via target relation
        h = heuristic_transe(start, E_rel, entity_embeddings, node2id)
        heapq.heappush(heap, (h, start, [start], 0))

    visited = set()
    answers = []

    while heap:
        f, current, path, g = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)

        # Check neighbors via target relation
        for neighbor in graph.neighbors(current):
            edge_rel = graph[current][neighbor]['relation']

            # If this edge uses our target relation → answer!
            if edge_rel == target_relation:
                answers.append((neighbor, path + [neighbor]))

            # Continue exploring
            if neighbor not in visited:
                g_new = g + 1
                h = heuristic_transe(neighbor, E_rel,
                                    entity_embeddings, node2id)
                f_new = g_new + h
                heapq.heappush(heap, (f_new, neighbor,
                                     path + [neighbor], g_new))

    return answers

def heuristic_transe(node, E_relation, entity_embeddings, node2id):
    """
    h(n) = ||E_node + E_relation - E_goal||

    For QA, we don't know E_goal ahead of time, so we estimate:
    - E_goal ≈ E_node + E_relation (ideal translation)
    - h(n) = distance to this ideal point

    Alternatively, if we have sample goals from the relation,
    use: h(n) = min over goals: ||E_node + E_rel - E_goal||
    """
    idx = node2id[node]
    E_node = entity_embeddings[idx]

    # Predicted goal location
    E_predicted = E_node + E_relation

    # For simplicity, use norm of prediction as heuristic
    # (Lower norm → closer to typical answer entities)
    return np.linalg.norm(E_predicted)
```

**Alternative Heuristic (Better):**
If we can identify potential goal entities (e.g., all directors for "who directed X?"):
```python
def heuristic_transe_with_goals(node, E_relation, goal_candidates,
                                 entity_embeddings, node2id):
    """Use actual goal entities for better heuristic"""
    idx = node2id[node]
    E_node = entity_embeddings[idx]

    # Find closest goal via translation
    min_dist = float('inf')
    for goal in goal_candidates:
        goal_idx = node2id[goal]
        E_goal = entity_embeddings[goal_idx]

        # TransE score: ||h + r - t||
        dist = np.linalg.norm(E_node + E_relation - E_goal)
        min_dist = min(min_dist, dist)

    return min_dist
```

**Expected Performance:**
- Significantly fewer nodes expanded than BFS/FastRP
- High accuracy on 2-hop and 3-hop questions
- Relation-aware: Prioritizes correct edge types

---

### 3. Question Parsing

**Challenge:** Extract target relation from natural language question

**Examples:**
```
"who directed movies about Brad Pitt?" → target_relation = "directed_by"
"what movies star Tom Hanks?" → target_relation = "starred_actors"
"what language is Parasite in?" → target_relation = "in_language"
```

**Strategy 1: Rule-Based (Simple, Start Here)**
```python
def extract_relation(question):
    """Map question patterns to relations"""
    question_lower = question.lower()

    if 'directed' in question_lower or 'director' in question_lower:
        return 'directed_by'
    elif 'star' in question_lower or 'actor' in question_lower:
        return 'starred_actors'
    elif 'written' in question_lower or 'writer' in question_lower:
        return 'written_by'
    elif 'language' in question_lower:
        return 'in_language'
    elif 'genre' in question_lower:
        return 'has_genre'
    elif 'year' in question_lower or 'released' in question_lower:
        return 'release_year'
    else:
        return None  # Unknown
```

**Strategy 2: LLM-Based (For Variant 4, 5)**
```python
def extract_relation_llm(question, openai_client):
    """Use GPT-4 to extract relation"""
    relations = [
        "directed_by", "starred_actors", "written_by",
        "in_language", "has_genre", "release_year",
        "has_tags", "has_imdb_rating", "has_imdb_votes"
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Question: '{question}'\n"
                      f"Which relation from {relations} is being asked about?\n"
                      f"Return ONLY the relation name."
        }]
    )

    return response.choices[0].message.content.strip()
```

---

### 4. Evaluation Framework

**Metrics to Track:**

```python
class EvaluationMetrics:
    def __init__(self):
        self.total_questions = 0
        self.correct_answers = 0
        self.nodes_expanded = []
        self.search_times = []
        self.api_calls = 0
        self.total_cost = 0.0

    def compute(self):
        return {
            'accuracy': self.correct_answers / self.total_questions,
            'avg_nodes_expanded': np.mean(self.nodes_expanded),
            'avg_search_time_ms': np.mean(self.search_times) * 1000,
            'total_api_calls': self.api_calls,
            'total_cost_usd': self.total_cost,
            'cost_per_query_usd': self.total_cost / self.total_questions
        }
```

**Per-Question Evaluation:**
```python
def evaluate_answer(predicted, ground_truth):
    """
    MetaQA answers are in format: "answer1|answer2|answer3"
    Success if ANY predicted answer matches ground truth
    """
    gt_set = set(ground_truth.split('|'))
    pred_set = set(predicted)

    # Exact match
    if pred_set & gt_set:  # Intersection
        return True

    # Partial match (for multi-answer questions)
    overlap = len(pred_set & gt_set)
    recall = overlap / len(gt_set) if gt_set else 0

    return recall > 0.5  # At least 50% of answers correct
```

---

### 5. Configuration

```python
# config.py

class Config:
    # Paths
    GRAPH_PATH = "data/metaqa/graph.pkl"
    NODE2ID_PATH = "embeddings/node2id.json"

    # Embeddings
    FASTRP_PATH = "embeddings/fastrp_embeddings_iter3.npy"
    TRANSE_ENTITY_PATH = "embeddings/transe_embeddings_epochs100.npy"
    TRANSE_RELATION_PATH = "embeddings/transe_relation_embeddings_epochs100.npy"

    # Datasets
    QA_1HOP_TRAIN = "data/metaqa/1-hop/vanilla/qa_train.txt"
    QA_2HOP_TRAIN = "data/metaqa/2-hop/vanilla/qa_train.txt"

    # Evaluation
    NUM_TEST_QUESTIONS = 1000  # Sample size per dataset
    MAX_SEARCH_DEPTH = 3
    TIMEOUT_SECONDS = 30

    # OpenAI (for Variants 4, 5)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_EMBED = "text-embedding-3-small"
    OPENAI_MODEL_CHAT = "gpt-4o-mini"

    # Costs (for tracking)
    COST_PER_EMBEDDING = 0.00000002  # text-embedding-3-small
    COST_PER_CHAT = 0.0001  # gpt-4o-mini average
```

---

## Implementation Phases

### Phase 2A: Core Infrastructure (Day 1)
- [ ] Set up `qa_system/` directory structure
- [ ] Implement `ExactMatcher` entity linker
- [ ] Implement rule-based relation extraction
- [ ] Create base evaluation framework
- [ ] Write unit tests for entity linking

### Phase 2B: Variants 1-3 (Day 2-3)
- [ ] Implement BFS baseline (Variant 1)
- [ ] Implement A* with FastRP (Variant 2)
- [ ] Implement A* with TransE (Variant 3) ⭐
- [ ] Test on 100 1-hop questions
- [ ] Debug and optimize

### Phase 2C: OpenAI Integration (Day 4)
- [ ] Implement `OpenAILinker`
- [ ] Cache entity embeddings
- [ ] Implement Variants 4 and 5
- [ ] Cost tracking infrastructure

### Phase 2D: Experiments (Day 5)
- [ ] Run all 5 variants on 1-hop (1K questions)
- [ ] Run all 5 variants on 2-hop (1K questions)
- [ ] Generate results tables
- [ ] Create comparison plots

---

## Expected Results

### Predicted Accuracy (1-hop questions)

| Variant | Accuracy | Nodes Expanded | Time/Query |
|---------|----------|----------------|------------|
| 1. BFS | 45% | 5000 | 800ms |
| 2. FastRP A* | 58% | 1200 | 150ms |
| **3. TransE A*** | **72%** | **450** | **180ms** |
| 4. OpenAI+FastRP | 62% | 1100 | 250ms |
| 5. OpenAI+TransE | 76% | 400 | 280ms |

### Predicted Accuracy (2-hop questions)

| Variant | Accuracy | Nodes Expanded | Time/Query |
|---------|----------|----------------|------------|
| 1. BFS | 18% | 15000 | 2500ms |
| 2. FastRP A* | 28% | 3500 | 450ms |
| **3. TransE A*** | **48%** | **800** | **520ms** |
| 4. OpenAI+FastRP | 32% | 3200 | 600ms |
| 5. OpenAI+TransE | 54% | 700 | 680ms |

### Cost Analysis (1000 questions)

| Variant | Training | Inference | Total |
|---------|----------|-----------|-------|
| 1. BFS | $0 | $0 | $0 |
| 2. FastRP A* | $0 | $0 | $0 |
| **3. TransE A*** | **$0** | **$0** | **$0** |
| 4. OpenAI+FastRP | $0.02 | $0.10 | $0.12 |
| 5. OpenAI+TransE | $0.02 | $0.10 | $0.12 |

**Scale to 1M questions:**
- Variant 3: $0
- Variant 5: $100

---

## Paper Narrative

### Abstract Claim
*"We demonstrate that TransE-guided A* search achieves 89% of the accuracy of expensive LLM-based methods while maintaining zero inference cost, making knowledge graph reasoning practical for large-scale applications."*

### Key Contributions
1. **Relation-aware heuristic:** TransE embeddings encode relational structure, enabling efficient multi-hop reasoning
2. **Zero-cost inference:** No API calls during search (unlike LLM-based methods)
3. **Comprehensive ablation:** 5 variants isolating the impact of entity linking vs. search heuristic

### Ablation Insights
- **BFS → FastRP A*:** Structural embeddings reduce search space by 76%
- **FastRP A* → TransE A*:** Relation awareness improves 2-hop accuracy by 71%
- **TransE A* → OpenAI+TransE:** Expensive entity linking only adds 8% accuracy

### Conclusion
*"For knowledge graphs with structured entity names (like MetaQA), relation-aware embeddings provide the primary performance gain, while expensive LLM entity linking offers diminishing returns. Our method achieves practical accuracy at zero marginal cost."*

---

## Next Steps

1. **Start with Variants 1-3** (no OpenAI dependency)
2. **Validate on 1-hop questions** (simpler, faster iteration)
3. **Add OpenAI variants** once core system works
4. **Scale to 2-hop and 3-hop** after validation

---

## Open Questions

### Q1: How to handle unknown goal entities?
**Problem:** A* needs a goal to compute heuristic, but we don't know the answer beforehand.

**Solutions:**
- **Option A:** Sample candidate goals from the target relation (e.g., all directors)
- **Option B:** Use relation embedding alone: `h(n) = ||E_node + E_rel||`
- **Option C:** Multi-goal search: Explore towards top-k most likely entities

**Recommendation:** Start with Option A (sample 100 random entities of the correct type)

### Q2: How to extract entity types from questions?
**Problem:** "Who directed X?" → need to know answer type is "person" (director)

**Solutions:**
- **Option A:** Rule-based (keywords: "who" → person, "what movies" → movies)
- **Option B:** Use graph schema (directors are entities with outgoing `directed_by⁻¹` edges)
- **Option C:** LLM extraction (for Variants 4, 5)

**Recommendation:** Option A for Variants 1-3, Option C for 4-5

### Q3: Should we cache search results?
**Answer:** Yes! Many questions share entities (e.g., "Brad Pitt" appears in multiple questions). Cache paths from common entities.

---

## Files to Create

### Priority 1 (Core)
- [x] `phase2_context.md` (this file)
- [ ] `qa_system/__init__.py`
- [ ] `qa_system/config.py`
- [ ] `qa_system/utils.py`
- [ ] `qa_system/entity_linkers/exact_matcher.py`
- [ ] `qa_system/search_algorithms/bfs_search.py`
- [ ] `qa_system/search_algorithms/astar_transe.py`
- [ ] `run_ablation.py`

### Priority 2 (Ablations)
- [ ] `qa_system/search_algorithms/astar_fastrp.py`
- [ ] `qa_system/entity_linkers/openai_linker.py`
- [ ] `qa_system/evaluator.py`

### Priority 3 (Analysis)
- [ ] `analyze_results.py`
- [ ] `visualize_search_paths.py`

---

## References

- **TransE Paper:** Bordes et al. (2013) - "Translating Embeddings for Modeling Multi-relational Data"
- **A* Search:** Hart et al. (1968) - "A Formal Basis for the Heuristic Determination of Minimum Cost Paths"
- **MetaQA:** Zhang et al. (2018) - "Variational Reasoning for Question Answering with Knowledge Graph"
- **PyKEEN Docs:** https://pykeen.readthedocs.io/

---

**End of Phase 2 Context Document**