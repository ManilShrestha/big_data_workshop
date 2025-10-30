# Variant 3 Ablation Studies

## Overview

To validate that the dual embedding approach (text + graph) is actually learning from both modalities and not just relying on text embeddings, we created three ablation studies. Each ablation removes different components to measure their individual contribution.

## Baseline: Dual Model (Full Model)

**File**: `variant3_edge_scorer_dual.py` / `variant3_train_edge_scorer_dual.py`

**Components**:
- Question: text_emb (1536)
- Node: text_emb (1536) + graph_emb (256)
- Edge: text_emb (1536) + graph_emb (256)
- Target: text_emb (1536) + graph_emb (256)
- Hop: learned_emb (512)

**Total**: 5 components with dual embeddings for entities/relations

## Ablation 1: Text Only

**Files**:
- Model: `variant3_edge_scorer_abl1_text_only.py`
- Trainer: `variant3_train_edge_scorer_abl1_text_only.py`

**Components**:
- Question: text_emb (1536)
- Node: text_emb (1536)
- Edge: text_emb (1536)
- Target: text_emb (1536)

**Removed**: Graph embeddings, hop information

**Purpose**: Tests if the model can perform well using ONLY semantic (text) information. If this ablation performs similarly to the dual model, it would suggest that graph embeddings and hop information don't add value.

**Expected Result**: Lower performance than dual model, especially on structural reasoning tasks.

## Ablation 2: Graph Only

**Files**:
- Model: `variant3_edge_scorer_abl2_graph_only.py`
- Trainer: `variant3_train_edge_scorer_abl2_graph_only.py`

**Components**:
- Query: learned_emb (256) - shared across all questions
- Node: graph_emb (256)
- Edge: graph_emb (256)
- Target: graph_emb (256)

**Removed**: Text embeddings, hop information

**Purpose**: Tests if the model can perform well using ONLY structural (graph) information. Since questions don't have graph embeddings, we use a learned query embedding that represents "question context" in the graph space.

**Expected Result**: Significantly lower performance than dual model, since questions have no semantic grounding in the graph structure. This should demonstrate the critical importance of text embeddings.

## Ablation 3: Text + Hops

**Files**:
- Model: `variant3_edge_scorer_abl3_text_hops.py`
- Trainer: `variant3_train_edge_scorer_abl3_text_hops.py`

**Components**:
- Question: text_emb (1536)
- Node: text_emb (1536)
- Edge: text_emb (1536)
- Target: text_emb (1536)
- Hop: learned_emb (512)

**Removed**: Graph embeddings

**Purpose**: Tests if adding hop information to text embeddings is sufficient, or if graph embeddings provide additional value. This is the most important ablation to compare against the dual model.

**Expected Result**: Performance between Ablation 1 and the dual model. If graph embeddings are valuable, the dual model should outperform this ablation. If not, they should perform similarly.

## Comparison Matrix

| Model | Text Emb | Graph Emb | Hop Info | Components | Purpose |
|-------|----------|-----------|----------|------------|---------|
| **Dual (Full)** | ✓ | ✓ | ✓ | 5 | Baseline - full model |
| **Ablation 1** | ✓ | ✗ | ✗ | 4 | Tests text-only performance |
| **Ablation 2** | ✗ | ✓ | ✗ | 4 | Tests graph-only performance |
| **Ablation 3** | ✓ | ✗ | ✓ | 5 | Tests if graph adds value to text+hop |

## Training Configuration

All ablations use **identical** training configuration for fair comparison:

```python
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10
VAL_SPLIT = 0.2
HIDDEN_DIM = 512
DROPOUT = 0.3
OPTIMIZER = AdamW (weight_decay=1e-5)
SCHEDULER = CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
LOSS = BCEWithLogitsLoss (with pos_weight for class imbalance)
```

## Running the Ablations

```bash
# Ablation 1: Text only
python variant3/variant3_train_edge_scorer_abl1_text_only.py

# Ablation 2: Graph only
python variant3/variant3_train_edge_scorer_abl2_graph_only.py

# Ablation 3: Text + Hops
python variant3/variant3_train_edge_scorer_abl3_text_hops.py
```

## Output Files

Each ablation saves the following files to `models/`:

1. **Best model checkpoint**: `variant3_edge_scorer_abl{1,2,3}_*.pt`
2. **Training history (JSON)**: `variant3_training_history_abl{1,2,3}_*.json`
3. **Training history (CSV)**: `variant3_training_history_abl{1,2,3}_*.csv`
4. **Training summary**: `variant3_training_summary_abl{1,2,3}_*.json`
5. **Validation results**: `variant3_validation_results_abl{1,2,3}_*.json`

## Expected Ablation Insights

### If Dual Model is Superior:

1. **Dual > Abl3 > Abl1 > Abl2**: This would show that:
   - Graph embeddings add value beyond text+hop
   - Hop information is important
   - Text is more important than graph structure
   - The combination of all three is best

### If Text Dominates:

1. **Dual ≈ Abl3 ≈ Abl1 >> Abl2**: This would show that:
   - Text embeddings contain most of the signal
   - Graph embeddings don't add much value
   - Hop information doesn't add much value
   - The graph structure is relatively unimportant

### If Graph is Critical:

1. **Dual > Abl2 > Abl3 ≈ Abl1**: This would show that:
   - Graph embeddings are crucial
   - Text alone is insufficient
   - Structural information is key to the task

## Metrics to Compare

For each ablation, compare:

1. **Validation F1** (primary metric)
2. **Validation AUC-ROC**
3. **Per-hop performance** (where applicable)
4. **Training convergence speed**
5. **Final validation loss**

## WandB Tracking

All ablations log to WandB with clear naming:
- `variant3-ablation1-text-only`
- `variant3-ablation2-graph-only`
- `variant3-ablation3-text-hops`

This allows easy comparison in the WandB dashboard.
