# 6-Bit Hop Encoding Implementation

**Date**: 2025-10-30
**Goal**: Give the model context about question hop count, not just current hop

## The Problem

Previous model only knew:
- `hop = 2` → "I'm at hop 2"
- But didn't know: "Is this a 2-hop question (2/2 done) or 3-hop question (2/3 done)?"

## The Solution

6-bit one-hot encoding: `[q1, q2, q3, c1, c2, c3]`
- First 3 bits: Question hop count (one-hot)
- Second 3 bits: Current hop (one-hot)

**Examples**:
- `[1,0,0, 1,0,0]` = 1-hop question, at hop 1 (done!)
- `[0,1,0, 0,1,0]` = 2-hop question, at hop 2 (done!)
- `[0,0,1, 0,1,0]` = 3-hop question, at hop 2 (1 more to go!)
- `[0,0,1, 0,0,1]` = 3-hop question, at hop 3 (final hop!)

## Files Modified

### 1. **variant3_edge_scorer_dual.py**
- Changed `hop_embed` (Embedding layer) → `hop_context_proj` (MLP: 6→512)
- Updated `forward()` to accept `hop_context: [B, 6]` instead of `hop: [B]`
- Updated `predict()` and `get_gates()` signatures

### 2. **variant3_train_edge_scorer_dual.py**
- Added `question_hop_counts` dict in `EdgeScorerDualDataset.__init__()`
  - Infers question hop count from max(hop) per question_id
- Modified `__getitem__()` to create 6-bit hop context
- Updated `train_epoch()` and `evaluate()` to use `hop_context` tensor
- Changed model initialization to use `hop_context_dim=6`

### 3. **edge_scorer_ranker.py**
- Added `question_hop_count` parameter to `score_edges_batch()`
- Creates 6-bit hop context before model inference
- Updated model loading to use `hop_context_dim=6`

### 4. **edge_scorer_bfs.py**
- Passes `question_hop_count=max_hops` when calling `score_edges_batch()`

## Training Data

**No changes needed!** Training data already has:
- `'hop'`: Current hop (1, 2, or 3)
- Question hop count is inferred as `max(hop)` per `question_id`

## Benefits

Model can now learn hop-specific strategies:
- **1-hop questions**: Direct traversal to answer
- **2-hop, hop 1**: Initial exploration phase
- **2-hop, hop 2**: Final answer selection
- **3-hop, hop 1**: Start of chain reasoning
- **3-hop, hop 2**: Middle of chain (need to continue)
- **3-hop, hop 3**: Final hop (answer is close!)

## Next Steps

1. Train new model: `python variant3/variant3_train_edge_scorer_dual.py`
2. Evaluate on test sets
3. Compare F1 scores with baseline (especially 3-hop)

## Backup

Original working version backed up to:
```
variant3/backup_20251030_043148/
```

To restore if needed:
```bash
cp variant3/backup_20251030_043148/*.py variant3/
cp variant3/backup_20251030_043148/edge_scorer_*.py variant3/eval_qa/
cp variant3/backup_20251030_043148/variant3_qa_evaluator.py variant3/eval_qa/
```
