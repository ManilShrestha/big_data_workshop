# Backup: Working Version Before 6-bit Hop Encoding

**Date**: 2025-10-30 04:31:48
**Reason**: Implementing 6-bit hop encoding to give model context about question hop count

## Performance (Current Working Version)

### 1-hop
- Accuracy: 0.9800
- F1: 0.8767

### 2-hop
- Accuracy: 0.9800
- F1: 0.7945

### 3-hop
- Accuracy: 0.8700
- F1: 0.5211 ⚠️

## What's Changing

### Current Approach
- Single 3-bit hop encoding: 0, 1, or 2 (representing hop 1, 2, or 3)
- Model only knows: "I am currently at hop N"
- Model does NOT know: "This is a N-hop question"

### New Approach (6-bit One-Hot Encoding)
- First 3 bits: Question hop count (one-hot: [1,0,0] = 1-hop, [0,1,0] = 2-hop, [0,0,1] = 3-hop)
- Second 3 bits: Current hop (one-hot: [1,0,0] = hop 1, [0,1,0] = hop 2, [0,0,1] = hop 3)

**Examples**:
- `[1,0,0, 1,0,0]` = 1-hop question, currently at hop 1
- `[0,0,1, 0,1,0]` = 3-hop question, currently at hop 2
- `[0,1,0, 0,0,1]` = 2-hop question, currently at hop 3 (impossible, but encoded)

### Hypothesis
Model can learn different strategies:
- "If I'm at hop 1/3 in a 3-hop question, I should focus on initial traversal relations"
- "If I'm at hop 3/3 in a 3-hop question, I should focus on answer-type relations"
- "If I'm at hop 1/1 in a 1-hop question, go directly to answer"

## Files Backed Up
1. `variant3_edge_scorer_dual.py` - Model architecture
2. `variant3_train_edge_scorer_dual.py` - Training script
3. `edge_scorer_ranker.py` - Inference wrapper
4. `edge_scorer_bfs.py` - Search algorithm
5. `variant3_qa_evaluator.py` - Evaluation script

## To Restore
If the new approach doesn't work:
```bash
cp variant3/backup_20251030_043148/*.py variant3/
cp variant3/backup_20251030_043148/edge_scorer_*.py variant3/eval_qa/
cp variant3/backup_20251030_043148/variant3_qa_evaluator.py variant3/eval_qa/
```
