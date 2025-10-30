# Variant3 Text Embedding Cache - Overview

## What We Built

The `variant3_cache_embeddings.py` script generates **text embeddings** for all unique strings in the training data using OpenAI's `text-embedding-3-small` model.

## Key Features

### 1. Smart Deduplication
Instead of embedding 5.7M training samples, we:
- Extract unique texts from 4 fields: `question_text`, `node_id`, `edge_relation`, `edge_target`
- Deduplicate across fields (some nodes appear as targets too)
- **Result**: Only ~319K unique texts to embed (17.8× reduction!)

### 2. Batch API Calls
- Uses OpenAI's batch embedding API (up to 2048 texts per call)
- Batch size: 2000 texts per request
- ~160 API calls total (vs 319K individual calls!)

### 3. Checkpoint & Resume
- Saves progress every 10 batches
- If interrupted, resumes from checkpoint
- Automatically cleans up checkpoint file when done

### 4. Cost Efficient
- **Total texts**: 319,477
- **Estimated tokens**: ~4.8M
- **Estimated cost**: ~$0.096 (less than 10 cents!)

## Output Structure

### Main Cache File: `variant3_text_embeddings_cache.pkl`

```python
{
    # Questions (276K unique)
    "what movies are about [ginger rogers]": np.array([...], dtype=float32),  # (1536,)
    "what films can be described by [new zealand]": np.array([...]),
    
    # Node IDs (42K unique)
    "ginger rogers": np.array([...]),
    "new zealand": np.array([...]),
    "Kitty Foyle": np.array([...]),
    
    # Edge Relations (9 unique)
    "has_tags": np.array([...]),
    "starred_actors": np.array([...]),
    "directed_by": np.array([...]),
    "written_by": np.array([...]),
    "has_genre": np.array([...]),
    "in_language": np.array([...]),
    "release_year": np.array([...]),
    "has_imdb_rating": np.array([...]),
    "has_imdb_votes": np.array([...]),
    
    # Edge Targets (43K unique)
    "The World's Fastest Indian": np.array([...]),
    "drama": np.array([...]),
    # ... more targets
}
```

**File size**: ~1.8 GB (319,477 texts × 1536 dims × 4 bytes)

### Statistics File: `variant3_embedding_stats.pkl`

```python
{
    'total_training_samples': 5691906,
    'unique_questions': 276234,
    'unique_nodes': 41741,
    'unique_relations': 9,
    'unique_targets': 43231,
    'total_unique_texts': 319477,
    'embedding_dimension': 1536,
    'model': 'text-embedding-3-small'
}
```

## How to Use

### Running the Script

```bash
# Make sure you have .env with OPENAI_API_KEY
python -m variant3.variant3_cache_embeddings
```

The script will:
1. Load training data (~5.7M samples)
2. Extract and deduplicate unique texts
3. Show cost estimate
4. Generate embeddings in batches (with progress bar)
5. Save cache and statistics

### Using the Cache in Training

```python
import pickle
import numpy as np

# Load cache once
with open('embeddings/variant3_text_embeddings_cache.pkl', 'rb') as f:
    text_cache = pickle.load(f)

# During training, lookup embeddings
for sample in training_data:
    q_emb = text_cache[sample['question_text']]      # (1536,)
    n_emb = text_cache[sample['node_id']]            # (1536,)
    e_emb = text_cache[sample['edge_relation']]      # (1536,)
    t_emb = text_cache[sample['edge_target']]        # (1536,)
    
    # Concatenate for model input
    features = np.concatenate([q_emb, n_emb, e_emb, t_emb])  # (6144,)
```

## Integration with EdgeScorerDual Model

The cached text embeddings are ONE component of the dual architecture:

```
Model Inputs:
├── Text Embeddings (from cache):
│   ├── question_text_emb   [B, 1536]  ✅ From cache
│   ├── node_text_emb       [B, 1536]  ✅ From cache
│   ├── edge_text_emb       [B, 1536]  ✅ From cache
│   └── target_text_emb     [B, 1536]  ✅ From cache
│
├── Graph Embeddings (already exist):
│   ├── node_graph_emb      [B, 256]   ← transe_embeddings_epochs100.npy
│   ├── edge_graph_emb      [B, 256]   ← transe_relation_embeddings_epochs100.npy
│   └── target_graph_emb    [B, 256]   ← transe_embeddings_epochs100.npy
│
└── Hop (learned in model):
    └── hop                 [B]        ← Learned embedding layer
```

## Performance Notes

### Deduplication Savings

| Field | Unique Count | Notes |
|-------|-------------|-------|
| question_text | 276,234 | Many questions have same template |
| node_id | 41,741 | Knowledge graph entities |
| edge_relation | 9 | Only 9 relation types in MetaQA |
| edge_target | 43,231 | Target nodes (slightly more than sources) |
| **After dedup** | **319,477** | 41,738 texts appear in multiple fields |

### Why Deduplication Matters

Without deduplication:
- Total: 276,234 + 41,741 + 9 + 43,231 = **361,215 texts**
- Cost: ~$0.108

With deduplication:
- Total: **319,477 texts**
- Cost: ~$0.096
- **Savings**: 41,738 texts, $0.012

### Additional Memory Savings During Training

By using a lookup cache:
- **Without cache**: 5.7M samples × 1536 dims × 4 bytes = **35 GB** of duplicated embeddings
- **With cache**: 319K texts × 1536 dims × 4 bytes = **1.8 GB** + lookup overhead
- **Memory reduction**: 19.4× smaller!

## Next Steps

After generating the cache:

1. **Train EdgeScorer**:
   ```bash
   python -m variant3.variant3_train_edge_scorer_dual
   ```

2. **Evaluate on test set**:
   ```bash
   python -m variant3.variant3_evaluate
   ```

3. **Integrate into QA system**:
   ```bash
   python -m variant5_openai_guided  # Use trained scorer for relation ranking
   ```

## Troubleshooting

### Missing API Key
```
ValueError: OPENAI_API_KEY not found in environment variables
```
**Solution**: Make sure `.env` file exists and contains `OPENAI_API_KEY=sk-...`

### Rate Limit Errors
The script includes automatic retry with exponential backoff. If you hit rate limits:
- Reduce `batch_size` in the script (line 226)
- The checkpoint system ensures no work is lost

### Out of Memory
If loading the cache causes OOM:
- The cache is ~1.8 GB
- Make sure you have at least 4 GB available RAM
- During training, use smaller batch sizes

## Files Created

```
embeddings/
├── variant3_text_embeddings_cache.pkl           # Main cache (~1.8 GB)
├── variant3_embedding_stats.pkl                 # Statistics (~1 KB)
└── variant3_text_embeddings_cache.checkpoint.pkl # Temp file (deleted when done)
```

## Comparison with Other Variants

| Variant | Embedding Source | Dimension | Cost |
|---------|-----------------|-----------|------|
| Variant 0 | None (pure LLM) | N/A | $0.10 per query |
| Variant 1 | FastRP | 128 | Free |
| Variant 2 | TransE | 256 | Free |
| Variant 3 (text only) | OpenAI | 1536 | $0.096 (one-time) |
| **Variant 3 (dual)** | **OpenAI + TransE** | **1536 + 256** | **$0.096 (one-time)** |
| Variant 5 | Qwen + scoring | N/A | $0 (local) |

Variant 3 dual gets the best of both worlds: semantic understanding (text) + structural reasoning (graph)!
