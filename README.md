# Efficient Multi-Hop Question Answering over Knowledge Graphs via LLM Planning and Embedding-Guided Search

A multi-hop question answering system over knowledge graphs using learned edge scoring models to guide graph traversal.

## Overview

This project implements several variants for answering natural language questions over the MetaQA knowledge graph:

- **Variant 0**: LLM baseline (GPT-5-mini, Qwen) - direct QA without graph traversal
- **Variant 1**: Rule-based graph traversal with heuristic edge scoring
- **Variant 3**: Neural edge scorer combining text embeddings (OpenAI) + graph embeddings (TransE) + hop context
- **Variant 5**: LLM-guided graph traversal using structured prompts
- **Variant 6**: Question decomposition approach
- **Variant 7**: Fine-tuned Qwen model for graph-guided QA

## Architecture

The core approach (Variant 3) uses a neural edge scorer that combines:
1. **Text Embeddings** (1536-dim): Semantic representations from OpenAI's text-embedding-3-small
2. **Graph Embeddings** (256-dim): Structural knowledge from TransE trained on the KG
3. **Hop Context**: Position encoding for multi-hop reasoning

The scorer guides beam search through the knowledge graph, predicting which edges are most likely to lead to correct answers.

## Project Structure

```
├── qa_system/           # Core QA system components
│   ├── entity_linkers/  # Entity linking (exact match)
│   ├── llm_qa/          # LLM-based QA variants
│   └── utils/           # Data loaders, config
├── variant3/            # Neural edge scorer implementation
│   ├── eval_qa/         # BFS/beam search evaluation
│   └── variant3_train_* # Training scripts for ablations
├── variant5/            # LLM-guided traversal
├── variant6/            # Question decomposition
├── variant7/            # Fine-tuned model approach
├── ablations_train.py   # Unified ablation training
├── ablations_test.py    # Unified ablation evaluation
└── train_embeddings.py  # TransE graph embedding training
```

## Key Results

Performance on MetaQA test sets (Hit Rate, F1):

| Method | 1-hop | 2-hop | 3-hop |
|--------|-------|-------|-------|
| Full Model (TE+GE+HC) | 99.6%, 95.2% | 97.3%, 77.4% | 91.1%, 55.3% |
| Text+Hop (Abl1) | 99.9%, 45.9% | 99.6%, 48.6% | 89.7%, 25.5% |
| Graph+Hop (Abl3) | 94.8%, 41.8% | 88.1%, 28.0% | 46.1%, 7.9% |


## Data

Uses the MetaQA dataset with:
- 43,234 entities (movies, actors, directors, etc.)
- 134,741 edges (9 relation types)
- 1/2/3-hop question-answer pairs


