"""Utilities for loading graph, embeddings, and datasets"""

import pickle
import json
import numpy as np
import networkx as nx
from typing import Tuple, List, Dict
from ..core.question import Question

def load_graph(graph_path: str) -> nx.DiGraph:
    """
    Load NetworkX graph from pickle file

    Args:
        graph_path: Path to graph.pkl

    Returns:
        NetworkX DiGraph
    """
    print(f"  Loading graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    print(f"  Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph

def load_node2id(node2id_path: str) -> Dict[str, int]:
    """
    Load node-to-ID mapping

    Args:
        node2id_path: Path to node2id.json

    Returns:
        Dictionary mapping entity names to indices
    """
    print(f"  Loading node2id from {node2id_path}...")
    with open(node2id_path, 'r') as f:
        node2id = json.load(f)

    print(f"  Loaded {len(node2id)} entity mappings")
    return node2id

def load_transe_embeddings(
    entity_path: str,
    relation_path: str,
    metadata_path: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load TransE embeddings

    Args:
        entity_path: Path to entity embeddings (.npy)
        relation_path: Path to relation embeddings (.npy)
        metadata_path: Path to metadata (.json)

    Returns:
        (entity_embeddings, relation_embeddings, relation2id)
    """
    print(f"  Loading TransE entity embeddings from {entity_path}...")
    entity_emb = np.load(entity_path)
    print(f"  Entity embeddings shape: {entity_emb.shape}")

    print(f"  Loading TransE relation embeddings from {relation_path}...")
    relation_emb = np.load(relation_path)
    print(f"  Relation embeddings shape: {relation_emb.shape}")

    print(f"  Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    relation2id = metadata['relation_to_id']
    print(f"  Loaded {len(relation2id)} relation mappings")

    return entity_emb, relation_emb, relation2id

def load_qa_dataset(
    dataset_path: str,
    hop_count: int,
    limit: int = None
) -> List[Question]:
    """
    Load QA dataset from file

    Args:
        dataset_path: Path to qa_*.txt file
        hop_count: Number of hops (1, 2, or 3)
        limit: Maximum number of questions to load (None = all)

    Returns:
        List of Question objects
    """
    print(f"  Loading {hop_count}-hop questions from {dataset_path}...")

    questions = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                question = Question.from_line(line, question_id=i, hop_count=hop_count)
                questions.append(question)
            except Exception as e:
                print(f"  Warning: Failed to parse line {i}: {e}")
                continue

    print(f"  Loaded {len(questions)} questions")
    return questions
