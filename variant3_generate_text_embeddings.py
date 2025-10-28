"""
Generate text embeddings for all entities and relations in the knowledge graph.

This script:
1. Loads the knowledge graph
2. Extracts unique entities and relations
3. Generates OpenAI text embeddings for each
4. Caches embeddings for fast lookup during training

Cost estimation:
- ~50K entities × 2 tokens avg = 100K tokens = $0.002
- ~500 relations × 2 tokens avg = 1K tokens = $0.00002
Total: ~$0.002
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Set
from tqdm import tqdm
import openai
import time

from qa_system.config import Config
from qa_system.graph.graph_loader import load_graph


def get_text_from_entity(entity_id: str) -> str:
    """
    Convert entity ID to human-readable text.

    Examples:
        'm.0d05w3' -> 'person entity'  (generic fallback)
        'Tom_Hanks' -> 'Tom Hanks'
        'Forrest_Gump' -> 'Forrest Gump'
    """
    # Remove common prefixes
    text = entity_id.replace('m.', '').replace('g.', '')

    # Replace underscores with spaces
    text = text.replace('_', ' ')

    # If still looks like an ID (alphanumeric hash), use generic
    if len(text) < 20 and not any(c.isspace() for c in text):
        # Likely an ID, use generic
        return "entity"

    return text


def get_text_from_relation(relation: str) -> str:
    """
    Convert relation name to human-readable text.

    Examples:
        'directed_by' -> 'directed by'
        'starred_actors' -> 'starred actors'
        'film.film.genre' -> 'film genre'
    """
    # Remove domain prefixes like 'film.film.'
    parts = relation.split('.')
    if len(parts) > 1:
        # Take last part (most specific)
        relation = parts[-1]

    # Replace underscores with spaces
    text = relation.replace('_', ' ')

    return text


def batch_embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 2048,  # OpenAI max
    max_retries: int = 3
) -> Dict[str, np.ndarray]:
    """
    Batch embed texts using OpenAI API.

    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model
        batch_size: Number of texts per API call
        max_retries: Max retry attempts for failed batches

    Returns:
        Dict mapping text -> embedding vector (np.ndarray)
    """
    embeddings = {}

    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches", total=num_batches):
        batch_texts = texts[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                response = openai.embeddings.create(
                    input=batch_texts,
                    model=model
                )

                # Store embeddings
                for j, text in enumerate(batch_texts):
                    embedding = response.data[j].embedding
                    embeddings[text] = np.array(embedding, dtype=np.float32)

                break  # Success

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\nBatch {i // batch_size + 1} failed (attempt {attempt + 1}): {e}")
                    print(f"Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"\nBatch {i // batch_size + 1} failed after {max_retries} attempts: {e}")
                    # Skip this batch
                    break

        # Rate limiting
        time.sleep(0.1)

    return embeddings


def main():
    print("=" * 80)
    print("Variant 3: Generate Text Embeddings for Entities and Relations")
    print("=" * 80)

    # Set OpenAI API key
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    openai.api_key = Config.OPENAI_API_KEY

    # Load graph
    print("\n[1/5] Loading knowledge graph...")
    graph = load_graph()
    print(f"   Nodes: {graph.number_of_nodes():,}")
    print(f"   Edges: {graph.number_of_edges():,}")

    # Extract unique entities and relations
    print("\n[2/5] Extracting unique entities and relations...")

    entities: Set[str] = set()
    relations: Set[str] = set()

    for node in graph.nodes():
        entities.add(node)

    for _, _, data in graph.edges(data=True):
        relation = data.get('relation', 'unknown')
        relations.add(relation)

    print(f"   Unique entities: {len(entities):,}")
    print(f"   Unique relations: {len(relations):,}")

    # Convert to text
    print("\n[3/5] Converting to human-readable text...")

    entity_texts = {}
    for entity_id in tqdm(entities, desc="Processing entities"):
        entity_texts[entity_id] = get_text_from_entity(entity_id)

    relation_texts = {}
    for relation in relations:
        relation_texts[relation] = get_text_from_relation(relation)

    # Get unique text strings to embed
    unique_entity_texts = sorted(set(entity_texts.values()))
    unique_relation_texts = sorted(set(relation_texts.values()))

    print(f"   Unique entity texts: {len(unique_entity_texts):,}")
    print(f"   Unique relation texts: {len(unique_relation_texts):,}")

    # Estimate cost
    total_texts = len(unique_entity_texts) + len(unique_relation_texts)
    avg_tokens_per_text = 2  # Most are 1-3 tokens
    total_tokens = total_texts * avg_tokens_per_text
    cost = total_tokens * Config.COST_PER_EMBEDDING

    print(f"\n[Cost Estimation]")
    print(f"   Total texts: {total_texts:,}")
    print(f"   Estimated tokens: {total_tokens:,}")
    print(f"   Estimated cost: ${cost:.4f}")

    # Confirm
    user_input = input("\nProceed with API calls? (yes/no): ")
    if user_input.lower() not in ['yes', 'y']:
        print("Aborted by user")
        return

    # Generate embeddings
    print("\n[4/5] Generating embeddings...")

    print("   Embedding entities...")
    entity_text_embeddings = batch_embed_texts(unique_entity_texts, Config.OPENAI_MODEL_EMBED)

    print("   Embedding relations...")
    relation_text_embeddings = batch_embed_texts(unique_relation_texts, Config.OPENAI_MODEL_EMBED)

    # Map back to original IDs
    print("\n   Mapping embeddings to IDs...")

    entity_embeddings = {}
    for entity_id, text in entity_texts.items():
        if text in entity_text_embeddings:
            entity_embeddings[entity_id] = entity_text_embeddings[text]
        else:
            # Fallback: zero vector
            entity_embeddings[entity_id] = np.zeros(1536, dtype=np.float32)

    relation_embeddings = {}
    for relation, text in relation_texts.items():
        if text in relation_text_embeddings:
            relation_embeddings[relation] = relation_text_embeddings[text]
        else:
            # Fallback: zero vector
            relation_embeddings[relation] = np.zeros(1536, dtype=np.float32)

    print(f"   Entity embeddings: {len(entity_embeddings):,}")
    print(f"   Relation embeddings: {len(relation_embeddings):,}")

    # Save to disk
    print("\n[5/5] Saving embeddings...")
    output_dir = Config.EMBEDDINGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    entity_emb_path = output_dir / "variant3_entity_text_embeddings.pkl"
    relation_emb_path = output_dir / "variant3_relation_text_embeddings.pkl"

    with open(entity_emb_path, 'wb') as f:
        pickle.dump(entity_embeddings, f)

    with open(relation_emb_path, 'wb') as f:
        pickle.dump(relation_embeddings, f)

    print(f"   Entity embeddings saved: {entity_emb_path}")
    print(f"   Relation embeddings saved: {relation_emb_path}")

    # Save text mappings for reference
    text_mapping_path = output_dir / "variant3_text_mappings.pkl"
    with open(text_mapping_path, 'wb') as f:
        pickle.dump({
            'entity_texts': entity_texts,
            'relation_texts': relation_texts
        }, f)
    print(f"   Text mappings saved: {text_mapping_path}")

    # Verify
    sample_entity = list(entity_embeddings.keys())[0]
    sample_relation = list(relation_embeddings.keys())[0]

    print(f"\n[Verification]")
    print(f"   Sample entity: '{sample_entity}' -> '{entity_texts[sample_entity]}'")
    print(f"   Embedding shape: {entity_embeddings[sample_entity].shape}")
    print(f"   Sample relation: '{sample_relation}' -> '{relation_texts[sample_relation]}'")
    print(f"   Embedding shape: {relation_embeddings[sample_relation].shape}")

    print("\n" + "=" * 80)
    print("Text embedding generation complete!")
    print("=" * 80)
    print(f"\nNext step: Run variant3_create_training_data.py to generate training samples")


if __name__ == "__main__":
    main()