"""Configuration for QA system"""

import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "metaqa"
    EMBEDDINGS_DIR = BASE_DIR / "embeddings"
    RESULTS_DIR = BASE_DIR / "results"

    # Graph and embeddings
    GRAPH_PATH = str(DATA_DIR / "graph.pkl")
    NODE2ID_PATH = str(EMBEDDINGS_DIR / "node2id.json")

    # FastRP embeddings
    FASTRP_PATH = str(EMBEDDINGS_DIR / "fastrp_embeddings_iter3.npy")

    # TransE embeddings (100 epochs - good performance)
    TRANSE_ENTITY_PATH = str(EMBEDDINGS_DIR / "transe_embeddings_epochs100.npy")
    TRANSE_RELATION_PATH = str(EMBEDDINGS_DIR / "transe_relation_embeddings_epochs100.npy")
    TRANSE_METADATA_PATH = str(EMBEDDINGS_DIR / "transe_metadata_epochs100.json")

    # QA datasets
    QA_1HOP_TRAIN = str(DATA_DIR / "1-hop" / "vanilla" / "qa_train.txt")
    QA_1HOP_DEV = str(DATA_DIR / "1-hop" / "vanilla" / "qa_dev.txt")
    QA_1HOP_TEST = str(DATA_DIR / "1-hop" / "vanilla" / "qa_test.txt")

    QA_2HOP_TRAIN = str(DATA_DIR / "2-hop" / "vanilla" / "qa_train.txt")
    QA_2HOP_DEV = str(DATA_DIR / "2-hop" / "vanilla" / "qa_dev.txt")
    QA_2HOP_TEST = str(DATA_DIR / "2-hop" / "vanilla" / "qa_test.txt")

    QA_3HOP_TRAIN = str(DATA_DIR / "3-hop" / "vanilla" / "qa_train.txt")
    QA_3HOP_DEV = str(DATA_DIR / "3-hop" / "vanilla" / "qa_dev.txt")
    QA_3HOP_TEST = str(DATA_DIR / "3-hop" / "vanilla" / "qa_test.txt")

    # Search parameters
    MAX_SEARCH_DEPTH = 3
    TIMEOUT_SECONDS = 30

    # Relation ranking
    NUM_TOP_RELATIONS = 9  # Use top-k relations from ranker (use all 9 for MetaQA)

    # OpenAI configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_EMBED = "text-embedding-3-small"
    OPENAI_MODEL_CHAT = "gpt-5-mini"  # Latest model for relation planning
    OPENAI_BATCH_SIZE = 50  # Number of questions to batch in parallel LLM calls
    OPENAI_MAX_WORKERS = 10  # ThreadPoolExecutor max workers for parallel API calls
    OPENAI_MAX_RETRIES = 5  # Number of retries for malformed JSON responses

    # Cost tracking (USD)
    COST_PER_EMBEDDING = 0.00000002  # text-embedding-3-small: $0.02 per 1M tokens
    COST_PER_CHAT_TOKEN = 0.0000005  # gpt-4o-mini average

    # MetaQA relations
    METAQA_RELATIONS = [
        "directed_by",
        "starred_actors",
        "written_by",
        "in_language",
        "has_genre",
        "release_year",
        "has_tags",
        "has_imdb_rating",
        "has_imdb_votes"
    ]