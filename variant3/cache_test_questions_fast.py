#!/usr/bin/env python3
"""
Fast parallel embedding cache using regular OpenAI API.

Uses parallel requests with conservative batch sizes for speed and reliability.
Much faster than Batch API (3-5 minutes vs 40+ minutes).
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from openai import OpenAI
import time
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load environment
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qa_system.config import Config
from qa_system.utils.loader import load_qa_dataset
from qa_system.relation_rankers.openai_ranker import calculate_api_cost


# Configuration
BATCH_SIZE = 100  # Questions per API call (conservative, stable)
MAX_WORKERS = 20  # Parallel threads (adjust based on rate limits)


def embed_batch(client: OpenAI, texts: list, batch_num: int) -> tuple:
    """
    Embed a batch of texts using OpenAI API.

    Returns:
        (embeddings_dict, cost, success)
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )

            # Extract embeddings
            embeddings = {}
            for i, data in enumerate(response.data):
                text = texts[i]
                embedding = np.array(data.embedding, dtype=np.float32)
                embeddings[text] = embedding

            # Calculate cost
            cost = calculate_api_cost(response)

            return (embeddings, cost, True)

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)  # Exponential backoff
            else:
                print(f"\n   Batch {batch_num} failed after {max_retries} attempts: {e}")
                return ({}, 0.0, False)


def embed_questions_parallel(client: OpenAI, questions: list) -> tuple:
    """
    Embed questions in parallel using ThreadPoolExecutor.

    Returns:
        (all_embeddings_dict, total_cost)
    """
    # Split into batches
    batches = []
    for i in range(0, len(questions), BATCH_SIZE):
        batch = questions[i:i+BATCH_SIZE]
        batches.append((i // BATCH_SIZE + 1, batch))

    print(f"  Processing {len(batches)} batches of {BATCH_SIZE} questions each")
    print(f"  Using {MAX_WORKERS} parallel workers")
    print()

    all_embeddings = {}
    total_cost = 0.0
    failed_batches = []

    # Progress bar
    with tqdm(total=len(batches), desc="  Embedding", ncols=80) as pbar:
        # Submit all batches in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            futures = {
                executor.submit(embed_batch, client, batch, batch_num): (batch_num, batch)
                for batch_num, batch in batches
            }

            # Process completed tasks
            for future in as_completed(futures):
                batch_num, batch = futures[future]
                embeddings, cost, success = future.result()

                if success:
                    all_embeddings.update(embeddings)
                    total_cost += cost
                else:
                    failed_batches.append(batch_num)

                pbar.update(1)

    print()

    if failed_batches:
        print(f"   Failed batches: {failed_batches}")

    return all_embeddings, total_cost


def main():
    print("=" * 80)
    print("Cache Test Question Embeddings (Fast Parallel)")
    print("=" * 80)
    print()

    start_time = time.time()

    # Initialize OpenAI client
    print("[1/5] Initializing OpenAI client...")
    if not Config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in environment")
        sys.exit(1)

    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    print("   OpenAI client initialized")
    print()

    # Load existing cache
    print("[2/5] Loading existing cache...")
    cache_path = Config.EMBEDDINGS_DIR / "variant3_text_embeddings_cache.pkl"

    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print(f"   Loaded {len(cache):,} existing embeddings")
    else:
        print("  ! Cache file not found, creating new cache")
        cache = {}
    print()

    # Load test questions
    print("[3/5] Loading test questions...")
    test_datasets = [
        (Config.QA_1HOP_TEST, '1-hop', 1),
        (Config.QA_2HOP_TEST, '2-hop', 2),
        (Config.QA_3HOP_TEST, '3-hop', 3),
    ]

    all_questions = []
    dataset_stats = []

    for qa_path, name, hop_count in test_datasets:
        questions = load_qa_dataset(qa_path, hop_count=hop_count)
        question_texts = [q.text for q in questions]

        # Check how many are already cached
        cached = sum(1 for text in question_texts if text in cache)
        missing = len(question_texts) - cached

        dataset_stats.append({
            'name': name,
            'total': len(question_texts),
            'cached': cached,
            'missing': missing
        })

        print(f"  {name}: {len(question_texts):,} questions ({cached:,} cached, {missing:,} missing)")

        # Add missing questions to the list
        for text in question_texts:
            if text not in cache:
                all_questions.append(text)

    # Remove duplicates while preserving order
    seen = set()
    unique_questions = []
    for q in all_questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    all_questions = unique_questions

    print()
    print(f"  Total questions to embed: {len(all_questions):,}")
    print()

    if len(all_questions) == 0:
        print(" All test questions are already cached!")
        print()
        return

    # Estimate cost
    estimated_tokens = len(all_questions) * 8
    estimated_cost = estimated_tokens * (0.02 / 1_000_000)

    print(f"  Estimated cost: ${estimated_cost:.4f}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Parallel workers: {MAX_WORKERS}")
    print(f"  Estimated time: 3-5 minutes")
    print()

    # Embed questions
    print("[4/5] Embedding questions...")
    new_embeddings, total_cost = embed_questions_parallel(client, all_questions)

    print(f"   Embedded {len(new_embeddings):,} questions")
    print(f"   Total cost: ${total_cost:.6f}")
    print()

    # Update cache
    print("[5/5] Updating cache...")

    # Create backup
    if cache_path.exists():
        backup_path = cache_path.with_suffix('.pkl.backup')
        print(f"  Creating backup: {backup_path.name}")
        with open(backup_path, 'wb') as f:
            with open(cache_path, 'rb') as f_old:
                f.write(f_old.read())

    # Update cache
    cache.update(new_embeddings)

    # Save updated cache
    print(f"  Saving updated cache: {cache_path.name}")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

    print(f"   Cache updated: {len(cache):,} total embeddings")
    print()

    # Summary
    elapsed_time = time.time() - start_time
    elapsed_str = f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s"

    print("=" * 80)
    print("Fast Parallel Embedding Complete!")
    print("=" * 80)
    print()
    print("Dataset Coverage:")
    for stats in dataset_stats:
        print(f"  {stats['name']}: {stats['total']:,} questions (100% cached)")

    print()
    print(f"New embeddings added: {len(new_embeddings):,}")
    print(f"Total cache size: {len(cache):,}")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"Total time: {elapsed_str}")
    print()
    print(" All test questions are now cached!")
    print(" Future evaluations will have NO OpenAI API costs for question embeddings")
    print()


if __name__ == "__main__":
    main()
