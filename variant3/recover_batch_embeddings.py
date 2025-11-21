#!/usr/bin/env python3
"""
Recovery script to download and cache completed batch embeddings.

Use this when the batch job completed but download failed.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from openai import OpenAI
import json
import time
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qa_system.config import Config
from qa_system.utils.loader import load_qa_dataset


def download_batch_results_with_retry(client: OpenAI, output_file_id: str, texts: list, max_retries: int = 5) -> dict:
    """Download and parse batch results with retry logic"""

    for attempt in range(max_retries):
        try:
            print(f"\n  Attempt {attempt + 1}/{max_retries}: Downloading results...")
            print(f"  File ID: {output_file_id}")
            print(f"  Expected size: ~781 MB (818,963,249 bytes)")

            # Try to download with longer timeout
            file_content = client.files.content(output_file_id)

            print(f"   Download successful!")
            print(f"  Parsing embeddings...")

            # Parse JSONL results
            embeddings = {}
            line_count = 0

            for line in file_content.text.strip().split('\n'):
                line_count += 1
                if line_count % 5000 == 0:
                    print(f"    Parsed {line_count:,} lines...", end='\r')

                result = json.loads(line)

                # Extract custom_id and embedding
                custom_id = result['custom_id']
                request_idx = int(custom_id.split('-')[1])
                text = texts[request_idx]

                # Get embedding from response
                response_body = result['response']['body']
                embedding_data = response_body['data'][0]['embedding']
                embedding = np.array(embedding_data, dtype=np.float32)

                embeddings[text] = embedding

            print(f"\n   Parsed {len(embeddings):,} embeddings")
            return embeddings

        except Exception as e:
            print(f"\n   Download failed: {e}")

            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Exponential backoff: 10s, 20s, 30s, etc.
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\n   Failed after {max_retries} attempts")
                raise


def main():
    print("=" * 80)
    print("Recovery: Download Batch Embeddings")
    print("=" * 80)
    print()

    # Initialize OpenAI client
    print("[1/5] Initializing OpenAI client...")
    if not Config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in environment")
        sys.exit(1)

    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    print("   OpenAI client initialized")
    print()

    # Find the most recent batch info file
    print("[2/5] Finding batch job info...")
    batch_info_files = sorted(Config.EMBEDDINGS_DIR.glob("batch_info_*.json"))

    if not batch_info_files:
        print("   No batch info files found!")
        print("  Please provide the batch ID manually:")
        batch_id = input("  Batch ID: ").strip()
    else:
        latest_batch_info = batch_info_files[-1]
        print(f"  Found: {latest_batch_info.name}")

        with open(latest_batch_info, 'r') as f:
            batch_info = json.load(f)

        batch_id = batch_info['batch_id']
        print(f"  Batch ID: {batch_id}")

    print()

    # Get batch job status
    print("[3/5] Checking batch job status...")
    batch_job = client.batches.retrieve(batch_id)

    print(f"  Status: {batch_job.status}")
    print(f"  Completed: {batch_job.request_counts.completed}/{batch_job.request_counts.total}")

    if batch_job.status != "completed":
        print(f"\n   Batch job not completed yet (status: {batch_job.status})")
        print(f"  Please wait for completion before running this script")
        sys.exit(1)

    if not batch_job.output_file_id:
        print(f"\n   No output file available")
        sys.exit(1)

    output_file_id = batch_job.output_file_id
    print(f"  Output file ID: {output_file_id}")
    print()

    # Load test questions (to map back to text)
    print("[4/5] Loading test questions...")
    test_datasets = [
        (Config.QA_1HOP_TEST, '1-hop', 1),
        (Config.QA_2HOP_TEST, '2-hop', 2),
        (Config.QA_3HOP_TEST, '3-hop', 3),
    ]

    all_questions = []

    for qa_path, name, hop_count in test_datasets:
        questions = load_qa_dataset(qa_path, hop_count=hop_count)
        question_texts = [q.text for q in questions]
        all_questions.extend(question_texts)

    # Remove duplicates while preserving order
    seen = set()
    unique_questions = []
    for q in all_questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    all_questions = unique_questions

    print(f"  Loaded {len(all_questions):,} questions")
    print()

    # Download and parse results
    print("[5/5] Downloading and parsing batch results...")
    new_embeddings = download_batch_results_with_retry(client, output_file_id, all_questions, max_retries=5)
    print()

    # Load existing cache
    print("[6/6] Updating cache...")
    cache_path = Config.EMBEDDINGS_DIR / "variant3_text_embeddings_cache.pkl"

    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print(f"  Loaded {len(cache):,} existing embeddings")
    else:
        print("  ! Cache file not found, creating new cache")
        cache = {}

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
    print("=" * 80)
    print("Recovery Complete!")
    print("=" * 80)
    print()
    print(f"New embeddings added: {len(new_embeddings):,}")
    print(f"Total cache size: {len(cache):,}")
    print()
    print(" All test questions are now cached!")
    print(" Future evaluations will have NO OpenAI API costs for question embeddings")
    print()


if __name__ == "__main__":
    main()
