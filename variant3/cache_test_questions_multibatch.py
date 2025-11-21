#!/usr/bin/env python3
"""
Pre-compute and cache embeddings for test questions using multiple parallel batch jobs.

Splits questions into smaller batches (5000 each) for faster processing.
Each batch completes in ~10-15 minutes instead of 90+ minutes for a single large batch.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from openai import OpenAI
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qa_system.config import Config
from qa_system.utils.loader import load_qa_dataset


BATCH_SIZE = 5000  # Questions per batch job


def create_batch_file(texts: list, output_path: Path, start_idx: int = 0) -> Path:
    """
    Create a JSONL batch file for OpenAI Batch API.

    Args:
        texts: List of texts to embed
        output_path: Path to save JSONL file
        start_idx: Starting index for custom_id (for tracking across batches)
    """
    with open(output_path, 'w') as f:
        for i, text in enumerate(texts):
            request = {
                "custom_id": f"request-{start_idx + i}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": "text-embedding-3-small",
                    "input": text,
                    "encoding_format": "float"
                }
            }
            f.write(json.dumps(request) + '\n')

    return output_path


def submit_batch_job(client: OpenAI, batch_file_path: Path, batch_num: int) -> str:
    """Upload batch file and create batch job"""
    # Upload file
    with open(batch_file_path, 'rb') as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )

    # Create batch job
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={
            "description": f"Cache test questions - batch {batch_num}",
            "batch_number": str(batch_num)
        }
    )

    return batch_job.id


def check_batch_status(client: OpenAI, batch_ids: list) -> dict:
    """Check status of all batch jobs"""
    statuses = {}
    for batch_id in batch_ids:
        batch_job = client.batches.retrieve(batch_id)
        statuses[batch_id] = {
            'status': batch_job.status,
            'completed': batch_job.request_counts.completed if batch_job.request_counts else 0,
            'total': batch_job.request_counts.total if batch_job.request_counts else 0,
            'failed': batch_job.request_counts.failed if batch_job.request_counts else 0,
            'output_file_id': batch_job.output_file_id
        }
    return statuses


def wait_for_all_batches(client: OpenAI, batch_ids: list, poll_interval: int = 30):
    """Poll all batch jobs until all complete"""
    print(f"\n  Waiting for {len(batch_ids)} batch jobs to complete...")
    print(f"  (Polling every {poll_interval} seconds)")
    print()

    start_time = time.time()
    completed_batches = set()

    while len(completed_batches) < len(batch_ids):
        statuses = check_batch_status(client, batch_ids)

        elapsed = int(time.time() - start_time)
        elapsed_str = f"{elapsed//60}m {elapsed%60}s"

        # Check each batch
        all_done = True
        status_lines = []

        for i, batch_id in enumerate(batch_ids):
            status = statuses[batch_id]
            batch_status = status['status']
            progress = f"{status['completed']}/{status['total']}"

            if batch_status == "completed":
                completed_batches.add(batch_id)
                status_lines.append(f"    Batch {i+1}/{len(batch_ids)}:  completed ({progress})")
            elif batch_status == "failed":
                status_lines.append(f"    Batch {i+1}/{len(batch_ids)}:  FAILED")
            elif batch_status in ["cancelled", "expired"]:
                status_lines.append(f"    Batch {i+1}/{len(batch_ids)}:  {batch_status}")
            else:
                all_done = False
                status_lines.append(f"    Batch {i+1}/{len(batch_ids)}: {batch_status} ({progress})")

        # Print status
        print(f"\r  [{elapsed_str}] Progress: {len(completed_batches)}/{len(batch_ids)} batches completed", end='')

        if all_done:
            print("\n")
            for line in status_lines:
                print(line)
            print()
            print(f"   All batch jobs completed!")
            print(f"   Time elapsed: {elapsed_str}")
            return statuses

        # Check for failures
        failed = [i for i, bid in enumerate(batch_ids) if statuses[bid]['status'] in ['failed', 'cancelled', 'expired']]
        if failed:
            print("\n")
            for line in status_lines:
                print(line)
            raise Exception(f"Batch jobs failed: {failed}")

        time.sleep(poll_interval)


def download_batch_results(client: OpenAI, output_file_id: str, texts: list, start_idx: int) -> dict:
    """Download and parse batch results with retry logic"""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Download content
            file_content = client.files.content(output_file_id)

            # Parse JSONL results
            embeddings = {}

            for line in file_content.text.strip().split('\n'):
                result = json.loads(line)

                # Extract custom_id and embedding
                custom_id = result['custom_id']
                request_idx = int(custom_id.split('-')[1]) - start_idx
                text = texts[request_idx]

                # Get embedding from response
                response_body = result['response']['body']
                embedding_data = response_body['data'][0]['embedding']
                embedding = np.array(embedding_data, dtype=np.float32)

                embeddings[text] = embedding

            return embeddings

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"\n    Retry in {wait_time}s... ({e})")
                time.sleep(wait_time)
            else:
                raise


def main():
    print("=" * 80)
    print("Cache Test Question Embeddings (Multi-Batch - Faster)")
    print("=" * 80)
    print()

    # Initialize OpenAI client
    print("[1/6] Initializing OpenAI client...")
    if not Config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in environment")
        sys.exit(1)

    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    print("   OpenAI client initialized")
    print()

    # Load existing cache
    print("[2/6] Loading existing cache...")
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
    print("[3/6] Loading test questions...")
    test_datasets = [
        (Config.QA_1HOP_TEST, '1-hop', 1),
        (Config.QA_2HOP_TEST, '2-hop', 2),
        (Config.QA_3HOP_TEST, '3-hop', 3),
    ]

    all_questions = []

    for qa_path, name, hop_count in test_datasets:
        questions = load_qa_dataset(qa_path, hop_count=hop_count)
        question_texts = [q.text for q in questions]

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

    print(f"  Total questions to embed: {len(all_questions):,}")
    print()

    if len(all_questions) == 0:
        print(" All test questions are already cached!")
        print()
        return

    # Split into batches
    num_batches = (len(all_questions) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Splitting into {num_batches} batches of ~{BATCH_SIZE} questions each")
    print()

    # Estimate cost (50% discount for batch API)
    estimated_tokens = len(all_questions) * 8
    estimated_cost_batch = estimated_tokens * (0.01 / 1_000_000)
    print(f"Estimated cost (batch API): ${estimated_cost_batch:.4f} (50% savings)")
    print()

    # Create batch files and submit jobs
    print("[4/6] Creating and submitting batch jobs...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_ids = []
    batch_info = []

    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(all_questions))
        batch_questions = all_questions[start_idx:end_idx]

        print(f"  Batch {i+1}/{num_batches}: {len(batch_questions):,} questions")

        # Create batch file
        batch_file_path = Config.EMBEDDINGS_DIR / f"batch_{timestamp}_part{i+1}.jsonl"
        create_batch_file(batch_questions, batch_file_path, start_idx)

        # Submit batch job
        batch_id = submit_batch_job(client, batch_file_path, i+1)
        batch_ids.append(batch_id)

        batch_info.append({
            'batch_id': batch_id,
            'batch_num': i + 1,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'num_questions': len(batch_questions)
        })

        print(f"     Submitted: {batch_id}")

        # Clean up batch file
        batch_file_path.unlink()

        # Small delay between submissions
        time.sleep(0.5)

    print()
    print(f"   All {num_batches} batch jobs submitted")
    print()

    # Save batch info
    batch_info_path = Config.EMBEDDINGS_DIR / f"multibatch_info_{timestamp}.json"
    with open(batch_info_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'num_batches': num_batches,
            'total_questions': len(all_questions),
            'batch_size': BATCH_SIZE,
            'batches': batch_info
        }, f, indent=2)

    print(f"   Batch info saved: {batch_info_path.name}")
    print()

    # Wait for all batches to complete
    print("[5/6] Waiting for batch completion...")
    final_statuses = wait_for_all_batches(client, batch_ids, poll_interval=20)

    # Download all results
    print("[6/6] Downloading results...")
    all_embeddings = {}

    for i, batch_id in enumerate(batch_ids):
        info = batch_info[i]
        output_file_id = final_statuses[batch_id]['output_file_id']

        if not output_file_id:
            print(f"   Batch {i+1}: No output file")
            continue

        print(f"  Batch {i+1}/{num_batches}: Downloading...")

        batch_questions = all_questions[info['start_idx']:info['end_idx']]
        embeddings = download_batch_results(client, output_file_id, batch_questions, info['start_idx'])

        all_embeddings.update(embeddings)
        print(f"     Downloaded {len(embeddings):,} embeddings")

    print()
    print(f"   Total embeddings downloaded: {len(all_embeddings):,}")
    print()

    # Update cache
    print("[7/7] Updating cache...")

    # Create backup
    if cache_path.exists():
        backup_path = cache_path.with_suffix('.pkl.backup')
        print(f"  Creating backup: {backup_path.name}")
        with open(backup_path, 'wb') as f:
            with open(cache_path, 'rb') as f_old:
                f.write(f_old.read())

    # Update cache
    cache.update(all_embeddings)

    # Save updated cache
    print(f"  Saving updated cache: {cache_path.name}")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

    print(f"   Cache updated: {len(cache):,} total embeddings")
    print()

    # Summary
    print("=" * 80)
    print("Multi-Batch Complete!")
    print("=" * 80)
    print()
    print(f"Batches processed: {num_batches}")
    print(f"New embeddings added: {len(all_embeddings):,}")
    print(f"Total cache size: {len(cache):,}")
    print(f"Estimated cost: ${estimated_cost_batch:.6f}")
    print()
    print(" All test questions are now cached!")
    print(" Future evaluations will have NO OpenAI API costs for question embeddings")
    print()


if __name__ == "__main__":
    main()
