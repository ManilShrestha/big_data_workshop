"""
Phase 1 (Updated): Decompose 2-hop and 3-hop Training Questions using OpenAI Batch API

Changes from v1:
- Skip 1-hop questions (already 1-hop, no decomposition needed)
- Use only 50% of 2-hop and 3-hop data (shuffled)
- Use 25K batch size instead of 50K
- Wait for new API key to be configured

Process:
1. Load 2-hop and 3-hop training questions
2. Sample 50% (shuffled)
3. Create batch API requests (max 25K per batch)
4. Submit batch jobs to OpenAI
5. Poll for completion
6. Download and save results

Estimated: ~7k 2-hop + ~7k 3-hop = ~14k questions total
"""

import json
import sys
import time
import random
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from qa_system.config import Config
from qa_system.utils.loader import load_qa_dataset

# MetaQA Relations
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


def build_decomposition_prompt(question_text: str, hop_count: int) -> str:
    """Build prompt for LLM decomposition (same as Phase 0)."""

    relations_str = ', '.join(METAQA_RELATIONS)

    # Build example based on hop count
    if hop_count == 2:
        example = '''Example (2-hop):
Question: "What genre are movies directed by the director of [The Matrix]?"
{
  "hop_1": {
    "question": "Who directed [The Matrix]?",
    "relation": "directed_by"
  },
  "hop_2": {
    "question": "What genre are movies directed by [answer_from_hop_1]?",
    "relation": "has_genre"
  }
}'''
    else:  # 3-hop
        example = '''Example (3-hop):
Question: "When did films written by [Send Me No Flowers] writers release?"
{
  "hop_1": {
    "question": "Who wrote [Send Me No Flowers]?",
    "relation": "written_by"
  },
  "hop_2": {
    "question": "What films were written by [answer_from_hop_1]?",
    "relation": "written_by"
  },
  "hop_3": {
    "question": "When did [answer_from_hop_2] release?",
    "relation": "release_year"
  }
}'''

    prompt = f"""Given a {hop_count}-hop knowledge graph question, decompose it into {hop_count} sequential 1-hop questions.

Available relations: {relations_str}

Question: "{question_text}"

Rules:
1. Each subquestion is a standalone 1-hop question
2. Use [entity_name] for entities from the original question (already in brackets)
3. Use [answer_from_hop_N] as placeholder for answers from previous hops
4. Select ONE relation per hop from the available relations list

{example}

Now decompose the question above. Respond in valid JSON format only."""

    return prompt


def create_batch_requests(questions: List, hop_count: int) -> List[Dict]:
    """
    Create batch API requests for all questions of a given hop count.

    Returns:
        List of batch request dictionaries
    """
    batch_requests = []

    for question in tqdm(questions, desc=f"Creating {hop_count}-hop requests"):
        prompt = build_decomposition_prompt(question.text, hop_count)

        request = {
            "custom_id": f"{question.question_id}_{hop_count}hop",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": Config.OPENAI_MODEL_CHAT,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
        }

        batch_requests.append(request)

    return batch_requests


def save_batch_requests(batch_requests: List[Dict], output_path: Path):
    """Save batch requests to JSONL file."""
    with open(output_path, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')

    print(f"   Saved {len(batch_requests)} requests to {output_path}")


def submit_batch_job(client: OpenAI, batch_file_path: Path) -> str:
    """
    Upload batch file and submit job to OpenAI.

    Returns:
        Batch job ID
    """
    print(f"\n   Uploading batch file: {batch_file_path}")

    with open(batch_file_path, 'rb') as f:
        batch_file = client.files.create(
            file=f,
            purpose='batch'
        )

    print(f"   File uploaded: {batch_file.id}")

    print("   Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    print(f"   Batch job created: {batch_job.id}")
    print(f"   Status: {batch_job.status}")

    return batch_job.id


def poll_batch_completion(client: OpenAI, batch_job_id: str, poll_interval: int = 60):
    """
    Poll batch job until completion.

    Args:
        poll_interval: Seconds between polls (default: 60)
    """
    print(f"\n   Polling batch job {batch_job_id}...")
    print(f"   (Checking every {poll_interval} seconds)")

    while True:
        batch_job = client.batches.retrieve(batch_job_id)
        status = batch_job.status

        print(f"\n   Status: {status}")

        if hasattr(batch_job, 'request_counts'):
            counts = batch_job.request_counts
            print(f"   Requests - Total: {counts.total}, "
                  f"Completed: {counts.completed}, "
                  f"Failed: {counts.failed}")

        if status == 'completed':
            print(f"\n    Batch job completed!")
            return batch_job
        elif status in ['failed', 'expired', 'cancelled']:
            print(f"\n    Batch job {status}!")
            if hasattr(batch_job, 'errors'):
                print(f"   Errors: {batch_job.errors}")
            return None

        # Still in progress
        print(f"   Waiting {poll_interval} seconds...")
        time.sleep(poll_interval)


def download_batch_results(client: OpenAI, batch_job) -> List[Dict]:
    """
    Download and parse batch results.

    Returns:
        List of result dictionaries
    """
    print(f"\n   Downloading results...")

    output_file_id = batch_job.output_file_id

    # Download file content
    file_response = client.files.content(output_file_id)
    content = file_response.read().decode('utf-8')

    # Parse JSONL
    results = []
    for line in content.strip().split('\n'):
        if line:
            results.append(json.loads(line))

    print(f"   Downloaded {len(results)} results")

    return results


def parse_batch_results(batch_results: List[Dict]) -> Dict:
    """
    Parse batch results into decomposed questions.

    Returns:
        Dict mapping question_id -> decomposition
    """
    decompositions = {}

    for result in batch_results:
        custom_id = result['custom_id']
        question_id = custom_id.rsplit('_', 1)[0]  # Remove "_Nhop" suffix

        if result['response']['status_code'] == 200:
            # Successful response
            body = result['response']['body']
            content = body['choices'][0]['message']['content']
            decomposition = json.loads(content)

            decompositions[question_id] = {
                'success': True,
                'decomposition': decomposition,
                'usage': body['usage']
            }
        else:
            # Error response
            decompositions[question_id] = {
                'success': False,
                'error': result['response']['body']
            }

    return decompositions


def main():
    print("=" * 80)
    print("Variant 6 - Phase 1 (v2): Decompose 2-hop & 3-hop Training Questions")
    print("=" * 80)
    print()

    # Initialize OpenAI client
    print("Initializing OpenAI client...")
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    print(f"Model: {Config.OPENAI_MODEL_CHAT}")
    print()

    # Create output directory
    output_dir = Path(__file__).parent / "data" / "phase1_batches_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training questions
    print("=" * 80)
    print("Step 1: Load and Sample Training Questions")
    print("=" * 80)

    # Load full datasets
    questions_2hop_full = load_qa_dataset(str(Config.QA_2HOP_TRAIN), hop_count=2)
    questions_3hop_full = load_qa_dataset(str(Config.QA_3HOP_TRAIN), hop_count=3)

    print(f"\nLoaded training questions:")
    print(f"   2-hop (full): {len(questions_2hop_full):,}")
    print(f"   3-hop (full): {len(questions_3hop_full):,}")

    # Shuffle and sample 50%
    random.seed(42)  # For reproducibility

    questions_2hop_shuffled = list(questions_2hop_full)
    random.shuffle(questions_2hop_shuffled)
    questions_2hop = questions_2hop_shuffled[:len(questions_2hop_shuffled)//2]

    questions_3hop_shuffled = list(questions_3hop_full)
    random.shuffle(questions_3hop_shuffled)
    questions_3hop = questions_3hop_shuffled[:len(questions_3hop_shuffled)//2]

    total_questions = len(questions_2hop) + len(questions_3hop)

    print(f"\nSampled 50% (shuffled, seed=42):")
    print(f"   2-hop: {len(questions_2hop):,}")
    print(f"   3-hop: {len(questions_3hop):,}")
    print(f"   Total: {total_questions:,}")
    print()

    # Estimate cost
    avg_tokens_per_question = 500  # Estimated from Phase 0
    total_tokens = total_questions * avg_tokens_per_question
    cost_per_million = 0.15 * 0.5 + 0.60 * 0.5  # Batch API 50% discount
    estimated_cost = (total_tokens / 1_000_000) * cost_per_million

    print(f"Estimated cost (with 50% batch discount): ${estimated_cost:.2f}")
    print()

    # Submit and complete batches SEQUENTIALLY to respect 2M token queue limit
    print("=" * 80)
    print("Step 2-5: Process Each Hop Count Sequentially")
    print("=" * 80)
    print("Note: Submitting one batch at a time to respect 2M token queue limit")
    print()

    MAX_BATCH_SIZE = 25000  # 25K per batch
    all_decompositions = {}
    completed_jobs = []

    for questions, hop_count in [(questions_2hop, 2), (questions_3hop, 3)]:
        print(f"\n{'#' * 80}")
        print(f"# Processing {hop_count}-HOP Questions ({len(questions):,} questions)")
        print(f"{'#' * 80}\n")

        # Create batch requests
        print(f"[Step 2.{hop_count}] Creating batch requests...")
        batch_requests = create_batch_requests(questions, hop_count)

        # Split into chunks of 25k
        num_batches = (len(batch_requests) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
        print(f"   Splitting into {num_batches} batches (max {MAX_BATCH_SIZE} requests each)")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * MAX_BATCH_SIZE
            end_idx = min((batch_idx + 1) * MAX_BATCH_SIZE, len(batch_requests))
            batch_chunk = batch_requests[start_idx:end_idx]

            batch_name = f"{hop_count}hop_batch{batch_idx+1}"
            print(f"\n{'=' * 80}")
            print(f"[{batch_name}] Requests {start_idx+1:,} to {end_idx:,} ({len(batch_chunk):,} total)")
            print(f"{'=' * 80}")

            # Save to JSONL file
            batch_file_path = output_dir / f"batch_requests_{batch_name}.jsonl"
            save_batch_requests(batch_chunk, batch_file_path)

            # Submit batch job
            print(f"\n   [{batch_name}] Submitting to OpenAI...")
            batch_job_id = submit_batch_job(client, batch_file_path)

            # Save batch job ID
            job_info = {
                'batch_job_id': batch_job_id,
                'hop_count': hop_count,
                'batch_index': batch_idx + 1,
                'total_batches': num_batches,
                'total_requests': len(batch_chunk),
                'request_range': f"{start_idx+1}-{end_idx}",
                'submitted_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            job_info_path = output_dir / f"batch_job_info_{batch_name}.json"
            with open(job_info_path, 'w') as f:
                json.dump(job_info, f, indent=2)

            print(f"   [{batch_name}] Job ID: {batch_job_id}")

            # Wait for THIS batch to complete before submitting next
            print(f"\n   [{batch_name}] Waiting for completion before submitting next batch...")
            batch_job = poll_batch_completion(client, batch_job_id, poll_interval=60)

            if batch_job is None:
                print(f"\n    [{batch_name}] Batch job failed!")
                print(f"    Stopping processing. Fix issues and restart from this batch.")

                # Save partial results before exiting
                if len(completed_jobs) > 0:
                    print("\n   Saving partial results from completed batches...")
                    for job_info_temp in completed_jobs:
                        batch_results = download_batch_results(client, job_info_temp['batch_job'])
                        decompositions = parse_batch_results(batch_results)
                        all_decompositions.update(decompositions)

                break

            print(f"\n    [{batch_name}] Completed successfully!")
            completed_jobs.append({
                'batch_name': batch_name,
                'batch_job': batch_job
            })

            # Download results immediately after completion
            print(f"   [{batch_name}] Downloading results...")
            batch_results = download_batch_results(client, batch_job)

            print(f"   [{batch_name}] Parsing results...")
            decompositions = parse_batch_results(batch_results)
            all_decompositions.update(decompositions)

            print(f"   [{batch_name}] Processed {len(decompositions)} decompositions")

        # If a batch failed, stop processing
        if batch_job is None:
            break

    if len(completed_jobs) == 0:
        print("\n All batch jobs failed!")
        return

    # Results already downloaded during sequential processing
    print("\n" + "=" * 80)
    print("Step 5: Results Downloaded")
    print("=" * 80)
    print(f"Downloaded and parsed {len(all_decompositions)} decompositions from {len(completed_jobs)} batches")

    # Create final output format
    print("\n   Creating final output...")

    final_output = []

    # Map question IDs to original questions
    all_questions_dict = {}
    for q in questions_2hop + questions_3hop:
        all_questions_dict[q.question_id] = q

    success_count = 0
    failed_count = 0

    for question_id, decomp_result in all_decompositions.items():
        if question_id not in all_questions_dict:
            continue

        question = all_questions_dict[question_id]

        entry = {
            'question_id': question_id,
            'question_text': question.text,
            'hop_count': question.hop_count
        }

        if decomp_result['success']:
            entry['decomposition'] = decomp_result['decomposition']
            entry['success'] = True
            success_count += 1
        else:
            entry['success'] = False
            entry['error'] = decomp_result['error']
            failed_count += 1

        final_output.append(entry)

    # Save final output
    output_path = Path(__file__).parent / "data" / "variant6_decomposed_questions_v2.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"   Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total questions: {len(final_output):,}")
    print(f"Successful decompositions: {success_count:,} ({100*success_count/len(final_output):.1f}%)")
    print(f"Failed decompositions: {failed_count:,}")
    print()

    if success_count == len(final_output):
        print(" All decompositions successful! Ready for Phase 2.")
    else:
        print(f" {failed_count} decompositions failed. Review errors in output file.")

    print()


if __name__ == "__main__":
    main()
