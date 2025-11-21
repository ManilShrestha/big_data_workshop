"""
Check status of all OpenAI batch jobs submitted in Phase 1.
"""

import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from qa_system.config import Config

def main():
    print("=" * 80)
    print("Checking Batch Job Status")
    print("=" * 80)
    print()

    # Initialize OpenAI client
    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    # Find all batch job info files
    batch_dir = Path(__file__).parent / "data" / "phase1_batches"
    job_info_files = sorted(batch_dir.glob("batch_job_info_*hop_batch*.json"))

    print(f"Found {len(job_info_files)} batch jobs\n")

    all_jobs_info = []

    for job_file in job_info_files:
        with open(job_file, 'r') as f:
            job_info = json.load(f)

        batch_job_id = job_info['batch_job_id']
        batch_name = job_file.stem.replace('batch_job_info_', '')

        print(f"[{batch_name}]")
        print(f"  Job ID: {batch_job_id}")
        print(f"  Submitted: {job_info['submitted_at']}")
        print(f"  Requests: {job_info['total_requests']:,}")

        # Check status
        try:
            batch_job = client.batches.retrieve(batch_job_id)
            status = batch_job.status

            print(f"  Status: {status}")

            if hasattr(batch_job, 'request_counts'):
                counts = batch_job.request_counts
                print(f"  Progress: {counts.completed:,}/{counts.total:,} completed")
                if counts.failed > 0:
                    print(f"  Failed: {counts.failed:,}")

            if hasattr(batch_job, 'created_at'):
                print(f"  Created: {batch_job.created_at}")

            if hasattr(batch_job, 'completed_at') and batch_job.completed_at:
                print(f"  Completed: {batch_job.completed_at}")

            all_jobs_info.append({
                'batch_name': batch_name,
                'batch_job_id': batch_job_id,
                'status': status,
                'request_counts': {
                    'total': counts.total if hasattr(batch_job, 'request_counts') else 0,
                    'completed': counts.completed if hasattr(batch_job, 'request_counts') else 0,
                    'failed': counts.failed if hasattr(batch_job, 'request_counts') else 0,
                }
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            all_jobs_info.append({
                'batch_name': batch_name,
                'batch_job_id': batch_job_id,
                'status': 'error',
                'error': str(e)
            })

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    status_counts = {}
    total_completed = 0
    total_requests = 0
    total_failed = 0

    for job in all_jobs_info:
        status = job['status']
        status_counts[status] = status_counts.get(status, 0) + 1

        if 'request_counts' in job:
            total_completed += job['request_counts']['completed']
            total_requests += job['request_counts']['total']
            total_failed += job['request_counts']['failed']

    print(f"Total jobs: {len(all_jobs_info)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    print(f"\nTotal requests: {total_requests:,}")
    print(f"Completed: {total_completed:,} ({100*total_completed/total_requests:.1f}%)")

    if total_failed > 0:
        print(f"Failed: {total_failed:,}")

    print()

    # Check if all completed
    all_completed = all(job['status'] == 'completed' for job in all_jobs_info if job['status'] != 'error')
    any_running = any(job['status'] in ['validating', 'in_progress', 'finalizing'] for job in all_jobs_info)

    if all_completed:
        print(" All batch jobs completed! Ready to download results.")
    elif any_running:
        print("⏳ Some batch jobs are still running. Check back later.")
    else:
        print(" Check individual job statuses above.")

    print()


if __name__ == "__main__":
    main()
