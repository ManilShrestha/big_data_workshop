#!/usr/bin/env python3
"""
Batch Manager - Manage OpenAI Batch API jobs

This utility helps you check, list, and retrieve batch jobs without waiting.
Useful for debugging and monitoring long-running batches.
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from qa_system.llm_qa.batch_qa import BatchLLMQA


def list_batches(llm_qa: BatchLLMQA, limit: int = 10):
    """List recent batches"""
    print(f"\n{'='*80}")
    print(f"RECENT BATCHES (limit: {limit})")
    print(f"{'='*80}\n")

    batches = llm_qa.client.batches.list(limit=limit)

    if not batches.data:
        print("No batches found.")
        return

    for i, batch in enumerate(batches.data, 1):
        print(f"{i}. Batch ID: {batch.id}")
        print(f"   Status: {batch.status}")
        print(f"   Created: {batch.created_at}")
        print(f"   Requests: {batch.request_counts.total} total, "
              f"{batch.request_counts.completed} completed, "
              f"{batch.request_counts.failed} failed")
        if batch.metadata:
            desc = batch.metadata.get('description', 'N/A')
            print(f"   Description: {desc}")
        print()


def check_status(llm_qa: BatchLLMQA, batch_id: str):
    """Check status of a specific batch"""
    print(f"\n{'='*80}")
    print(f"BATCH STATUS: {batch_id}")
    print(f"{'='*80}\n")

    status = llm_qa.check_batch_status(batch_id, verbose=True)

    if status['status'] == 'completed':
        print(f"\n✅ Batch is complete!")
        print(f"\nTo retrieve results and continue evaluation, run:")
        print(f"  python variant0_llm_baseline.py --mode batch --batch-id {batch_id}")
    elif status['status'] == 'failed':
        print(f"\n❌ Batch failed!")
    elif status['status'] in ['validating', 'in_progress', 'finalizing']:
        print(f"\n⏳ Batch is still processing...")
        print(f"\nTo wait for completion, run:")
        print(f"  python variant0_llm_baseline.py --mode batch --batch-id {batch_id}")
    else:
        print(f"\nStatus: {status['status']}")


def cancel_batch(llm_qa: BatchLLMQA, batch_id: str):
    """Cancel a batch"""
    print(f"\n{'='*80}")
    print(f"CANCELLING BATCH: {batch_id}")
    print(f"{'='*80}\n")

    try:
        batch = llm_qa.client.batches.cancel(batch_id)
        print(f"✅ Batch cancelled successfully!")
        print(f"   New status: {batch.status}")
    except Exception as e:
        print(f"❌ Failed to cancel batch: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='OpenAI Batch API Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List recent batches
  python batch_manager.py --list

  # Check status of a specific batch
  python batch_manager.py --status batch_abc123

  # Cancel a batch
  python batch_manager.py --cancel batch_abc123
        """
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List recent batches'
    )
    parser.add_argument(
        '--status',
        type=str,
        metavar='BATCH_ID',
        help='Check status of a specific batch'
    )
    parser.add_argument(
        '--cancel',
        type=str,
        metavar='BATCH_ID',
        help='Cancel a batch'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of batches to list (default: 10)'
    )

    args = parser.parse_args()

    # Initialize BatchLLMQA
    llm_qa = BatchLLMQA()

    # Execute command
    if args.list:
        list_batches(llm_qa, limit=args.limit)
    elif args.status:
        check_status(llm_qa, args.status)
    elif args.cancel:
        cancel_batch(llm_qa, args.cancel)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
