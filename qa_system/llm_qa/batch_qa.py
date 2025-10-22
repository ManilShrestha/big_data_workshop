"""Batch LLM QA using OpenAI Batch API - 50% cheaper, async processing"""

import json
import time
import tempfile
from typing import List, Tuple
from pathlib import Path
from openai import OpenAI
from ..config import Config
from ..relation_rankers.openai_ranker import calculate_api_cost
from .prompts import build_direct_qa_prompt


class BatchLLMQA:
    """
    Variant 0: Direct LLM QA using OpenAI Batch API

    Uses async batch processing for 50% cost savings.
    Batches can take minutes to hours to complete.
    """

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Batch LLM QA

        Args:
            api_key: OpenAI API key (defaults to Config.OPENAI_API_KEY)
            model: Model to use (defaults to Config.OPENAI_MODEL_CHAT)
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model or Config.OPENAI_MODEL_CHAT
        self.total_cost = 0.0

        print(f"  [BatchLLM] Initialized with model: {self.model}")

    def _build_qa_prompt(self, question: str) -> str:
        """Build the prompt for direct QA (uses shared prompt)"""
        return build_direct_qa_prompt(question)

    def create_batch(
        self,
        questions: List[str],
        batch_description: str = "Movie QA Batch",
        verbose: bool = True
    ) -> str:
        """
        Create a batch job for answering questions

        Args:
            questions: List of question texts
            batch_description: Description for the batch
            verbose: Print progress

        Returns:
            Batch ID for tracking
        """
        if verbose:
            print(f"\n  [BatchLLM] Creating batch for {len(questions)} questions...")

        # Create JSONL file with batch requests
        batch_requests = []
        for i, question in enumerate(questions):
            prompt = self._build_qa_prompt(question)
            request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                }
            }
            batch_requests.append(request)

        # Write to temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
            temp_file = f.name

        if verbose:
            print(f"  [BatchLLM] Created batch file: {temp_file}")

        # Upload file to OpenAI
        with open(temp_file, 'rb') as f:
            file_response = self.client.files.create(
                file=f,
                purpose='batch'
            )

        if verbose:
            print(f"  [BatchLLM] Uploaded file: {file_response.id}")

        # Create batch
        batch_response = self.client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": batch_description,
                "num_questions": str(len(questions))
            }
        )

        if verbose:
            print(f"  [BatchLLM] Batch created: {batch_response.id}")
            print(f"  [BatchLLM] Status: {batch_response.status}")
            print(f"  [BatchLLM] Completion window: 24h (required by API, but typically completes in minutes)")
            print(f"  [BatchLLM] Processing asynchronously - poll for status updates")

        # Clean up temp file
        Path(temp_file).unlink()

        return batch_response.id

    def check_batch_status(self, batch_id: str, verbose: bool = True) -> dict:
        """
        Check status of a batch job

        Args:
            batch_id: Batch ID to check
            verbose: Print status

        Returns:
            Batch status dict
        """
        batch = self.client.batches.retrieve(batch_id)

        if verbose:
            print(f"\n  [BatchLLM] Batch {batch_id} status:")
            print(f"    Status: {batch.status}")
            print(f"    Total requests: {batch.request_counts.total}")
            print(f"    Completed: {batch.request_counts.completed}")
            print(f"    Failed: {batch.request_counts.failed}")

        return {
            'id': batch.id,
            'status': batch.status,
            'total': batch.request_counts.total,
            'completed': batch.request_counts.completed,
            'failed': batch.request_counts.failed,
            'created_at': batch.created_at,
            'completed_at': batch.completed_at,
            'output_file_id': batch.output_file_id,
            'error_file_id': batch.error_file_id,
        }

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = 60,
        verbose: bool = True
    ) -> dict:
        """
        Wait for batch to complete

        Args:
            batch_id: Batch ID to wait for
            poll_interval: Seconds between status checks
            verbose: Print progress

        Returns:
            Final batch status
        """
        if verbose:
            print(f"\n  [BatchLLM] Waiting for batch {batch_id} to complete...")

        while True:
            status = self.check_batch_status(batch_id, verbose=verbose)

            if status['status'] in ['completed', 'failed', 'expired', 'cancelled']:
                if verbose:
                    print(f"\n  [BatchLLM] Batch finished with status: {status['status']}")
                return status

            if verbose:
                print(f"  [BatchLLM] Still processing... waiting {poll_interval}s")
            time.sleep(poll_interval)

    def retrieve_batch_results(
        self,
        batch_id: str,
        verbose: bool = True
    ) -> List[Tuple[List[str], float]]:
        """
        Retrieve results from completed batch

        Args:
            batch_id: Batch ID to retrieve
            verbose: Print progress

        Returns:
            List of (answers, cost) tuples, one per question
        """
        # Get batch info
        batch = self.client.batches.retrieve(batch_id)

        if batch.status != 'completed':
            raise ValueError(f"Batch {batch_id} is not completed. Status: {batch.status}")

        if not batch.output_file_id:
            raise ValueError(f"Batch {batch_id} has no output file")

        if verbose:
            print(f"\n  [BatchLLM] Retrieving results from batch {batch_id}...")

        # Download output file
        output_content = self.client.files.content(batch.output_file_id)
        output_lines = output_content.text.strip().split('\n')

        if verbose:
            print(f"  [BatchLLM] Retrieved {len(output_lines)} results")

        # Parse results
        results = {}
        total_cost = 0.0

        for line in output_lines:
            result = json.loads(line)
            custom_id = result['custom_id']
            idx = int(custom_id.split('-')[1])

            response_body = result['response']['body']

            # Calculate cost (batch API is 50% cheaper)
            # We'll estimate based on usage if available
            if 'usage' in response_body:
                # Create a mock response object for cost calculation
                class MockResponse:
                    def __init__(self, model, usage):
                        self.model = model
                        self.usage = usage

                class MockUsage:
                    def __init__(self, usage_dict):
                        self.prompt_tokens = usage_dict.get('prompt_tokens', 0)
                        self.completion_tokens = usage_dict.get('completion_tokens', 0)
                        self.total_tokens = usage_dict.get('total_tokens', 0)

                mock_response = MockResponse(
                    response_body.get('model', self.model),
                    MockUsage(response_body['usage'])
                )
                cost = calculate_api_cost(mock_response) * 0.5  # 50% discount for batch
                total_cost += cost
            else:
                cost = 0.0

            # Extract answers
            try:
                content = response_body['choices'][0]['message']['content']
                parsed = json.loads(content)
                answers = parsed.get('answers', [])

                if not isinstance(answers, list):
                    answers = []

                results[idx] = (answers, cost)

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                if verbose:
                    print(f"  [BatchLLM] Warning: Failed to parse result {idx}: {e}")
                results[idx] = ([], cost)

        # Sort by index and return as list
        sorted_results = [results.get(i, ([], 0.0)) for i in range(len(output_lines))]

        self.total_cost += total_cost

        if verbose:
            print(f"  [BatchLLM] Total cost: ${total_cost:.6f}")
            print(f"  [BatchLLM] Cost per question: ${total_cost/len(sorted_results):.6f}")

        return sorted_results

    def answer_batch_sync(
        self,
        questions: List[str],
        poll_interval: int = 60,
        verbose: bool = True
    ) -> List[Tuple[List[str], float]]:
        """
        Answer questions using batch API and wait for completion

        Args:
            questions: List of question texts
            poll_interval: Seconds between status checks
            verbose: Print progress

        Returns:
            List of (answers, cost) tuples, one per question
        """
        # Create batch
        batch_id = self.create_batch(questions, verbose=verbose)

        # Wait for completion
        self.wait_for_batch(batch_id, poll_interval=poll_interval, verbose=verbose)

        # Retrieve results
        return self.retrieve_batch_results(batch_id, verbose=verbose)

    def get_cost(self) -> float:
        """Return total accumulated cost"""
        return self.total_cost
