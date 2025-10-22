"""Direct LLM QA (no graph traversal) - Variant 0 baseline"""

import json
import time
from typing import List, Tuple
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..config import Config
from ..relation_rankers.openai_ranker import calculate_api_cost
from .prompts import build_direct_qa_prompt


class DirectLLMQA:
    """
    Variant 0: Direct LLM QA without graph traversal

    Simply asks GPT to answer the question directly using its knowledge.
    This serves as a baseline to show:
    1. Cost of pure LLM approach
    2. Hallucination issues (answers not in KB)
    3. Value of graph-grounded reasoning
    """

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Direct LLM QA

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

        print(f"  [DirectLLM] Initialized with model: {self.model}")

    def _build_qa_prompt(self, question: str) -> str:
        """Build the prompt for direct QA (uses shared prompt)"""
        return build_direct_qa_prompt(question)

    def answer_single(self, question: str) -> Tuple[List[str], float]:
        """
        Answer a single question directly with LLM

        Args:
            question: Question text

        Returns:
            Tuple of (list of answers, cost in USD)
        """
        prompt = self._build_qa_prompt(question)

        max_retries = Config.OPENAI_MAX_RETRIES

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )

                # Calculate cost
                cost = calculate_api_cost(response)
                self.total_cost += cost

                # Parse response
                try:
                    raw_content = response.choices[0].message.content
                    result = json.loads(raw_content)
                    answers = result.get('answers', [])

                    if not isinstance(answers, list):
                        raise ValueError("'answers' field is not a list")

                    # Log if empty response
                    if len(answers) == 0:
                        print(f"\n  [DirectLLM] ⚠️  WARNING: Empty answer list received")
                        print(f"  [DirectLLM] Question: {question}")
                        print(f"  [DirectLLM] Model: {self.model}")
                        print(f"  [DirectLLM] Raw response: {raw_content}")
                        print(f"  [DirectLLM] Prompt sent: {prompt[:300]}...")

                    # Success!
                    return answers, cost

                except (json.JSONDecodeError, ValueError) as e:
                    # JSON parsing failed - retry
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  [DirectLLM] Parse error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  [DirectLLM] Failed after {max_retries} retries")
                        return [], cost

            except Exception as e:
                # API call failed - retry
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  [DirectLLM] API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  [DirectLLM] Failed after {max_retries} retries")
                    return [], 0.0

        # Should never reach here
        return [], 0.0

    def answer_batch(
        self,
        questions: List[str],
        batch_size: int = None,
        max_workers: int = None,
        verbose: bool = True
    ) -> List[Tuple[List[str], float]]:
        """
        Answer multiple questions in parallel

        Args:
            questions: List of question texts
            batch_size: Questions per batch (None = use Config.OPENAI_BATCH_SIZE)
            max_workers: ThreadPoolExecutor workers (None = use Config.OPENAI_MAX_WORKERS)
            verbose: Print progress

        Returns:
            List of (answers, cost) tuples, one per question
        """
        if not questions:
            return []

        # Use config defaults if not specified
        if batch_size is None:
            batch_size = Config.OPENAI_BATCH_SIZE
        if max_workers is None:
            max_workers = Config.OPENAI_MAX_WORKERS

        num_batches = (len(questions) + batch_size - 1) // batch_size

        if verbose:
            print(f"  [DirectLLM] Processing {len(questions)} questions in {num_batches} batch(es) (batch_size={batch_size}, max_workers={max_workers})...")

        # Build prompts for all questions
        messages_batch = []
        for question in questions:
            prompt = self._build_qa_prompt(question)
            messages_batch.append([{"role": "user", "content": prompt}])

        # Helper function for parallel API calls
        def call_api_with_retries(args):
            """Call API with retry logic for individual question"""
            idx, messages = args
            question_text = questions[idx] if idx < len(questions) else "Unknown"
            max_retries = Config.OPENAI_MAX_RETRIES

            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        response_format={"type": "json_object"},
                    )

                    # Try to parse JSON immediately
                    try:
                        raw_content = response.choices[0].message.content

                        # Debug: print full response if empty
                        if not raw_content and verbose:
                            print(f"\n  [DirectLLM] ⚠️  Question {idx+1}: Empty content")
                            print(f"  [DirectLLM] Question: {question_text}")
                            print(f"  [DirectLLM] Full response object: {response}")
                            print(f"  [DirectLLM] Message object: {response.choices[0].message}")
                            print(f"  [DirectLLM] Finish reason: {response.choices[0].finish_reason}")

                        result = json.loads(raw_content)
                        answers = result.get('answers', [])

                        if not isinstance(answers, list):
                            raise ValueError("'answers' field is not a list")

                        # Log if empty response (for debugging)
                        if len(answers) == 0 and verbose:
                            print(f"\n  [DirectLLM] ⚠️  Question {idx+1}: Empty answer list")
                            print(f"  [DirectLLM] Question: {question_text}")
                            print(f"  [DirectLLM] Raw response: {raw_content}")

                        # Success!
                        return idx, response, answers, None

                    except (json.JSONDecodeError, ValueError) as e:
                        # JSON parsing failed - retry
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            if verbose:
                                print(f"\n  [DirectLLM] ⚠️  Question {idx+1} JSON parse error (attempt {attempt+1}/{max_retries}): {e}")
                                print(f"  [DirectLLM] Question: {question_text}")
                                print(f"  [DirectLLM] Raw response: '{raw_content}'")
                                print(f"  [DirectLLM] Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        else:
                            if verbose:
                                print(f"\n  [DirectLLM] ❌ Question {idx+1} failed after {max_retries} retries")
                                print(f"  [DirectLLM] Question: {question_text}")
                                print(f"  [DirectLLM] Final response: '{raw_content}'")
                            return idx, response, [], "FAILED"

                except Exception as e:
                    # API call failed - retry
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        if verbose:
                            print(f"  [DirectLLM] Question {idx+1} API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        if verbose:
                            print(f"  [DirectLLM] Question {idx+1} failed after {max_retries} retries")
                        return idx, None, [], "FAILED"

            # Should never reach here
            return idx, None, [], "FAILED"

        # Process in parallel with progress tracking
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(call_api_with_retries, (i, msg)): i for i, msg in enumerate(messages_batch)}

            # Collect results with progress tracking
            results = [None] * len(messages_batch)
            completed = 0
            last_print_pct = 0
            failed_count = 0

            for future in as_completed(futures):
                idx, response, answers, error = future.result()

                if error == "FAILED":
                    failed_count += 1
                    results[idx] = ([], 0.0)
                else:
                    cost = calculate_api_cost(response)
                    results[idx] = (answers, cost)

                completed += 1

                # Print progress every 10% or every batch_size questions
                if verbose:
                    current_pct = (completed * 100) // len(messages_batch)
                    if current_pct >= last_print_pct + 10 or completed % batch_size == 0 or completed == len(messages_batch):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(messages_batch) - completed) / rate if rate > 0 else 0
                        print(f"  [DirectLLM] Progress: {completed}/{len(messages_batch)} ({current_pct}%) | Rate: {rate:.1f} q/s | ETA: {eta:.0f}s")
                        last_print_pct = current_pct

        # Calculate total cost
        total_cost = sum(cost for _, cost in results)
        self.total_cost += total_cost

        if verbose:
            if failed_count > 0:
                print(f"  [DirectLLM] Warning: {failed_count} question(s) failed")
            print(f"  [DirectLLM] Batch complete - {len(questions)} questions")
            print(f"  [DirectLLM] Model used: {self.model}")
            print(f"  [DirectLLM] Total cost: ${total_cost:.6f} (${total_cost/len(questions):.6f} per question)")

        return results

    def get_cost(self) -> float:
        """Return total accumulated cost"""
        return self.total_cost
