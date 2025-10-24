"""Qwen LLM QA (no graph traversal) - Variant 0B baseline"""

import json
import re
import time
from typing import List, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..config import Config
from .prompts import build_qwen_qa_prompt


class QwenLLMQA:
    """
    Variant 0B: Direct LLM QA using Qwen 30B model

    Similar to DirectLLMQA but uses self-hosted Qwen model instead of OpenAI.
    Key differences:
    1. No cost (self-hosted)
    2. May have lower accuracy than GPT-4o
    3. More likely to produce malformed JSON - robust parsing needed
    4. Demonstrates model quality dependence
    """

    def __init__(self, base_url: str = None, model: str = None):
        """
        Initialize Qwen LLM QA

        Args:
            base_url: Base URL for Qwen API endpoint
            model: Model name/path
        """
        # Default to the endpoint from test_connections.ipynb
        self.base_url = base_url or "http://96.245.177.243:12302/v1"
        self.model = model or "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-8bit"
        self.total_cost = 0.0  # Keep for compatibility (always $0)

        print(f"  [QwenLLM] Initialized with model: {self.model}")
        print(f"  [QwenLLM] Endpoint: {self.base_url}")

    def _clean_answers(self, answers: List[str]) -> List[str]:
        """
        Clean and deduplicate answers to reduce hallucinations

        Args:
            answers: Raw answer list from model

        Returns:
            Cleaned answer list
        """
        if not answers:
            return []

        cleaned = []
        seen = set()

        for ans in answers:
            # Skip placeholder responses
            if ans.lower() in {'item1', 'item2', 'movie 1', 'movie 2', 'movies', 'films'}:
                continue

            # Skip if it looks like a generic template
            if ans.lower().startswith('the matrix:') and len(cleaned) > 3:
                # Stop after 3 Matrix-related answers (likely hallucinating)
                continue

            # Normalize for deduplication
            normalized = ans.lower().strip()

            # Skip duplicates
            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned.append(ans)

        return cleaned

    def _extract_json_robust(self, text: str) -> dict:
        """
        Robust JSON extraction with multiple fallback strategies

        Args:
            text: Response text that may contain JSON

        Returns:
            Parsed dict with 'answers' key, or {'answers': []} if all fail
        """
        # Strategy 1: Direct JSON parse
        try:
            result = json.loads(text)
            if 'answers' in result and isinstance(result['answers'], list):
                result['answers'] = self._clean_answers(result['answers'])
            return result
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find JSON object between curly braces
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                if 'answers' in result and isinstance(result['answers'], list):
                    result['answers'] = self._clean_answers(result['answers'])
                return result
        except (json.JSONDecodeError, AttributeError):
            pass

        # Strategy 3: Extract answers array specifically
        try:
            match = re.search(r'"answers"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if match:
                answers_str = match.group(1)
                # Parse individual quoted strings
                answers = re.findall(r'"([^"]*)"', answers_str)
                if answers:
                    return {'answers': self._clean_answers(answers)}
        except Exception:
            pass

        # Strategy 4: Look for any quoted strings that might be answers
        try:
            # Find all quoted strings (potential movie titles/names)
            potential_answers = re.findall(r'"([^"]{3,})"', text)
            # Filter out common JSON keys
            json_keys = {'answers', 'question', 'result', 'response', 'data'}
            filtered = [ans for ans in potential_answers if ans.lower() not in json_keys]
            if filtered:
                return {'answers': self._clean_answers(filtered[:20])}  # Limit to 20 max
        except Exception:
            pass

        # All strategies failed
        return {'answers': []}

    def _build_qa_prompt_pass1(self, question: str) -> str:
        """Build the prompt for Pass 1: Get raw answers (no JSON required)"""
        return f"""Answer this movie question with a simple list of movies or people. Just list the names, one per line.

Question: {question}

Instructions:
- List ONLY actual movies/people you are confident about
- If you don't know, say "I don't know"
- One name per line
- Movie titles WITHOUT years
- Do NOT make up fake titles

Your answer:"""

    def _build_formatting_prompt_pass2(self, question: str, raw_answer: str) -> str:
        """Build the prompt for Pass 2: Convert raw answer to JSON"""
        return f"""Convert this answer into JSON format.

Question: {question}
Answer: {raw_answer}

Convert the answer above into this exact JSON format:
{{"answers": ["item1", "item2", "item3"]}}

Rules:
- Each line becomes one item in the array
- Skip lines like "I don't know" or empty lines
- Keep the exact text, just put it in JSON format
- If there are no valid answers, return: {{"answers": []}}

JSON output:"""

    def _build_qa_prompt(self, question: str) -> str:
        """Build the prompt for direct QA (uses Qwen-specific prompt)"""
        return build_qwen_qa_prompt(question)

    def answer_single(self, question: str, use_two_pass: bool = True) -> Tuple[List[str], float]:
        """
        Answer a single question using Qwen API with two-pass approach

        Args:
            question: Question text
            use_two_pass: If True, use two-pass (answer → JSON). If False, one-pass.

        Returns:
            Tuple of (list of answers, cost in USD) - cost is always 0.0
        """
        max_retries = Config.OPENAI_MAX_RETRIES

        if not use_two_pass:
            # Single-pass mode (original behavior)
            prompt = self._build_qa_prompt(question)

            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.base_url}/completions",
                        headers={'Content-Type': 'application/json'},
                        json={
                            'model': self.model,
                            'prompt': prompt,
                            'max_tokens': 300,
                            'temperature': 0.1,
                            'top_p': 0.9,
                            'frequency_penalty': 0.5,
                            'presence_penalty': 0.3,
                            'stop': ['\n\n', 'Question:', 'Q:', 'Example']
                        },
                        timeout=30
                    )

                    if response.status_code != 200:
                        raise ValueError(f"API returned status {response.status_code}: {response.text}")

                    result = response.json()
                    completion_text = result['choices'][0]['text']
                    parsed = self._extract_json_robust(completion_text)
                    answers = parsed.get('answers', [])

                    if not isinstance(answers, list):
                        raise ValueError("'answers' field is not a list")

                    return answers, 0.0

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  [QwenLLM] Error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  [QwenLLM] Failed after {max_retries} retries: {e}")
                        return [], 0.0

        # TWO-PASS MODE (default)
        # Pass 1: Get raw answer (plain text, no JSON)
        prompt_pass1 = self._build_qa_prompt_pass1(question)
        raw_answer = None

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/completions",
                    headers={'Content-Type': 'application/json'},
                    json={
                        'model': self.model,
                        'prompt': prompt_pass1,
                        'max_tokens': 200,
                        'temperature': 0.1,
                        'top_p': 0.9,
                        'frequency_penalty': 0.5,
                        'presence_penalty': 0.3,
                        'stop': ['\n\nQuestion:', 'Q:']
                    },
                    timeout=30
                )

                if response.status_code != 200:
                    raise ValueError(f"API returned status {response.status_code}")

                result = response.json()
                raw_answer = result['choices'][0]['text'].strip()

                # Success - got raw answer
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  [QwenLLM] Pass1 error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  [QwenLLM] Pass1 failed after {max_retries} retries: {e}")
                    return [], 0.0

        if not raw_answer or raw_answer.lower() in {'i don\'t know', 'i dont know', 'unknown'}:
            return [], 0.0

        # Pass 2: Convert raw answer to JSON
        prompt_pass2 = self._build_formatting_prompt_pass2(question, raw_answer)

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/completions",
                    headers={'Content-Type': 'application/json'},
                    json={
                        'model': self.model,
                        'prompt': prompt_pass2,
                        'max_tokens': 300,
                        'temperature': 0,  # Deterministic for formatting
                        'stop': ['\n\n', 'Question:']
                    },
                    timeout=30
                )

                if response.status_code != 200:
                    raise ValueError(f"API returned status {response.status_code}")

                result = response.json()
                json_text = result['choices'][0]['text'].strip()

                # Extract JSON
                parsed = self._extract_json_robust(json_text)
                answers = parsed.get('answers', [])

                if not isinstance(answers, list):
                    raise ValueError("'answers' field is not a list")

                return answers, 0.0

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  [QwenLLM] Pass2 error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Fallback: Parse raw answer manually
                    print(f"  [QwenLLM] Pass2 failed, parsing raw answer manually")
                    lines = [line.strip() for line in raw_answer.split('\n') if line.strip()]
                    filtered = [line for line in lines if line.lower() not in {'i don\'t know', 'unknown', ''}]
                    return self._clean_answers(filtered), 0.0

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
            print(f"  [QwenLLM] Processing {len(questions)} questions in {num_batches} batch(es) (batch_size={batch_size}, max_workers={max_workers})...")

        # Helper function for parallel API calls
        def call_api_with_retries(args):
            """Call API with retry logic for individual question"""
            idx, question = args
            max_retries = Config.OPENAI_MAX_RETRIES

            for attempt in range(max_retries):
                try:
                    prompt = self._build_qa_prompt(question)

                    response = requests.post(
                        f"{self.base_url}/completions",
                        headers={'Content-Type': 'application/json'},
                        json={
                            'model': self.model,
                            'prompt': prompt,
                            'max_tokens': 300,
                            'temperature': 0.1,
                            'top_p': 0.9,
                            'frequency_penalty': 0.5,
                            'presence_penalty': 0.3,
                            'stop': ['\n\n', 'Question:', 'Q:', 'Example']
                        },
                        timeout=30
                    )

                    if response.status_code != 200:
                        raise ValueError(f"API returned status {response.status_code}")

                    result = response.json()
                    completion_text = result['choices'][0]['text']

                    # Use robust JSON extraction
                    parsed = self._extract_json_robust(completion_text)
                    answers = parsed.get('answers', [])

                    if not isinstance(answers, list):
                        raise ValueError("'answers' field is not a list")

                    # Log parsing issues
                    if len(answers) == 0 and verbose:
                        print(f"\n  [QwenLLM] ⚠️  Question {idx+1}: Empty answer list")
                        print(f"  [QwenLLM] Raw response: {completion_text[:200]}...")

                    # Success!
                    return idx, answers, None

                except Exception as e:
                    # Retry with exponential backoff
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        if verbose:
                            print(f"  [QwenLLM] Question {idx+1} error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        if verbose:
                            print(f"  [QwenLLM] Question {idx+1} failed after {max_retries} retries: {e}")
                        return idx, [], "FAILED"

            # Should never reach here
            return idx, [], "FAILED"

        # Process in parallel with progress tracking
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(call_api_with_retries, (i, q)): i for i, q in enumerate(questions)}

            # Collect results with progress tracking
            results = [None] * len(questions)
            completed = 0
            last_print_pct = 0
            failed_count = 0

            for future in as_completed(futures):
                idx, answers, error = future.result()

                if error == "FAILED":
                    failed_count += 1
                    results[idx] = ([], 0.0)
                else:
                    results[idx] = (answers, 0.0)  # No cost

                completed += 1

                # Print progress every 10% or every batch_size questions
                if verbose:
                    current_pct = (completed * 100) // len(questions)
                    if current_pct >= last_print_pct + 10 or completed % batch_size == 0 or completed == len(questions):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(questions) - completed) / rate if rate > 0 else 0
                        print(f"  [QwenLLM] Progress: {completed}/{len(questions)} ({current_pct}%) | Rate: {rate:.1f} q/s | ETA: {eta:.0f}s")
                        last_print_pct = current_pct

        if verbose:
            if failed_count > 0:
                print(f"  [QwenLLM] Warning: {failed_count} question(s) failed")
            print(f"  [QwenLLM] Batch complete - {len(questions)} questions")
            print(f"  [QwenLLM] Model used: {self.model}")
            print(f"  [QwenLLM] Total cost: $0.00 (self-hosted model)")

        return results

    def get_cost(self) -> float:
        """Return total accumulated cost (always $0 for self-hosted)"""
        return 0.0
