"""Qwen-based relation ranking and planning"""

from typing import List, Tuple
import json
import re
import time
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..core.base_relation_ranker import BaseRelationRanker
from ..config import Config


class QwenRelationRanker(BaseRelationRanker):
    """
    Use Qwen model for relation ranking and planning

    Key differences from OpenAIRelationRanker:
    1. No cost (self-hosted)
    2. No embeddings API - use text generation for ranking
    3. More likely to produce malformed JSON
    4. Demonstrates model quality impact
    """

    # Natural language descriptions of each relation (same as OpenAI)
    RELATION_TEXTS = {
        'directed_by': 'directed by, film director, movie director, who directed',
        'starred_actors': 'starring, actors, cast members, acted in, stars, featured',
        'written_by': 'written by, screenplay, screenwriter, author, wrote',
        'in_language': 'language, spoken language, in language, what language',
        'has_genre': 'genre, type of movie, category, kind of film',
        'release_year': 'release year, released in, year made, when released',
        'has_tags': 'tags, keywords, themes, topics, about',
        'has_imdb_rating': 'rating, score, imdb rating, rated',
        'has_imdb_votes': 'votes, popularity, famous, well-known, popular',
    }

    def __init__(self, base_url: str = None, model: str = None):
        """
        Initialize Qwen ranker

        Args:
            base_url: Base URL for Qwen API endpoint
            model: Model name/path
        """
        self.base_url = base_url or Config.QWEN_BASE_URL
        self.model = model or Config.QWEN_MODEL
        self.total_cost = 0.0  # Keep for compatibility (always $0)
        self.last_query_cost = 0.0

        print(f"  [QwenRanker] Initialized with model: {self.model}")
        print(f"  [QwenRanker] Endpoint: {self.base_url}")
        print(f"  [QwenRanker] Note: Using text-based ranking (no embeddings)")

    def _extract_json_robust(self, text: str) -> dict:
        """
        Robust JSON extraction with multiple fallback strategies

        Args:
            text: Response text that may contain JSON

        Returns:
            Parsed dict, or empty dict if all fail
        """
        # Strategy 1: Direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find FIRST JSON object between curly braces (non-greedy)
        # This prevents matching repeated patterns like {...}JSON: {...}JSON: {...}
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except (json.JSONDecodeError, AttributeError):
            pass

        # Strategy 3: Extract specific fields (reasoning + hops)
        try:
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, re.DOTALL)
            hops_match = re.search(r'"hops"\s*:\s*(\[.*?\])', text, re.DOTALL)

            if reasoning_match and hops_match:
                reasoning = reasoning_match.group(1)
                hops_str = hops_match.group(1)
                hops = json.loads(hops_str)
                return {'reasoning': reasoning, 'hops': hops}
        except Exception:
            pass

        # All strategies failed
        return {}

    def rank_relations(self, question: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Rank relations using LLM-based scoring (fallback - less accurate than embeddings)

        For Qwen, we use a simple prompt-based approach rather than embeddings.
        This is less accurate but demonstrates model dependence.

        Args:
            question: Question text
            top_k: Return only top-k (None = all)

        Returns:
            List of (relation, score) sorted by relevance
        """
        # Simple heuristic: keyword matching (fast, no API call needed)
        # This is intentionally simple to show the gap vs. OpenAI embeddings
        question_lower = question.lower()

        scores = []
        for rel in Config.METAQA_RELATIONS:
            rel_text = self.RELATION_TEXTS[rel].lower()
            keywords = rel_text.split(', ')

            # Count keyword matches
            match_count = sum(1 for kw in keywords if kw in question_lower)
            score = match_count / len(keywords) if keywords else 0.0

            scores.append((rel, score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k if specified
        if top_k is not None:
            scores = scores[:top_k]

        return scores

    def get_cost(self) -> float:
        """Return total accumulated cost (always $0 for self-hosted)"""
        return 0.0

    def get_last_query_cost(self) -> float:
        """Return cost of last query only (always $0 for self-hosted)"""
        return 0.0

    def _build_relation_planning_prompt_pass1(self, question: str, max_hops: int) -> str:
        """Build Pass 1 prompt: plain text reasoning (no JSON)"""
        relations_list = "\n".join([f"- {rel}" for rel in Config.METAQA_RELATIONS])

        return f"""You are analyzing a knowledge graph question to determine which relations to traverse.

Available relations in the knowledge graph:
{relations_list}

Question: "{question}"

This is a {max_hops}-hop question. You need to select which relation(s) might be relevant at EACH hop.

Think about the logical path needed to answer the question. For each hop, select 1-2 most relevant relations from the list above.

Examples:
- "Who directed movies starring [Actor]?" → Hop 1: starred_actors, Hop 2: directed_by
- "What genre are films written by [Writer]?" → Hop 1: written_by, Hop 2: has_genre
- "What year were movies by the director of [Movie]?" → Hop 1: directed_by, Hop 2: release_year

Provide your analysis as plain text (one relation per hop, one line per hop):
Hop 1: relation_name
Hop 2: relation_name
...

Your analysis:"""

    def _build_relation_planning_prompt_pass2(self, question: str, max_hops: int, raw_answer: str) -> str:
        """Build Pass 2 prompt: convert plain text to JSON"""
        return f"""Convert the following relation planning to JSON format.

Question: "{question}"
Max hops: {max_hops}

Analysis:
{raw_answer}

Convert to this exact JSON format:
{{
  "reasoning": "brief explanation of the path",
  "hops": [
    ["relation1"],
    ["relation2"],
    ...
  ]
}}

JSON:"""

    def plan_relation_sequence(self, question: str, max_hops: int) -> Tuple[List[List[str]], str]:
        """
        Use Qwen to plan which relations to use for multi-hop questions.
        Uses two-pass approach: first get reasoning, then convert to JSON.

        Args:
            question: Question text
            max_hops: Number of hops needed

        Returns:
            Tuple of (List of relation lists for each hop, reasoning string)
            Returns (None, "FAILED") if all retries fail
        """
        # PASS 1: Get plain text reasoning (retry API errors)
        prompt_pass1 = self._build_relation_planning_prompt_pass1(question, max_hops)
        raw_answer = None

        for attempt in range(Config.QWEN_MAX_RETRIES_API):
            try:
                response = requests.post(
                    f"{self.base_url}/completions",
                    headers={'Content-Type': 'application/json'},
                    json={
                        'model': self.model,
                        'prompt': prompt_pass1,
                        'max_tokens': 200,
                        'temperature': 0.1,
                        'frequency_penalty': 1.0,  # Prevent repetition
                        'presence_penalty': 0.5,   # Prevent repetition
                        'stop': ['\n\nQuestion:', 'Q:', '\n\n']
                    },
                    timeout=Config.QWEN_TIMEOUT
                )

                if response.status_code != 200:
                    raise ValueError(f"API returned status {response.status_code}")

                result = response.json()
                raw_answer = result['choices'][0]['text'].strip()
                break  # Success

            except Exception as e:
                if attempt < Config.QWEN_MAX_RETRIES_API - 1:
                    wait_time = min(2 ** attempt, 4)
                    print(f"  [QwenRanker] Pass1 API error (attempt {attempt+1}/{Config.QWEN_MAX_RETRIES_API}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  [QwenRanker] Pass1 failed after {Config.QWEN_MAX_RETRIES_API} API retries - marking as FAILED")
                    return None, "FAILED"

        if not raw_answer:
            return None, "FAILED"

        # PASS 2: Convert to JSON (retry API errors, limited parse retries)
        prompt_pass2 = self._build_relation_planning_prompt_pass2(question, max_hops, raw_answer)

        for attempt in range(Config.QWEN_MAX_RETRIES_API):
            try:
                response = requests.post(
                    f"{self.base_url}/completions",
                    headers={'Content-Type': 'application/json'},
                    json={
                        'model': self.model,
                        'prompt': prompt_pass2,
                        'max_tokens': 300,
                        'temperature': 0,
                        'frequency_penalty': 1.0,  # Prevent repetition
                        'presence_penalty': 0.5,   # Prevent repetition
                        'stop': ['\n\nQuestion:', '\n\nConvert', '}\n', 'JSON:']
                    },
                    timeout=Config.QWEN_TIMEOUT
                )

                if response.status_code != 200:
                    raise ValueError(f"API returned status {response.status_code}")

                result = response.json()
                completion_text = result['choices'][0]['text']

                # Use robust JSON extraction
                parse_attempt = 0
                parsed = None

                while parse_attempt < Config.QWEN_MAX_RETRIES_PARSE:
                    try:
                        parsed = self._extract_json_robust(completion_text)

                        reasoning = parsed.get('reasoning', raw_answer[:100])  # Fallback to raw
                        hops = parsed.get('hops', [])

                        # Validate hops structure
                        if not isinstance(hops, list):
                            raise ValueError("'hops' field is not a list")

                        # Filter hops to only include lists
                        filtered_hops = [hop for hop in hops if isinstance(hop, list)]

                        if len(filtered_hops) == 0:
                            raise ValueError("No valid hop lists found")

                        # Success!
                        print(f"  [QwenRanker] Relation planning - Reasoning: {reasoning}")
                        print(f"  [QwenRanker] Planned hops: {filtered_hops}")

                        return filtered_hops, reasoning

                    except (ValueError, KeyError) as e:
                        # Parse error - limited retries (re-request from API)
                        parse_attempt += 1
                        if parse_attempt < Config.QWEN_MAX_RETRIES_PARSE:
                            print(f"  [QwenRanker] Pass2 parse error (attempt {parse_attempt}/{Config.QWEN_MAX_RETRIES_PARSE}): {e}")
                            print(f"  [QwenRanker] Full response: {completion_text}")
                            print(f"  [QwenRanker] Re-requesting from API...")
                            break  # Break inner loop to retry API call
                        else:
                            # Final fallback: parse raw_answer manually
                            print(f"  [QwenRanker] Pass2 parse failed after {Config.QWEN_MAX_RETRIES_PARSE} attempts")
                            print(f"  [QwenRanker] Last response: {completion_text}")
                            print(f"  [QwenRanker] Attempting manual parse of raw answer...")
                            return self._parse_raw_planning(raw_answer, max_hops)

                # If we successfully parsed, we would have returned above
                # If we broke out to retry API call, continue the outer loop
                if parse_attempt < Config.QWEN_MAX_RETRIES_PARSE:
                    continue

            except Exception as e:
                # API error - full retries
                if attempt < Config.QWEN_MAX_RETRIES_API - 1:
                    wait_time = min(2 ** attempt, 4)
                    print(f"  [QwenRanker] Pass2 API error (attempt {attempt+1}/{Config.QWEN_MAX_RETRIES_API}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Final fallback: parse raw_answer manually
                    print(f"  [QwenRanker] Pass2 API failed after {Config.QWEN_MAX_RETRIES_API} retries")
                    print(f"  [QwenRanker] Attempting manual parse of raw answer...")
                    return self._parse_raw_planning(raw_answer, max_hops)

        # Should never reach here
        return None, "FAILED"

    def _parse_raw_planning(self, raw_answer: str, max_hops: int) -> Tuple[List[List[str]], str]:
        """
        Fallback: Parse plain text planning into hop structure

        Expected format:
        Hop 1: relation_name
        Hop 2: relation_name
        """
        import re

        lines = raw_answer.split('\n')
        hops = []

        for i in range(1, max_hops + 1):
            # Look for "Hop N: relation_name"
            pattern = rf'Hop\s*{i}\s*:\s*(\w+)'
            for line in lines:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    relation = match.group(1).strip()
                    # Validate it's a real relation
                    if relation in Config.METAQA_RELATIONS:
                        hops.append([relation])
                    break

        if len(hops) > 0:
            print(f"  [QwenRanker] Manual parse successful: {hops}")
            return hops, raw_answer[:100]
        else:
            print(f"  [QwenRanker] Manual parse failed - marking as FAILED")
            return None, "FAILED"

    def plan_relation_sequence_batch(self, questions: List[str], max_hops: int, batch_size: int = None, max_workers: int = None) -> List[Tuple[List[List[str]], str]]:
        """
        Batch version: Plan relations for multiple questions at once.
        Processes questions in parallel for speed.

        Args:
            questions: List of question texts
            max_hops: Number of hops needed
            batch_size: Max questions per batch (None = uses Config.QWEN_BATCH_SIZE)
            max_workers: ThreadPoolExecutor workers (None = uses Config.QWEN_MAX_WORKERS)

        Returns:
            List of (hops, reasoning) tuples, one per question
        """
        if not questions:
            return []

        # Use config defaults if not specified
        if batch_size is None:
            batch_size = Config.QWEN_BATCH_SIZE
        if max_workers is None:
            max_workers = Config.QWEN_MAX_WORKERS

        num_batches = (len(questions) + batch_size - 1) // batch_size
        print(f"  [QwenRanker] Processing {len(questions)} questions in {num_batches} batch(es) (batch_size={batch_size}, max_workers={max_workers})...")

        def call_api_with_retries(args):
            """Call API with retry logic for individual question - uses two-pass approach"""
            idx, question = args

            # Use the single-question plan_relation_sequence method (already has two-pass logic)
            try:
                hops, reasoning = self.plan_relation_sequence(question, max_hops)
                if hops is None:
                    return idx, None, "FAILED"
                return idx, {'hops': hops, 'reasoning': reasoning}, None
            except Exception as e:
                print(f"  [QwenRanker] Question {idx+1} failed: {e}")
                return idx, None, "FAILED"

        # Process in parallel with progress tracking
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(call_api_with_retries, (i, q)): i for i, q in enumerate(questions)}

            # Collect results with progress tracking
            responses = [None] * len(questions)
            failed_indices = []
            completed = 0
            last_print_pct = 0

            for future in as_completed(futures):
                idx, parsed, error = future.result()

                if error == "FAILED":
                    failed_indices.append(idx)
                    responses[idx] = None
                else:
                    responses[idx] = parsed

                completed += 1

                # Print progress every 10% or every batch_size questions
                current_pct = (completed * 100) // len(questions)
                if current_pct >= last_print_pct + 10 or completed % batch_size == 0 or completed == len(questions):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(questions) - completed) / rate if rate > 0 else 0
                    print(f"  [QwenRanker] Progress: {completed}/{len(questions)} ({current_pct}%) | Rate: {rate:.1f} q/s | ETA: {eta:.0f}s")
                    last_print_pct = current_pct

        if failed_indices:
            print(f"  [QwenRanker] Warning: {len(failed_indices)} question(s) failed and will be marked as FAILED")

        # Parse all responses
        results = []
        for i, parsed in enumerate(responses):
            if parsed is None:
                # This question failed
                results.append((None, "FAILED"))
                continue

            reasoning = parsed.get('reasoning', 'N/A')
            hops = parsed.get('hops', [])

            # Filter hops to only include lists
            filtered_hops = [hop for hop in hops if isinstance(hop, list)]

            results.append((filtered_hops, reasoning))

        print(f"  [QwenRanker] Batch complete - {len(questions)} questions")
        print(f"  [QwenRanker] Model used: {self.model}")
        print(f"  [QwenRanker] Total cost: $0.00 (self-hosted model)")

        return results
