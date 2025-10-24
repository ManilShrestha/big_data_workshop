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
        self.base_url = base_url or "http://96.245.177.243:12302/v1"
        self.model = model or "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-8bit"
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

        # Strategy 2: Find JSON object between curly braces
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
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

    def _build_relation_planning_prompt(self, question: str, max_hops: int) -> str:
        """Build the prompt for relation planning (same as OpenAI)"""
        relations_list = "\n".join([f"- {rel}" for rel in Config.METAQA_RELATIONS])

        return f"""You are analyzing a knowledge graph question to determine which relations to traverse.

Available relations in the knowledge graph:
{relations_list}

Question: "{question}"

This is a {max_hops}-hop question. You need to select which relation(s) might be relevant at EACH hop.

For each hop, select 1-2 most relevant relations from the list above. Think about the logical path needed to answer the question.

Examples:
- "Who directed movies starring [Actor]?" → Hop 1: starred_actors, Hop 2: directed_by
- "What genre are films written by [Writer]?" → Hop 1: written_by, Hop 2: has_genre
- "What year were movies by the director of [Movie]?" → Hop 1: directed_by, Hop 2: release_year
- "What films can be described by [occupation]?" → Hop 1: has_tags (films point TO tags/themes)
- "What films can be described by [Person Name]?" → Hop 1: has_tags (films associated with person as tag)

Respond in JSON format:
{{
  "reasoning": "brief explanation of the path",
  "hops": [
    ["relation1"],
    ["relation2"],
    ...
  ]
}}

Provide your analysis:"""

    def plan_relation_sequence(self, question: str, max_hops: int) -> Tuple[List[List[str]], str]:
        """
        Use Qwen to plan which relations to use for multi-hop questions.

        Args:
            question: Question text
            max_hops: Number of hops needed

        Returns:
            Tuple of (List of relation lists for each hop, reasoning string)
            Returns (None, "FAILED") if all retries fail
        """
        prompt = self._build_relation_planning_prompt(question, max_hops)

        max_retries = Config.OPENAI_MAX_RETRIES

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/completions",
                    headers={'Content-Type': 'application/json'},
                    json={
                        'model': self.model,
                        'prompt': prompt,
                        'max_tokens': 300,
                        'temperature': 0,
                        'stop': ['\n\nQuestion:', 'Q:', 'Examples:']
                    },
                    timeout=30
                )

                if response.status_code != 200:
                    raise ValueError(f"API returned status {response.status_code}")

                result = response.json()
                completion_text = result['choices'][0]['text']

                # Use robust JSON extraction
                try:
                    parsed = self._extract_json_robust(completion_text)

                    reasoning = parsed.get('reasoning', 'N/A')
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
                    # Parsing failed - retry
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  [QwenRanker] Parse error (attempt {attempt+1}/{max_retries}): {e}")
                        print(f"  [QwenRanker] Response preview: {completion_text[:200]}...")
                        print(f"  [QwenRanker] Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  [QwenRanker] All {max_retries} retries failed - marking as FAILED")
                        return None, "FAILED"

            except Exception as e:
                # API call failed - retry
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  [QwenRanker] Error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  [QwenRanker] All {max_retries} retries failed - marking as FAILED")
                    return None, "FAILED"

        # Should never reach here
        return None, "FAILED"

    def plan_relation_sequence_batch(self, questions: List[str], max_hops: int, batch_size: int = None, max_workers: int = None) -> List[Tuple[List[List[str]], str]]:
        """
        Batch version: Plan relations for multiple questions at once.
        Processes questions in parallel for speed.

        Args:
            questions: List of question texts
            max_hops: Number of hops needed
            batch_size: Max questions per batch (None = uses Config.OPENAI_BATCH_SIZE)
            max_workers: ThreadPoolExecutor workers (None = uses Config.OPENAI_MAX_WORKERS)

        Returns:
            List of (hops, reasoning) tuples, one per question
        """
        if not questions:
            return []

        # Use config defaults if not specified
        if batch_size is None:
            batch_size = Config.OPENAI_BATCH_SIZE
        if max_workers is None:
            max_workers = Config.OPENAI_MAX_WORKERS

        num_batches = (len(questions) + batch_size - 1) // batch_size
        print(f"  [QwenRanker] Processing {len(questions)} questions in {num_batches} batch(es) (batch_size={batch_size}, max_workers={max_workers})...")

        max_retries = Config.OPENAI_MAX_RETRIES

        def call_api_with_retries(args):
            """Call API with retry logic for individual question"""
            idx, question = args

            for attempt in range(max_retries):
                try:
                    prompt = self._build_relation_planning_prompt(question, max_hops)

                    response = requests.post(
                        f"{self.base_url}/completions",
                        headers={'Content-Type': 'application/json'},
                        json={
                            'model': self.model,
                            'prompt': prompt,
                            'max_tokens': 300,
                            'temperature': 0,
                            'stop': ['\n\nQuestion:', 'Q:', 'Examples:']
                        },
                        timeout=30
                    )

                    if response.status_code != 200:
                        raise ValueError(f"API returned status {response.status_code}")

                    result = response.json()
                    completion_text = result['choices'][0]['text']

                    # Try to parse JSON
                    try:
                        parsed = self._extract_json_robust(completion_text)

                        # Validate structure
                        hops = parsed.get('hops', [])
                        if not isinstance(hops, list):
                            raise ValueError("'hops' field is not a list")

                        filtered_hops = [hop for hop in hops if isinstance(hop, list)]
                        if len(filtered_hops) == 0:
                            raise ValueError("No valid hop lists found")

                        # Success!
                        return idx, parsed, None

                    except (ValueError, KeyError) as e:
                        # Parsing failed - retry
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"  [QwenRanker] Question {idx+1} failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"  [QwenRanker] Question {idx+1} failed after {max_retries} retries - marking as FAILED")
                            return idx, None, "FAILED"

                except Exception as e:
                    # API call failed - retry
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  [QwenRanker] Question {idx+1} API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  [QwenRanker] Question {idx+1} failed after {max_retries} retries - marking as FAILED")
                        return idx, None, "FAILED"

            # Should never reach here
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
