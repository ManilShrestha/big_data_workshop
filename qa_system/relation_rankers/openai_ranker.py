"""OpenAI-based relation ranking"""

from typing import List, Tuple
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from ..core.base_relation_ranker import BaseRelationRanker
from ..config import Config

# OpenAI API Pricing (per 1M tokens)
# Source: https://openai.com/api/pricing/ (as of 2025)
OPENAI_PRICING = {
    'text-embedding-3-small': {'input': 0.02, 'output': 0.0},
    'text-embedding-3-large': {'input': 0.13, 'output': 0.0},
    'gpt-4o-2024-08-06': {'input': 2.50, 'output': 10.00},
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
    'gpt-4o-mini-2024-07-18': {'input': 0.15, 'output': 0.60},  # Specific version
    'gpt-5-mini': {'input': 0.15, 'output': 0.60},
    'gpt-5-mini-2025-08-07': {'input': 0.15, 'output': 0.60},  # GPT-5 Mini (versioned)
    'chatgpt-4o-latest': {'input': 2.50, 'output': 10.00},
    'gpt-4.1-mini-2025-04-14': {'input': 0.40, 'output': 1.60},
}

def calculate_api_cost(response) -> float:
    """
    Calculate cost from OpenAI API response.

    Args:
        response: OpenAI API response object with .model and .usage attributes

    Returns:
        Cost in USD
    """
    model = response.model
    usage = response.usage

    # Get pricing for this model
    if model not in OPENAI_PRICING:
        # Calculate cost based on token usage even if model not in table
        # Use gpt-4o-mini pricing as fallback (most similar to unknown models)
        print(f"  [Warning] Unknown model '{model}', using gpt-4o-mini pricing as estimate")
        print(f"  [Info] Usage: {usage.prompt_tokens} input tokens, {getattr(usage, 'completion_tokens', 0)} output tokens")
        pricing = OPENAI_PRICING['gpt-4o-mini']
        input_cost = usage.prompt_tokens * pricing['input'] / 1_000_000
        output_cost = getattr(usage, 'completion_tokens', 0) * pricing['output'] / 1_000_000
        return input_cost + output_cost

    pricing = OPENAI_PRICING[model]

    # Calculate cost
    input_cost = usage.prompt_tokens * pricing['input'] / 1_000_000
    output_cost = usage.completion_tokens * pricing['output'] / 1_000_000 if hasattr(usage, 'completion_tokens') else 0.0

    return input_cost + output_cost

class OpenAIRelationRanker(BaseRelationRanker):
    """
    Use OpenAI embeddings to rank relations by relevance to question

    One-time cost: ~$0.0001 to embed 9 relations
    Per-query cost: ~$0.00001 to embed question
    """

    # Natural language descriptions of each relation
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

    def __init__(self, api_key: str = None):
        """
        Initialize OpenAI ranker

        Args:
            api_key: OpenAI API key (defaults to Config.OPENAI_API_KEY)
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.relation_embeddings = None
        self.total_cost = 0.0
        self.last_query_cost = 0.0  # Track cost of last query for incremental tracking

        # Embed relations once
        self._embed_relations()

    def _embed_relations(self):
        """One-time embedding of all relations"""
        print("  [OpenAI] Embedding 9 relations...")

        texts = [self.RELATION_TEXTS[rel] for rel in Config.METAQA_RELATIONS]

        response = self.client.embeddings.create(
            model=Config.OPENAI_MODEL_EMBED,
            input=texts
        )

        # Store embeddings
        self.relation_embeddings = {}
        for rel, emb_data in zip(Config.METAQA_RELATIONS, response.data):
            self.relation_embeddings[rel] = np.array(emb_data.embedding)

        # Track cost from API response
        cost = calculate_api_cost(response)
        self.total_cost += cost
        print(f"  [OpenAI] Relations embedded. Tokens: {response.usage.total_tokens}, Cost: ${cost:.6f}")

    def rank_relations(self, question: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Rank all relations by similarity to question

        Args:
            question: Question text
            top_k: Return only top-k (None = all)

        Returns:
            List of (relation, score) sorted by relevance
        """
        # Embed question
        response = self.client.embeddings.create(
            model=Config.OPENAI_MODEL_EMBED,
            input=[question]
        )

        q_emb = np.array(response.data[0].embedding).reshape(1, -1)

        # Track cost from API response
        cost = calculate_api_cost(response)
        self.total_cost += cost
        self.last_query_cost = cost

        # Compute similarities
        scores = []
        for rel in Config.METAQA_RELATIONS:
            rel_emb = self.relation_embeddings[rel].reshape(1, -1)
            sim = cosine_similarity(q_emb, rel_emb)[0][0]
            scores.append((rel, float(sim)))

        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k if specified
        if top_k is not None:
            scores = scores[:top_k]

        return scores

    def get_cost(self) -> float:
        """Return total accumulated cost"""
        return self.total_cost

    def get_last_query_cost(self) -> float:
        """Return cost of last query only (for incremental tracking)"""
        return self.last_query_cost

    def _build_relation_planning_prompt(self, question: str, max_hops: int) -> str:
        """Build the prompt for relation planning"""
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
        Use LLM to intelligently plan which relations to use for multi-hop questions.

        This is the UPPER BOUND approach - using LLM reasoning to select optimal relations.

        Args:
            question: Question text
            max_hops: Number of hops needed

        Returns:
            Tuple of (List of relation lists for each hop, reasoning string)
            Returns (None, "FAILED") if all retries fail
        """
        import json
        import time

        prompt = self._build_relation_planning_prompt(question, max_hops)
        print(prompt)

        max_retries = Config.OPENAI_MAX_RETRIES

        for attempt in range(max_retries):
            try:
                # Call OpenAI chat model for structured JSON output
                response = self.client.chat.completions.create(
                    model=Config.OPENAI_MODEL_CHAT,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )

                # Track cost from API response
                cost = calculate_api_cost(response)
                self.total_cost += cost
                self.last_query_cost = cost

                # Parse response
                try:
                    result = json.loads(response.choices[0].message.content)
                except json.JSONDecodeError as e:
                    # JSON parsing failed - retry with exponential backoff
                    wait_time = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                    print(f"  [OpenAI LLM] JSON parse error (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"  [OpenAI LLM] Response preview: {response.choices[0].message.content[:200]}...")

                    if attempt < max_retries - 1:
                        print(f"  [OpenAI LLM] Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # All retries exhausted - return failed result
                        print(f"  [OpenAI LLM] All {max_retries} retries failed - marking as FAILED")
                        return None, "FAILED"

                reasoning = result.get('reasoning', 'N/A')
                hops = result.get('hops', [])

                # Validate hops structure
                if not isinstance(hops, list):
                    print(f"  [OpenAI LLM] Invalid hops structure (attempt {attempt+1}/{max_retries}): 'hops' field is not a list")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  [OpenAI LLM] Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  [OpenAI LLM] All {max_retries} retries failed - marking as FAILED")
                        return None, "FAILED"

                # Filter hops to only include lists (ignore any string descriptions)
                filtered_hops = [hop for hop in hops if isinstance(hop, list)]

                if len(filtered_hops) == 0:
                    print(f"  [OpenAI LLM] No valid hop lists (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  [OpenAI LLM] Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  [OpenAI LLM] All {max_retries} retries failed - marking as FAILED")
                        return None, "FAILED"

                # Success!
                print(f"  [OpenAI LLM] Relation planning - Reasoning: {reasoning}")
                print(f"  [OpenAI LLM] Planned hops: {filtered_hops}")
                print(f"  [OpenAI LLM] Cost: ${cost:.6f}")

                return filtered_hops, reasoning  # Return both for storage in metadata

            except Exception as e:
                # Unexpected error - retry
                print(f"  [OpenAI LLM] Unexpected error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  [OpenAI LLM] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  [OpenAI LLM] All {max_retries} retries failed - marking as FAILED")
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
        print(f"  [OpenAI LLM] Processing {len(questions)} questions in {num_batches} batch(es) (batch_size={batch_size}, max_workers={max_workers})...")

        # Build prompts for all questions
        messages_batch = []
        for question in questions:
            prompt = self._build_relation_planning_prompt(question, max_hops)
            messages_batch.append([{"role": "user", "content": prompt}])

        # Make parallel API calls with progress tracking
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        import json

        max_retries = Config.OPENAI_MAX_RETRIES

        def call_api_with_retries(args):
            """Call API with retry logic for individual question"""
            idx, messages = args

            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=Config.OPENAI_MODEL_CHAT,
                        messages=messages,
                        response_format={"type": "json_object"}
                    )

                    # Try to parse JSON immediately
                    try:
                        result = json.loads(response.choices[0].message.content)

                        # Validate structure
                        hops = result.get('hops', [])
                        if not isinstance(hops, list):
                            raise ValueError("'hops' field is not a list")

                        filtered_hops = [hop for hop in hops if isinstance(hop, list)]
                        if len(filtered_hops) == 0:
                            raise ValueError("No valid hop lists found")

                        # Success! Return valid response
                        return idx, response, None

                    except (json.JSONDecodeError, ValueError) as e:
                        # JSON parsing or validation failed - retry
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"  [OpenAI LLM] Question {idx+1} failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        else:
                            # All retries exhausted - return error marker
                            print(f"  [OpenAI LLM] Question {idx+1} failed after {max_retries} retries - marking as FAILED")
                            return idx, None, "FAILED"

                except Exception as e:
                    # API call failed - retry
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  [OpenAI LLM] Question {idx+1} API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  [OpenAI LLM] Question {idx+1} failed after {max_retries} retries - marking as FAILED")
                        return idx, None, "FAILED"

            # Should never reach here, but just in case
            return idx, None, "FAILED"

        # Process in parallel using thread pool with progress
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(call_api_with_retries, (i, msg)): i for i, msg in enumerate(messages_batch)}

            # Collect results with progress tracking
            responses = [None] * len(messages_batch)
            failed_indices = []
            completed = 0
            last_print_pct = 0

            for future in as_completed(futures):
                idx, response, error = future.result()

                if error == "FAILED":
                    failed_indices.append(idx)
                    responses[idx] = None
                else:
                    responses[idx] = response

                completed += 1

                # Print progress every 10% or every batch_size questions
                current_pct = (completed * 100) // len(messages_batch)
                if current_pct >= last_print_pct + 10 or completed % batch_size == 0 or completed == len(messages_batch):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(messages_batch) - completed) / rate if rate > 0 else 0
                    print(f"  [OpenAI LLM] Progress: {completed}/{len(messages_batch)} ({current_pct}%) | Rate: {rate:.1f} q/s | ETA: {eta:.0f}s")
                    last_print_pct = current_pct

        if failed_indices:
            print(f"  [OpenAI LLM] Warning: {len(failed_indices)} question(s) failed and will be marked as FAILED")

        # Parse all responses
        results = []
        total_input_tokens = 0
        total_output_tokens = 0

        for i, response in enumerate(responses):
            if response is None:
                # This question failed - return failed marker
                results.append((None, "FAILED"))
                continue

            # Parse JSON response (already validated in call_api_with_retries)
            result = json.loads(response.choices[0].message.content)
            reasoning = result.get('reasoning', 'N/A')
            hops = result.get('hops', [])

            # Filter hops to only include lists (already validated)
            filtered_hops = [hop for hop in hops if isinstance(hop, list)]

            results.append((filtered_hops, reasoning))

            # Track tokens
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens

        # Calculate total cost for batch (skip None responses)
        total_cost = 0.0
        for response in responses:
            if response is not None:
                total_cost += calculate_api_cost(response)

        self.total_cost += total_cost
        self.last_query_cost = total_cost

        print(f"  [OpenAI LLM] Batch complete - {len(questions)} questions")
        print(f"  [OpenAI LLM] Total tokens: {total_input_tokens + total_output_tokens} (input: {total_input_tokens}, output: {total_output_tokens})")
        print(f"  [OpenAI LLM] Cost: ${total_cost:.6f} (${total_cost/len(questions):.6f} per question)")

        return results

    def plan_relation_sequence_async_batch(
        self,
        questions: List[str],
        max_hops: int,
        poll_interval: int = 60,
        verbose: bool = True
    ) -> Tuple[str, List[Tuple[List[List[str]], str]]]:
        """
        Create async batch for relation planning using OpenAI Batch API (50% cheaper).

        Args:
            questions: List of question texts
            max_hops: Number of hops needed
            poll_interval: Seconds between status checks
            verbose: Print progress

        Returns:
            Tuple of (batch_id, results_list)
            - batch_id: ID for tracking/resuming
            - results_list: List of (hops, reasoning) tuples, one per question
        """
        import tempfile

        if not questions:
            return ("", [])

        if verbose:
            print(f"\n  [Async Batch] Creating batch for {len(questions)} questions...")

        # Build batch requests in JSONL format
        batch_requests = []
        for i, question in enumerate(questions):
            prompt = self._build_relation_planning_prompt(question, max_hops)
            request = {
                "custom_id": f"plan-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": Config.OPENAI_MODEL_CHAT,
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
            print(f"  [Async Batch] Created batch file: {temp_file}")

        # Upload file to OpenAI
        with open(temp_file, 'rb') as f:
            file_response = self.client.files.create(
                file=f,
                purpose='batch'
            )

        if verbose:
            print(f"  [Async Batch] Uploaded file: {file_response.id}")

        # Create batch
        batch_response = self.client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"Relation planning for {len(questions)} questions",
                "num_questions": str(len(questions)),
                "max_hops": str(max_hops)
            }
        )

        batch_id = batch_response.id

        if verbose:
            print(f"  [Async Batch] Batch created: {batch_id}")
            print(f"  [Async Batch] Status: {batch_response.status}")
            print(f"  [Async Batch] Waiting for completion (polling every {poll_interval}s)...")

        # Clean up temp file
        from pathlib import Path
        Path(temp_file).unlink()

        # Wait for batch to complete
        import time
        while True:
            batch = self.client.batches.retrieve(batch_id)

            if verbose:
                print(f"  [Async Batch] Status: {batch.status} | Completed: {batch.request_counts.completed}/{batch.request_counts.total}")

            if batch.status in ['completed', 'failed', 'expired', 'cancelled']:
                break

            time.sleep(poll_interval)

        if batch.status != 'completed':
            raise ValueError(f"Batch {batch_id} failed with status: {batch.status}")

        # Retrieve results
        if verbose:
            print(f"  [Async Batch] Retrieving results...")

        output_content = self.client.files.content(batch.output_file_id)
        output_lines = output_content.text.strip().split('\n')

        # Parse results and handle JSON parse failures
        results_dict = {}
        failed_indices = []
        total_cost = 0.0

        for line in output_lines:
            result = json.loads(line)
            custom_id = result['custom_id']
            idx = int(custom_id.split('-')[1])

            response_body = result['response']['body']

            # Calculate cost (50% discount for batch API)
            if 'usage' in response_body:
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
                    response_body.get('model', Config.OPENAI_MODEL_CHAT),
                    MockUsage(response_body['usage'])
                )
                cost = calculate_api_cost(mock_response) * 0.5  # 50% discount
                total_cost += cost

            # Try to parse JSON response
            try:
                content = response_body['choices'][0]['message']['content']
                parsed = json.loads(content)

                reasoning = parsed.get('reasoning', 'N/A')
                hops = parsed.get('hops', [])

                # Validate structure
                if not isinstance(hops, list):
                    raise ValueError("'hops' field is not a list")

                for hop_idx, hop in enumerate(hops):
                    if not isinstance(hop, list):
                        raise ValueError(f"Hop {hop_idx} is not a list")

                results_dict[idx] = (hops, reasoning)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if verbose:
                    print(f"  [Async Batch] Warning: Failed to parse result {idx}: {e}")
                failed_indices.append(idx)
                results_dict[idx] = None

        # Retry failed parses
        if failed_indices and verbose:
            print(f"  [Async Batch] Retrying {len(failed_indices)} failed question(s)...")

        retry_questions = [questions[i] for i in failed_indices]
        if retry_questions:
            # Use synchronous batch with retries for failed questions
            retry_results = self.plan_relation_sequence_batch(
                retry_questions,
                max_hops=max_hops
            )

            for orig_idx, retry_result in zip(failed_indices, retry_results):
                results_dict[orig_idx] = retry_result

        # Build final results list in order
        results = [results_dict.get(i, (None, "FAILED")) for i in range(len(questions))]

        self.total_cost += total_cost
        self.last_query_cost = total_cost

        if verbose:
            print(f"  [Async Batch] Complete!")
            print(f"  [Async Batch] Cost: ${total_cost:.6f} (${total_cost/len(questions):.6f} per question)")
            print(f"  [Async Batch] Savings: 50% vs regular API")

        return (batch_id, results)
