"""Evaluation utilities"""

import json
from typing import List, Dict
from pathlib import Path
from ..core.question import Question
from ..core.search_result import SearchResult
from ..core.base_entity_linker import BaseEntityLinker
from ..core.base_relation_ranker import BaseRelationRanker
from ..core.base_search import BaseSearch

class Evaluator:
    """Evaluate QA system on a dataset"""

    def __init__(
        self,
        entity_linker: BaseEntityLinker,
        relation_ranker: BaseRelationRanker,
        search_algo: BaseSearch,
        variant_name: str,
        use_llm_planning: bool = False,
        batch_planning: bool = True,
        poll_interval: int = 60,
        incremental_save_path: str = None
    ):
        """
        Initialize evaluator

        Args:
            entity_linker: Entity linking component
            relation_ranker: Relation ranking component
            search_algo: Search algorithm
            variant_name: Name of this variant (e.g., "variant5_openai_transe")
            use_llm_planning: Use GPT-4o for intelligent relation planning (upper bound)
            batch_planning: If True and use_llm_planning=True, batch all planning calls (cheaper)
            poll_interval: Seconds between batch status checks (for async batch mode)
            incremental_save_path: If provided, save results incrementally to this JSON file
        """
        self.entity_linker = entity_linker
        self.relation_ranker = relation_ranker
        self.search_algo = search_algo
        self.variant_name = variant_name
        self.use_llm_planning = use_llm_planning
        self.batch_planning = batch_planning
        self.poll_interval = poll_interval
        self.incremental_save_path = incremental_save_path

    def evaluate(
        self,
        questions: List[Question],
        top_k_relations: int = 3,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate on a list of questions

        Args:
            questions: List of Question objects
            top_k_relations: Use top-k relations from ranker
            verbose: Print progress

        Returns:
            Dictionary with aggregate metrics
        """
        import time
        eval_start_time = time.time()

        results = []
        total_cost = 0.0
        batch_llm_cost = 0.0  # Track total cost of all batch LLM calls

        # ASYNC BATCHED LLM PLANNING: If using LLM planning with batch mode, use async Batch API
        llm_plans = {}  # Maps question index -> (hop_relations, reasoning)
        if self.use_llm_planning and self.batch_planning and hasattr(self.relation_ranker, 'plan_relation_sequence_async_batch'):
            if verbose:
                print("  [Async Batch Planning] Planning relations for all questions using OpenAI Batch API (50% cheaper)...")

            # Get unique hop counts
            hop_counts = list(set(q.hop_count for q in questions))

            # Batch by hop count (since max_hops is a parameter)
            for hop_count in hop_counts:
                questions_for_hop = [(i, q) for i, q in enumerate(questions) if q.hop_count == hop_count]
                question_texts = [q.text for _, q in questions_for_hop]

                if verbose:
                    print(f"  [Async Batch Planning] Submitting {len(question_texts)} {hop_count}-hop questions...")

                # Track cost before batch call
                cost_before = self.relation_ranker.get_cost()

                # Use async batch API (50% cheaper)
                batch_id, batch_results = self.relation_ranker.plan_relation_sequence_async_batch(
                    question_texts,
                    max_hops=hop_count,
                    poll_interval=60,
                    verbose=verbose
                )

                # Track cost after batch call
                cost_after = self.relation_ranker.get_cost()
                batch_llm_cost += (cost_after - cost_before)

                if verbose:
                    print(f"  [Async Batch Planning] Batch {batch_id} completed for {hop_count}-hop questions")

                # Store results mapped by original question index
                for (orig_idx, _), (hop_relations, reasoning) in zip(questions_for_hop, batch_results):
                    llm_plans[orig_idx] = (hop_relations, reasoning)

            if verbose:
                print(f"  [Async Batch Planning] Complete! Planned {len(llm_plans)} questions")
                print(f"  [Async Batch Planning] Total LLM cost: ${batch_llm_cost:.6f} (50% discount applied)\n")

        # Now process each question
        for i, question in enumerate(questions):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(questions)} questions")

            # Step 1: Entity linking
            start_nodes = self.entity_linker.extract_and_link(question.text)

            # Step 2: Relation ranking
            gpt_reasoning = None  # Track GPT reasoning for metadata

            if self.use_llm_planning:
                if self.batch_planning:
                    # BATCH MODE: Use pre-computed batch plan
                    if i not in llm_plans:
                        raise RuntimeError(f"Question {i+1} missing from batch LLM plans. This should not happen.")

                    # Use pre-computed batch plan
                    hop_relations, reasoning = llm_plans[i]
                    gpt_reasoning = reasoning
                else:
                    # DIRECT MODE: Plan this question individually
                    if verbose and (i + 1) % 10 == 0:
                        print(f"  [Direct Planning] Planning question {i+1}/{len(questions)}...")

                    hop_relations, reasoning = self.relation_ranker.plan_relation_sequence(
                        question.text,
                        max_hops=question.hop_count
                    )
                    gpt_reasoning = reasoning

                # Check if LLM planning failed after all retries
                if hop_relations is None and reasoning == "FAILED":
                    if verbose:
                        print(f"  [Question {i+1}] LLM planning failed after all retries - marking as incorrect")

                    # Create a failed search result
                    result = SearchResult(
                        question=question,
                        found_answers=[],
                        expected_answers=question.answers,
                        success=False,
                        nodes_expanded=0,
                        search_time_ms=0.0,
                        metadata={'error': 'LLM planning failed after retries', 'llm_reasoning': 'FAILED'}
                    )

                    # Track costs (no relation cost since LLM failed)
                    entity_cost = self.entity_linker.get_last_query_cost() if hasattr(self.entity_linker, 'get_last_query_cost') else 0.0
                    result.cost_usd = entity_cost
                    total_cost += result.cost_usd

                    results.append(result)

                    # Incremental save
                    if self.incremental_save_path:
                        self._save_incremental(results, total_cost, eval_start_time)

                    continue  # Skip to next question

                # Validate hop_relations format
                if not isinstance(hop_relations, list):
                    raise ValueError(f"Question {i+1}: Invalid hop relations format: {hop_relations}. Expected list of lists.")

                for hop_idx, hop_rels in enumerate(hop_relations):
                    if not isinstance(hop_rels, list):
                        raise ValueError(f"Question {i+1}, Hop {hop_idx}: Invalid format: {hop_rels}. Expected list of relation names.")

                if not hop_relations or not any(hop_relations):
                    raise ValueError(f"Question {i+1}: Empty LLM plan returned. Check LLM output.")

                # Pass the sequence directly (NOT flattened!)
                target_relations = hop_relations

            else:
                # Use embedding similarity ranking
                target_relations = self.relation_ranker.rank_relations(
                    question.text,
                    top_k=top_k_relations
                )

            # Step 3: Search
            result = self.search_algo.search(
                question=question,
                start_nodes=start_nodes,
                target_relations=target_relations,
                max_hops=question.hop_count
            )

            # Track costs
            entity_cost = self.entity_linker.get_last_query_cost() if hasattr(self.entity_linker, 'get_last_query_cost') else 0.0

            # For batched planning, cost will be distributed equally at the end
            # For direct planning or embedding-based ranking, get last query cost immediately
            if self.batch_planning and i in llm_plans:
                # Cost will be distributed from batch_llm_cost later
                relation_cost = 0.0
            else:
                # Direct LLM planning or embedding-based ranking
                relation_cost = self.relation_ranker.get_last_query_cost()

            result.cost_usd = entity_cost + relation_cost
            total_cost += result.cost_usd

            # Add GPT reasoning to metadata if available
            if gpt_reasoning:
                result.metadata['llm_reasoning'] = gpt_reasoning
                result.metadata['llm_planned_hops'] = hop_relations if self.use_llm_planning else None

            results.append(result)

            # Incremental save: write results after each question
            # NOTE: Cost doesn't include batch_llm_cost yet - will be added after loop
            if self.incremental_save_path:
                self._save_incremental(results, total_cost, eval_start_time)

        # Distribute batch LLM cost across all questions that used LLM planning
        if llm_plans and batch_llm_cost > 0:
            cost_per_query = batch_llm_cost / len(llm_plans)
            for i in llm_plans.keys():
                results[i].cost_usd += cost_per_query
            total_cost += batch_llm_cost

            # Re-save with correct costs included
            if self.incremental_save_path:
                self._save_incremental(results, total_cost, eval_start_time)

        # Compute total evaluation time
        total_eval_time = time.time() - eval_start_time

        # Compute aggregate metrics
        metrics = self._compute_metrics(results, total_cost, total_eval_time)

        if verbose:
            self._print_metrics(metrics)

        return {
            'variant_name': self.variant_name,
            'metrics': metrics,
            'results': results
        }

    def _compute_metrics(self, results: List[SearchResult], total_cost: float, total_eval_time: float) -> Dict:
        """Compute aggregate metrics for paper reporting"""
        total = len(results)
        if total == 0:
            return {}

        correct = sum(1 for r in results if r.is_correct)
        successful = sum(1 for r in results if r.success)

        nodes_expanded = [r.nodes_expanded for r in results]
        search_times = [r.search_time_ms for r in results]

        # Precision, Recall, F1 metrics (per-question)
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        f1_scores = [r.f1_score for r in results]

        # Answer quality metrics
        num_correct_answers = [r.num_correct for r in results]
        num_incorrect_answers = [r.num_incorrect for r in results]
        num_missed_answers = [r.num_missed for r in results]

        # Micro-averaging: aggregate all predictions first, then calculate P/R/F1
        total_tp = sum(num_correct_answers)  # True positives
        total_fp = sum(num_incorrect_answers)  # False positives
        total_fn = sum(num_missed_answers)  # False negatives

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # Additional metrics for paper
        import statistics

        return {
            # Core accuracy metrics
            'total_questions': total,
            'correct_answers': correct,
            'accuracy': correct / total,
            'successful_searches': successful,
            'success_rate': successful / total,

            # Macro-averaging: average P/R/F1 across all questions (treats all questions equally)
            'macro_avg_precision': sum(precisions) / total,
            'macro_avg_recall': sum(recalls) / total,
            'macro_avg_f1_score': sum(f1_scores) / total,
            'macro_median_precision': statistics.median(precisions) if precisions else 0,
            'macro_median_recall': statistics.median(recalls) if recalls else 0,
            'macro_median_f1_score': statistics.median(f1_scores) if f1_scores else 0,

            # Micro-averaging: aggregate all predictions then calculate (weights by answer count)
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1_score': micro_f1,

            # Answer quality metrics
            'avg_correct_per_question': sum(num_correct_answers) / total,
            'avg_incorrect_per_question': sum(num_incorrect_answers) / total,
            'avg_missed_per_question': sum(num_missed_answers) / total,
            'total_correct_answers': sum(num_correct_answers),
            'total_incorrect_answers': sum(num_incorrect_answers),
            'total_missed_answers': sum(num_missed_answers),

            # Efficiency metrics
            'avg_nodes_expanded': sum(nodes_expanded) / total,
            'min_nodes_expanded': min(nodes_expanded) if nodes_expanded else 0,
            'max_nodes_expanded': max(nodes_expanded) if nodes_expanded else 0,
            'median_nodes_expanded': statistics.median(nodes_expanded) if nodes_expanded else 0,

            # Timing metrics
            'avg_search_time_ms': sum(search_times) / total,
            'min_search_time_ms': min(search_times) if search_times else 0,
            'max_search_time_ms': max(search_times) if search_times else 0,
            'median_search_time_ms': statistics.median(search_times) if search_times else 0,
            'total_search_time_sec': sum(search_times) / 1000,  # Convert to seconds

            # Total evaluation time (includes all overhead: entity linking, relation ranking, search)
            'total_eval_time_sec': total_eval_time,
            'total_eval_time_min': total_eval_time / 60,
            'avg_time_per_query_sec': total_eval_time / total,

            # Cost metrics
            'total_cost_usd': total_cost,
            'cost_per_query_usd': total_cost / total,

            # Throughput
            'queries_per_second': total / total_eval_time if total_eval_time > 0 else 0,
            'queries_per_minute': (total / total_eval_time) * 60 if total_eval_time > 0 else 0,
        }

    def _print_metrics(self, metrics: Dict):
        """Print metrics in a nice format"""
        print("\n" + "="*80)
        print(f"  EVALUATION RESULTS: {self.variant_name}")
        print("="*80)

        print("\n  ACCURACY METRICS:")
        print(f"    Total questions:     {metrics['total_questions']}")
        print(f"    Correct answers:     {metrics['correct_answers']}")
        print(f"    Accuracy:            {metrics['accuracy']:.2%}")
        print(f"    Success rate:        {metrics['success_rate']:.2%}")

        print("\n  ANSWER QUALITY METRICS (Macro-Averaging):")
        print(f"    Avg Precision:       {metrics['macro_avg_precision']:.2%}  (median: {metrics['macro_median_precision']:.2%})")
        print(f"    Avg Recall:          {metrics['macro_avg_recall']:.2%}  (median: {metrics['macro_median_recall']:.2%})")
        print(f"    Avg F1 Score:        {metrics['macro_avg_f1_score']:.2%}  (median: {metrics['macro_median_f1_score']:.2%})")

        print("\n  ANSWER QUALITY METRICS (Micro-Averaging):")
        print(f"    Precision:           {metrics['micro_precision']:.2%}")
        print(f"    Recall:              {metrics['micro_recall']:.2%}")
        print(f"    F1 Score:            {metrics['micro_f1_score']:.2%}")

        print("\n  ANSWER COUNTS:")
        print(f"    Correct/question:    {metrics['avg_correct_per_question']:.2f}  (total: {metrics['total_correct_answers']})")
        print(f"    Incorrect/question:  {metrics['avg_incorrect_per_question']:.2f}  (total: {metrics['total_incorrect_answers']})")
        print(f"    Missed/question:     {metrics['avg_missed_per_question']:.2f}  (total: {metrics['total_missed_answers']})")

        print("\n  EFFICIENCY METRICS (Nodes Expanded):")
        print(f"    Average:             {metrics['avg_nodes_expanded']:.2f}")
        print(f"    Median:              {metrics['median_nodes_expanded']:.2f}")
        print(f"    Min / Max:           {metrics['min_nodes_expanded']} / {metrics['max_nodes_expanded']}")

        print("\n  TIMING METRICS:")
        print(f"    Total eval time:     {metrics['total_eval_time_min']:.2f} min ({metrics['total_eval_time_sec']:.1f} sec)")
        print(f"    Avg time/query:      {metrics['avg_time_per_query_sec']:.3f} sec")
        print(f"    Avg search time:     {metrics['avg_search_time_ms']:.2f} ms")
        print(f"    Median search time:  {metrics['median_search_time_ms']:.2f} ms")
        print(f"    Throughput:          {metrics['queries_per_minute']:.1f} queries/min")

        print("\n  COST METRICS:")
        print(f"    Total cost:          ${metrics['total_cost_usd']:.6f}")
        print(f"    Cost per query:      ${metrics['cost_per_query_usd']:.6f}")

        print("="*80 + "\n")

    def _save_incremental(self, results: List[SearchResult], total_cost: float, eval_start_time: float):
        """
        Save intermediate results incrementally during evaluation.
        This allows monitoring progress while the evaluation is running.

        Args:
            results: Current list of results
            total_cost: Cost accumulated so far
            eval_start_time: When evaluation started
        """
        import time

        # Compute current metrics
        current_eval_time = time.time() - eval_start_time
        metrics = self._compute_metrics(results, total_cost, current_eval_time)

        # Prepare output
        output = {
            'variant_name': self.variant_name,
            'metrics': metrics,
            'results': [r.to_dict() for r in results],
            'status': 'in_progress',  # Mark as incomplete
            'last_updated': time.time()
        }

        # Create directory if needed
        Path(self.incremental_save_path).parent.mkdir(parents=True, exist_ok=True)

        # Write to file (overwrite each time)
        with open(self.incremental_save_path, 'w') as f:
            json.dump(output, f, indent=2)

    def save_results(self, evaluation: Dict, output_path: str):
        """
        Save evaluation results to JSON

        Args:
            evaluation: Output from evaluate()
            output_path: Path to save JSON file
        """
        # Convert SearchResult objects to dicts
        output = {
            'variant_name': evaluation['variant_name'],
            'metrics': evaluation['metrics'],
            'results': [r.to_dict() for r in evaluation['results']],
            'status': 'completed'  # Mark as complete
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"  Results saved to: {output_path}")
