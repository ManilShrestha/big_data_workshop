"""LLM-guided BFS search algorithm"""

import time
from typing import List, Tuple, Set, Dict
import networkx as nx
from collections import deque
from ..core.base_search import BaseSearch
from ..core.search_result import SearchResult
from ..core.question import Question

class LLMGuidedBFS(BaseSearch):
    """
    LLM-guided BFS search algorithm

    Uses LLM (GPT) to plan which relations to use at each hop,
    then performs pure BFS following that relation sequence.

    For N-hop questions:
    - 1-hop: Simple traversal
    - 2-hop+: Follow LLM-planned relation sequence per hop
    """

    def __init__(self, graph: nx.DiGraph):
        """
        Initialize LLM-guided BFS search

        Args:
            graph: NetworkX directed graph
        """
        super().__init__(graph)

    def search(
        self,
        question: Question,
        start_nodes: List[str],
        target_relations,  # Can be List[Tuple[str, float]] OR List[List[str]] (sequence per hop)
        max_hops: int
    ) -> SearchResult:
        """
        Perform backward search

        Args:
            question: Question object
            start_nodes: Starting entities from question
            target_relations: Either:
                - List[Tuple[str, float]]: Ranked list of (relation, score) for flat search
                - List[List[str]]: Sequence of relations per hop [[hop1_rels], [hop2_rels], ...]
            max_hops: Maximum hops to search

        Returns:
            SearchResult with answers and metrics
        """
        start_time = time.time()
        self.reset_metrics()

        if not start_nodes:
            # No entities found
            return SearchResult(
                question_id=question.question_id,
                question_text=question.text,
                predicted_answers=[],
                ground_truth_answers=question.ground_truth_answers,
                nodes_expanded=0,
                search_time_ms=0.0,
                success=False,
                relations_used=[],
                metadata={'error': 'no_start_nodes'}
            )

        # Check if target_relations is a sequence per hop or flat list
        relations_to_try = []  # Initialize default
        if isinstance(target_relations, list) and len(target_relations) > 0:
            if isinstance(target_relations[0], list):
                # Sequence per hop: [[hop1_rels], [hop2_rels], ...]
                relation_sequence = target_relations
                relations_to_log = [rel for hop_rels in relation_sequence for rel in hop_rels]
            else:
                # Flat list of (relation, score) tuples
                relations_to_try = [rel for rel, score in target_relations]
                relation_sequence = None
                relations_to_log = relations_to_try
        else:
            relation_sequence = None
            relations_to_log = []

        # Perform backward search
        if max_hops == 1:
            if relation_sequence:
                relations_to_try = relation_sequence[0] if len(relation_sequence) > 0 else []
            answers, path = self._backward_1hop(start_nodes, relations_to_try)
        else:
            # Only support sequenced search for multi-hop
            if relation_sequence:
                answers, path = self._backward_multihop_sequenced(start_nodes, relation_sequence, max_hops)
            else:
                # Flat relation list not supported without LLM planning
                answers, path = [], []

        # Compute metrics
        search_time_ms = (time.time() - start_time) * 1000

        return SearchResult(
            question_id=question.question_id,
            question_text=question.text,
            predicted_answers=answers,
            ground_truth_answers=question.ground_truth_answers,
            nodes_expanded=self.nodes_expanded,
            search_time_ms=search_time_ms,
            success=len(answers) > 0,
            reasoning_path=path,
            relations_used=relations_to_log[:3]  # Log top 3
        )

    def _backward_1hop(
        self,
        start_nodes: List[str],
        target_relations: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        1-hop backward search (trivial lookup)

        Question: "Who directed [The Matrix]?"
        Start: ["The Matrix"]
        Target: "directed_by"
        Search: Both directions for matching relations

        Returns:
            (answers, path)
        """
        answers = []
        path = []

        for start_entity in start_nodes:
            if start_entity not in self.graph:
                continue

            # Check all incoming edges (predecessors)
            # MultiDiGraph: need to iterate over all edges between nodes
            for pred, _, key, data in self.graph.in_edges(start_entity, keys=True, data=True):
                self.nodes_expanded += 1

                # Get edge relation
                edge_relation = data.get('relation', '')

                # Check if this relation matches
                if edge_relation in target_relations:
                    answers.append(pred)
                    path.append(f"{pred} --{edge_relation}--> {start_entity}")

            # Check all outgoing edges (successors)
            # MultiDiGraph: need to iterate over all edges between nodes
            for _, succ, key, data in self.graph.out_edges(start_entity, keys=True, data=True):
                self.nodes_expanded += 1

                # Get edge relation
                edge_relation = data.get('relation', '')

                # Check if this relation matches
                if edge_relation in target_relations:
                    answers.append(succ)
                    path.append(f"{start_entity} --{edge_relation}--> {succ}")

        return list(set(answers)), path  # Deduplicate

    def _backward_multihop_sequenced(
        self,
        start_nodes: List[str],
        relation_sequence: List[List[str]],
        max_hops: int
    ) -> Tuple[List[str], List[str]]:
        """
        Multi-hop backward search following a specific relation sequence

        Pure BFS following LLM-planned relations in order per hop

        Question: "When did movies release whose actors appear in [Cast a Deadly Spell]?"
        Start: ["Cast a Deadly Spell"]
        Hops: 3
        Relation sequence: [["starred_actors"], ["starred_actors"], ["release_year"]]

        Path:
        1. Hop 0->1: Use "starred_actors" to find actors
        2. Hop 1->2: Use "starred_actors" to find other movies
        3. Hop 2->3: Use "release_year" to find years (answers!)

        Returns:
            (answers, path)
        """
        # BFS queue: (current_node, depth, path_so_far_with_edges)
        queue = deque()
        for start in start_nodes:
            if start in self.graph:
                queue.append((start, 0, [(start, None)]))  # (node, edge_used_to_reach_it)

        visited = set()
        answers = []
        answer_paths = []

        while queue:
            current, depth, path_with_edges = queue.popleft()

            # If we've reached max depth, this node is a potential answer
            # At the final depth, we don't need to expand, just collect answers
            if depth == max_hops:
                answers.append(current)
                # Build path string with edges
                path_str = path_with_edges[0][0]  # Start node
                for i in range(1, len(path_with_edges)):
                    node, edge = path_with_edges[i]
                    path_str += f" --{edge}--> {node}"
                answer_paths.append(path_str)
                continue  # Don't expand further

            # Use (node, depth) for visited to avoid redundant expansion
            # But we don't check this for final depth (above) to collect all answers
            visit_key = (current, depth)
            if visit_key in visited:
                continue
            visited.add(visit_key)
            self.nodes_expanded += 1

            # Get the relations to use for this hop
            if depth >= len(relation_sequence):
                # No more relations specified, stop
                continue

            current_hop_relations = relation_sequence[depth]

            # Get nodes already in current path to avoid cycles
            nodes_in_path = set(node for node, _ in path_with_edges)

            # Explore neighbors in both directions - NO BEAM SEARCH, PURE BFS
            neighbors = []

            # Check incoming edges (predecessors)
            for pred, _, key, data in self.graph.in_edges(current, keys=True, data=True):
                # Skip if already visited at this depth
                if (pred, depth + 1) in visited:
                    continue

                # Skip if this node is already in the current path (avoid cycles)
                if pred in nodes_in_path:
                    continue

                edge_relation = data.get('relation', '')

                # Only follow relations specified for this hop
                if edge_relation not in current_hop_relations:
                    continue

                neighbors.append((pred, edge_relation, 'incoming'))

            # Check outgoing edges (successors)
            for _, succ, key, data in self.graph.out_edges(current, keys=True, data=True):
                # Skip if already visited at this depth
                if (succ, depth + 1) in visited:
                    continue

                # Skip if this node is already in the current path (avoid cycles)
                if succ in nodes_in_path:
                    continue

                edge_relation = data.get('relation', '')

                # Only follow relations specified for this hop
                if edge_relation not in current_hop_relations:
                    continue

                neighbors.append((succ, edge_relation, 'outgoing'))

            # Add ALL neighbors to queue (pure BFS, no beam)
            for neighbor, edge_rel, direction in neighbors:
                # Build new path with edges
                new_path_with_edges = path_with_edges + [(neighbor, edge_rel)]
                queue.append((neighbor, depth + 1, new_path_with_edges))

        return list(set(answers)), answer_paths
