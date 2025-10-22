"""Exact string matching entity linker"""

import re
from typing import List, Set
from ..core.base_entity_linker import BaseEntityLinker

class ExactMatcher(BaseEntityLinker):
    """
    Extract entities from questions using bracket notation and exact matching

    MetaQA questions have entities in brackets: "who directed [The Matrix]"
    """

    def __init__(self, node2id: dict):
        """
        Initialize exact matcher

        Args:
            node2id: Dictionary mapping entity names to IDs
        """
        self.node2id = node2id
        self.entities = set(node2id.keys())

    def extract_and_link(self, question: str) -> List[str]:
        """
        Extract entities from [brackets] and find exact matches in graph

        Args:
            question: Question text (e.g., "who directed [The Matrix]")

        Returns:
            List of matched entity names from the graph
        """
        # Extract text within brackets
        bracket_matches = re.findall(r'\[([^\]]+)\]', question)

        linked_entities = []

        for entity_str in bracket_matches:
            # Case-insensitive matching: find ALL matching entities
            # (Graph may have duplicates with different cases - include them all for multi-source BFS)

            entity_lower = entity_str.lower()

            # Find ALL entities that match (case-insensitive)
            for graph_entity in self.entities:
                if graph_entity.lower() == entity_lower:
                    linked_entities.append(graph_entity)

        return linked_entities

    def get_cost(self) -> float:
        """Exact matching is free"""
        return 0.0
