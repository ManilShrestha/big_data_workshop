"""Base class for search algorithms"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import networkx as nx
from .search_result import SearchResult
from .question import Question

class BaseSearch(ABC):
    """Abstract base class for search algorithms"""

    def __init__(self, graph: nx.DiGraph):
        """
        Initialize search algorithm

        Args:
            graph: NetworkX directed graph
        """
        self.graph = graph
        self.nodes_expanded = 0  # Track for metrics

    @abstractmethod
    def search(
        self,
        question: Question,
        start_nodes: List[str],
        target_relations: List[Tuple[str, float]],
        max_hops: int
    ) -> SearchResult:
        """
        Perform search from start nodes

        Args:
            question: Question object
            start_nodes: Starting entity nodes
            target_relations: List of (relation, score) tuples
            max_hops: Maximum search depth

        Returns:
            SearchResult object with answers and metrics
        """
        pass

    def reset_metrics(self):
        """Reset search metrics"""
        self.nodes_expanded = 0
