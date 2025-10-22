"""Base class for relation ranking"""

from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseRelationRanker(ABC):
    """Abstract base class for relation ranking"""

    @abstractmethod
    def rank_relations(self, question: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Rank relations by relevance to the question

        Args:
            question: Natural language question
            top_k: Return only top-k relations (None = all)

        Returns:
            List of (relation_name, score) tuples sorted by relevance (descending)
        """
        pass

    def get_cost(self) -> float:
        """
        Return total accumulated cost

        Returns:
            Cost in USD
        """
        return 0.0  # Override in subclasses that use APIs

    def get_last_query_cost(self) -> float:
        """
        Return cost of last query only (for incremental tracking)

        Returns:
            Cost in USD for last query
        """
        return 0.0  # Override in subclasses that use APIs
