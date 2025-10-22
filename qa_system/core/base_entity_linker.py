"""Base class for entity linking"""

from abc import ABC, abstractmethod
from typing import List

class BaseEntityLinker(ABC):
    """Abstract base class for entity linking"""

    @abstractmethod
    def extract_and_link(self, question: str) -> List[str]:
        """
        Extract entity strings from question and link to graph nodes

        Args:
            question: Natural language question

        Returns:
            List of graph node IDs (entity names)
        """
        pass

    def get_cost(self) -> float:
        """
        Return cost of entity linking operation

        Returns:
            Cost in USD
        """
        return 0.0  # Override in subclasses that use APIs
