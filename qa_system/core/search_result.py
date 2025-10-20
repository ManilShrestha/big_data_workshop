"""Search result data structure"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json

@dataclass
class SearchResult:
    """Result from a search algorithm"""
    question_id: int
    question_text: str
    predicted_answers: List[str]
    ground_truth_answers: List[str]

    # Search metrics
    nodes_expanded: int
    search_time_ms: float
    success: bool

    # Path information (optional)
    reasoning_path: Optional[List[str]] = None

    # Relation information
    relations_used: List[str] = field(default_factory=list)

    # Cost tracking
    api_calls: int = 0
    cost_usd: float = 0.0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_correct(self) -> bool:
        """Check if any predicted answer matches ground truth"""
        if not self.predicted_answers:
            return False

        pred_set = set(self.predicted_answers)
        gt_set = set(self.ground_truth_answers)

        return len(pred_set & gt_set) > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'question_id': self.question_id,
            'question_text': self.question_text,
            'predicted_answers': self.predicted_answers,
            'ground_truth_answers': self.ground_truth_answers,
            'is_correct': self.is_correct,
            'nodes_expanded': self.nodes_expanded,
            'search_time_ms': self.search_time_ms,
            'success': self.success,
            'reasoning_path': self.reasoning_path,
            'relations_used': self.relations_used,
            'api_calls': self.api_calls,
            'cost_usd': self.cost_usd,
            'metadata': self.metadata
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
