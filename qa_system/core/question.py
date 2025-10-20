"""Question data structure"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Question:
    """Represents a QA question"""
    text: str
    ground_truth_answers: List[str]
    hop_count: int
    question_id: Optional[int] = None

    @classmethod
    def from_line(cls, line: str, question_id: int, hop_count: int) -> 'Question':
        """
        Parse a line from MetaQA dataset

        Format: question[TAB]answer1|answer2|...
        Example: "who directed [The Matrix]\tWachowski Sisters"
        """
        parts = line.strip().split('\t')
        if len(parts) != 2:
            raise ValueError(f"Invalid line format: {line}")

        question_text = parts[0]
        answers = parts[1].split('|')

        return cls(
            text=question_text,
            ground_truth_answers=answers,
            hop_count=hop_count,
            question_id=question_id
        )

    def __repr__(self):
        return f"Question(id={self.question_id}, hops={self.hop_count}, text='{self.text[:50]}...')"
