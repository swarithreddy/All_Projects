from __future__ import annotations

import json
import logging
from pathlib import Path

from quiz_wizard.models.question import Question, QuestionBank
from quiz_wizard.paths import questions_dir

logger = logging.getLogger(__name__)


class QuestionRepositoryError(Exception):
    """Raised when a question bank cannot be loaded."""


class QuestionRepository:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or questions_dir()

    def path_for(self, category: str, difficulty: str) -> Path:
        return self.base_dir / category / f"{difficulty}.json"

    def load(self, category: str, difficulty: str) -> QuestionBank:
        path = self.path_for(category, difficulty)
        if not path.exists():
            raise QuestionRepositoryError(f"Question bank not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise QuestionRepositoryError(f"Corrupt question bank: {path}") from exc
        try:
            questions = [
                Question(
                    id=int(item["id"]),
                    prompt=str(item["prompt"]),
                    options=tuple(item["options"]),  # type: ignore[arg-type]
                    answer_index=int(item["answer_index"]),
                    explanation=str(item["explanation"]),
                )
                for item in raw["questions"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise QuestionRepositoryError(f"Invalid question bank schema: {path}") from exc
        for q in questions:
            if len(q.options) != 4:
                raise QuestionRepositoryError(f"Question {q.id} must have 4 options")
            if q.answer_index not in (1, 2, 3, 4):
                raise QuestionRepositoryError(f"Question {q.id} has invalid answer_index")
        return QuestionBank(
            category=str(raw.get("category", category)),
            difficulty=str(raw.get("difficulty", difficulty)),
            questions=questions,
        )
