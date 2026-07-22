from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Question:
    id: int
    prompt: str
    options: tuple[str, str, str, str]
    answer_index: int  # 1–4
    explanation: str

    def is_correct(self, choice: int) -> bool:
        return choice == self.answer_index


@dataclass
class QuestionBank:
    category: str
    difficulty: str
    questions: list[Question] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.questions)

    def get(self, index: int) -> Question:
        return self.questions[index]
