from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from quiz_wizard.config import DIFFICULTY_AUTO, DIFFICULTY_EASY
from quiz_wizard.models.question import Question, QuestionBank
from quiz_wizard.repositories.questions import QuestionRepository, QuestionRepositoryError
from quiz_wizard.services.auto_difficulty import adjust_difficulty
from quiz_wizard.services.scoring import points_for_answer


@dataclass
class AnswerResult:
    correct: bool
    points_awarded: int
    explanation: str
    score: int
    finished: bool
    difficulty_changed: bool = False
    current_difficulty: str = ""
    recent_results: list[int] = field(default_factory=list)


@dataclass
class QuizEngine:
    """Fixed or Auto quiz session driven by a QuestionRepository."""

    category: str
    mode: str  # easy|medium|hard|auto
    question_repo: QuestionRepository
    bank: QuestionBank = field(init=False)
    score: int = 0
    question_index: int = 0
    correct_streak: int = 0
    wrong_streak: int = 0
    results: list[int] = field(default_factory=list)
    current_difficulty: str = field(init=False)
    finished: bool = False
    _awaiting_continue: bool = False

    def __post_init__(self) -> None:
        if self.mode == DIFFICULTY_AUTO:
            self.current_difficulty = DIFFICULTY_EASY
        else:
            self.current_difficulty = self.mode
        self.bank = self.question_repo.load(self.category, self.current_difficulty)

    @property
    def is_auto(self) -> bool:
        return self.mode == DIFFICULTY_AUTO

    @property
    def total_questions(self) -> int:
        return len(self.bank)

    @property
    def has_current(self) -> bool:
        return not self.finished and self.question_index < len(self.bank)

    def current_question(self) -> Question:
        if not self.has_current:
            raise RuntimeError("No current question")
        return self.bank.get(self.question_index)

    def recent_results(self, n: int = 3) -> list[int]:
        return self.results[-n:]

    def submit_answer(self, choice: int) -> AnswerResult:
        if self.finished:
            raise RuntimeError("Quiz already finished")
        if self._awaiting_continue:
            raise RuntimeError("Call continue_after_feedback before next answer")

        question = self.current_question()
        correct = question.is_correct(choice)
        points = points_for_answer(correct)
        self.score += points
        self.results.append(1 if correct else 0)

        if correct:
            self.correct_streak += 1
            self.wrong_streak = 0
        else:
            self.wrong_streak += 1
            self.correct_streak = 0

        self._awaiting_continue = True
        return AnswerResult(
            correct=correct,
            points_awarded=points,
            explanation=question.explanation,
            score=self.score,
            finished=False,
            difficulty_changed=False,
            current_difficulty=self.current_difficulty,
            recent_results=self.recent_results(),
        )

    def continue_after_feedback(self) -> AnswerResult:
        """Advance after showing explanation; may swap Auto bank or finish."""
        if self.finished:
            return self._done_result(difficulty_changed=False)

        if not self._awaiting_continue:
            raise RuntimeError("No pending feedback to continue from")

        self._awaiting_continue = False
        difficulty_changed = False

        if self.is_auto:
            new_diff = adjust_difficulty(
                self.current_difficulty, self.correct_streak, self.wrong_streak
            )
            if new_diff != self.current_difficulty:
                self.current_difficulty = new_diff
                self.correct_streak = 0
                self.wrong_streak = 0
                self.bank = self.question_repo.load(self.category, self.current_difficulty)
                difficulty_changed = True

        self.question_index += 1
        if self.question_index >= len(self.bank):
            self.finished = True
            return self._done_result(difficulty_changed=difficulty_changed)

        return AnswerResult(
            correct=False,
            points_awarded=0,
            explanation="",
            score=self.score,
            finished=False,
            difficulty_changed=difficulty_changed,
            current_difficulty=self.current_difficulty,
            recent_results=self.recent_results(),
        )

    def quit(self) -> AnswerResult:
        """Early exit (Esc); keep partial score."""
        self.finished = True
        self._awaiting_continue = False
        return self._done_result(difficulty_changed=False)

    def _done_result(self, difficulty_changed: bool) -> AnswerResult:
        return AnswerResult(
            correct=False,
            points_awarded=0,
            explanation="",
            score=self.score,
            finished=True,
            difficulty_changed=difficulty_changed,
            current_difficulty=self.current_difficulty,
            recent_results=self.recent_results(),
        )


def run_perfect_game(
    engine: QuizEngine,
    choose: Callable[[Question], int] | None = None,
) -> int:
    """Helper for tests: answer every question (default: always correct)."""
    chooser = choose or (lambda q: q.answer_index)
    while engine.has_current and not engine.finished:
        q = engine.current_question()
        engine.submit_answer(chooser(q))
        engine.continue_after_feedback()
    return engine.score
