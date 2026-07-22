from __future__ import annotations

from pathlib import Path

import pytest

from quiz_wizard.config import (
    CATEGORY_GENERAL,
    DIFFICULTY_AUTO,
    DIFFICULTY_EASY,
)
from quiz_wizard.repositories.questions import QuestionRepository
from quiz_wizard.services.quiz_engine import QuizEngine, run_perfect_game


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "questions"


@pytest.fixture
def repo() -> QuestionRepository:
    if not (DATA / "general_knowledge" / "easy.json").exists():
        pytest.skip("JSON banks not migrated yet")
    return QuestionRepository(DATA)


def test_perfect_fixed_game_score_200(repo: QuestionRepository):
    engine = QuizEngine(CATEGORY_GENERAL, DIFFICULTY_EASY, repo)
    score = run_perfect_game(engine)
    assert score == 200
    assert engine.finished


def test_quit_mid_quiz_keeps_partial_score(repo: QuestionRepository):
    engine = QuizEngine(CATEGORY_GENERAL, DIFFICULTY_EASY, repo)
    q = engine.current_question()
    engine.submit_answer(q.answer_index)
    engine.continue_after_feedback()
    result = engine.quit()
    assert result.finished
    assert result.score == 10


def test_wrong_answer_no_points(repo: QuestionRepository):
    engine = QuizEngine(CATEGORY_GENERAL, DIFFICULTY_EASY, repo)
    q = engine.current_question()
    wrong = 1 if q.answer_index != 1 else 2
    result = engine.submit_answer(wrong)
    assert result.correct is False
    assert result.score == 0


def test_auto_upgrades_after_three_correct(repo: QuestionRepository):
    engine = QuizEngine(CATEGORY_GENERAL, DIFFICULTY_AUTO, repo)
    assert engine.current_difficulty == DIFFICULTY_EASY
    for _ in range(3):
        q = engine.current_question()
        engine.submit_answer(q.answer_index)
        engine.continue_after_feedback()
    assert engine.current_difficulty == "medium"
    assert engine.question_index == 3


def test_auto_resets_streak_after_upgrade(repo: QuestionRepository):
    engine = QuizEngine(CATEGORY_GENERAL, DIFFICULTY_AUTO, repo)
    for _ in range(3):
        q = engine.current_question()
        engine.submit_answer(q.answer_index)
        engine.continue_after_feedback()
    assert engine.correct_streak == 0
    q = engine.current_question()
    engine.submit_answer(q.answer_index)
    engine.continue_after_feedback()
    assert engine.current_difficulty == "medium"
