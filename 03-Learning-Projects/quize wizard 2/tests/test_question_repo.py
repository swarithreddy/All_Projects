from __future__ import annotations

from pathlib import Path

import pytest

from quiz_wizard.config import CATEGORY_GENERAL, DIFFICULTY_EASY
from quiz_wizard.repositories.questions import QuestionRepository, QuestionRepositoryError


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "questions"


@pytest.fixture
def repo() -> QuestionRepository:
    if not (DATA / "general_knowledge" / "easy.json").exists():
        pytest.skip("JSON banks not migrated yet")
    return QuestionRepository(DATA)


def test_load_all_banks(repo: QuestionRepository):
    for cat in ("general_knowledge", "technical", "geopolitical"):
        for diff in ("easy", "medium", "hard"):
            bank = repo.load(cat, diff)
            assert bank.category == cat
            assert len(bank) == 20
            assert all(len(q.options) == 4 for q in bank.questions)
            assert all(q.answer_index in (1, 2, 3, 4) for q in bank.questions)


def test_missing_bank_raises(tmp_path: Path):
    repo = QuestionRepository(tmp_path)
    with pytest.raises(QuestionRepositoryError):
        repo.load(CATEGORY_GENERAL, DIFFICULTY_EASY)
