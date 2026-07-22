from __future__ import annotations

from pathlib import Path

import pytest

from quiz_wizard.config import (
    CATEGORY_GENERAL,
    DIFFICULTY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_MEDIUM,
)
from quiz_wizard.repositories.migrator import migrate_legacy_directory, parse_legacy_txt
from quiz_wizard.repositories.questions import QuestionRepository


ROOT = Path(__file__).resolve().parent.parent
LEGACY = ROOT / "legacy"
SAMPLE = """1
What is the capital of France?
    Paris
    London
    Berlin
    Madrid
1
Explanation: Paris is the capital.

2
Who wrote Hamlet?
    Dickens
    Shakespeare
    Twain
    Rowling
2
Explanation: William Shakespeare wrote Hamlet.
"""


def test_parse_legacy_txt_basic():
    questions = parse_legacy_txt(SAMPLE)
    assert len(questions) == 2
    assert questions[0].prompt.startswith("What is the capital")
    assert questions[0].options[0] == "Paris"
    assert questions[0].answer_index == 1
    assert questions[0].explanation == "Paris is the capital."
    assert "Explanation:" not in questions[0].explanation
    assert questions[1].answer_index == 2


def test_migrate_all_legacy_banks(tmp_path: Path):
    if not (LEGACY / "gene.txt").exists():
        pytest.skip("legacy banks not present")
    written = migrate_legacy_directory(LEGACY, tmp_path)
    assert len(written) == 9
    repo = QuestionRepository(tmp_path)
    total = 0
    for cat in ("general_knowledge", "technical", "geopolitical"):
        for diff in ("easy", "medium", "hard"):
            bank = repo.load(cat, diff)
            assert len(bank) == 20
            total += len(bank)
    assert total == 180


def test_utf8_content_preserved():
    path = LEGACY / "gene.txt"
    if not path.exists():
        pytest.skip("legacy gene.txt missing")
    text = path.read_text(encoding="utf-8")
    questions = parse_legacy_txt(text)
    # Ensure parser accepts full file
    assert len(questions) == 20
