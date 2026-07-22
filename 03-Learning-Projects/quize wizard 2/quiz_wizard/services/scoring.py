from __future__ import annotations

from quiz_wizard.config import POINTS_PER_CORRECT


def points_for_answer(correct: bool) -> int:
    return POINTS_PER_CORRECT if correct else 0
