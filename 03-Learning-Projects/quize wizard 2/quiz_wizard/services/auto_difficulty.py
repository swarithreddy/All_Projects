from __future__ import annotations

from quiz_wizard.config import (
    AUTO_STREAK_THRESHOLD,
    DIFFICULTY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_MEDIUM,
)

_ORDER = (DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD)


def adjust_difficulty(
    current: str,
    correct_streak: int,
    wrong_streak: int,
    threshold: int = AUTO_STREAK_THRESHOLD,
) -> str:
    """Step Easy↔Medium↔Hard when streak reaches threshold."""
    if current not in _ORDER:
        return current
    idx = _ORDER.index(current)
    if correct_streak >= threshold and idx < len(_ORDER) - 1:
        return _ORDER[idx + 1]
    if wrong_streak >= threshold and idx > 0:
        return _ORDER[idx - 1]
    return current
