from quiz_wizard.config import DIFFICULTY_EASY, DIFFICULTY_HARD, DIFFICULTY_MEDIUM
from quiz_wizard.services.auto_difficulty import adjust_difficulty


def test_upgrade_on_streak():
    assert adjust_difficulty(DIFFICULTY_EASY, 3, 0) == DIFFICULTY_MEDIUM
    assert adjust_difficulty(DIFFICULTY_MEDIUM, 3, 0) == DIFFICULTY_HARD
    assert adjust_difficulty(DIFFICULTY_HARD, 3, 0) == DIFFICULTY_HARD


def test_downgrade_on_streak():
    assert adjust_difficulty(DIFFICULTY_HARD, 0, 3) == DIFFICULTY_MEDIUM
    assert adjust_difficulty(DIFFICULTY_MEDIUM, 0, 3) == DIFFICULTY_EASY
    assert adjust_difficulty(DIFFICULTY_EASY, 0, 3) == DIFFICULTY_EASY


def test_no_change_below_threshold():
    assert adjust_difficulty(DIFFICULTY_EASY, 2, 0) == DIFFICULTY_EASY
    assert adjust_difficulty(DIFFICULTY_HARD, 0, 2) == DIFFICULTY_HARD
