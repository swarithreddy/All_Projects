from quiz_wizard.config import POINTS_PER_CORRECT
from quiz_wizard.services.scoring import points_for_answer


def test_points_correct():
    assert points_for_answer(True) == POINTS_PER_CORRECT
    assert points_for_answer(True) == 10


def test_points_wrong():
    assert points_for_answer(False) == 0
