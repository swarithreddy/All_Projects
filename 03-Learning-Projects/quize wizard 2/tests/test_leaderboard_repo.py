from __future__ import annotations

from pathlib import Path

from quiz_wizard.models.leaderboard import Leaderboard, ScoreEntry
from quiz_wizard.repositories.leaderboard import LeaderboardRepository


def test_leaderboard_spaces_in_name(tmp_path: Path):
    path = tmp_path / "leaderboard.json"
    repo = LeaderboardRepository(path)
    repo.add("Ada Lovelace", 36, 80)
    repo.add("shiva", 19, 30)
    board = repo.load()
    entries = board.sorted_entries()
    assert entries[0].name == "Ada Lovelace"
    assert entries[0].score == 80
    assert entries[1].name == "shiva"


def test_leaderboard_sort_descending(tmp_path: Path):
    path = tmp_path / "lb.json"
    repo = LeaderboardRepository(path)
    repo.add("a", 10, 10)
    repo.add("b", 10, 50)
    repo.add("c", 10, 30)
    scores = [e.score for e in repo.load().sorted_entries()]
    assert scores == [50, 30, 10]


def test_import_legacy_txt(tmp_path: Path):
    legacy = tmp_path / "leaderboard.txt"
    legacy.write_text("1 Ada Lovelace 36 80\n2 bob 12 20\n", encoding="utf-8")
    repo = LeaderboardRepository(tmp_path / "leaderboard.json")
    board = repo.import_legacy_txt(legacy)
    assert board.sorted_entries()[0].name == "Ada Lovelace"
    assert board.sorted_entries()[0].score == 80
