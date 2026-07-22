from __future__ import annotations

import json
import logging
from pathlib import Path

from quiz_wizard.models.leaderboard import Leaderboard, ScoreEntry
from quiz_wizard.paths import leaderboard_path

logger = logging.getLogger(__name__)


class LeaderboardRepositoryError(Exception):
    """Raised when leaderboard persistence fails."""


class LeaderboardRepository:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or leaderboard_path()

    def load(self) -> Leaderboard:
        if not self.path.exists():
            return Leaderboard()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return Leaderboard.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise LeaderboardRepositoryError(
                f"Corrupt leaderboard file: {self.path}"
            ) from exc

    def save(self, board: Leaderboard) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(board.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise LeaderboardRepositoryError(
                f"Could not write leaderboard: {self.path}"
            ) from exc

    def add(self, name: str, age: int, score: int) -> Leaderboard:
        board = self.load()
        board.entries.append(ScoreEntry(name=name, age=age, score=score))
        board.entries = board.sorted_entries()
        self.save(board)
        return board

    def import_legacy_txt(self, legacy_path: Path) -> Leaderboard:
        """One-shot import of V1 whitespace leaderboard lines."""
        if not legacy_path.exists():
            return self.load()
        board = self.load()
        for line in legacy_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            # V1: serial name age score — name has no spaces in practice;
            # tolerate multi-token names by taking last two ints as age/score.
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                score = int(parts[-1])
                age = int(parts[-2])
                name = " ".join(parts[1:-2])
            except ValueError:
                continue
            if not name:
                continue
            board.entries.append(ScoreEntry(name=name, age=age, score=score))
        board.entries = board.sorted_entries()
        self.save(board)
        return board


def maybe_import_legacy_leaderboard(
    repo: LeaderboardRepository, legacy_txt: Path
) -> None:
    marker = repo.path.parent / ".legacy_leaderboard_imported"
    if marker.exists() or not legacy_txt.exists():
        return
    if repo.path.exists():
        marker.write_text("skipped\n", encoding="utf-8")
        return
    try:
        repo.import_legacy_txt(legacy_txt)
        marker.write_text("done\n", encoding="utf-8")
    except LeaderboardRepositoryError:
        logger.exception("Failed to import legacy leaderboard")
