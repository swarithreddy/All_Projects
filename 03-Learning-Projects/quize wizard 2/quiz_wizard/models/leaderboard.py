from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ScoreEntry:
    name: str
    age: int
    score: int
    played_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "age": self.age,
            "score": self.score,
            "played_at": self.played_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ScoreEntry:
        return cls(
            name=str(data["name"]),
            age=int(data["age"]),
            score=int(data["score"]),
            played_at=str(data.get("played_at") or datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class Leaderboard:
    entries: list[ScoreEntry] = field(default_factory=list)

    def sorted_entries(self) -> list[ScoreEntry]:
        return sorted(
            self.entries,
            key=lambda e: (-e.score, e.played_at),
        )

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self.sorted_entries()]}

    @classmethod
    def from_dict(cls, data: dict) -> Leaderboard:
        raw = data.get("entries") or []
        return cls(entries=[ScoreEntry.from_dict(item) for item in raw])
