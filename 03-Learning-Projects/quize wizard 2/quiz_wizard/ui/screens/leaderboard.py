from __future__ import annotations

import logging

import customtkinter as ctk

from quiz_wizard.repositories.leaderboard import LeaderboardRepositoryError
from quiz_wizard.ui.theme import (
    COLORS,
    FONT_BODY,
    FONT_HEADING,
    FONT_SMALL,
    secondary_button,
)

logger = logging.getLogger(__name__)


class LeaderboardScreen(ctk.CTkFrame):
    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color=COLORS["bg"])
        self.app = app

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=32, pady=(28, 8))
        ctk.CTkLabel(
            top, text="Leaderboard", font=FONT_HEADING, text_color=COLORS["text"]
        ).pack(side="left")
        secondary_button(top, "Back", self._back, width=120).pack(side="right")

        header = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=8)
        header.pack(fill="x", padx=32, pady=(8, 4))
        for col, weight in (("Rank", 1), ("Name", 4), ("Age", 1), ("Score", 1)):
            ctk.CTkLabel(
                header, text=col, font=FONT_SMALL, text_color=COLORS["muted"]
            ).pack(side="left", expand=True if weight > 1 else False, padx=12, pady=10)

        self.list_frame = ctk.CTkScrollableFrame(
            self, fg_color=COLORS["surface"], corner_radius=8
        )
        self.list_frame.pack(fill="both", expand=True, padx=32, pady=(4, 28))

        self.empty_label = ctk.CTkLabel(
            self.list_frame,
            text="No scores yet. Play a quiz to appear here!",
            font=FONT_BODY,
            text_color=COLORS["muted"],
        )

    def _back(self) -> None:
        # Prefer results if we just finished, else home
        if self.app.session.final_score is not None and self.app.session.player:
            self.app.show("results")
        else:
            self.app.show("home")

    def on_show(self) -> None:
        for child in self.list_frame.winfo_children():
            child.destroy()
        try:
            board = self.app.leaderboard_repo.load()
        except LeaderboardRepositoryError as exc:
            logger.exception("Leaderboard load failed")
            self.app.show_error(str(exc))
            ctk.CTkLabel(
                self.list_frame,
                text="Could not load leaderboard.",
                font=FONT_BODY,
                text_color=COLORS["danger"],
            ).pack(pady=24)
            return

        entries = board.sorted_entries()
        if not entries:
            ctk.CTkLabel(
                self.list_frame,
                text="No scores yet. Play a quiz to appear here!",
                font=FONT_BODY,
                text_color=COLORS["muted"],
            ).pack(pady=24)
            return

        for rank, entry in enumerate(entries, start=1):
            row = ctk.CTkFrame(self.list_frame, fg_color=COLORS["surface_alt"], corner_radius=6)
            row.pack(fill="x", pady=4, padx=4)
            values = (str(rank), entry.name, str(entry.age), str(entry.score))
            for value in values:
                ctk.CTkLabel(
                    row, text=value, font=FONT_BODY, text_color=COLORS["text"]
                ).pack(side="left", expand=True, padx=12, pady=10)
