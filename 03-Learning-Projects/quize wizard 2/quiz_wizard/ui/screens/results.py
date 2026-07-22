from __future__ import annotations

import logging

import customtkinter as ctk

from quiz_wizard.config import CATEGORY_LABELS
from quiz_wizard.repositories.leaderboard import LeaderboardRepositoryError
from quiz_wizard.ui.theme import (
    COLORS,
    FONT_BODY,
    FONT_HEADING,
    FONT_TITLE,
    primary_button,
    secondary_button,
)

logger = logging.getLogger(__name__)


class ResultsScreen(ctk.CTkFrame):
    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color=COLORS["bg"])
        self.app = app

        panel = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=12)
        panel.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            panel,
            text="Quiz Completed!",
            font=FONT_TITLE,
            text_color=COLORS["accent"],
        ).pack(padx=48, pady=(36, 12))

        self.summary = ctk.CTkLabel(
            panel,
            text="",
            font=FONT_BODY,
            text_color=COLORS["text"],
            justify="left",
        )
        self.summary.pack(padx=48, pady=8)

        self.score_label = ctk.CTkLabel(
            panel, text="", font=FONT_HEADING, text_color=COLORS["accent"]
        )
        self.score_label.pack(padx=48, pady=(8, 20))

        self.status = ctk.CTkLabel(
            panel, text="", font=FONT_BODY, text_color=COLORS["muted"]
        )
        self.status.pack(padx=48, pady=(0, 12))

        btn_row = ctk.CTkFrame(panel, fg_color="transparent")
        btn_row.pack(padx=48, pady=(8, 36))
        primary_button(
            btn_row, "Leaderboard", lambda: app.show("leaderboard"), width=160
        ).pack(side="left", padx=6)
        secondary_button(btn_row, "Home", lambda: app.show("home"), width=160).pack(
            side="left", padx=6
        )

    def on_show(self) -> None:
        session = self.app.session
        player = session.player
        if player is None or session.final_score is None:
            self.app.show("home")
            return

        cat = CATEGORY_LABELS.get(session.category or "", session.category or "")
        self.summary.configure(
            text=(
                f"Player: {player.name}\n"
                f"Age: {player.age}\n"
                f"Category: {cat}\n"
                f"Difficulty: {session.final_difficulty_label or session.difficulty_mode}"
            )
        )
        self.score_label.configure(text=f"Final score: {session.final_score}")

        if session.extras.get("score_saved"):
            self.status.configure(text="Score saved to the leaderboard.")
            return

        try:
            self.app.leaderboard_repo.add(player.name, player.age, session.final_score)
            session.extras["score_saved"] = True
            self.status.configure(text="Score saved to the leaderboard.")
        except LeaderboardRepositoryError as exc:
            logger.exception("Failed to save score")
            self.status.configure(text="Could not save score.")
            self.app.show_error(str(exc))
