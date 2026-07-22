from __future__ import annotations

import customtkinter as ctk

from quiz_wizard.config import (
    DIFFICULTIES,
    DIFFICULTY_AUTO,
    DIFFICULTY_LABELS,
)
from quiz_wizard.ui.theme import COLORS, FONT_HEADING, primary_button, secondary_button


class DifficultyScreen(ctk.CTkFrame):
    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color=COLORS["bg"])
        self.app = app

        panel = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=12)
        panel.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            panel,
            text="Choose Difficulty",
            font=FONT_HEADING,
            text_color=COLORS["text"],
        ).pack(padx=40, pady=(32, 20))

        for key in DIFFICULTIES:
            primary_button(
                panel,
                DIFFICULTY_LABELS[key],
                lambda k=key: self._start(k),
                width=300,
            ).pack(padx=40, pady=6)

        primary_button(
            panel,
            DIFFICULTY_LABELS[DIFFICULTY_AUTO],
            lambda: self._start(DIFFICULTY_AUTO),
            width=300,
        ).pack(padx=40, pady=6)

        secondary_button(
            panel, "Back", lambda: app.show("category"), width=300
        ).pack(padx=40, pady=(16, 32))

    def _start(self, mode: str) -> None:
        self.app.session.difficulty_mode = mode
        self.app.show("quiz")

    def on_show(self) -> None:
        pass
