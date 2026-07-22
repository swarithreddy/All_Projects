from __future__ import annotations

import customtkinter as ctk

from quiz_wizard.config import APP_NAME, APP_VERSION
from quiz_wizard.ui.theme import COLORS, FONT_BODY, FONT_TITLE, primary_button, secondary_button


class HomeScreen(ctk.CTkFrame):
    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color=COLORS["bg"])
        self.app = app

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            inner,
            text=APP_NAME,
            font=FONT_TITLE,
            text_color=COLORS["accent"],
        ).pack(pady=(0, 8))
        ctk.CTkLabel(
            inner,
            text="Test your knowledge across General Knowledge, Technical, and Geopolitical quizzes.",
            font=FONT_BODY,
            text_color=COLORS["muted"],
            wraplength=520,
        ).pack(pady=(0, 28))

        primary_button(inner, "Play", lambda: app.show("player_setup"), width=260).pack(
            pady=6
        )
        secondary_button(
            inner, "Leaderboard", lambda: app.show("leaderboard"), width=260
        ).pack(pady=6)
        secondary_button(inner, "Tutorial", lambda: app.show("tutorial"), width=260).pack(
            pady=6
        )
        secondary_button(inner, "Exit", app.destroy, width=260).pack(pady=6)

        ctk.CTkLabel(
            self,
            text=f"v{APP_VERSION}",
            font=FONT_BODY,
            text_color=COLORS["muted"],
        ).place(relx=0.98, rely=0.98, anchor="se")

    def on_show(self) -> None:
        self.app.session.clear_play()
