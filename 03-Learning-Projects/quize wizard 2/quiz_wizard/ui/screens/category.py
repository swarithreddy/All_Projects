from __future__ import annotations

import customtkinter as ctk

from quiz_wizard.config import CATEGORIES, CATEGORY_LABELS
from quiz_wizard.ui.theme import COLORS, FONT_HEADING, primary_button, secondary_button


class CategoryScreen(ctk.CTkFrame):
    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color=COLORS["bg"])
        self.app = app

        panel = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=12)
        panel.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            panel,
            text="Choose a Category",
            font=FONT_HEADING,
            text_color=COLORS["text"],
        ).pack(padx=40, pady=(32, 20))

        for key in CATEGORIES:
            primary_button(
                panel,
                CATEGORY_LABELS[key],
                lambda k=key: self._select(k),
                width=300,
            ).pack(padx=40, pady=6)

        secondary_button(
            panel, "Cancel", lambda: app.show("home"), width=300
        ).pack(padx=40, pady=(16, 32))

    def _select(self, category: str) -> None:
        self.app.session.category = category
        self.app.show("difficulty")

    def on_show(self) -> None:
        pass
