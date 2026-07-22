from __future__ import annotations

import customtkinter as ctk

from quiz_wizard.models.player import Player
from quiz_wizard.ui.theme import (
    COLORS,
    FONT_BODY,
    FONT_HEADING,
    FONT_SMALL,
    primary_button,
    secondary_button,
)
from quiz_wizard.utils.validation import validate_age, validate_name


class PlayerSetupScreen(ctk.CTkFrame):
    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color=COLORS["bg"])
        self.app = app

        panel = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=12)
        panel.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            panel, text="Player Setup", font=FONT_HEADING, text_color=COLORS["text"]
        ).pack(padx=40, pady=(32, 16))

        ctk.CTkLabel(panel, text="Name", font=FONT_SMALL, text_color=COLORS["muted"]).pack(
            anchor="w", padx=40
        )
        self.name_entry = ctk.CTkEntry(panel, width=320, height=40, font=FONT_BODY)
        self.name_entry.pack(padx=40, pady=(4, 12))

        ctk.CTkLabel(panel, text="Age", font=FONT_SMALL, text_color=COLORS["muted"]).pack(
            anchor="w", padx=40
        )
        self.age_entry = ctk.CTkEntry(panel, width=320, height=40, font=FONT_BODY)
        self.age_entry.pack(padx=40, pady=(4, 8))

        self.error_label = ctk.CTkLabel(
            panel, text="", font=FONT_SMALL, text_color=COLORS["danger"]
        )
        self.error_label.pack(padx=40, pady=(0, 12))

        btn_row = ctk.CTkFrame(panel, fg_color="transparent")
        btn_row.pack(padx=40, pady=(8, 32))
        secondary_button(btn_row, "Back", lambda: app.show("home"), width=140).pack(
            side="left", padx=6
        )
        primary_button(btn_row, "Continue", self._continue, width=140).pack(
            side="left", padx=6
        )

    def on_show(self) -> None:
        self.error_label.configure(text="")
        self.name_entry.delete(0, "end")
        self.age_entry.delete(0, "end")
        self.name_entry.focus_set()

    def _continue(self) -> None:
        name_err = validate_name(self.name_entry.get())
        age, age_err = validate_age(self.age_entry.get())
        if name_err:
            self.error_label.configure(text=name_err)
            return
        if age_err:
            self.error_label.configure(text=age_err)
            return
        assert age is not None
        self.app.session.player = Player(name=self.name_entry.get().strip(), age=age)
        self.app.show("category")
