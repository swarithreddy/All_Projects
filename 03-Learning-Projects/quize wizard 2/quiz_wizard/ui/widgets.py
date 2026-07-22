from __future__ import annotations

import customtkinter as ctk

from quiz_wizard.ui.theme import COLORS, FONT_BODY, secondary_button


def option_button(master, text: str, command, **kwargs) -> ctk.CTkButton:
    return ctk.CTkButton(
        master,
        text=text,
        command=command,
        font=FONT_BODY,
        fg_color=COLORS["surface_alt"],
        hover_color=COLORS["border"],
        text_color=COLORS["text"],
        anchor="w",
        corner_radius=8,
        height=48,
        **kwargs,
    )


def make_back_row(master, on_back, label: str = "Back") -> ctk.CTkFrame:
    row = ctk.CTkFrame(master, fg_color="transparent")
    secondary_button(row, label, on_back, width=120).pack(side="left")
    return row
