from __future__ import annotations

import customtkinter as ctk

# Avoid purple/indigo AI-default look: deep teal + warm sand accents
COLORS = {
    "bg": "#0f1c1a",
    "surface": "#1a2e2a",
    "surface_alt": "#243d37",
    "text": "#f2ebe3",
    "muted": "#a8b5b0",
    "accent": "#d4a373",
    "accent_hover": "#e0b588",
    "success": "#3d9a6a",
    "danger": "#c45c4a",
    "border": "#2f4a44",
}

FONT_FAMILY = "Segoe UI"
FONT_TITLE = (FONT_FAMILY, 36, "bold")
FONT_HEADING = (FONT_FAMILY, 24, "bold")
FONT_BODY = (FONT_FAMILY, 15)
FONT_SMALL = (FONT_FAMILY, 13)
FONT_BUTTON = (FONT_FAMILY, 15, "bold")


def apply_theme() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")


def primary_button(master, text: str, command, **kwargs) -> ctk.CTkButton:
    return ctk.CTkButton(
        master,
        text=text,
        command=command,
        font=FONT_BUTTON,
        fg_color=COLORS["accent"],
        hover_color=COLORS["accent_hover"],
        text_color=COLORS["bg"],
        corner_radius=8,
        height=44,
        **kwargs,
    )


def secondary_button(master, text: str, command, **kwargs) -> ctk.CTkButton:
    return ctk.CTkButton(
        master,
        text=text,
        command=command,
        font=FONT_BUTTON,
        fg_color=COLORS["surface_alt"],
        hover_color=COLORS["border"],
        text_color=COLORS["text"],
        corner_radius=8,
        height=44,
        **kwargs,
    )
