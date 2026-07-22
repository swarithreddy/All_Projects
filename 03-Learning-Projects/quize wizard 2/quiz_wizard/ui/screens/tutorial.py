from __future__ import annotations

import logging

import customtkinter as ctk

from quiz_wizard.paths import tutorial_path
from quiz_wizard.ui.theme import COLORS, FONT_BODY, FONT_HEADING, secondary_button

logger = logging.getLogger(__name__)

_FALLBACK = """# Quiz Wizard Tutorial

## Main Menu
- **Play** — enter your name and age, pick a category and difficulty, then answer questions.
- **Leaderboard** — see high scores ranked from highest to lowest.
- **Tutorial** — you are here.
- **Exit** — close the application.

## Categories
General Knowledge, Technical, and Geopolitical.

## Difficulty
Easy, Medium, Hard, or Auto.

## Scoring
Each correct answer awards **10** points. Wrong answers award 0. There is no time limit or lives system.

## Auto Mode
Auto starts on Easy. Three correct answers in a row increase difficulty; three wrong answers in a row decrease it. Streaks reset when difficulty changes. The quiz continues at the same question index in the new bank. The last three results are shown as you play.

## Leaving early
Press Esc (or Quit) during a quiz to finish early. Your score so far is still saved to the leaderboard.
"""


class TutorialScreen(ctk.CTkFrame):
    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color=COLORS["bg"])
        self.app = app

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=32, pady=(28, 8))
        ctk.CTkLabel(
            top, text="Tutorial", font=FONT_HEADING, text_color=COLORS["text"]
        ).pack(side="left")
        secondary_button(top, "Back", lambda: app.show("home"), width=120).pack(
            side="right"
        )

        self.text = ctk.CTkTextbox(
            self,
            font=FONT_BODY,
            fg_color=COLORS["surface"],
            text_color=COLORS["text"],
            wrap="word",
        )
        self.text.pack(fill="both", expand=True, padx=32, pady=(8, 28))

    def on_show(self) -> None:
        path = tutorial_path()
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Tutorial file missing at %s", path)
            content = _FALLBACK
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self.text.configure(state="disabled")
