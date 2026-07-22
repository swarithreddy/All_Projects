from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import customtkinter as ctk
from tkinter import messagebox

from quiz_wizard.config import APP_NAME, DEFAULT_WINDOW_SIZE, MIN_WINDOW_SIZE
from quiz_wizard.models.player import Player
from quiz_wizard.paths import log_file_path
from quiz_wizard.repositories.leaderboard import LeaderboardRepository
from quiz_wizard.repositories.questions import QuestionRepository
from quiz_wizard.ui.theme import COLORS, apply_theme

logger = logging.getLogger(__name__)


@dataclass
class AppSession:
    player: Player | None = None
    category: str | None = None
    difficulty_mode: str | None = None
    final_score: int | None = None
    final_difficulty_label: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def clear_play(self) -> None:
        self.player = None
        self.category = None
        self.difficulty_mode = None
        self.final_score = None
        self.final_difficulty_label = None
        self.extras.clear()


class App(ctk.CTk):
    def __init__(
        self,
        question_repo: QuestionRepository | None = None,
        leaderboard_repo: LeaderboardRepository | None = None,
    ) -> None:
        apply_theme()
        super().__init__()
        self.title(APP_NAME)
        self.geometry(DEFAULT_WINDOW_SIZE)
        self.minsize(*MIN_WINDOW_SIZE)
        self.configure(fg_color=COLORS["bg"])

        self.question_repo = question_repo or QuestionRepository()
        self.leaderboard_repo = leaderboard_repo or LeaderboardRepository()
        self.session = AppSession()

        self._container = ctk.CTkFrame(self, fg_color=COLORS["bg"])
        self._container.pack(fill="both", expand=True)
        self._screens: dict[str, ctk.CTkFrame] = {}
        self._current: str | None = None

        self._register_screens()
        self.show("home")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _register_screens(self) -> None:
        from quiz_wizard.ui.screens.category import CategoryScreen
        from quiz_wizard.ui.screens.difficulty import DifficultyScreen
        from quiz_wizard.ui.screens.home import HomeScreen
        from quiz_wizard.ui.screens.leaderboard import LeaderboardScreen
        from quiz_wizard.ui.screens.player_setup import PlayerSetupScreen
        from quiz_wizard.ui.screens.quiz import QuizScreen
        from quiz_wizard.ui.screens.results import ResultsScreen
        from quiz_wizard.ui.screens.tutorial import TutorialScreen

        builders = {
            "home": HomeScreen,
            "player_setup": PlayerSetupScreen,
            "category": CategoryScreen,
            "difficulty": DifficultyScreen,
            "quiz": QuizScreen,
            "results": ResultsScreen,
            "leaderboard": LeaderboardScreen,
            "tutorial": TutorialScreen,
        }
        for name, cls in builders.items():
            frame = cls(self._container, self)
            frame.grid(row=0, column=0, sticky="nsew")
            self._screens[name] = frame
        self._container.grid_rowconfigure(0, weight=1)
        self._container.grid_columnconfigure(0, weight=1)

    def show(self, name: str) -> None:
        screen = self._screens.get(name)
        if screen is None:
            logger.error("Unknown screen: %s", name)
            return
        if hasattr(screen, "on_show"):
            try:
                screen.on_show()
            except Exception:
                logger.exception("Error preparing screen %s", name)
                messagebox.showerror(
                    APP_NAME,
                    "Something went wrong loading this screen. Returning home.",
                )
                if name != "home":
                    self.show("home")
                return
        screen.tkraise()
        self._current = name

    def show_error(self, message: str, title: str | None = None) -> None:
        messagebox.showerror(title or APP_NAME, message)

    def show_info(self, message: str, title: str | None = None) -> None:
        messagebox.showinfo(title or APP_NAME, message)

    def confirm(self, message: str, title: str | None = None) -> bool:
        return bool(messagebox.askyesno(title or APP_NAME, message))

    def _on_close(self) -> None:
        self.destroy()


def setup_logging() -> None:
    log_path = log_file_path()
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    root.addHandler(fh)
    root.addHandler(sh)


def run() -> None:
    setup_logging()
    logger.info("Starting %s", APP_NAME)
    app = App()
    app.mainloop()
