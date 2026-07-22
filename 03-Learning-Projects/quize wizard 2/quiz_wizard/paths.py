"""Resolve bundled asset paths and writable AppData paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    """Source-tree root (dev) or folder containing the exe (frozen onedir)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_root() -> Path:
    """Read-only bundled resources (PyInstaller _MEIPASS or project root)."""
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return project_root()


def questions_dir() -> Path:
    return resource_root() / "data" / "questions"


def assets_dir() -> Path:
    return resource_root() / "assets"


def tutorial_path() -> Path:
    return assets_dir() / "tutorial.md"


def app_data_dir() -> Path:
    """Writable user data directory."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            path = Path(base) / "QuizWizard"
        else:
            path = Path.home() / "AppData" / "Local" / "QuizWizard"
    else:
        path = Path.home() / ".local" / "share" / "QuizWizard"
    path.mkdir(parents=True, exist_ok=True)
    return path


def leaderboard_path() -> Path:
    return app_data_dir() / "leaderboard.json"


def settings_path() -> Path:
    return app_data_dir() / "settings.json"


def logs_dir() -> Path:
    path = app_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_file_path() -> Path:
    return logs_dir() / "app.log"
