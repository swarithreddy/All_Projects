from __future__ import annotations

import customtkinter as ctk

from quiz_wizard.config import CATEGORY_LABELS, DIFFICULTY_LABELS
from quiz_wizard.repositories.questions import QuestionRepositoryError
from quiz_wizard.services.quiz_engine import QuizEngine
from quiz_wizard.ui.theme import (
    COLORS,
    FONT_BODY,
    FONT_HEADING,
    FONT_SMALL,
    primary_button,
    secondary_button,
)
from quiz_wizard.ui.widgets import option_button


class QuizScreen(ctk.CTkFrame):
    def __init__(self, master, app) -> None:
        super().__init__(master, fg_color=COLORS["bg"])
        self.app = app
        self.engine: QuizEngine | None = None
        self._option_buttons: list[ctk.CTkButton] = []
        self._feedback_visible = False

        header = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=0)
        header.pack(fill="x")
        self.meta_label = ctk.CTkLabel(
            header, text="", font=FONT_SMALL, text_color=COLORS["muted"]
        )
        self.meta_label.pack(side="left", padx=24, pady=16)
        self.score_label = ctk.CTkLabel(
            header, text="Score: 0", font=FONT_BODY, text_color=COLORS["accent"]
        )
        self.score_label.pack(side="right", padx=24, pady=16)

        self.streak_label = ctk.CTkLabel(
            self, text="", font=FONT_SMALL, text_color=COLORS["muted"]
        )
        self.streak_label.pack(anchor="w", padx=32, pady=(12, 0))

        self.question_label = ctk.CTkLabel(
            self,
            text="",
            font=FONT_HEADING,
            text_color=COLORS["text"],
            wraplength=900,
            justify="left",
        )
        self.question_label.pack(anchor="w", padx=32, pady=(20, 16))

        self.options_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.options_frame.pack(fill="x", padx=32)

        self.feedback_frame = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=10)
        self.feedback_title = ctk.CTkLabel(
            self.feedback_frame, text="", font=FONT_BODY, text_color=COLORS["text"]
        )
        self.feedback_title.pack(anchor="w", padx=20, pady=(16, 4))
        self.feedback_body = ctk.CTkLabel(
            self.feedback_frame,
            text="",
            font=FONT_SMALL,
            text_color=COLORS["muted"],
            wraplength=860,
            justify="left",
        )
        self.feedback_body.pack(anchor="w", padx=20, pady=(0, 12))
        self.continue_btn = primary_button(
            self.feedback_frame, "Continue", self._on_continue, width=180
        )
        self.continue_btn.pack(anchor="e", padx=20, pady=(0, 16))

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=32, pady=20)
        secondary_button(footer, "Quit (Esc)", self._on_quit, width=160).pack(side="left")
        self._esc_bound = False

    def on_show(self) -> None:
        session = self.app.session
        if not session.player or not session.category or not session.difficulty_mode:
            self.app.show("home")
            return
        try:
            self.engine = QuizEngine(
                category=session.category,
                mode=session.difficulty_mode,
                question_repo=self.app.question_repo,
            )
        except QuestionRepositoryError as exc:
            self.app.show_error(str(exc))
            self.app.show("difficulty")
            return
        self._feedback_visible = False
        self.feedback_frame.pack_forget()
        if not self._esc_bound:
            self.app.bind("<Escape>", self._esc_handler)
            self._esc_bound = True
        self._render_question()

    def _esc_handler(self, _event=None):
        if self.app._current == "quiz":
            self._on_quit()
        return "break"

    def _render_question(self) -> None:
        assert self.engine is not None
        if not self.engine.has_current:
            self._finish()
            return

        q = self.engine.current_question()
        cat = CATEGORY_LABELS.get(self.engine.category, self.engine.category)
        diff = DIFFICULTY_LABELS.get(
            self.engine.current_difficulty, self.engine.current_difficulty
        )
        mode_note = " (Auto)" if self.engine.is_auto else ""
        self.meta_label.configure(
            text=f"{cat}  ·  {diff}{mode_note}  ·  Q{self.engine.question_index + 1}"
        )
        self.score_label.configure(text=f"Score: {self.engine.score}")

        if self.engine.is_auto:
            recent = self.engine.recent_results()
            symbols = " ".join("✓" if r else "✗" for r in recent) if recent else "—"
            self.streak_label.configure(text=f"Last 3 results: {symbols}")
            self.streak_label.pack(anchor="w", padx=32, pady=(12, 0))
        else:
            self.streak_label.configure(text="")
            self.streak_label.pack_forget()

        self.question_label.configure(text=f"{q.id}. {q.prompt}")

        for btn in self._option_buttons:
            btn.destroy()
        self._option_buttons.clear()

        for idx, option in enumerate(q.options, start=1):
            btn = option_button(
                self.options_frame,
                f"{idx}. {option}",
                lambda i=idx: self._on_answer(i),
                width=860,
            )
            btn.pack(fill="x", pady=6)
            self._option_buttons.append(btn)

        self._set_options_enabled(True)
        self.feedback_frame.pack_forget()
        self._feedback_visible = False

    def _set_options_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for btn in self._option_buttons:
            btn.configure(state=state)

    def _on_answer(self, choice: int) -> None:
        if not self.engine or self._feedback_visible:
            return
        result = self.engine.submit_answer(choice)
        self._set_options_enabled(False)
        self.score_label.configure(text=f"Score: {result.score}")
        title = "Correct!" if result.correct else "Wrong!"
        color = COLORS["success"] if result.correct else COLORS["danger"]
        self.feedback_title.configure(text=title, text_color=color)
        self.feedback_body.configure(text=result.explanation)
        self.feedback_frame.pack(fill="x", padx=32, pady=20)
        self._feedback_visible = True

    def _on_continue(self) -> None:
        if not self.engine or not self._feedback_visible:
            return
        result = self.engine.continue_after_feedback()
        if result.finished:
            self._finish()
            return
        self._render_question()

    def _on_quit(self) -> None:
        if not self.engine or self.engine.finished:
            return
        if not self.app.confirm("Leave the quiz early? Your score so far will be saved."):
            return
        self.engine.quit()
        self._finish()

    def _finish(self) -> None:
        assert self.engine is not None
        if self._esc_bound:
            self.app.unbind("<Escape>")
            self._esc_bound = False
        session = self.app.session
        session.final_score = self.engine.score
        if self.engine.is_auto:
            ended = DIFFICULTY_LABELS.get(
                self.engine.current_difficulty, self.engine.current_difficulty
            )
            session.final_difficulty_label = f"Auto ({ended})"
        else:
            session.final_difficulty_label = DIFFICULTY_LABELS.get(
                self.engine.mode, self.engine.mode
            )
        self.app.show("results")
