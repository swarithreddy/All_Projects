Quiz Wizard

A Python-based quiz game that presents questions from various categories
(tech, geography, mathematics, etc.) and tracks scores. The project consists
of multiple scripts for question generation, difficulty selection, and game
logic. Data files (text) store the questions.

Structure

```
quize wizard/
├── add_data.py          <- scripts to add new questions
├── main.py              <- entry point for playing quizzes
├── play.py              <- game loop and score handling
├── difficulty_choice.py <- choose difficulty level
├── type_choice.py       <- choose category/type
├── tutorial.txt         <- user instructions
├── *.txt                <- question datasets (gene, geom, techh, etc.)
└── __pycache__/         <- compiled Python files (ignored)
```

Usage

Run `main.py` with Python 3 to start the quiz.

Git setup

This folder has been initialized as a Git repository (see `.gitignore`).

License

No license specified; add one if sharing publicly.
