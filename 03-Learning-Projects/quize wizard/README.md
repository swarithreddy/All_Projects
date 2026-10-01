# Quiz Wizard

A terminal quiz game with category and difficulty selection, a tutorial, and a text-file leaderboard. Question banks and scores are stored in text files beside the Python scripts.

## Features

- Choose General Knowledge, Technical, or Geopolitical questions
- Choose a difficulty level; the game also includes an adaptive Auto mode
- Record player name and age before playing
- Save scores to a local leaderboard
- View the included tutorial

## Project Structure

```text
quize wizard/
├── main.py                 # Menu and entry point
├── play.py                 # Player details and quiz selection
├── main_operation.py       # Quiz gameplay
├── auto.py                 # Adaptive difficulty mode
├── add_data.py             # Leaderboard operations
├── type_choice.py          # Category selection
├── difficulty_choice.py    # Difficulty selection
├── *_*.txt                 # Category/difficulty question banks
├── leaderboard.txt         # Local score data
└── tutorial.txt            # In-app tutorial text
```

## Requirements

- Python 3
- No third-party packages

## Run

Run from this project directory so the program can find its question, tutorial, and leaderboard files:

```bash
python main.py
```

Choose a menu option and follow the prompts. The player name, age, and score are written to `leaderboard.txt` after a quiz.

## Notes

- Question banks and leaderboard data are plain text files and can be edited locally.
- Scores are stored on the same machine; there is no database or account system.
- The project does not include an automated test suite or dependency manifest.
