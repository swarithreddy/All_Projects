# Quiz Wizard — Windows desktop edition (CustomTkinter)

Version **2.0.0** turns the original terminal quiz into an installable Windows app.

## Features

- Play quizzes in **General Knowledge**, **Technical**, and **Geopolitical**
- Difficulties: **Easy**, **Medium**, **Hard**, and adaptive **Auto**
- +10 points per correct answer; explanations after each question
- Leaderboard stored in `%LOCALAPPDATA%\QuizWizard\`
- Tutorial screen and early quit (Esc) with partial score saved

## Requirements

- Python 3.10+
- Windows recommended (packaging targets Windows)

```bash
python -m pip install -r requirements.txt
```

## Run (development)

```bash
python scripts/migrate_questions.py   # once, if data/questions is empty
python main.py
```

## Tests

```bash
python -m pytest tests -q
```

## Build Windows executable (PyInstaller)

```bash
python -m PyInstaller pyinstaller.spec --noconfirm
```

Output: `dist/QuizWizard/QuizWizard.exe`

## Installer (Inno Setup)

1. Install [Inno Setup](https://jrsoftware.org/isinfo.php)
2. Build the exe with PyInstaller (above)
3. Open `installer/quiz_wizard.iss` and compile

Output: `dist/installer/QuizWizard-Setup-2.0.0.exe`

## Project layout

- `quiz_wizard/` — application package (UI, services, repositories)
- `data/questions/` — shipped JSON question banks
- `assets/` — tutorial and icons
- `legacy/` — Version 1 terminal sources and `.txt` banks
- `tests/` — unit tests for domain and data layers

## License

No license specified; add one if sharing publicly.
