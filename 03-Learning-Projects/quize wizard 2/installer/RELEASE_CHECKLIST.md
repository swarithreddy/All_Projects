# Quiz Wizard 2.0.0 — Release checklist

## Pre-release

- [ ] `python -m pytest tests -q` passes
- [ ] Manual play: Easy / Medium / Hard / Auto for one category
- [ ] Esc quit saves partial score
- [ ] Leaderboard shows spaced names and survives restart
- [ ] Tutorial opens; Exit closes app
- [ ] Fault injection: rename a JSON bank → friendly error dialog

## Build

```bat
python -m pip install -r requirements.txt
python scripts/migrate_questions.py
python -m PyInstaller pyinstaller.spec --noconfirm
```

## Installer

1. Open `installer/quiz_wizard.iss` in Inno Setup Compiler
2. Build → `dist/installer/QuizWizard-Setup-2.0.0.exe`

## Clean install smoke test

- [ ] Install under a fresh Windows user (or clear AppData QuizWizard)
- [ ] Start from Start Menu
- [ ] Complete one quiz; confirm `%LOCALAPPDATA%\QuizWizard\leaderboard.json`
- [ ] Uninstall; confirm program files gone; AppData scores remain
