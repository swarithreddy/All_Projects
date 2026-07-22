@echo off
REM Build Quiz Wizard V2 (exe + optional installer)
cd /d "%~dp0.."
python -m pip install -r requirements.txt
python scripts\migrate_questions.py
python -m pytest tests -q
python -m PyInstaller pyinstaller.spec --noconfirm
echo.
echo Exe: dist\QuizWizard\QuizWizard.exe
echo Compile installer with Inno Setup: installer\quiz_wizard.iss
pause
