# Python Mini Projects

A collection of small, independent Python command-line and desktop projects. Each project has its own source file and README with project-specific setup instructions.

## Projects

| Project | Description | Dependencies |
|---------|-------------|--------------|
| [File Organizer](file-organizer/) | Moves files in the current directory into extension-based folders | Python standard library |
| [Password Manager](password-manager/) | Saves and retrieves website/password pairs in a local text file | `pyperclip` |
| [PDF Merger](pdf-merger/) | Tkinter interface for combining selected PDF files | `pypdf`, Tkinter |
| [Quiz App](quiz-app/) | Four-question multiple-choice terminal quiz | Python standard library |
| [Typing Speed Test](typing-speed-test/) | Measures typing time, approximate WPM, and character accuracy | Python standard library |
| [Water Reminder](water-reminder/) | Sends a desktop notification every hour | `plyer` |

## Run a Project

Use Python 3 and follow the README inside the project you want to run. For example:

```bash
cd water-reminder
python -m pip install plyer
python main.py
```

The `file-organizer` and `password-manager` projects modify local files. Read their project-specific safety notes before running them on important data.