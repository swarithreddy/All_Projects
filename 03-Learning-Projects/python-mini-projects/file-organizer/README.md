# File Organizer

A simple Python script to organize files in a directory based on their file extensions.

\#\# Features

- Automatically creates folders for different file types (Images, Documents, Audio, Videos, Archives, Scripts)
- Moves files into appropriate folders based on their extensions
- Skips existing folders to avoid moving directories

\#\# Requirements

- Python 3.x
- Standard library modules: `os`, `shutil`

\#\# How to Run

1. Place the script in the directory you want to organize
2. Run `python main.py`
3. The script will organize all files in the current directory

\#\# Supported File Types

- \*\*Images\*\*: .jpg, .jpeg, .png, .gif, .bmp, .webp
- \*\*Documents\*\*: .pdf, .docx, .doc, .txt, .xlsx, .pptx, .md
- \*\*Audio\*\*: .mp3, .wav, .aac, .flac
- \*\*Videos\*\*: .mp4, .avi, .mov, .mkv
- \*\*Archives\*\*: .zip, .rar, .tar, .gz
- \*\*Scripts\*\*: .js, .sh, .bat

\#\# Note

The script organizes files in the current working directory. Make sure to run it from the correct location.