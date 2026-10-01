# File Organizer

A Python script that moves files from the current working directory into folders based on a fixed set of extensions.

## Requirements

- Python 3
- No third-party packages; uses `os` and `shutil`

## Run Safely

The script organizes the directory returned by `os.getcwd()`. First make a backup, then open a terminal in the directory you intend to organize and run the script by its full path:

```bash
python "C:/path/to/file-organizer/main.py"
```

It creates category folders and moves matching files. It has no preview, undo, or collision-management interface. Do not run it in a project, downloads, or personal folder unless you want the files moved.

## File Categories

| Folder | Extensions |
|--------|------------|
| `Images` | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp` |
| `Documents` | `.pdf`, `.docx`, `.doc`, `.txt`, `.xlsx`, `.pptx`, `.md` |
| `Audio` | `.mp3`, `.wav`, `.aac`, `.flac` |
| `Videos` | `.mp4`, `.avi`, `.mov`, `.mkv` |
| `Archives` | `.zip`, `.rar`, `.tar`, `.gz` |
| `Scripts` | `.js`, `.sh`, `.bat` |

Files with other extensions are left in place. Existing directories are skipped.