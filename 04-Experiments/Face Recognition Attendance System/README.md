RAOP

A simple Flask-based attendance tracking application that appears to use face
images and QR code verification. The project includes a Python server script
(`app.py`), static assets, and HTML templates for login, scan and result pages.

Structure

```
RAOP/
├── app.py                <- main Flask application
├── attendance.xlsx       <- sample attendance spreadsheet
├── known_faces/          <- folder containing reference face images
├── static/
│   └── style.css         <- basic styling
├── templates1/           <- HTML views for various pages
└── tempCodeRunnerFile.py
```

Usage

Install Python dependencies (Flask, face-recognition, etc.) then run
`app.py`. Open the local web server in a browser to interact with the UI.

Git setup

This directory has been initialized as a Git repository. See `.gitignore` for
ignored files.

License

No license specified; feel free to add one before sharing.
