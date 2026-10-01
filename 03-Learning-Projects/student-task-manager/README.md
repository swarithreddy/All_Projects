# Student Task Manager

A Flask web application for registering and signing in, managing dated and prioritized tasks, and filtering tasks by search text or priority. User credentials and tasks are stored in a local SQLite database.

> This is a learning project. The current hard-coded session secret and task mutation routes are not suitable for a public deployment.

## Features

- Register accounts and sign in with password hashes
- Create tasks with optional due dates and priority values
- Edit tasks and view a user's task list
- Search tasks by content and filter by priority
- Return the signed-in user's tasks from `GET /api/tasks`
- Persist users and tasks in `database.db`

## Technology Stack

| Area | Technology |
|------|------------|
| Web framework | Flask |
| Authentication session | Flask-Login |
| Password hashing | Werkzeug security helpers |
| Database | SQLite (`sqlite3`) |
| Templates and static assets | Jinja2, HTML, CSS, JavaScript |

## Project Structure

```text
student-task-manager/
├── app.py             # Routes, login, task operations, and schema setup
├── database.db        # Local SQLite database
├── requirements.txt   # Flask dependencies (Flask-Login is currently missing)
├── Procfile           # Process command for compatible hosting platforms
├── templates/         # Login, registration, task list, and edit pages
└── static/            # Browser-side styles and scripts
```

## Requirements and Installation

- Python 3
- Dependencies from `requirements.txt`
- Flask-Login, which is imported by the app but is not currently listed in `requirements.txt`

Create a virtual environment and install dependencies from the project directory:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install Flask-Login
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install Flask-Login
```

## Run Locally

Start the development server:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), register an account, and sign in to manage tasks. The app initializes its SQLite tables when `app.py` starts.

## Routes

| Route | Behavior |
|-------|----------|
| `/register` | Register a username and password |
| `/login` | Authenticate and create a login session |
| `/logout` | End the current session |
| `/` | List, create, search, and filter the signed-in user's tasks |
| `/edit/<id>` | Edit a task belonging to the signed-in user |
| `/api/tasks` | Return the signed-in user's tasks as JSON |
| `/delete/<id>` | Delete a task |
| `/complete/<id>` | Mark a task complete |

## Data

The app creates `users` and `tasks` tables in `database.db` at startup. Passwords are hashed with Werkzeug before being stored. The SQLite connection uses a relative filename, so run the app from the project directory to use the expected database file.

## Security and Limitations

- `app.secret_key` is hard-coded. Replace it with a strong secret loaded from the environment before deployment.
- `/delete/<id>` and `/complete/<id>` do not require login and do not verify task ownership. Do not expose this app to untrusted users until those routes are protected and scoped to the signed-in user.
- State-changing actions use GET routes for delete and complete; production code should use authenticated POST/DELETE requests and CSRF protection.
- The app binds to `0.0.0.0`; use only in a trusted local environment until production hosting and security settings are configured.
- `requirements.txt` does not currently include Flask-Login, which is required by `app.py`.
- No automated tests are configured.
