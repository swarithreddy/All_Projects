from flask import Flask, render_template, request, redirect
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

import sqlite3

app = Flask(__name__)

app.secret_key = "supersecretkey"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if user:
        return User(user["id"], user["username"], user["password"])
    return None

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        conn = get_db_connection()
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return redirect("/login")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            login_user(User(user["id"], user["username"], user["password"]))
            return redirect("/")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

@app.route("/edit/<int:id>", methods=["GET", "POST"])
@login_required
def edit(id):
    conn = get_db_connection()

    if request.method == "POST":
        content = request.form["task"]
        due_date = request.form["due_date"]
        priority = request.form["priority"]

        conn.execute("""
            UPDATE tasks
            SET content=?, due_date=?, priority=?
            WHERE id=? AND user_id=?
        """, (content, due_date, priority, id, current_user.id))
        conn.commit()
        conn.close()
        return redirect("/")

    task = conn.execute("SELECT * FROM tasks WHERE id=? AND user_id=?",
                        (id, current_user.id)).fetchone()
    conn.close()

    return render_template("edit.html", task=task)


def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

# Create table if not exists
def init_db():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            completed INTEGER DEFAULT 0,
            due_date TEXT,
            priority TEXT,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    conn.commit()
    conn.close()




init_db()

@app.route("/api/tasks")
@login_required
def api_tasks():
    conn = get_db_connection()
    tasks = conn.execute(
        "SELECT * FROM tasks WHERE user_id=?",
        (current_user.id,)
    ).fetchall()
    conn.close()

    return {"tasks": [dict(t) for t in tasks]}


@app.route("/delete/<int:id>")
def delete(id):
    conn = get_db_connection()
    conn.execute("DELETE FROM tasks WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return redirect("/")

@app.route("/complete/<int:id>")
def complete(id):
    conn = get_db_connection()
    conn.execute(
        "UPDATE tasks SET completed = 1 WHERE id = ?", (id,)
    )
    conn.commit()
    conn.close()
    return redirect("/")

@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    conn = get_db_connection()

    if request.method == "POST":
        task = request.form.get("task")
        due_date = request.form.get("due_date")
        priority = request.form.get("priority")

        conn.execute(
            "INSERT INTO tasks (content, due_date, priority, user_id) VALUES (?, ?, ?, ?)",
            (task, due_date, priority, current_user.id),
        )
        conn.commit()


    search = request.args.get("search", "")
    filter_priority = request.args.get("filter", "")

    query = "SELECT * FROM tasks WHERE user_id=?"
    params = [current_user.id]

    if search:
        query += " AND content LIKE ?"
        params.append(f"%{search}%")

    if filter_priority:
        query += " AND priority=?"
        params.append(filter_priority)

    tasks = conn.execute(query, params).fetchall()
    conn.close()

    return render_template("index.html", tasks=tasks)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
