import sqlite3 as s

def get_db_connection():
    conn = s.connect("attandance.db")
    conn.row_factory = s.Row
    return conn

def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faculty (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            login_id TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rollno TEXT NOT NULL UNIQUE,
            year INTEGER NOT NULL,
            branch TEXT NOT NULL,
            section TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rollno TEXT NOT NULL,
            date TEXT NOT NULL,
            forenoon INTEGER NOT NULL,
            afternoon INTEGER NOT NULL,
            FOREIGN KEY (rollno) REFERENCES students (rollno)
        )
    ''')
    conn.commit()
    conn.close()

create_tables()
