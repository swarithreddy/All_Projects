import sqlite3

def create_database():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # Create a table for faculty login
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS faculty (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        login_id TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')

    # Create a table for students
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        rollno INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        year INTEGER,
        branch TEXT,
        section TEXT
    )
    ''')

    # Create a table for attendance
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        rollno INTEGER,
        date TEXT,
        forenoon INTEGER,
        afternoon INTEGER,
        FOREIGN KEY (rollno) REFERENCES students(rollno)
    )
    ''')

    conn.commit()
    conn.close()

create_database()
