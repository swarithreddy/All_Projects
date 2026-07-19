import sqlite3

def create_database_and_tables(db_name):
    # Connect to SQLite database (it will create a new database if it doesn't exist)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create students table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        rollno INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        c1 TEXT,
        c2 TEXT,
        c3 TEXT,
        c4 TEXT,
        c5 TEXT,
        c6 TEXT,
        c7 TEXT,
        period TEXT,
        year INTEGER,
        branch TEXT,
        section TEXT
    )
    ''')

    # Create admins table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        password TEXT NOT NULL
    )
    ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# Example usage
create_database_and_tables('students_and_admins.db')
