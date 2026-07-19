import sqlite3 as s
def get_db_connection():
    conn = s.connect("attandance.db")
    conn.row_factory = s.Row
    return conn




def insert_attendance(rollno, date, forenoon, afternoon):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO attendance (rollno, date, forenoon, afternoon)
        VALUES (?, ?, ?, ?)
    ''', (rollno, date, forenoon, afternoon))
    conn.commit()
    conn.close()
insert_attendance(1, '2024-12-12', 1, 1)