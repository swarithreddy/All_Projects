import sqlite3 as s
def get_db_connection():
    conn = s.connect("attandance.db")
    conn.row_factory = s.Row
    return conn



def insert_student(rollno, year, branch, section):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO students (rollno, year, branch, section)
        VALUES (?, ?, ?, ?)
    ''', (rollno, year, branch, section))
    conn.commit()
    conn.close()
insert_student(1, 2, 'cse', 'c')