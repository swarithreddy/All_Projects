import sqlite3

DATABASE = 'attendance.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def insert_student(rollno, year, branch, section):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO students (rollno, year, branch, section) VALUES (?, ?, ?, ?)', (rollno, year, branch, section))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    rollno = input("Enter Student Roll Number: ")
    year = input("Enter Student Year: ")
    branch = input("Enter Student Branch: ")
    section = input("Enter Student Section: ")
    insert_student(rollno, year, branch, section)
    print("Student details inserted successfully.")
