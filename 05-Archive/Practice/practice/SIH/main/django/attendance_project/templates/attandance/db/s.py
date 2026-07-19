import sqlite3 as s
def get_db_connection():
    conn = s.connect("attandance.db")
    conn.row_factory = s.Row
    return conn



def insert_faculty(login_id, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO faculty (login_id, password)
        VALUES (?, ?)
    ''', (login_id, password))
    conn.commit()
    conn.close()
insert_faculty(1,123)