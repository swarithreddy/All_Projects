from flask import Flask, render_template, request, redirect, url_for
import sqlite3
from werkzeug.security import check_password_hash  # Update if using password hashing

app = Flask(__name__)

DATABASE = 'attendance.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def check_faculty_login(login_id, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM faculty WHERE login_id = ?', (login_id,))
    faculty = cursor.fetchone()
    conn.close()
    
    if faculty and check_password_hash(faculty['password'], password):
        return True
    return False

def get_students(year, branch, section):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM students WHERE year = ? AND branch = ? AND section = ?', (year, branch, section))
    students = cursor.fetchall()
    conn.close()
    return students

@app.route('/faculty_login', methods=['GET', 'POST'])
def faculty_login():
    if request.method == 'POST':
        login_id = request.form['login_id']
        password = request.form['password']
        
        if check_faculty_login(login_id, password):
            return redirect(url_for('select_attendance'))
        else:
            return "Login Failed. Please try again."

    return render_template('faculty_login.html')

@app.route('/select_attendance', methods=['GET', 'POST'])
def select_attendance():
    if request.method == 'POST':
        year = request.form['year']
        branch = request.form['branch']
        section = request.form['section']
        return redirect(url_for('attendance_page', year=year, branch=branch, section=section))
    
    return render_template('select_attendance.html')

@app.route('/attendance_page', methods=['GET', 'POST'])
def attendance_page():
    year = request.args.get('year')
    branch = request.args.get('branch')
    section = request.args.get('section')

    students = get_students(year, branch, section)

    if request.method == 'POST':
        conn = get_db_connection()
        cursor = conn.cursor()

        for student in students:
            rollno = student['rollno']
            forenoon = 1 if request.form.get(f'forenoon_{rollno}') else 0
            afternoon = 1 if request.form.get(f'afternoon_{rollno}') else 0
            
            cursor.execute("INSERT INTO attendance (rollno, date, forenoon, afternoon) VALUES (?, date('now'), ?, ?)", (rollno, forenoon, afternoon))
        
        conn.commit()
        conn.close()
        return redirect(url_for('success'))

    return render_template('attendance_page.html', students=students)

@app.route('/success')
def success():
    return render_template('success.html')

if __name__ == '__main__':
    app.run(debug=True)
