from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
import sqlite3 as s
from django.views.decorators.csrf import csrf_protect

def get_db_connection():
    conn = s.connect('attandance.py')
    conn.row_factory = s.Row
    return conn

def faculty_login(request):
    if request.method == 'POST':
        login_id = request.POST['login_id']
        password = request.POST['password']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM faculty WHERE login_id = ? AND password = ?
        ''', (login_id, password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            # Assuming you have a session management system
            request.session['user_id'] = user['id']
            return redirect('select_attendance')
        else:
            return render(request, 'faculty_login.html', {'error': 'Login Failed. Please try again.'})
    return render(request, 'faculty_login.html')

def select_attendance(request):
    if request.method == 'POST':
        year = request.POST['year']
        branch = request.POST['branch']
        section = request.POST['section']
        
        # Assuming you have a URL pattern named 'attendance_page' that accepts year, branch, and section as parameters
        return redirect('attendance_page', year=year, branch=branch, section=section)
    
    return render(request, 'select_attendance.html')


def select_attendance(request):
    if request.method == 'POST':
        year = request.POST['year']
        branch = request.POST['branch']
        section = request.POST['section']
        
        # Assuming you have a URL pattern named 'attendance_page' that accepts year, branch, and section as parameters
        return redirect('attendance_page', year=year, branch=branch, section=section)
    
    return render(request, 'select_attendance.html')

def insert_attendance(rollno, date, forenoon, afternoon):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO attendance (rollno, date, forenoon, afternoon)
        VALUES (?, ?, ?, ?)
    ''', (rollno, date, forenoon, afternoon))
    conn.commit()
    conn.close()

def insert_attendance_view(request):
    if request.method == 'POST':
        rollno = request.POST['rollno']
        date = request.POST['date']
        forenoon = int(request.POST['forenoon'])
        afternoon = int(request.POST['afternoon'])
        
        insert_attendance(rollno, date, forenoon, afternoon)
        return redirect('attendance_success')  # Redirect to a success page or another view
    
    return render(request, 'attendance_page.html')
@csrf_protect
def insert_attendance_view(request):
    if request.method == 'POST':
        rollno = request.POST['rollno']
        date = request.POST['date']
        forenoon = int(request.POST['forenoon'])
        afternoon = int(request.POST['afternoon'])
        
        insert_attendance(rollno, date, forenoon, afternoon)
        return redirect('attendance_success')  # Redirect to a success page or another view
    
    return render(request, 'attendance_form.html')
def success(request):
    return render(request, 'success.html')
