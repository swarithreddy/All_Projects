from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from .models import Student, Attendance
from .forms import FacultyLoginForm

def faculty_login(request):
    if request.method == 'POST':
        form = FacultyLoginForm(request, data=request.POST)
        if form.is_valid():
            login_id = form.cleaned_data.get('login_id')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=login_id, password=password)
            if user is not None:
                login(request, user)
                return redirect('select_attendance')
            else:
                return render(request, 'faculty_login.html', {'form': form, 'error': 'Invalid login credentials'})
    else:
        form = FacultyLoginForm()
    return render(request, 'faculty_login.html', {'form': form})

def select_attendance(request):
    if request.method == 'POST':
        year = request.POST['year']
        branch = request.POST['branch']
        section = request.POST['section']
        return redirect('attendance_page', year=year, branch=branch, section=section)
    return render(request, 'select_attendance.html')

def attendance_page(request, year, branch, section):
    students = Student.objects.filter(year=year, branch=branch, section=section)
    if request.method == 'POST':
        for student in students:
            rollno = student.rollno
            forenoon = 'forenoon_{}'.format(rollno) in request.POST
            afternoon = 'afternoon_{}'.format(rollno) in request.POST
            Attendance.objects.create(rollno=student, forenoon=forenoon, afternoon=afternoon)
        return redirect('success')
    return render(request, 'attendance_page.html', {'students': students})

def success(request):
    return render(request, 'success.html')
