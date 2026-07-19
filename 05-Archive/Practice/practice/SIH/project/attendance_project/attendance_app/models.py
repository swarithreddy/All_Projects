from django.db import models
from django.contrib.auth.models import User

class Faculty(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    login_id = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=100)

class Student(models.Model):
    rollno = models.CharField(max_length=10, unique=True)
    year = models.IntegerField()
    branch = models.CharField(max_length=100)
    section = models.CharField(max_length=10)

class Attendance(models.Model):
    rollno = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    forenoon = models.BooleanField(default=False)
    afternoon = models.BooleanField(default=False)
