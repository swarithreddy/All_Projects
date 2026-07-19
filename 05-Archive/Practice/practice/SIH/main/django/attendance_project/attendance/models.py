from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager

class FacultyManager(BaseUserManager):
    def create_user(self, login_id, password=None):
        if not login_id:
            raise ValueError('Faculty must have a login ID')
        user = self.model(login_id=login_id)
        user.set_password(password)
        user.save(using=self._db)
        return user

class Faculty(AbstractBaseUser):
    login_id = models.CharField(max_length=255, unique=True)
    password = models.CharField(max_length=255)

    objects = FacultyManager()

    USERNAME_FIELD = 'login_id'

class Student(models.Model):
    rollno = models.CharField(max_length=255, unique=True)
    year = models.IntegerField()
    branch = models.CharField(max_length=255)
    section = models.CharField(max_length=255)

class Attendance(models.Model):
    rollno = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    forenoon = models.BooleanField(default=False)
    afternoon = models.BooleanField(default=False)
