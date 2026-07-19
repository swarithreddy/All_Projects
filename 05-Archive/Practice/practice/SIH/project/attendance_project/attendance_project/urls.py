from django.urls import path
from . import views

urlpatterns = [
    path('faculty_login/', views.faculty_login, name='faculty_login'),
    path('select_attendance/', views.select_attendance, name='select_attendance'),
    path('attendance_page/', views.attendance_page, name='attendance_page'),
    path('success/', views.success, name='success'),
]
