from django.urls import path
from . import views

urlpatterns = [
    path('faculty_login/', views.faculty_login, name='faculty_login'),
    path('select_attendance/', views.select_attendance, name='select_attendance'),
    path('insert_attendance/', views.insert_attendance_view, name='insert_attendance'),
    path('success/', views.success, name='success'),    
    path('insert_attendance/', views.insert_attendance_view, name='insert_attendance'),
]
