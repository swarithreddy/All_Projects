# billing/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.billing_view, name='billing'),
    path('success/', views.success_view, name='success'),
]
