from django import forms
from django.contrib.auth.forms import AuthenticationForm

class FacultyLoginForm(AuthenticationForm):
    login_id = forms.CharField(max_length=100)
    password = forms.CharField(widget=forms.PasswordInput)
