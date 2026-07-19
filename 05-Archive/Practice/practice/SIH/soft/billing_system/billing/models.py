# billing/models.py

from django.db import models

class Billing(models.Model):
    username = models.CharField(max_length=100)
    phone_number = models.CharField(max_length=15)
    item_quantity = models.IntegerField()
    item_id = models.CharField(max_length=20)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.username} - {self.item_id}'
