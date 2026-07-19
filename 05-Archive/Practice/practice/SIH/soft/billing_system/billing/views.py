from django.shortcuts import render, redirect
import pywhatkit as kit
import qrcode
from io import BytesIO
import os

def billing_view(request):
    if request.method == 'POST':
        # Retrieve form data
        customer_name = request.POST.get('customer_name')
        phone_number = request.POST.get('phone_number')
        date_p = request.POST.get('date_p')
        n_o_i = request.POST.get('n_o_i')

        # Validate input data
        if not customer_name or not phone_number or not date_p or not n_o_i:
            return render(request, 'billing/billing_form.html', {
                'error_message': 'Please fill out all fields.'
            })

        try:
            n_o_i = int(n_o_i)
        except ValueError:
            return render(request, 'billing/billing_form.html', {
                'error_message': 'Number of items must be an integer.'
            })

        items = []
        total_amount = 0

        for i in range(1, n_o_i + 1):
            item_name = request.POST.get(f'item_name_{i}')
            quantity = request.POST.get(f'quantity_{i}')
            price_per_unit = request.POST.get(f'price_per_unit_{i}')

            if not item_name or not quantity or not price_per_unit:
                return render(request, 'billing/billing_form.html', {
                    'error_message': f'Incomplete data for item {i}.'
                })

            try:
                quantity = int(quantity)
                price_per_unit = float(price_per_unit)
            except ValueError:
                return render(request, 'billing/billing_form.html', {
                    'error_message': f'Invalid quantity or price for item {i}.'
                })

            total_cost = quantity * price_per_unit
            total_amount += total_cost

            items.append({
                'item_name': item_name,
                'quantity': quantity,
                'price_per_unit': price_per_unit,
                'total_cost': total_cost
            })

        # Generate the billing message
        bill_message = f"Hi {customer_name},\n\nHere is your bill:\n"
        bill_message += "Item Name\tQuantity\tPrice-Per Unit\tTotal Cost\n"
        for item in items:
            bill_message += f"{item['item_name']}\t\t{item['quantity']}\t\t{item['price_per_unit']:.2f}\t\t{item['total_cost']:.2f}\n"
        bill_message += f"\nTotal Bill Amount: {total_amount:.2f}\n\nThank you for shopping with us!"

        # Ensure phone number is in the correct format
        if not phone_number.startswith('+91'):
            phone_number = '+91' + phone_number.strip().lstrip('0')
        if len(phone_number) != 13:
            return render(request, 'billing/billing_form.html', {
                'error_message': 'Invalid phone number. Please enter a valid Indian phone number with 10 digits.'
            })

        # Send WhatsApp message with the bill details
        try:
            kit.sendwhatmsg_instantly(
                phone_number,
                bill_message,
                10  # Send the message after 10 seconds
            )
        except Exception as e:
            return render(request, 'billing/billing_form.html', {
                'error_message': f'Failed to send WhatsApp message: {e}'
            })

        return redirect('success')

    return render(request, 'billing/billing_form.html')

def success_view(request):
    return render(request, 'billing/success.html')
