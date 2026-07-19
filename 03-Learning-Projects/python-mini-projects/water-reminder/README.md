# Water Reminder

A desktop notification app that reminds you to drink water at regular intervals.

\#\# Features

- Sends desktop notifications every hour
- Customizable reminder message
- Runs continuously in the background
- Uses system notifications

\#\# Requirements

- Python 3.x
- `plyer` library for notifications

Install dependencies:
```
pip install plyer
```

\#\# How to Run

1. Run `python main.py`
2. The app will start sending notifications every hour
3. Keep the terminal window open to continue receiving reminders

\#\# Customization

- Change the reminder interval by modifying `time.sleep(3600)` (3600 seconds = 1 hour)
- Modify the notification title and message in the `notification.notify()` call
- For testing, uncomment the `time.sleep(3)` line to get reminders every 3 seconds

\#\# Note

The app uses the `plyer` library which supports notifications on Windows, macOS, and Linux. Make sure your system allows notifications from Python applications.