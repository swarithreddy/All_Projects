# Water Reminder

A small Python script that sends a desktop notification, waits one hour, and repeats while the process is running.

## Requirements

- Python 3
- `plyer`
- Desktop notifications enabled in the operating system

Install the dependency:

```bash
python -m pip install plyer
```

## Run

```bash
python main.py
```

Keep the process running to receive reminders. Stop it with `Ctrl+C` in the terminal.

## Customize

Edit the notification title, message, and `time.sleep(3600)` interval in `main.py`. The current title and message are fixed in the source; the script does not provide a settings interface.

## Limitations

- This is a foreground Python process, not a configured background service.
- The script sends its first notification immediately, then waits an hour before the next one.