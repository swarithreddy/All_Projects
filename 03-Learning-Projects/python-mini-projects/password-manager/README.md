# Password Manager Demo

A small terminal program that appends website/password pairs to `passwords.txt` and can copy a matching password to the clipboard.

## Requirements

- Python 3
- `pyperclip`

Install the dependency:

```bash
python -m pip install pyperclip
```

## Run

```bash
python main.py
```

Choose **Save Password**, **Get Password**, or **Exit**. Saved records are stored in `passwords.txt` in the current working directory. A password must be saved before the lookup option can read that file.

## Security Warning

This is an educational example, not a secure password manager. It stores passwords as plaintext, has no encryption or master password, and performs a simple substring lookup. Do not store real credentials in it. Use a reputable password manager for personal accounts.