from __future__ import annotations


def validate_name(name: str) -> str | None:
    cleaned = name.strip()
    if not cleaned:
        return "Please enter a name."
    if len(cleaned) > 60:
        return "Name must be 60 characters or fewer."
    return None


def validate_age(age_text: str) -> tuple[int | None, str | None]:
    text = age_text.strip()
    if not text:
        return None, "Please enter an age."
    if not text.isdigit():
        return None, "Age must be a whole number."
    age = int(text)
    if age < 1 or age > 120:
        return None, "Age must be between 1 and 120."
    return age, None
