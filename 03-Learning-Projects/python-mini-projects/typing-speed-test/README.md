# Typing Speed Test

A command-line exercise that selects one of three built-in sentences, times a typing attempt, and reports approximate words per minute and character accuracy.

## Requirements

- Python 3
- No third-party packages

## Run

```bash
python main.py
```

Read the displayed sentence, press Enter when ready, then type the sentence and press Enter again.

## Scoring

- Elapsed time is measured around the typing input.
- WPM uses the number of words in the prompt divided by elapsed minutes; it does not count the words actually typed.
- Accuracy compares matching character positions against the prompt length. Extra characters beyond the prompt are not included in the comparison.

The three sample prompts are defined in `main.py` and can be edited there.