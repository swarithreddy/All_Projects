import msvcrt
import os

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def center_text(text):
    terminal_width = os.get_terminal_size().columns
    return text.center(terminal_width)

def display_header(selected_quiz, selected_difficulty, score):
    clear_terminal()
    print(center_text(selected_quiz))
    print(center_text(selected_difficulty))
    print(f"Score: {score}\n")

def read_questions(fname):
    with open(fname, 'r') as file:
        lines = file.readlines()
    questions = []
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():
            question = {
                'number': lines[i].strip(),
                'question': lines[i+1].strip(),
                'options': [lines[i+2][3:].strip(), lines[i+3][3:].strip(), lines[i+4][3:].strip(), lines[i+5][3:].strip()],
                'answer': lines[i+6].strip(),
                'explanation': lines[i+7].strip()
            }
            questions.append(question)
            i += 8
        else:
            i += 1
    return questions

def wait_for_key():
    print(center_text("\nPress 'Esc' to go to the home page or any other key to continue..."))
    while True:
        key = msvcrt.getch()
        if key == b'\x1b':  # ESC key
            return 'esc'
        else:
            return 'continue'

def main(name, age, selected_quiz, selected_difficulty, fname):
    score = 0
    questions = read_questions(fname)

    for question in questions:
        display_header(selected_quiz, selected_difficulty, score)
        print(f"Question {question['number']}: {question['question']}")
        for idx, option in enumerate(question['options'], 1):
            print(f"{idx}. {option}")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice == question['answer']:
            print("\nCorrect!")
            score += 10
        else:
            print("\nWrong!")
        
        print(f"Explanation: {question['explanation']}")
        if wait_for_key() == 'esc':
            break

    clear_terminal()
    print(center_text(selected_quiz))
    print(center_text(selected_difficulty))
    print(center_text(name))
    print(center_text(age))
    print(center_text("Quiz Completed!"))
    print(center_text(f"Your final score is: {score}"))
    wait_for_key()  # Wait for user input before clearing the terminal
    clear_terminal()
    return score

