import msvcrt
import os

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def center_text(text):
    terminal_width = os.get_terminal_size().columns
    return text.center(terminal_width)

def display_header(selected_quiz, selected_difficulty, score, results):
    clear_terminal()
    print(center_text(selected_quiz))
    print(center_text(selected_difficulty))
    print(f"Score: {score}\n")
    print(center_text("Last 3 Results: " + " ".join(map(str, results[-3:]))))

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

def adjust_difficulty(current_difficulty, correct_streak, wrong_streak):
    if correct_streak >= 3:
        if current_difficulty == 'e':
            return 'm'
        elif current_difficulty == 'm':
            return 'h'
    elif wrong_streak >= 3:
        if current_difficulty == 'h':
            return 'm'
        elif current_difficulty == 'm':
            return 'e'
    return current_difficulty

def main(name, age, selected_quiz, selected_difficulty, fname):
    score = 0
    correct_streak = 0
    wrong_streak = 0
    results = []
    current_difficulty = fname[-5]  # Get the current difficulty level from the filename
    question_index = 0

    while True:
        questions = read_questions(fname)
        while question_index < len(questions):
            question = questions[question_index]
            display_header(selected_quiz, selected_difficulty, score, results)
            print(f"Question {question['number']}: {question['question']}")
            for idx, option in enumerate(question['options'], 1):
                print(f"{idx}. {option}")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            if choice == question['answer']:
                print("\nCorrect!")
                score += 10
                correct_streak += 1
                wrong_streak = 0
                results.append(1)
            else:
                print("\nWrong!")
                correct_streak = 0
                wrong_streak += 1
                results.append(0)
            
            print(f"Explanation: {question['explanation']}")
            if wait_for_key() == 'esc':
                break

            question_index += 1

            # Adjust difficulty if needed
            new_difficulty = adjust_difficulty(current_difficulty, correct_streak, wrong_streak)
            if new_difficulty != current_difficulty:
                current_difficulty = new_difficulty
                fname = fname[:-5] + current_difficulty + ".txt"
                selected_difficulty = {"e": "Easy", "m": "Medium", "h": "Hard"}[current_difficulty]
                break  # Break to reload questions with new difficulty

        # If all questions are answered or ESC is pressed, break the outer loop
        if question_index >= len(questions) or wait_for_key() == 'esc':
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


