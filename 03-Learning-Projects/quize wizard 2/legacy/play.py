from type_choice import choose_quiz_type
from difficulty_choice import choose_difficulty
import main_operation 
import add_data
import os
import shutil

def filename(selected_quiz,selected_difficulty):
    file_name=""
    if selected_quiz == 'General Knowledge':
        file_name=file_name+'gen'
        if selected_difficulty == 'Easy':
            file_name=file_name+'e'
        elif selected_difficulty == 'Medium':
            file_name=file_name+'m'
        elif selected_difficulty == 'Hard':
            file_name=file_name+'h'
        else:
            file_name=file_name+'e'
    elif selected_quiz == 'Technical':
        file_name=file_name+'tech'
        if selected_difficulty == 'Easy':
            file_name=file_name+'e'
        elif selected_difficulty == 'Medium':
            file_name=file_name+'m'
        elif selected_difficulty == 'Hard':
            file_name=file_name+'h'
        else:
            file_name=file_name+'e'
    elif selected_quiz == 'Geopolitical':
        file_name=file_name+'geo'
        if selected_difficulty == 'Easy':
            file_name=file_name+'e'
        elif selected_difficulty == 'Medium':
            file_name=file_name+'m'
        elif selected_difficulty == 'Hard':
            file_name=file_name+'h'
        else:
            file_name=file_name+'e'
    return file_name

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def center_text(text):
    terminal_width = shutil.get_terminal_size().columns
    return text.center(terminal_width)

def get_player_info():
    clear_screen()
    
    # Get terminal size
    terminal_size = shutil.get_terminal_size()
    terminal_width = terminal_size.columns
    
    # Calculate vertical padding
    vertical_padding = "\n" * (shutil.get_terminal_size().lines // 2 - 3)
    
    # Print vertical padding
    print(vertical_padding)
    
    # Center the input prompts and values
    name = input("Enter player's name: ".center(terminal_width))
    age = input("Enter player's age: ".center(terminal_width))
    
    input("Press any key to continue...".center(terminal_width))
    
    clear_screen()
    
    # Display the entered name and age in the center of the screen
    print(vertical_padding)
    print(center_text(f"Player Name: {name}"))
    print(center_text(f"Player Age: {age}"))
    input("Press any key to continue...".center(terminal_width))
    return name, age

def main():
    name, age = get_player_info()  #taking player data
    clear_screen()
    selected_quiz = choose_quiz_type()   # taking quize type data
    

    clear_screen()

    selected_difficulty = choose_difficulty()   # taking difficulty data
    

    
    fname=filename(selected_quiz,selected_difficulty)   # making file name
    fname=fname+'.txt'
    

    if selected_difficulty == 'Auto':
        import auto
        score=auto.main(name, age, selected_quiz, selected_difficulty, fname)
        add_data.add_player_data('leaderboard.txt',name,age,score)   # adding data to the leaderboard after the game and arranging then in order
    else:
        score=main_operation.main(name,age,selected_quiz,selected_difficulty,fname)  #taking score after gameplay
        add_data.add_player_data('leaderboard.txt',name,age,score)   # adding data to the leaderboard after the game and arranging then in order