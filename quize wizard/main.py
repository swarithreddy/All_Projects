def display_menu():
    print("Welcome to Quiz Wizard")
    print("======================")
    print("1. Play")
    print("2. Leaderboard")
    print("3. Tutorial")
    print("4. Exit")

def clear_screen():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def opentut():
    try:
        with open('tutorial.txt', 'r') as file:
            content = file.read()
            print(content)
    except FileNotFoundError:
        print("File 'tutorial' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    input("Press any key to continue...")
    clear_screen()

def main():
    while True:
        

        display_menu()
        choice = input("Choose an option: ")

        if choice == '1':
            print("Starting the quiz...")
            import play
            play.main()
        elif choice == '2':
            print("Showing the leaderboard...")
            import add_data
            add_data.display_leaderboard('leaderboard.txt')
        elif choice == '3':
            clear_screen()
            print("Showing the tutorial...")
            opentut()
        elif choice == '4':
            clear_screen()
            print("Exiting...")
            break
        else:
            clear_screen()
            print("Invalid choice. Please try again.")
        
main()
