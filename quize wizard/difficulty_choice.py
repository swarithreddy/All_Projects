def choose_difficulty():
    difficulty_levels = {
        "1": "Easy",
        "2": "Medium",
        "3": "Hard",
        "4": "Auto",
        "5": "Exit"
    }

    print("Available Difficulty Levels:")
    for key, value in difficulty_levels.items():
        print(f"{key}. {value}")
    while True:
        choice = input("Choose a difficulty level by entering the corresponding number: ")

        if choice in difficulty_levels and choice is not '5':
            return difficulty_levels[choice]
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")     