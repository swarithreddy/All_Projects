def choose_quiz_type():
    quiz_types = {
        "1": "General Knowledge",
        "2": "Technical",
        "3": "Geopolitical",
        "4": "Exit"
    }

    print("Available Quiz Types:")
    for key, value in quiz_types.items():
        print(f"{key}. {value}")
    while True:
        choice = input("Choose a quiz type by entering the corresponding number: ")

        if choice in quiz_types and choice is not '4':
            return quiz_types[choice]
        elif choice is '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
            
