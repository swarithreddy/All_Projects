def add_player_data(file_name, name, age, score):
    # Read existing data
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        lines = []

    # Add new player data
    sno = len(lines) + 1
    new_line = f"{sno} {name} {age} {score}\n"
    lines.append(new_line)

    # Sort data by score in decreasing order
    lines.sort(key=lambda x: int(x.split()[3]), reverse=True)

    # Update serial numbers
    for i, line in enumerate(lines):
        parts = line.split()
        parts[0] = str(i + 1)
        lines[i] = ' '.join(parts) + '\n'

    # Write sorted data back to file
    with open(file_name, 'w') as file:
        file.writelines(lines)


def display_leaderboard(file_name):
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            for line in lines:
                print(line.strip())
    except FileNotFoundError:
        print(f"The file {file_name} does not exist.")
    
    
    input("Press any key to continue...")
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

