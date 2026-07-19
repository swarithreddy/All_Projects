Task Tactics

A console application combining a To‑Do List and Tic‑Tac‑Toe game.

Originally authored by R. Swarith Reddy (23911A05J5) as part of a data
structures course project, Task Tactics is written in plain C and demonstrates
the use of linked lists and simple game logic.

🚀 Features

- ✅ To‑Do List
  - Add tasks with unique IDs and descriptions
  - Delete tasks by ID
  - Display current task list
  - Tasks are stored in a singly linked list and freed on exit

- 🎮 Tic‑Tac‑Toe Game
  - Two‑player local game played in the terminal
  - Automatic win/draw detection
  - Board updates after every move

🛠️ Getting Started

Prerequisites

- A C compiler such as `gcc` (MinGW, TDM‑GCC, or similar) installed and on your
  `PATH`.
- A terminal (Command Prompt, PowerShell, Git Bash, etc.).

Building

```sh
cd "c:/Users/swarith reddy/OneDrive/Desktop/gtidemo/All_Projects/data structures  in c/"
gcc src/tasktactics.c -o tasktactics
```

Running

```sh
./tasktactics
```

Follow the on‑screen menus to manage your tasks or play Tic‑Tac‑Toe.

📂 Repository Structure

```
data structures  in c/
├── .gitignore
├── LICENSE
├── README.md
├── docs/
│   └── design.md         <- high‑level algorithm description
└── src/
    └── tasktactics.c     <- main source code
```

📄 Additional Documentation

See `docs/design.md` for the full algorithm breakdown that was originally
provided in the Word document.

🤝 Contributing

Feel free to open issues or submit pull requests if you'd like to extend or
refactor the code (e.g. add persistence, a GUI, or AI opponent).

📜 License

This project is released under the [MIT License](LICENSE).
