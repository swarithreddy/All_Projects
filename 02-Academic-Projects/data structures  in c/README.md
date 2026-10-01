# Task Tactics

A terminal-based C program with two menu-driven activities: a linked-list to-do list and a two-player Tic-Tac-Toe game.

## Features

- Add, list, and delete in-memory tasks by generated ID
- Play local two-player Tic-Tac-Toe with win and draw detection
- Free allocated task nodes when the program exits

## Technology

- C and the standard library
- Singly linked list for task storage
- Console input/output

## Project Structure

```text
data structures  in c/
├── docs/
│   └── design.md
├── src/
│   └── tasktactics.c
├── LICENSE
└── README.md
```

## Requirements

- A C compiler such as GCC (MinGW-w64 on Windows)
- A terminal

## Build and Run

Run these commands from the project directory:

```bash
gcc src/tasktactics.c -o tasktactics
```

On Windows, run `tasktactics.exe`; on macOS or Linux, run:

```bash
./tasktactics
```

Use the numbered menu. Tic-Tac-Toe rows and columns are numbered `0` through `2`.

## Notes

- Tasks exist only in memory and are lost when the program exits.
- The game supports two local players and has no computer opponent.
- See [docs/design.md](docs/design.md) for the design notes.

## License

See [LICENSE](LICENSE).
