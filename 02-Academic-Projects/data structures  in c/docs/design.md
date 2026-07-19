Design & Algorithm

This document describes the underlying data structures and algorithm used by
Task Tactics.  It is a straight transcription of the original project report.

To‑Do List

- Structure: Singly linked list of tasks.
- Node:
  ```c
  typedef struct Task {
      int id;
      char desc[256];
      struct Task *next;
  } Task;
  ```
- Add Task: Create a new node with unique ID, append to end of list.
- Delete Task: Search by ID, unlink the node, and free memory.
- Display Tasks: Iterate through the list, printing each `id.desc` pair.
- Memory Management: On exit, traverse the list freeing every node.

Tic‑Tac‑Toe Game

- Board: 3×3 `char board[3][3]`, initialized to spaces `' '`. 
- Display: Print row/column numbers and cell contents with dividers.
- Player Input: Prompt for row and column, validate range (0–2) and that the
  cell is empty.
- Update: Place `'X'` or `'O'` in the chosen cell.
- Win/Draw Checks:
  - Win: Three identical non‑space symbols in any row, column or either diagonal.
  - Draw: All cells filled and no winner.
- Loop: Alternate players until a win or draw is detected.

Main Menu Flow

1. Show main menu:
   - 1. To‑Do List
   - 2. Play Tic‑Tac‑Toe
   - 3. Exit
2. Based on choice, enter the corresponding subroutine.  Return to main menu
   after completing a sub‑task.

To‑Do List Menu
```
1. Add Task
2. Delete Task
3. Display Tasks
4. Back to Main Menu
```

Notes
- All user input is read from standard input.
- The program is single‑file C and compiles cleanly with `-Wall`.
