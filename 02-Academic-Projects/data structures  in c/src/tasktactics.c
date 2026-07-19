#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* linked list node for to-do items */
typedef struct Task {
    int id;
    char desc[256];
    struct Task *next;
} Task;

static Task *head = NULL;
static int nextId = 1;

/* prototypes */
void addTask(const char *desc);
void deleteTask(int id);
void displayTasks(void);
void freeTasks(void);

/* tic-tac-toe */
char board[3][3];

void initBoard(void);
void showBoard(void);
int checkWin(void);
int checkDraw(void);

int main(void) {
    int choice;
    while (1) {
        printf("\n=== TaskTactics Menu ===\n");
        printf("1. To-Do List\n");
        printf("2. Play Tic-Tac-Toe\n");
        printf("3. Exit\n");
        printf("Enter your choice: ");
        if (scanf("%d", &choice) != 1) break;
        getchar(); /* consume newline */
        switch (choice) {
            case 1: {
                int sub;
                while (1) {
                    printf("\n-- To-Do List --\n");
                    printf("1. Add Task\n");
                    printf("2. Delete Task\n");
                    printf("3. Display Tasks\n");
                    printf("4. Back to Main Menu\n");
                    printf("Enter your choice: ");
                    if (scanf("%d", &sub) != 1) break;
                    getchar();
                    if (sub == 1) {
                        char buf[256];
                        printf("Enter task description: ");
                        if (fgets(buf, sizeof(buf), stdin)) {
                            buf[strcspn(buf, "\n")] = '\0';
                            addTask(buf);
                        }
                    } else if (sub == 2) {
                        int id;
                        printf("Enter task ID to delete: ");
                        if (scanf("%d", &id) == 1) {
                            deleteTask(id);
                        }
                        getchar();
                    } else if (sub == 3) {
                        displayTasks();
                    } else if (sub == 4) {
                        break;
                    }
                }
                break;
            }
            case 2: {
                initBoard();
                char player = 'X';
                while (1) {
                    showBoard();
                    int r, c;
                    printf("Player %c, enter your move (row and column): ", player);
                    if (scanf("%d %d", &r, &c) != 2)
                        break;
                    getchar();
                    if (r < 0 || r > 2 || c < 0 || c > 2 || board[r][c] != ' ') {
                        printf("Invalid move\n");
                        continue;
                    }
                    board[r][c] = player;
                    if (checkWin()) {
                        showBoard();
                        printf("Player %c wins!\n", player);
                        break;
                    }
                    if (checkDraw()) {
                        showBoard();
                        printf("It's a draw!\n");
                        break;
                    }
                    player = (player == 'X' ? 'O' : 'X');
                }
                break;
            }
            case 3:
                freeTasks();
                return 0;
            default:
                printf("Invalid choice\n");
        }
    }
    freeTasks();
    return 0;
}

/* linked list functions */
void addTask(const char *desc) {
    Task *t = malloc(sizeof *t);
    if (!t) return;
    t->id = nextId++;
    strncpy(t->desc, desc, sizeof t->desc - 1);
    t->desc[sizeof t->desc - 1] = '\0';
    t->next = NULL;
    if (!head) {
        head = t;
    } else {
        Task *p = head;
        while (p->next) p = p->next;
        p->next = t;
    }
    printf("Task added: %s\n", desc);
}

void deleteTask(int id) {
    Task **pp = &head;
    while (*pp) {
        if ((*pp)->id == id) {
            Task *tofree = *pp;
            *pp = (*pp)->next;
            printf("Task deleted: %s\n", tofree->desc);
            free(tofree);
            return;
        }
        pp = &(*pp)->next;
    }
    printf("Task id %d not found\n", id);
}

void displayTasks(void) {
    if (!head) {
        printf("No tasks.\n");
        return;
    }
    printf("Tasks:\n");
    Task *p = head;
    while (p) {
        printf("%d. %s\n", p->id, p->desc);
        p = p->next;
    }
}

void freeTasks(void) {
    Task *p = head;
    while (p) {
        Task *next = p->next;
        free(p);
        p = next;
    }
    head = NULL;
}

/* tic tac toe helpers */

void initBoard(void) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            board[i][j] = ' ';
}

void showBoard(void) {
    printf("  0 1 2\n");
    for (int i = 0; i < 3; ++i) {
        printf("%d ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%c", board[i][j]);
            if (j < 2) printf("|");
        }
        printf("\n");
        if (i < 2) printf("  -+-+-\n");
    }
}

int checkWin(void) {
    for (int i = 0; i < 3; ++i) {
        if (board[i][0] != ' ' &&
            board[i][0] == board[i][1] &&
            board[i][1] == board[i][2])
            return 1;
        if (board[0][i] != ' ' &&
            board[0][i] == board[1][i] &&
            board[1][i] == board[2][i])
            return 1;
    }
    if (board[0][0] != ' ' &&
        board[0][0] == board[1][1] &&
        board[1][1] == board[2][2])
        return 1;
    if (board[0][2] != ' ' &&
        board[0][2] == board[1][1] &&
        board[1][1] == board[2][0])
        return 1;
    return 0;
}

int checkDraw(void) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (board[i][j] == ' ') return 0;
    return !checkWin();
}
