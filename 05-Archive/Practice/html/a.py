import sys
def apply_gravity(grid):
    if not grid:
        return
    R, C = len(grid), len(grid[0])
    for c in range(C):
        cnt = 0
        for r in range(R):
            if grid[r][c] == '*':
                cnt += 1
        for r in range(R - 1, -1, -1):
            if cnt > 0:
                grid[r][c] = '*'
                cnt -= 1
            else:
                grid[r][c] = '.'
def rotate_right(grid):
    reversed_grid = grid[::-1]
    result = []
    for cols in zip(*reversed_grid):
        result_row = []
        for val in cols:
            result_row.append(val)
        result.append(result_row)
    return result
def rotate_left(grid):
    transposed = list(zip(*grid))
    result = []
    for cols in transposed:
        result_row = []
        for val in cols:
            result_row.append(val)
        result.append(result_row)
    result.reverse()
    return result
def main():
    tokens = sys.stdin.read().split()
    if not tokens:
        return
    idx = 0
    m = int(tokens[idx])
    idx += 1
    n = int(tokens[idx])
    idx += 1
    grid = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(tokens[idx])
            idx += 1
        grid.append(row)
    k = int(tokens[idx])
    idx += 1
    instrs = []
    for i in range(k):
        instrs.append(tokens[idx])
        idx += 1
    apply_gravity(grid)
    for inst in instrs:
        if inst.lower() == 'right':
            grid = rotate_right(grid)
            apply_gravity(grid)
        elif inst.lower() == 'left':
            grid = rotate_left(grid)
            apply_gravity(grid)
        else:
            pass
    for row in grid:
        print(' '.join(row))
if __name__ == '__main__':
    main()