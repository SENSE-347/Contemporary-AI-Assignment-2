import os
from collections import deque

from maze_astar_task1 import astar, find_path

CASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cases_task1.txt')
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def load_cases(path):
    cases =[]
    case =None
    reading_grid =False

    with open(path, encoding='utf-8') as f:
        for raw in f:
            line= raw.strip()
            if not line:
                continue
            if line.startswith('#'):
                reading_grid = False
                if line.startswith('# name:'):
                    case = {'name': line[len('# name:'):].strip(), 'grid': []}
                    cases.append(case)
                continue
            if line == 'grid':
                reading_grid = True
            elif reading_grid:
                case['grid'].append(line.split())
            else:
                key, *vals = line.split()
                if key in ('start', 'goal'):
                    case[key] = [int(v) for v in vals]
                else:
                    case[key] = int(vals[0])
    return cases


def bfs_steps(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            return dist[cur]
        x, y = cur
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if not (0<= nx <rows and 0 <= ny < cols):
                continue
            if grid[nx][ny] == 'X' or (nx, ny) in dist:
                continue
            dist[(nx, ny)] = dist[cur] + 1
            queue.append((nx, ny))
    return None


def check_path_valid(path, start, goal, grid):
    assert path[0] == start and path[-1] == goal
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1])==1
        assert grid[b[0]][b[1]] != 'X'


def run_case(case):
    start = tuple(case['start'])
    goal = tuple(case['goal'])
    grid = case['grid']
    path = find_path(start, goal, grid)

    steps = bfs_steps(grid, start, goal)
    if steps is None:
        assert path == [], (case['name'], 'expected empty path')
    else:
        assert len(path) == steps + 1, (case['name'], len(path), steps + 1)
        check_path_valid(path, start, goal, grid)

    if 'expect_len' in case:
        assert len(path) == case['expect_len'], (case['name'], len(path))

    _, with_h = astar(start, goal, grid, use_heuristic=True)
    _, without_h = astar(start, goal, grid, use_heuristic=False)
    print(f"[{case['name']}] expanded, manhattan h: {with_h}, h = 0: {without_h}")


if __name__ == '__main__':
    cases = load_cases(CASE_FILE)
    for case in cases:
        run_case(case)
