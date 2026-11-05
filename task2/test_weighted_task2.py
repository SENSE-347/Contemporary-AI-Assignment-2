import heapq
import os
import random

from weighted_astar_task2 import (
    MOVES, GOAL, START, WALL, can_enter, enter_cost, heuristic,
    min_terrain_cost, astar, find_min_cost,
)

CASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cases_task2.txt')
INF = float('inf')


def load_cases(path):
    cases = []
    case = None
    reading_grid = False

    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
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
                case[key] = [int(v) for v in vals] if key in ('start', 'goal') else int(vals[0])
    return cases


def dijkstra_cost(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist[cur]:
            continue
        if cur == goal:
            return d
        x, y = cur
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if not can_enter(grid[nx][ny]):
                continue
            nd = d + enter_cost(grid[nx][ny])
            if nd < dist.get((nx, ny), INF):
                dist[(nx, ny)] = nd
                heapq.heappush(heap, (nd, (nx, ny)))
    return -1


def cost_to_goal(grid, goal):
    rows, cols = len(grid), len(grid[0])
    best = {goal: 0}
    heap = [(0, goal)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > best[cur]:
            continue
        # S cannot be entered, so no path goes backwards through it.
        if grid[cur[0]][cur[1]] == START:
            continue
        x, y = cur
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx][ny] == WALL:
                continue
            nd = d + enter_cost(grid[x][y])
            if nd < best.get((nx, ny), INF):
                best[(nx, ny)] = nd
                heapq.heappush(heap, (nd, (nx, ny)))
    return best


def check_heuristic(name, grid, goal):
    w_min = min_terrain_cost(grid)
    rows, cols = len(grid), len(grid[0])
    true_cost = cost_to_goal(grid, goal)

    for pos, real in true_cost.items():
        h = heuristic(pos, goal, w_min)
        assert h <= real, f'{name}: h{pos}={h} is larger than true cost {real}'

    for x in range(rows):
        for y in range(cols):
            if (x, y) not in true_cost:
                continue
            for dx, dy in MOVES:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < rows and 0 <= ny < cols):
                    continue
                if not can_enter(grid[nx][ny]):
                    continue
                h_here = heuristic((x, y), goal, w_min)
                h_next = heuristic((nx, ny), goal, w_min)
                assert h_here <= enter_cost(grid[nx][ny]) + h_next, \
                    f'{name}: inconsistent at {(x, y)} -> {(nx, ny)}'


def check_one(name, grid, start, goal, expect_cost=None):
    ref = dijkstra_cost(grid, start, goal)
    cost_h, _ = astar(start, goal, grid, use_heuristic=True)
    cost_0, _ = astar(start, goal, grid, use_heuristic=False)

    assert cost_h == ref, f'{name}: A* cost {cost_h} != Dijkstra {ref}'
    assert cost_0 == ref, f'{name}: h=0 cost {cost_0} != Dijkstra {ref}'
    if expect_cost is not None:
        assert cost_h == expect_cost, f'{name}: cost {cost_h} != expected {expect_cost}'

    check_heuristic(name, grid, goal)
    return ref


def run_case(case):
    start = tuple(case['start'])
    goal = tuple(case['goal'])
    return check_one(case['name'], case['grid'], start, goal, case.get('expect_cost'))


def report_expansions(case):
    start = tuple(case['start'])
    goal = tuple(case['goal'])
    _, with_h = astar(start, goal, case['grid'], use_heuristic=True)
    _, without_h = astar(start, goal, case['grid'], use_heuristic=False)
    print(f"[{case['name']}] expanded, manhattan h: {with_h}, h = 0: {without_h}")


def run_random(count=300, seed=2026):
    rng = random.Random(seed)
    for i in range(count):
        rows = rng.randint(1, 10)
        cols = rng.randint(1, 10)
        wall_prob = rng.choice([0.0, 0.15, 0.3])
        grid = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                if rng.random() < wall_prob:
                    row.append(WALL)
                else:
                    row.append(str(rng.randint(1, 9)))
            grid.append(row)
        start = (rng.randrange(rows), rng.randrange(cols))
        goal = (rng.randrange(rows), rng.randrange(cols))
        grid[start[0]][start[1]] = START
        grid[goal[0]][goal[1]] = GOAL
        check_one(f'random_{i}', grid, start, goal)
    print(f"random: {count} grids passed")


if __name__ == '__main__':
    cases = load_cases(CASE_FILE)
    for case in cases:
        run_case(case)
        report_expansions(case)
    print(f"{len(cases)} cases passed")
    run_random()
