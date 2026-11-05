import heapq

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WALL = 'X'
START = 'S'
GOAL = 'E'


def can_enter(cell):
    return cell not in (WALL, START)


def enter_cost(cell):
    if cell == GOAL:
        return 0
    return int(cell)


def min_terrain_cost(grid):
    costs = [int(cell) for row in grid for cell in row if cell.isdigit()]
    if not costs:
        return 1
    return min(costs)


def heuristic(pos, goal, min_cost):
    dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    return max(0, dist - 1) * min_cost


def find_min_cost(start, goal, grid, use_heuristic=True):
    cost, _ = astar(start, goal, grid, use_heuristic)
    return cost


def astar(start, goal, grid, use_heuristic=True):
    rows, cols = len(grid), len(grid[0])
    w_min = min_terrain_cost(grid)

    def h(pos):
        if use_heuristic:
            return heuristic(pos, goal, w_min)
        else:
            return 0

    g_score = {start: 0}
    open_heap = [(h(start), 0, start)]
    closed = set()
    expanded = 0

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)

        if cur in closed:
            continue

        if cur == goal:
            return g, expanded

        closed.add(cur)
        expanded += 1

        x, y = cur
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if not can_enter(grid[nx][ny]):
                continue

            nxt = (nx, ny)
            if nxt in closed:
                continue

            new_g = g + enter_cost(grid[nx][ny])
            if new_g < g_score.get(nxt, float('inf')):
                g_score[nxt] = new_g
                heapq.heappush(open_heap, (new_g + h(nxt), new_g, nxt))

    return -1, expanded
