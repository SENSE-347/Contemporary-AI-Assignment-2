import heapq

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WALL = 'X'


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_path(start, goal, grid, use_heuristic=True):
    path, _ = astar(start, goal, grid, use_heuristic)
    return path


def astar(start, goal, grid, use_heuristic=True):

    rows, cols = len(grid), len(grid[0])

    def h(pos):
        if use_heuristic:
            return manhattan(pos,goal)
        else:
            return 0

    g_score = {start: 0}
    came_from = {}
    open_heap = [(h(start), 0, start)]
    closed = set()
    expanded = 0

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)

        if cur in closed:
            continue

        if cur == goal:
            return rebuild_path(came_from, cur), expanded

        closed.add(cur)
        expanded += 1

        x, y = cur
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx][ny] == WALL:
                continue

            nxt = (nx, ny)
            if nxt in closed:
                continue

            new_g = g + 1
            if new_g < g_score.get(nxt, float('inf')):
                g_score[nxt] = new_g
                came_from[nxt] = cur
                heapq.heappush(open_heap, (new_g + h(nxt), new_g, nxt))

    return [], expanded


def rebuild_path(came_from, node):
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path
