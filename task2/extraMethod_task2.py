import heapq

from weighted_astar_task2 import (
    MOVES, GOAL, START, WALL, can_enter, enter_cost, heuristic, min_terrain_cost,
)

INF = float('inf')


def relaxed_enter(cell, w_min):
    if cell ==GOAL:
        return 0
    if cell.isdigit():
        return int(cell)
    return w_min


def reverse_costs(grid, goal, relaxed=False):
    rows, cols=len(grid), len(grid[0])
    w_min =min_terrain_cost(grid)
    best ={goal: 0}
    heap =[(0, goal)]

    while heap:
        d, cur= heapq.heappop(heap)
        if d > best[cur]:
            continue

        cur_cell = grid[cur[0]][cur[1]]
        if not relaxed and cur_cell == START:
            continue
        step = relaxed_enter(cur_cell, w_min) if relaxed else enter_cost(cur_cell)

        x, y = cur
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if not relaxed and grid[nx][ny] == WALL:
                continue
            nd = d + step
            if nd < best.get((nx, ny), INF):
                best[(nx, ny)] = nd
                heapq.heappush(heap, (nd, (nx, ny)))
    return best


def forward_costs(grid, src):
    # Minimum cost from src to every cell, relaxed graph.
    rows, cols = len(grid), len(grid[0])
    w_min =min_terrain_cost(grid)
    best ={src: 0}
    heap =[(0, src)]

    while heap:
        d, cur =heapq.heappop(heap)
        if d > best[cur]:
            continue
        x, y = cur
        for dx, dy in MOVES:
            nx, ny =x + dx, y + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            nd = d + relaxed_enter(grid[nx][ny], w_min)
            if nd < best.get((nx, ny), INF):
                best[(nx, ny)] = nd
                heapq.heappush(heap, (nd, (nx, ny)))
    return best


def landmark_corners(grid, goal):
    rows, cols = len(grid), len(grid[0])
    corners= [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    return [c for c in corners if c != goal]


def h_modified(grid, goal):
    w_min= min_terrain_cost(grid)
    return {(x, y): heuristic((x, y), goal, w_min)
            for x in range(len(grid)) for y in range(len(grid[0]))}


def h_relaxed(grid, goal):
    return reverse_costs(grid, goal, relaxed=True)


def h_exact(grid, goal):
    return reverse_costs(grid, goal, relaxed=False)


def h_landmark(grid, goal):
    rows, cols=len(grid), len(grid[0])
    best = {(x, y): 0 for x in range(rows) for y in range(cols)}

    for lm in landmark_corners(grid, goal):
        to_lm = reverse_costs(grid, lm, relaxed=True)
        from_lm = forward_costs(grid, lm)
        if goal not in to_lm or goal not in from_lm:
            continue
        for n in best:
            if n in to_lm and n in from_lm:
                bound = max(to_lm[n] - to_lm[goal], from_lm[goal] - from_lm[n])
                best[n]=max(best[n], bound)
    return best


def h_max(grid, goal):
    relaxed = h_relaxed(grid, goal)
    landmark= h_landmark(grid, goal)
    return {n: max(relaxed.get(n, 0), landmark.get(n, 0)) for n in set(relaxed) | set(landmark)}



METHODS = [
    ('modified', '修正曼哈顿（基准）', h_modified),
    ('relaxed', '松弛启发函数', h_relaxed),
    ('exact', '精确代价（理论上界）', h_exact),
    ('landmark', '地标启发函数', h_landmark),
    ('max', '松弛与地标取最大值', h_max),
]


def astar_with_table(start, goal, grid, table):
    rows, cols = len(grid), len(grid[0])
    g_score = {start: 0}
    open_heap = [(table.get(start, 0), 0, start)]
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
            nx, ny= x + dx, y + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if not can_enter(grid[nx][ny]):
                continue
            nxt =(nx, ny)
            if nxt in closed:
                continue
            new_g =g + enter_cost(grid[nx][ny])
            if new_g < g_score.get(nxt, INF):
                g_score[nxt] =new_g
                heapq.heappush(open_heap, (new_g + table.get(nxt, 0), new_g, nxt))
    return -1, expanded
