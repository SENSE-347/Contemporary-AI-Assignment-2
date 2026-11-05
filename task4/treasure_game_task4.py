import heapq
import math
import statistics

from scipy import stats as sstats

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WALL ='X'
INF =float('inf')
ORDERS =('left', 'right', 'gain')
ALGORITHMS =('minimax', 'alphabeta')


def cell_cost(cell):
    if cell.isdigit():
        return int(cell)
    return 1


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_points(grid):
    points = {}
    for x, row in enumerate(grid):
        for y, cell in enumerate(row):
            if cell in ('A', 'B') or cell.startswith('T'):
                points[cell] = (x, y)
    if 'A' not in points or 'B' not in points:
        raise ValueError('grid must contain both A and B')
    return points


def point_names(points):
    treasures = sorted(name for name in points if name.startswith('T'))
    return ['A', 'B'] + treasures


def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    g_score = {start: 0}
    open_heap = [(manhattan(start, goal), 0, start)]
    closed = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            return g
        closed.add(cur)

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
            new_g = g + cell_cost(grid[nx][ny])
            if new_g < g_score.get(nxt, INF):
                g_score[nxt] = new_g
                heapq.heappush(open_heap, (new_g + manhattan(nxt, goal), new_g, nxt))
    return INF


def distance_matrix(grid):
    points = find_points(grid)
    names = point_names(points)
    dist = [[0] * len(names) for _ in names]

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            cost = astar(grid, points[a], points[b])
            if cost == INF:
                raise ValueError(f'{a} cannot reach {b}')
            dist[i][j] = cost
    return names, dist


def value_list(names, values):
    if set(values) != set(names[2:]):
        raise ValueError('treasure values must match the treasures on the grid')
    return [0, 0] + [values[name] for name in names[2:]]


class GameState:
    def __init__(self, pos_a, pos_b, left, score_a, score_b, a_turn):
        self.pos_a = pos_a
        self.pos_b = pos_b
        self.left = left
        self.score_a = score_a
        self.score_b = score_b
        self.a_turn = a_turn


class TreasureGame:
    def __init__(self, names, dist, values):
        self.names = names
        self.dist = dist
        self.values = values

    def start_state(self):
        left = tuple(range(2, len(self.names)))
        return GameState(0, 1, left, 0, 0, True)

    def gain(self, pos, target):
        return self.values[target] - self.dist[pos][target]

    def ordered(self, state, order='left'):
        if order not in ORDERS:
            raise ValueError(f'unknown order: {order}')
        if order == 'left':
            return list(state.left)
        if order == 'right':
            return list(state.left)[::-1]
        pos = state.pos_a if state.a_turn else state.pos_b
        return sorted(state.left, key=lambda t: (-self.gain(pos, t), t))

    def apply(self, state, target):
        left = tuple(t for t in state.left if t != target)
        if state.a_turn:
            return GameState(target, state.pos_b, left,
                             state.score_a + self.gain(state.pos_a, target),
                             state.score_b, False)
        return GameState(state.pos_a, target, left,
                         state.score_a, state.score_b + self.gain(state.pos_b, target), True)

    def utility(self, state):
        return state.score_a - state.score_b


class LiveGame(TreasureGame):
    def __init__(self, grid, values):
        points = find_points(grid)
        names = point_names(points)
        super().__init__(names, None, value_list(names, values))
        self.grid = grid
        self.points = points

    def gain(self, pos, target):
        cost = astar(self.grid, self.points[self.names[pos]], self.points[self.names[target]])
        return self.values[target] - cost


def make_game(grid, values):
    names, dist = distance_matrix(grid)
    return TreasureGame(names, dist, value_list(names, values))


def tt_key(state, depth):
    return (state.pos_a, state.pos_b, state.left, state.a_turn, depth)


def minimax(game, state, depth, stats, order='left', table=None):
    stats['nodes'] += 1
    if depth == 0 or not state.left:
        return game.utility(state)

    util = game.utility(state)
    key = None
    if table is not None:
        key = tt_key(state, depth)
        if key in table:
            stats['tt_hits'] += 1
            return table[key] + util

    values = [minimax(game, game.apply(state, t), depth - 1, stats, order, table)
              for t in game.ordered(state, order)]
    best = max(values) if state.a_turn else min(values)

    if table is not None:
        table[key] = best - util
    return best


def alphabeta(game, state, depth, alpha, beta, stats, order='left', table=None):
    stats['nodes'] += 1
    if depth == 0 or not state.left:
        return game.utility(state)

    util = game.utility(state)
    alpha0, beta0 = alpha, beta
    key = None
    prev = (-INF, INF)

    if table is not None:
        key = tt_key(state, depth)
        if key in table:
            lo, hi = table[key]
            if lo == hi or lo >= beta - util or hi <= alpha - util:
                stats['tt_hits'] += 1
                if lo == hi:
                    return lo + util
                if lo >= beta - util:
                    return lo + util
                return hi + util
            prev = (lo, hi)

    if state.a_turn:
        best = -INF
        for t in game.ordered(state, order):
            v = alphabeta(game, game.apply(state, t), depth - 1, alpha, beta, stats, order, table)
            best = max(best, v)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
    else:
        best = INF
        for t in game.ordered(state, order):
            v = alphabeta(game, game.apply(state, t), depth - 1, alpha, beta, stats, order, table)
            best = min(best, v)
            beta = min(beta, best)
            if alpha >= beta:
                break

    if table is not None:
        if best <= alpha0:
            lo, hi = -INF, best - util
        elif best >= beta0:
            lo, hi = best - util, INF
        else:
            lo, hi = best - util, best - util
        table[key] = (max(prev[0], lo), min(prev[1], hi))
    return best


def root_search(game, state, depth, moves, algorithm, order, stats, table):
    maximizing = state.a_turn
    values = {}
    best = -INF if maximizing else INF

    for t in moves:
        child = game.apply(state, t)
        if algorithm =='minimax':
            v =minimax(game, child, depth - 1, stats, order, table)
        elif maximizing:
            v =alphabeta(game, child, depth - 1, best - 1, INF, stats, order, table)
        else:
            v =alphabeta(game, child, depth - 1, -INF, best + 1, stats, order, table)
        values[t] = v
        best = max(best, v) if maximizing else min(best, v)
    return values


def pick_best(values, maximizing):
    best =max(values.values()) if maximizing else min(values.values())
    target =min(t for t, v in values.items() if v == best)
    return target, best


def _result(game, target, value, stats):
    return {
        'best_target': game.names[target],
        'target': target,
        'best_value': value,
        'nodes': stats['nodes'],
        'tt_hits': stats['tt_hits'],
    }


def best_move(game, state, depth, algorithm='alphabeta', order='left', reverse=False, tt=False):
    if algorithm not in ALGORITHMS:
        raise ValueError(f'unknown algorithm: {algorithm}')
    if depth < 1:
        raise ValueError('depth must be at least 1')
    if not state.left:
        raise ValueError('no treasure left')

    order = 'right' if reverse else order
    moves =game.ordered(state, order)
    stats= {'nodes': 0, 'tt_hits': 0}
    table= {} if tt else None
    values = root_search(game, state, depth, moves, algorithm, order, stats, table)
    target, value = pick_best(values, state.a_turn)
    return _result(game, target, value, stats)


def iterative_deepening(game, state, depth, algorithm='alphabeta', order='left', tt=False):
    if depth < 1:
        raise ValueError('depth must be at least 1')
    stats = {'nodes': 0, 'tt_hits': 0}
    table = {} if tt else None
    moves = game.ordered(state, order)
    values = None

    for d in range(1, depth + 1):
        if values is not None:
            if state.a_turn:
                moves = sorted(values, key=lambda t: (-values[t], t))
            else:
                moves = sorted(values, key=lambda t: (values[t], t))
        values = root_search(game, state, d, moves, algorithm, order, stats, table)

    target, value = pick_best(values, state.a_turn)
    return _result(game, target, value, stats)


def solve_grid(grid, values, depth, use_alphabeta=True, reverse=False, order='left', tt=False):
    game = make_game(grid, values)
    algorithm = 'alphabeta' if use_alphabeta else 'minimax'
    return best_move(game, game.start_state(), depth, algorithm=algorithm,
                     order=order, reverse=reverse, tt=tt)


def choose_move(game, state, policy, depth, rng=None, tt=False):
    if policy == 'minimax':
        return best_move(game, state, depth, tt=tt)['target']
    if policy == 'greedy':
        pos = state.pos_a if state.a_turn else state.pos_b
        return min(state.left, key=lambda t: (-game.gain(pos, t), t))
    if policy == 'random':
        return rng.choice(sorted(state.left))
    raise ValueError(f'unknown policy: {policy}')


def play(game, depth, a_policy, b_policy, rng=None, tt=False):
    state = game.start_state()
    sequence = []
    while state.left:
        policy = a_policy if state.a_turn else b_policy
        target = choose_move(game, state, policy, depth, rng, tt)
        sequence.append(game.names[target])
        state = game.apply(state, target)
    return {'sequence': sequence, 'utility': game.utility(state)}


def generate_instance(rng, rows, cols, treasure_count, wall_prob, value_low, value_high):
    grid = [[WALL if rng.random() < wall_prob else str(rng.randint(1, 9))
             for _ in range(cols)] for _ in range(rows)]
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    labels = ['A', 'B'] + [f'T{i}' for i in range(1, treasure_count + 1)]
    if len(cells) < len(labels):
        raise ValueError('grid too small for the requested treasures')
    for label, (r, c) in zip(labels, cells):
        grid[r][c] = label
    values = {f'T{i}': rng.randint(value_low, value_high) for i in range(1, treasure_count + 1)}
    return [[str(cell) for cell in row] for row in grid], values


def mean(xs):
    return sum(xs) / len(xs)


def mean_ci(diffs, alpha=0.05):
    n = len(diffs)
    m = mean(diffs)
    if n < 2:
        return m, m, m
    s = statistics.stdev(diffs)
    t = sstats.t.ppf(1 - alpha / 2, n - 1)
    half = t * s / math.sqrt(n)
    return m, m - half, m + half


def sign_test_p(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    if pos + neg == 0:
        return 1.0
    return sstats.binomtest(min(pos, neg), pos + neg, 0.5).pvalue


def holm_adjust(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def required_n(diffs, alpha=0.05, power=0.80):
    delta = abs(mean(diffs))
    if delta == 0 or len(diffs) < 2:
        return None
    sd = statistics.stdev(diffs)
    z = sstats.norm.ppf(1 - alpha / 2) + sstats.norm.ppf(power)
    return math.ceil((z * sd / delta) ** 2)