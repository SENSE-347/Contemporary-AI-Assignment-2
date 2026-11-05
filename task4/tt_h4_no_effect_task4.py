import os
import random
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from treasure_game_task4 import TreasureGame, make_game, best_move, tt_key, mean, mean_ci, sign_test_p

INF = float('inf')
SEED = 2026
WALL = 'X'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(OUT_DIR, 'tt_h4_no_effect_task4.png')
REPORT_PATH = os.path.join(OUT_DIR, 'tt_h4_no_effect_task4_results.md')

K_EXHAUSTIVE = range(3, 8)
H_EXHAUSTIVE = range(2, 7)
K_RANDOM = (4, 5, 6)
H_RANDOM = (4, 5, 6)
N_MAPS = 200


def generate_instance(rng, rows, cols, treasure_count, wall_prob, value_low, value_high):
    grid = [[WALL if rng.random() < wall_prob else str(rng.randint(1, 9))
             for _ in range(cols)] for _ in range(rows)]
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    labels = ['A', 'B'] + [f'T{i}' for i in range(1, treasure_count + 1)]
    for label, (r, c) in zip(labels, cells):
        grid[r][c] = label
    values = {f'T{i}': rng.randint(value_low, value_high) for i in range(1, treasure_count + 1)}
    return [[str(cell) for cell in row] for row in grid], values


def sample_games(seed, count, treasures, rows=7, cols=7, wall_prob=0.2, low=6, high=9):
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        try:
            grid, values = generate_instance(rng, rows, cols, treasures, wall_prob, low, high)
            game = make_game(grid, values)
        except ValueError:
            continue
        out.append(game)
    return out


def structural_game(k):
    names = ['A', 'B'] + [f'T{i}' for i in range(1, k + 1)]
    dist = [[1] * len(names) for _ in names]
    return TreasureGame(names, dist, [0, 0] + [1] * k)


def repeated_visits(keys):
    counts = Counter(keys)
    return sum(v - 1 for v in counts.values() if v > 1)


def enumerate_repeats(k, H):
    game = structural_game(k)
    internal, terminal = [], []

    def walk(state, depth):
        if depth == 0 or not state.left:
            terminal.append(tt_key(state, 0))
            return
        internal.append(tt_key(state, depth))
        for t in state.left:
            walk(game.apply(state, t), depth - 1)

    walk(game.start_state(), H)
    return repeated_visits(internal), repeated_visits(terminal)


def first_depth(values):
    for H in H_EXHAUSTIVE:
        if values[H] > 0:
            return H
    return None


def alphabeta_count(game, state, depth, alpha, beta, stats, table, cache_terminal):
    util = game.utility(state)
    use_tt = table is not None

    if depth == 0 or not state.left:
        if use_tt and cache_terminal:
            key = tt_key(state, 0)
            if key in table:
                stats['terminal_hits'] += 1
                return table[key] + util
            table[key] = 0
        stats['terminal_evals'] += 1
        return util

    key = None
    prev = (-INF, INF)
    if use_tt:
        key = tt_key(state, depth)
        if key in table:
            lo, hi = table[key]
            if lo == hi or lo >= beta - util or hi <= alpha - util:
                stats['internal_hits'] += 1
                if lo == hi or lo >= beta - util:
                    return lo + util
                return hi + util
            prev = (lo, hi)

    stats['internal'] += 1
    alpha0, beta0 = alpha, beta

    if state.a_turn:
        best = -INF
        for t in state.left:
            v = alphabeta_count(game, game.apply(state, t), depth - 1, alpha, beta,
                                stats, table, cache_terminal)
            best = max(best, v)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
    else:
        best = INF
        for t in state.left:
            v = alphabeta_count(game, game.apply(state, t), depth - 1, alpha, beta,
                                stats, table, cache_terminal)
            best = min(best, v)
            beta = min(beta, best)
            if alpha >= beta:
                break

    if use_tt:
        if best <= alpha0:
            lo, hi = -INF, best - util
        elif best >= beta0:
            lo, hi = best - util, INF
        else:
            lo, hi = best - util, best - util
        table[key] = (max(prev[0], lo), min(prev[1], hi))
    return best


def search_root(game, state, depth, table, cache_terminal):
    stats = Counter()
    stats['internal'] += 1
    best = -INF
    for t in state.left:
        v = alphabeta_count(game, game.apply(state, t), depth - 1, best - 1, INF,
                            stats, table, cache_terminal)
        best = max(best, v)
    return best, stats


def fmt_ci(m, lo, hi):
    return f'{m:.1f}（95% 置信区间 {lo:.1f} 到 {hi:.1f}）'


def part_derivation(lines):
    lines.append('推导（不依赖地图）：')
    lines.append('1. 状态由 A 的位置、B 的位置、剩余宝藏、轮到谁、剩余深度决定。')
    lines.append('2. 中间状态（剩余深度大于 0）的重复，需要两条不同的走法得到相同的位置和剩余宝藏。'
                 '第一次可能出现在 4 步之后，且此时必须仍有宝藏可取，因此要求 H≥5 且宝藏数≥5。')
    lines.append('3. 终止状态的重复，只需要 4 步后两人的位置和剩余宝藏相同，因此要求 H≥4 且宝藏数≥4。')
    lines.append('4. 标准置换表只存储中间状态，终止状态直接返回评价值，所以 H=4 时没有可以命中的中间状态。')
    lines.append('')


def part_exhaustive(lines, table):
    lines.append('一、穷举验证（枚举所有走法，与地图无关）')
    for k in K_EXHAUSTIVE:
        internal, terminal = {}, {}
        for H in H_EXHAUSTIVE:
            internal[H], terminal[H] = enumerate_repeats(k, H)
            table[(k, H)] = (internal[H], terminal[H])

        first_internal = first_depth(internal)
        first_terminal = first_depth(terminal)
        derived_internal = 5 if k >= 5 else None
        derived_terminal = 4 if k >= 4 else None
        assert first_internal == derived_internal, (k, first_internal)
        assert first_terminal == derived_terminal, (k, first_terminal)

        lines.append(f'{k} 个宝藏：中间状态首次重复的深度 H = {first_internal or "无"}；'
                     f'终止状态首次重复的深度 H = {first_terminal or "无"}。')
    lines.append('穷举结果与推导一致。')
    lines.append('')


def part_random(lines, results):
    lines.append('二、随机地图实测（Alpha-beta，7×7 地图，每组 200 张）')
    for k in K_RANDOM:
        for H in H_RANDOM:
            games = sample_games(SEED + 10 * k + H, N_MAPS, k)
            rec = {'int_off': [], 'int_std': [], 'int_term': [],
                   'ev_off': [], 'ev_std': [], 'ev_term': [], 'term_hits': []}
            for g in games:
                st = g.start_state()
                ref = best_move(g, st, H)['best_value']

                v0, s0 = search_root(g, st, H, None, False)
                v1, s1 = search_root(g, st, H, {}, False)
                v2, s2 = search_root(g, st, H, {}, True)
                assert v0 == v1 == v2 == ref

                rec['int_off'].append(s0['internal'])
                rec['int_std'].append(s1['internal'])
                rec['int_term'].append(s2['internal'])
                rec['ev_off'].append(s0['terminal_evals'])
                rec['ev_std'].append(s1['terminal_evals'])
                rec['ev_term'].append(s2['terminal_evals'])
                rec['term_hits'].append(s2['terminal_hits'])

            results[(k, H)] = rec

            m1, lo1, hi1 = mean_ci([a - b for a, b in zip(rec['int_off'], rec['int_std'])])
            diffs = [a - b for a, b in zip(rec['ev_std'], rec['ev_term'])]
            m2, lo2, hi2 = mean_ci(diffs)
            p2 = sign_test_p(diffs)
            lines.append(f"宝藏 {k} 个、H={H}：标准置换表相对无置换表少展开内部节点 {fmt_ci(m1, lo1, hi1)}；"
                         f"缓存终止状态相对标准置换表少做终止评价 {fmt_ci(m2, lo2, hi2)}，"
                         f"符号检验 p = {p2:.2g}。")
    lines.append('')


def plot(table, results):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ks = list(K_EXHAUSTIVE)
    hs = list(H_EXHAUSTIVE)

    for ax, idx, title in ((axes[0], 0, 'internal repeats (exhaustive)'),
                           (axes[1], 1, 'terminal repeats (exhaustive)')):
        data = [[table[(k, H)][idx] for H in hs] for k in ks]
        im = ax.imshow(data, origin='lower', cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(hs)))
        ax.set_xticklabels(hs)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels(ks)
        ax.set_xlabel('depth H')
        ax.set_ylabel('treasures k')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    ax = axes[2]
    xs = list(range(len(H_RANDOM)))
    k = 6
    off = [mean(results[(k, H)]['ev_off']) for H in H_RANDOM]
    std = [mean(results[(k, H)]['ev_std']) for H in H_RANDOM]
    term = [mean(results[(k, H)]['ev_term']) for H in H_RANDOM]
    ax.bar([x - 0.27 for x in xs], off, width=0.27, label='no TT', color='#bbbbbb')
    ax.bar(xs, std, width=0.27, label='TT (internal only)', color='#8856a7')
    ax.bar([x + 0.27 for x in xs], term, width=0.27, label='TT (+ terminal)', color='#2ca25f')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'H={H}' for H in H_RANDOM])
    ax.set_ylabel('mean terminal evaluations (6 treasures, Alpha-beta)')
    ax.set_title('Terminal evaluations')
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)


def run():
    lines = []
    table = {}
    results = {}

    part_derivation(lines)
    part_exhaustive(lines, table)
    part_random(lines, results)

    text = '\n'.join(lines)
    print(text)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    plot(table, results)
    print(f'图：{PLOT_PATH}')


if __name__ == '__main__':
    run()