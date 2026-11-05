import os
import random
import statistics
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sstats

from treasure_game_task4 import (
    make_game, solve_grid, best_move, iterative_deepening, root_search, play,
    LiveGame, mean, mean_ci, sign_test_p,
)

SEED = 2026
WALL = 'X'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(OUT_DIR, 'test_treasure_game_task4.png')
REPORT_PATH = os.path.join(OUT_DIR, 'test_treasure_game_task4_results.md')

GIVEN_GRID = [l.split() for l in [
    'A 1 1 X 2 1 T1',
    '1 X 6 X 1 X 1',
    '1 1 T2 1 1 1 1',
    'X 1 X 8 X 1 X',
    '1 1 1 1 T3 1 1',
    '1 X 9 X 1 X 1',
    'T4 1 1 X 1 1 B',
]]
GIVEN_VALUES = {'T1': 9, 'T2': 6, 'T3': 7, 'T4': 7}

N_RANDOM = 200
N_TT = 100
N_COUNT = 30
N_CACHE_REPEAT = 5


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


def sample_games(seed, count, treasures, rows=7, cols=7, wall_prob=0.2, low=6, high=9):
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        try:
            grid, values = generate_instance(rng, rows, cols, treasures, wall_prob, low, high)
            game = make_game(grid, values)
        except ValueError:
            continue
        out.append((grid, values, game))
    return out


def prop_ci(k, n, alpha=0.05):
    z = sstats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt_ci(m, lo, hi):
    return f'{m:.2f}（95% 置信区间 {lo:.2f} 到 {hi:.2f}）'


def experiment_depth(lines):
    mm = solve_grid(GIVEN_GRID, GIVEN_VALUES, 2, use_alphabeta=False)
    ab = solve_grid(GIVEN_GRID, GIVEN_VALUES, 2, use_alphabeta=True)
    assert mm['best_target'] == ab['best_target'] == 'T2' and ab['best_value'] == -1
    mm = solve_grid(GIVEN_GRID, GIVEN_VALUES, 4, use_alphabeta=False)
    ab = solve_grid(GIVEN_GRID, GIVEN_VALUES, 4, use_alphabeta=True)
    assert mm['best_target'] == ab['best_target'] == 'T3' and ab['best_value'] == 3

    rows = []
    for h in range(1, 5):
        mm = solve_grid(GIVEN_GRID, GIVEN_VALUES, h, use_alphabeta=False)
        ab = solve_grid(GIVEN_GRID, GIVEN_VALUES, h, use_alphabeta=True)
        assert mm['best_target'] == ab['best_target'] and mm['best_value'] == ab['best_value']
        rows.append((h, ab['best_target'], ab['best_value'], mm['nodes'], ab['nodes']))
        lines.append(f"给定地图 H={h}：A 首步选 {ab['best_target']}，收益差 {ab['best_value']}；"
                     f"Minimax 访问 {mm['nodes']} 个节点，Alpha-beta 访问 {ab['nodes']} 个节点。")

    samples = sample_games(SEED + 2, N_RANDOM, 4)
    agree = {h: 0 for h in range(1, 5)}
    for _, _, g in samples:
        final = best_move(g, g.start_state(), 4)['best_target']
        for h in range(1, 5):
            if best_move(g, g.start_state(), h)['best_target'] == final:
                agree[h] += 1
    props = {}
    for h in range(1, 5):
        lo, hi = prop_ci(agree[h], len(samples))
        props[h] = (agree[h] / len(samples), lo, hi)
        if h < 4:
            lines.append(f'随机 {len(samples)} 局：H={h} 的首步与 H=4 相同的比例为 '
                         f'{agree[h] / len(samples):.1%}（95% 置信区间 {lo:.1%} 到 {hi:.1%}）。')
    return {'rows': rows, 'props': props}


def experiment_order(lines):
    game = make_game(GIVEN_GRID, GIVEN_VALUES)
    for h in (2, 4):
        nodes = {o: best_move(game, game.start_state(), h, order=o)['nodes'] for o in ('left', 'right', 'gain')}
        lines.append(f"给定地图 H={h}，Alpha-beta 节点数：从左到右 {nodes['left']}，"
                     f"从右到左 {nodes['right']}，按收益排序 {nodes['gain']}。")

    samples = sample_games(SEED + 3, N_RANDOM, 4)
    nodes = {o: [] for o in ('left', 'right', 'gain')}
    for _, _, g in samples:
        results = set()
        for o in nodes:
            r = best_move(g, g.start_state(), 4, order=o)
            nodes[o].append(r['nodes'])
            results.add((r['best_target'], r['best_value']))
        assert len(results) == 1, 'ordering changed the result'

    summary = {}
    for o in ('right', 'gain'):
        diffs = [a - b for a, b in zip(nodes['left'], nodes[o])]
        m, lo, hi = mean_ci(diffs)
        wins = sum(1 for d in diffs if d > 0)
        ties = sum(1 for d in diffs if d == 0)
        losses = sum(1 for d in diffs if d < 0)
        p = sign_test_p(diffs)
        summary[o] = (m, lo, hi)
        name = '从右到左' if o == 'right' else '按收益排序'
        lines.append(f'随机 {len(samples)} 局（H=4）：{name}相对从左到右平均少访问节点 '
                     f'{fmt_ci(m, lo, hi)}，少 {wins} 局、平 {ties} 局、多 {losses} 局，符号检验 p = {p:.2g}。')
    means = {o: mean(v) for o, v in nodes.items()}
    return {'means': means, 'summary': summary}


def experiment_tt(lines):
    samples = sample_games(SEED + 4, N_TT, 6)
    result = {}
    for h in (4, 5, 6):
        off, on, hits = [], [], []
        for _, _, g in samples:
            a = best_move(g, g.start_state(), h, tt=False)
            b = best_move(g, g.start_state(), h, tt=True)
            assert a['best_value'] == b['best_value'] and a['target'] == b['target']
            off.append(a['nodes'])
            on.append(b['nodes'])
            hits.append(b['tt_hits'])
        diffs = [a - b for a, b in zip(off, on)]
        m, lo, hi = mean_ci(diffs)
        result[h] = (mean(off), mean(on), m, lo, hi, mean(hits))
        lines.append(f'6 个宝藏、H={h}（{len(samples)} 局）：置换表前平均 {mean(off):.0f} 个节点，'
                     f'置换表后 {mean(on):.0f} 个节点，平均减少 {fmt_ci(m, lo, hi)}，'
                     f'平均命中 {mean(hits):.1f} 次。')
    return result


def experiment_iterative(lines):
    samples = sample_games(SEED + 5, N_RANDOM, 4)
    direct, iterative = [], []
    for _, _, g in samples:
        st = g.start_state()
        a = best_move(g, st, 4, tt=True)
        b = iterative_deepening(g, st, 4, tt=True)
        assert a['best_value'] == b['best_value'] and a['target'] == b['target']
        direct.append(a['nodes'])
        iterative.append(b['nodes'])
    diffs = [a - b for a, b in zip(direct, iterative)]
    m, lo, hi = mean_ci(diffs)
    lines.append(f'随机 {len(samples)} 局（H=4，均开置换表）：迭代加深相对直接搜索平均少访问节点 '
                 f'{fmt_ci(m, lo, hi)}（负值表示迭代加深更多）。')
    return {'direct': mean(direct), 'iterative': mean(iterative), 'summary': (m, lo, hi)}


class CountingLive(LiveGame):
    def __init__(self, grid, values):
        super().__init__(grid, values)
        self.calls = 0

    def gain(self, pos, target):
        self.calls += 1
        return super().gain(pos, target)


def experiment_cache(lines):
    cached_times, live_times = [], []
    for _ in range(N_CACHE_REPEAT):
        t0 = time.perf_counter()
        game = make_game(GIVEN_GRID, GIVEN_VALUES)
        best_move(game, game.start_state(), 4)
        cached_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        live = LiveGame(GIVEN_GRID, GIVEN_VALUES)
        best_move(live, live.start_state(), 4)
        live_times.append(time.perf_counter() - t0)

    counter = CountingLive(GIVEN_GRID, GIVEN_VALUES)
    best_move(counter, counter.start_state(), 4)
    cached_calls = len(game.names) * (len(game.names) - 1)
    lines.append(f'给定地图 H=4 完整搜索：缓存距离矩阵的中位耗时 {statistics.median(cached_times) * 1000:.2f} 毫秒，'
                 f'A* 调用 {cached_calls} 次；每次收益重新寻路的中位耗时 '
                 f'{statistics.median(live_times) * 1000:.2f} 毫秒，A* 调用 {counter.calls} 次。')
    return {
        'cached_ms': statistics.median(cached_times) * 1000,
        'live_ms': statistics.median(live_times) * 1000,
        'cached_calls': cached_calls,
        'live_calls': counter.calls,
    }


def experiment_policy(lines):
    samples = sample_games(SEED + 6, N_RANDOM, 4)
    policies = ('minimax', 'greedy', 'random')
    table = {}
    for a_pol in policies:
        for b_pol in policies:
            rng = random.Random(SEED + 7)
            utilities = []
            for _, _, g in samples:
                r = play(g, 4, a_pol, b_pol, rng=rng)
                utilities.append(r['utility'])
            m, lo, hi = mean_ci(utilities)
            table[(a_pol, b_pol)] = (m, lo, hi)
            lines.append(f'A 用 {a_pol}，B 用 {b_pol}：A 的平均收益差 {fmt_ci(m, lo, hi)}。')
    return table


def experiment_search_vs_greedy(lines):
    samples = sample_games(SEED + 8, N_RANDOM, 4)
    diffs = []
    for _, _, g in samples:
        u_search = play(g, 4, 'minimax', 'minimax')['utility']
        u_greedy = play(g, 4, 'greedy', 'minimax')['utility']
        diffs.append(u_search - u_greedy)
    m, lo, hi = mean_ci(diffs)
    better = sum(1 for d in diffs if d > 0)
    same = sum(1 for d in diffs if d == 0)
    worse = sum(1 for d in diffs if d < 0)
    p = sign_test_p(diffs)
    lines.append(f'随机 {len(samples)} 局（B 始终用最优搜索）：A 用搜索相对贪心的平均收益差 '
                 f'{fmt_ci(m, lo, hi)}，搜索更好 {better} 局、相同 {same} 局、更差 {worse} 局，'
                 f'符号检验 p = {p:.2g}。')
    return {'diffs': diffs, 'summary': (m, lo, hi)}


def experiment_count(lines):
    result = {}
    for k in (3, 4, 5, 6):
        samples = sample_games(SEED + 9 + k, N_COUNT, k)
        mm, ab = [], []
        for _, _, g in samples:
            st = g.start_state()
            mm.append(best_move(g, st, k, algorithm='minimax')['nodes'])
            ab.append(best_move(g, st, k, algorithm='alphabeta')['nodes'])
        result[k] = (mean(mm), mean(ab))
        lines.append(f'{k} 个宝藏（完整博弈，H={k}，{len(samples)} 局）：Minimax 平均 {mean(mm):.0f} 个节点，'
                     f'Alpha-beta 平均 {mean(ab):.0f} 个节点。')
    return result


def experiment_ties(lines):
    samples = sample_games(SEED + 20, N_RANDOM, 4)
    tie_cases = 0
    negative_gain = 0
    total_gain = 0
    for _, _, g in samples:
        st = g.start_state()
        moves = g.ordered(st, 'left')
        stats = {'nodes': 0, 'tt_hits': 0}
        values = root_search(g, st, 4, moves, 'minimax', 'left', stats, None)
        best = max(values.values())
        if sum(1 for v in values.values() if v == best) > 1:
            tie_cases += 1
        for t in moves:
            total_gain += 1
            if g.gain(st.pos_a, t) < 0:
                negative_gain += 1
    lines.append(f'随机 {len(samples)} 局（H=4）中，首步存在并列最优的局数为 {tie_cases}，'
                 f'占 {tie_cases / len(samples):.1%}；所有首步收益中为负的比例为 '
                 f'{negative_gain / total_gain:.1%}。')
    return {'tie_rate': tie_cases / len(samples), 'negative_rate': negative_gain / total_gain}


def plot(res):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0][0]
    hs = sorted(res['depth']['props'])
    props = [res['depth']['props'][h][0] for h in hs]
    errs = [[res['depth']['props'][h][0] - res['depth']['props'][h][1] for h in hs],
            [res['depth']['props'][h][2] - res['depth']['props'][h][0] for h in hs]]
    ax.bar([str(h) for h in hs], props, yerr=errs, capsize=5, color='#2b7bba')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('depth H')
    ax.set_ylabel('share of maps with same first move as H=4')
    ax.set_title('Decision agreement with H=4')

    ax = axes[0][1]
    keys = ['left', 'right', 'gain']
    means = [res['order']['means'][k] for k in keys]
    ax.bar(keys, means, color=['#999999', '#2b7bba', '#2ca25f'])
    ax.set_ylabel('mean nodes (Alpha-beta, H=4)')
    ax.set_title('Child ordering')

    ax = axes[0][2]
    hs = sorted(res['tt'])
    xs = list(range(len(hs)))
    off = [res['tt'][h][0] for h in hs]
    on = [res['tt'][h][1] for h in hs]
    ax.bar([x - 0.2 for x in xs], off, width=0.4, label='without TT', color='#bbbbbb')
    ax.bar([x + 0.2 for x in xs], on, width=0.4, label='with TT', color='#8856a7')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'H={h}' for h in hs])
    ax.set_ylabel('mean nodes (6 treasures)')
    ax.set_title('Transposition table')
    ax.legend()

    ax = axes[1][0]
    ks = sorted(res['count'])
    ax.plot(ks, [res['count'][k][0] for k in ks], marker='o', label='Minimax')
    ax.plot(ks, [res['count'][k][1] for k in ks], marker='s', label='Alpha-beta')
    ax.set_yscale('log')
    ax.set_xlabel('number of treasures (full game)')
    ax.set_ylabel('mean nodes (log scale)')
    ax.set_title('Search size vs treasures')
    ax.legend()

    ax = axes[1][1]
    pol = res['policy']
    labels = [f'A:{a[:3]} B:{b[:3]}' for (a, b) in pol]
    vals = [pol[k][0] for k in pol]
    ax.bar(range(len(vals)), vals, color='#999999')
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=60, fontsize=8)
    ax.set_ylabel('mean U = Score_A - Score_B')
    ax.set_title('Opponent policies')

    ax = axes[1][2]
    ax.hist(res['greedy']['diffs'], bins=range(min(res['greedy']['diffs']) - 1,
                                              max(res['greedy']['diffs']) + 2), color='#2ca25f')
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('U(search) - U(greedy)')
    ax.set_ylabel('number of games')
    ax.set_title('Search vs greedy for A')

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)


def run():
    lines = []

    lines.append('一、搜索深度对决策的影响')
    depth = experiment_depth(lines)
    lines.append('')

    lines.append('二、子节点访问顺序')
    order = experiment_order(lines)
    lines.append('')

    lines.append('三、置换表')
    tt = experiment_tt(lines)
    lines.append('')

    lines.append('四、迭代加深')
    iterative = experiment_iterative(lines)
    lines.append('')

    lines.append('五、距离缓存')
    cache = experiment_cache(lines)
    lines.append('')

    lines.append('六、对手策略')
    policy = experiment_policy(lines)
    lines.append('')

    lines.append('七、搜索与贪心')
    greedy = experiment_search_vs_greedy(lines)
    lines.append('')

    lines.append('八、宝藏数量与搜索规模')
    count = experiment_count(lines)
    lines.append('')

    lines.append('九、并列与负收益')
    ties = experiment_ties(lines)

    res = {'depth': depth, 'order': order, 'tt': tt, 'iterative': iterative,
           'cache': cache, 'policy': policy, 'greedy': greedy, 'count': count, 'ties': ties}

    text = '\n'.join(lines)
    print(text)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    plot(res)
    print(f'图：{PLOT_PATH}')


if __name__ == '__main__':
    run()