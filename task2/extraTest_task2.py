import itertools
import math
import os
import random
import statistics
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from weighted_astar_task2 import START, GOAL, can_enter, enter_cost
from extraMethod_task2 import METHODS, astar_with_table, h_exact, reverse_costs

SEED = 2026
PILOT_SIZE = 200
MIN_SIZE = 300
MAX_SIZE = 2000
ALPHA = 0.05
POWER = 0.80
TIME_REPEATS = 3
BASELINE = 'modified'
EXACT = 'exact'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(OUT_DIR, 'extraTest_task2.png')
REPORT_PATH = os.path.join(OUT_DIR, 'extraTest_task2_results.md')
KEYS = [key for key, _, _ in METHODS]
LABELS = {key: label for key, label, _ in METHODS}


class MapStream:
    # One seeded stream. Extending the sample continues the same sequence.
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def next_map(self):
        while True:
            grid, start, goal = self._raw_map()
            if start in reverse_costs(grid, goal, relaxed=False):
                return grid, start, goal

    def _raw_map(self):
        rng = self.rng
        rows = rng.randint(3, 12)
        cols = rng.randint(3, 12)
        wall_prob = rng.choice([0.0, 0.15, 0.3])
        grid = [['X' if rng.random() < wall_prob else str(rng.randint(1, 9))
                 for _ in range(cols)] for _ in range(rows)]
        start = (rng.randrange(rows), rng.randrange(cols))
        goal = (rng.randrange(rows), rng.randrange(cols))
        while goal == start:
            goal = (rng.randrange(rows), rng.randrange(cols))
        grid[start[0]][start[1]] = START
        grid[goal[0]][goal[1]] = GOAL
        return grid, start, goal


def check_table(grid, table, exact):
    admissible = all(table.get(c, 0) <= exact[c] for c in exact)
    consistent = True
    for c in exact:
        x, y = c
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (x + dx, y + dy)
            if nb not in exact or not can_enter(grid[nb[0]][nb[1]]):
                continue
            if table.get(c, 0) > enter_cost(grid[nb[0]][nb[1]]) + table.get(nb, 0):
                consistent = False
    return admissible, consistent


def evaluate(grid, start, goal):
    exact = h_exact(grid, goal)
    optimal = exact[start]
    record = {}

    for key, _, build in METHODS:
        pre_times, search_times = [], []
        table = None
        for _ in range(TIME_REPEATS):
            t0 = time.perf_counter()
            table = build(grid, goal)
            t1 = time.perf_counter()
            cost, expanded = astar_with_table(start, goal, grid, table)
            t2 = time.perf_counter()
            pre_times.append(t1 - t0)
            search_times.append(t2 - t1)

        assert cost == optimal, f'{key}: cost {cost} != optimal {optimal}'
        admissible, consistent = check_table(grid, table, exact)
        assert admissible, f'{key}: heuristic overestimates'

        record[key] = {
            'expanded': expanded,
            'pre': min(pre_times),
            'search': min(search_times),
            'consistent': consistent,
        }
    return record


def mean(xs):
    return sum(xs) / len(xs)


def paired_diffs(results, a, b):
    # Per map: nodes expanded by a minus nodes expanded by b.
    return [r[a]['expanded'] - r[b]['expanded'] for r in results]


def mean_ci(diffs):
    n = len(diffs)
    m = mean(diffs)
    if n < 2:
        return m, m, m
    s = statistics.stdev(diffs)
    t = stats.t.ppf(1 - ALPHA / 2, n - 1)
    half = t * s / math.sqrt(n)
    return m, m - half, m + half


def sign_test_p(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    if pos + neg == 0:
        return 1.0
    return stats.binomtest(min(pos, neg), pos + neg, 0.5).pvalue


def holm_adjust(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def required_n(diffs):
    # n >= ((z_alpha/2 + z_beta) * sd / |mean|)^2 for a paired comparison.
    delta = abs(mean(diffs))
    if delta == 0 or len(diffs) < 2:
        return None
    sd = statistics.stdev(diffs)
    z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)
    return math.ceil((z * sd / delta) ** 2)


def all_pairs():
    return list(itertools.combinations(KEYS, 2))


def sample_size_table(results):
    rows = []
    for a, b in all_pairs():
        diffs = paired_diffs(results, a, b)
        rows.append((a, b, mean(diffs), required_n(diffs)))
    return rows


def sample_size_needed(results):
    needed = [n for _, _, _, n in sample_size_table(results) if n is not None]
    return max(needed) if needed else None


def collect_sample():
    stream = MapStream(SEED)
    results = []
    target = max(PILOT_SIZE, MIN_SIZE)
    log = []

    while True:
        while len(results) < target:
            grid, start, goal = stream.next_map()
            results.append(evaluate(grid, start, goal))
        needed = sample_size_needed(results)
        log.append((len(results), needed))
        if needed is None or needed <= len(results) or len(results) >= MAX_SIZE:
            break
        target = min(MAX_SIZE, needed)
    return results, log


def summarize(results):
    base_exp = [r[BASELINE]['expanded'] for r in results]
    others = [k for k in KEYS if k != BASELINE]
    pvals = [sign_test_p(paired_diffs(results, BASELINE, k)) for k in others]
    adjusted = dict(zip(others, holm_adjust(pvals)))

    rows = []
    for key in KEYS:
        exp = [r[key]['expanded'] for r in results]
        pre = [r[key]['pre'] for r in results]
        search = [r[key]['search'] for r in results]
        consistent = all(r[key]['consistent'] for r in results)
        row = {
            'key': key,
            'label': LABELS[key],
            'mean_exp': mean(exp),
            'median_exp': statistics.median(exp),
            'pre_ms': statistics.median(pre) * 1000,
            'search_ms': statistics.median(search) * 1000,
            'total_ms': statistics.median([p + s for p, s in zip(pre, search)]) * 1000,
            'consistent': '是' if consistent else '否',
        }
        if key == BASELINE:
            row.update(diff='-', ci='-', win='-', p='-', med_ratio='-', gm_ratio='-')
        else:
            diffs = paired_diffs(results, BASELINE, key)
            m, lo, hi = mean_ci(diffs)
            ratios = [e / b for e, b in zip(exp, base_exp)]
            row.update(
                diff=f'{m:.2f}',
                ci=f'[{lo:.2f}, {hi:.2f}]',
                win=f'{sum(1 for d in diffs if d > 0)} / {sum(1 for d in diffs if d == 0)} / '
                    f'{sum(1 for d in diffs if d < 0)}',
                p=f'{adjusted[key]:.2g}',
                med_ratio=f'{statistics.median(ratios):.2f}',
                gm_ratio=f'{math.exp(mean([math.log(x) for x in ratios])):.2f}',
            )
        rows.append(row)
    return rows


def choose_best(results, rows):
    by_key = {row['key']: row for row in rows}
    practical = sorted([k for k in KEYS if k != EXACT], key=lambda k: by_key[k]['mean_exp'])
    top = practical[0]

    tied = [top]
    for k in practical[1:]:
        diffs = paired_diffs(results, k, top)
        _, lo, hi = mean_ci(diffs)
        if lo <= 0 <= hi:
            tied.append(k)

    winner = min(tied, key=lambda k: by_key[k]['total_ms'])
    if len(tied) == 1:
        reason = f'{LABELS[top]} 的平均扩展节点数显著少于其余实际可用方案'
    else:
        names = '、'.join(LABELS[k] for k in tied)
        reason = (f'{names} 与 {LABELS[top]} 的节点数差异不显著，'
                  f'在这些方案中选择预计算加搜索耗时最少者')
    overall = min(rows, key=lambda r: r['mean_exp'])
    return winner, reason, overall


def render_table(rows):
    lines = []
    for r in rows:
        if r['key'] == BASELINE:
            lines.append(f"{r['label']}：平均扩展 {r['mean_exp']:.1f} 个节点，"
                         f"预计算中位耗时 {r['pre_ms']:.3f} 毫秒，搜索中位耗时 {r['search_ms']:.3f} 毫秒")
        else:
            fewer, same, more = r['win'].split(' / ')
            lines.append(f"{r['label']}：平均扩展 {r['mean_exp']:.1f} 个节点，"
                         f"比基准平均少扩展 {r['diff']} 个，95% 置信区间为 {r['ci'].strip('[]').replace(', ', ' 到 ')}，"
                         f"相对基准少 {fewer} 张图、平 {same} 张图、多 {more} 张图，"
                         f"Holm 校正 p 值 {r['p']}，"
                         f"预计算中位耗时 {r['pre_ms']:.3f} 毫秒，搜索中位耗时 {r['search_ms']:.3f} 毫秒")
    return '\n'.join(lines)


def render_pair_table(results):
    lines = []
    for a, b, m, n in sample_size_table(results):
        _, lo, hi = mean_ci(paired_diffs(results, a, b))
        if n is None:
            lines.append(f"{LABELS[a]}与{LABELS[b]}：节点数相同")
        else:
            lines.append(f"{LABELS[a]}与{LABELS[b]}：配对平均差 {m:.2f}，"
                         f"95% 置信区间为 {lo:.2f} 到 {hi:.2f}，所需样本数 {n}")
    return '\n'.join(lines)


def plot(results, rows):
    keys = [r['key'] for r in rows]
    means = [r['mean_exp'] for r in rows]
    errs_low, errs_high = [], []
    for key in keys:
        exp = [r[key]['expanded'] for r in results]
        m, lo, hi = mean_ci(exp)
        errs_low.append(m - lo)
        errs_high.append(hi - m)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    xs = list(range(len(keys)))
    ax.bar(xs, means, yerr=[errs_low, errs_high], capsize=5,
           color=['#999999', '#2b7bba', '#bbbbbb', '#2ca25f', '#8856a7'])
    ax.set_xticks(xs)
    ax.set_xticklabels(keys, rotation=15)
    ax.set_ylabel('mean expanded nodes (95% CI)')
    ax.set_title('Expanded nodes per map')

    ax = axes[1]
    others = [k for k in keys if k != BASELINE]
    data = [paired_diffs(results, BASELINE, k) for k in others]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(others) + 1))
    ax.set_xticklabels(others, rotation=15)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_ylabel('baseline − method (nodes saved per map)')
    ax.set_title('Paired difference vs modified Manhattan')

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)


def run():
    results, log = collect_sample()
    rows = summarize(results)
    winner, reason, overall = choose_best(results, rows)
    needed = log[-1][1]

    lines = [
        f'随机种子 {SEED}，可达地图 {len(results)} 张（公式要求 {needed} 张）。'
        f'所有方案最优代价一致且可采纳。',
        '',
        render_pair_table(results),
        '',
        render_table(rows),
        '',
        f'实际可用最优：{LABELS[winner]}（{reason}）。',
        f'扩展节点最少（含理论上界）：{overall["label"]}，平均 {overall["mean_exp"]:.1f} 个节点。',
    ]

    text = '\n'.join(lines)
    print(text)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    plot(results, rows)
    print(f'图：{PLOT_PATH}')


if __name__ == '__main__':
    run()