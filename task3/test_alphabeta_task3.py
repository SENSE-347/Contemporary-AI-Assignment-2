import json
import os
import random

from alphabeta_task3 import best_score, build_tree

CASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cases_task3.txt')

def load_cases(path):
    cases =[]
    case =None

    with open(path, encoding='utf-8') as f:
        for raw in f:
            line =raw.strip()
            if not line:
                continue
            if line.startswith('# name:'):
                case ={'name': line[len('# name:'):].strip()}
                cases.append(case)
            elif line.startswith('#'):
                continue
            else:
                key, value =line.split(' ', 1)
                if key =='tree':
                    case['tree'] =json.loads(value)
                elif key == 'expect':
                    case['expect'] =int(value)
    return cases


def dict_score(data, is_max):
    """Reference score computed directly from the JSON dict."""
    if 'value' in data:
        return data['value']
    scores =[dict_score(child, not is_max) for child in data['children']]
    return max(scores) if is_max else min(scores)


def check_one(name, data, expect=None):
    root =build_tree(data)
    ref =dict_score(data, True)

    score_ab, nodes_ab =best_score(root, use_alphabeta=True)
    score_mm, nodes_mm = best_score(root, use_alphabeta=False)
    score_rev, _ =best_score(root, use_alphabeta=True, reverse=True)
    score_rev_mm, _ =best_score(root, use_alphabeta=False, reverse=True)

    assert score_ab == ref, f'{name}: alpha-beta {score_ab} != reference {ref}'
    assert score_mm == ref, f'{name}: minimax {score_mm} != reference {ref}'
    assert score_rev == ref, f'{name}: reversed alpha-beta {score_rev} != reference {ref}'
    assert score_rev_mm == ref, f'{name}: reversed minimax {score_rev_mm} != reference {ref}'
    assert nodes_ab <= nodes_mm, f'{name}: alpha-beta visited more nodes than minimax'
    if expect is not None:
        assert score_ab == expect, f'{name}: score {score_ab} != expected {expect}'
    return score_ab, nodes_ab, nodes_mm


def run_case(case):
    return check_one(case['name'], case['tree']['tree_structure'], case['expect'])


def report_orders(case):
    root = build_tree(case['tree']['tree_structure'])
    _, nodes_left = best_score(root, use_alphabeta=True)
    _, nodes_right = best_score(root, use_alphabeta=True, reverse=True)
    _, nodes_mm = best_score(root, use_alphabeta=False)
    print(f"[{case['name']}] visited: minimax={nodes_mm}, "
          f"alpha-beta left-to-right={nodes_left}, alpha-beta right-to-left={nodes_right}")


def random_tree(rng, level, max_level):
    if level == max_level or (level > 0 and rng.random() < 0.2):
        return {'value': rng.randint(-10, 10)}
    count = rng.randint(1, 4)
    return {
        'is_maximizing_player': level % 2 == 0,
        'children': [random_tree(rng, level + 1, max_level) for _ in range(count)],
    }


def run_random(count=300, seed=2026):
    rng = random.Random(seed)
    for i in range(count):
        data = random_tree(rng, 0, rng.randint(1, 5))
        check_one(f'random_{i}', data)
    print(f"random: {count} trees passed")


def expect_error(data):
    try:
        build_tree(data)
    except ValueError:
        return
    raise AssertionError(f'expected ValueError for {data}')


def run_bad_inputs():
    expect_error({'children': []})
    expect_error({'is_maximizing_player': True})
    expect_error({'is_maximizing_player': False, 'children': [{'value': 1}]})
    expect_error({'is_maximizing_player': True, 'value': 1, 'children': [{'value': 2}]})
    expect_error({'is_maximizing_player': True, 'children': [
        {'is_maximizing_player': True, 'children': [{'value': 1}]}]})
    print("bad inputs: 5 rejected")


if __name__ == '__main__':
    cases = load_cases(CASE_FILE)
    for case in cases:
        run_case(case)
        report_orders(case)
    print(f"{len(cases)} cases passed")
    run_random()
    run_bad_inputs()
