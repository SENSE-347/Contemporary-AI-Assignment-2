import math

INF = math.inf


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value= value
        self.children= children if children is not None else []


def build_tree(data, depth=0):
    if 'value' in data:
        if 'children' in data:
            raise ValueError('a node cannot have both value and children')
        return TreeNode(value=data['value'])

    children_data= data.get('children')
    if not children_data:
        raise ValueError('internal node needs at least one child')

    should_max= depth % 2==0
    if data.get('is_maximizing_player') !=should_max:
        raise ValueError(f'node at depth {depth} has the wrong player flag')

    children =[build_tree(child, depth + 1) for child in children_data]
    return TreeNode(children=children)


def minimax(node, is_max, stats, reverse=False):
    stats['nodes'] +=1
    if not node.children:
        return node.value

    children = node.children[::-1] if reverse else node.children
    if is_max:
        return max(minimax(child, False, stats, reverse) for child in children)
    return min(minimax(child, True, stats, reverse) for child in children)


def alphabeta(node, is_max, alpha, beta, stats, reverse=False):
    stats['nodes'] +=1
    if not node.children:
        return node.value

    children = node.children[::-1] if reverse else node.children
    if is_max:
        best = -INF
        for child in children:
            best = max(best, alphabeta(child, False, alpha, beta, stats, reverse))
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    best = INF
    for child in children:
        best =min(best, alphabeta(child, True, alpha, beta, stats, reverse))
        beta =min(beta, best)
        if alpha >= beta:
            break
    return best


def best_score(root, use_alphabeta=True, reverse=False):
    stats= {'nodes': 0}
    if use_alphabeta:
        score = alphabeta(root, True, -INF, INF, stats, reverse)
    else:
        score = minimax(root, True, stats, reverse)
    return score, stats['nodes']
