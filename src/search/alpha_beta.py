"""Alpha-beta pruning utility predicates."""


def should_prune(alpha: float, beta: float) -> bool:
    return beta <= alpha

