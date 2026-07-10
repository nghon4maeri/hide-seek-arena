"""Move ordering and depth heuristics for adversarial search."""

from __future__ import annotations

from math import isinf


def own_actions(state, role, analysis, legal_pacman_actions_fn, legal_ghost_actions_fn, simulate_fn, speed):
    if role == "seek":
        return ordered_pacman_actions(state, analysis, legal_pacman_actions_fn, simulate_fn, speed)
    return ordered_ghost_actions(state, analysis, legal_ghost_actions_fn, simulate_fn)


def opponent_actions(state, role, analysis, legal_pacman_actions_fn, legal_ghost_actions_fn, simulate_fn, speed):
    if role == "seek":
        return ordered_ghost_actions(state, analysis, legal_ghost_actions_fn, simulate_fn)
    return ordered_pacman_actions(state, analysis, legal_pacman_actions_fn, simulate_fn, speed)


def ordered_pacman_actions(state, analysis, legal_pacman_actions_fn, simulate_fn, speed):
    from environment import Move

    return tuple(sorted(legal_pacman_actions_fn(state.pacman, analysis, speed), key=lambda action: _pacman_priority(state, action, analysis, simulate_fn, Move.STAY)))


def ordered_ghost_actions(state, analysis, legal_ghost_actions_fn, simulate_fn):
    from environment import Move

    return tuple(sorted(legal_ghost_actions_fn(state.ghost, analysis), key=lambda action: _ghost_priority(state, action, analysis, simulate_fn, Move.STAY)))


def adaptive_depth_limit(state, analysis, own_branching, opponent_branching):
    distance = analysis.dist(state.pacman, state.ghost)
    branching = own_branching * max(1, opponent_branching)
    if isinf(distance):
        return 2
    depth = 4 if distance <= 4 else 3 if distance <= 8 else 2
    if branching > 40 and distance > 4:
        depth -= 1
    return max(2, min(depth, 4))


def opponent_branching(state, role, analysis, legal_pacman_actions_fn, legal_ghost_actions_fn, speed):
    if role == "seek":
        return len(legal_ghost_actions_fn(state.ghost, analysis))
    return len(legal_pacman_actions_fn(state.pacman, analysis, speed))


def _pacman_priority(state, action, analysis, simulate_fn, ghost_stay):
    new_pacman, _, captured = simulate_fn(state.pacman, state.ghost, action, ghost_stay, analysis)
    distance = analysis.dist(new_pacman, state.ghost)
    return (0 if captured else 1, distance if not isinf(distance) else 999, -action[1])


def _ghost_priority(state, action, analysis, simulate_fn, pacman_stay):
    _, new_ghost, captured = simulate_fn(state.pacman, state.ghost, (pacman_stay, 1), action, analysis)
    distance = analysis.dist(new_ghost, state.pacman)
    return (1 if captured else 0, -len(analysis.neighbors(new_ghost)), -analysis.region_size(new_ghost), -(distance if not isinf(distance) else 999))
