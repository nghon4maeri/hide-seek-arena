"""game_theory.py — Expectiminimax and decision-making under uncertainty.

Direction 1: Game Theory
- Expectiminimax with belief state (chance nodes weighted by belief)
- Evaluation functions based on distance, topology, escape routes
- Used by both Pacman (seek) and Ghost (evade) agents
"""

from typing import Optional, Tuple

import numpy as np

from pathfinding import (
    DIRS, _shape, _cell, _valid, _apply, _manhattan,
    _cell_exits, _legal, bfs_dist, astar,
)
from topology import TopologyAnalyzer, STATIC_FULL_MAP
from belief_state import BeliefState


def expectiminimax_ghost(
    ghost_pos: Tuple,
    belief: BeliefState,
    memory_map: np.ndarray,
    topo: TopologyAnalyzer,
    depth: int = 2,
    pacman_speed: int = 2,
) -> Tuple:
    """Expectiminimax for Ghost agent.

    Ghost chooses move to maximize survival.
    Pacman's response is weighted by belief state (chance node).

    Returns best delta (direction) for Ghost.
    """
    legal_moves = _legal(ghost_pos, memory_map)
    if not legal_moves:
        return (0, 0)

    best_move = legal_moves[0]
    best_score = float("-inf")

    for move in legal_moves:
        ghost_next = _apply(ghost_pos, move)

        # Chance node: Pacman's possible positions weighted by belief
        expected_score = 0.0
        total_prob = 0.0

        # Sample high-probability Pacman positions
        prob_cells = belief.highest_prob_cells(top_k=8)
        if not prob_cells:
            prob_cells = [((10, 10), 0.1)]

        for pac_pos, prob in prob_cells:
            if prob < 0.01:
                continue

            # Pacman's best response (minimize distance to ghost)
            pac_legal = _legal(pac_pos, memory_map)
            worst_pac_dist = float("inf")
            for pac_move in pac_legal:
                pac_next = _apply(pac_pos, pac_move)
                # Pacman can move up to pacman_speed steps in same direction
                for step in range(1, pacman_speed + 1):
                    pac_far = _apply(pac_next, (pac_move[0] * (step - 1), pac_move[1] * (step - 1)))
                    if not _valid(pac_far, memory_map):
                        break
                    d = _manhattan(ghost_next, pac_far)
                    worst_pac_dist = min(worst_pac_dist, d)

            score = _evaluate_ghost(ghost_next, pac_next if pac_legal else pac_pos,
                                    worst_pac_dist, memory_map, topo, belief)
            expected_score += prob * score
            total_prob += prob

        if total_prob > 0:
            expected_score /= total_prob

        if expected_score > best_score:
            best_score = expected_score
            best_move = move

    return best_move


def _evaluate_ghost(
    ghost_pos: Tuple,
    pacman_pos: Tuple,
    min_pac_dist: float,
    memory_map: np.ndarray,
    topo: TopologyAnalyzer,
    belief: BeliefState,
) -> float:
    """Evaluate Ghost position quality."""
    score = 0.0

    # Distance from Pacman (primary)
    dist = _manhattan(ghost_pos, pacman_pos)
    score += min_pac_dist * 100.0

    # Immediate capture check
    if dist < 2:
        score -= 10000.0

    # Topology bonus
    if topo.ready:
        exits = _cell_exits(ghost_pos, STATIC_FULL_MAP)
        score += exits * 150.0
        if ghost_pos in topo.junctions:
            score += 500.0
        if ghost_pos in topo.loops:
            score += 800.0
        if ghost_pos in topo.dead_ends:
            score -= 5000.0
        if ghost_pos in topo.corridor_cells:
            score -= 1000.0

    # Danger time bonus (more time = safer)
    danger = belief.danger_at(ghost_pos)
    score += danger * 50.0

    # Escape routes
    reachable = bfs_dist(memory_map, ghost_pos, max_dist=6)
    score += len(reachable) * 5.0

    return score


def expectiminimax_pacman(
    pacman_pos: Tuple,
    enemy_pos: Optional[Tuple],
    belief: BeliefState,
    memory_map: np.ndarray,
    topo: TopologyAnalyzer,
    depth: int = 2,
    pacman_speed: int = 2,
) -> Tuple:
    """Expectiminimax for Pacman agent.

    Pacman chooses move to minimize distance to Ghost.
    Ghost's response is weighted by belief state.

    Returns best delta (direction) for Pacman.
    """
    if enemy_pos is not None:
        # Direct chase with A*
        path = astar(memory_map, pacman_pos, enemy_pos)
        if path:
            return path[0]
        return (0, 0)

    # When enemy not visible, use belief to guide
    legal_moves = _legal(pacman_pos, memory_map)
    if not legal_moves:
        return (0, 0)

    best_move = legal_moves[0]
    best_score = float("-inf")

    for move in legal_moves:
        pac_next = _apply(pacman_pos, move)

        # Expected distance to Ghost based on belief
        expected_dist = 0.0
        total_prob = 0.0

        prob_cells = belief.highest_prob_cells(top_k=8)
        for ghost_pos, prob in prob_cells:
            if prob < 0.01:
                continue
            d = _manhattan(pac_next, ghost_pos)
            expected_dist += prob * d
            total_prob += prob

        if total_prob > 0:
            expected_dist /= total_prob
        else:
            expected_dist = 999

        # Score: minimize expected distance
        score = -expected_dist

        # Bonus for topology (junctions help catch Ghost)
        if topo.ready and pac_next in topo.junctions:
            score += 50.0

        if score > best_score:
            best_score = score
            best_move = move

    return best_move
