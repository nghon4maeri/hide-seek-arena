"""rollout.py — Short-horizon Monte Carlo rollout for decision validation.

Direction 5: Short-horizon Rollout (MCTS Lite)
- Simulate N steps forward from current state
- Evaluate outcome (capture/survival)
- Average over multiple rollouts for robustness
- Used to validate heuristic decisions before execution
"""

import random
from typing import Optional, Tuple

import numpy as np

from pathfinding import (
    DIRS, _shape, _valid, _apply, _manhattan,
    _cell_exits, _legal, bfs_dist,
)
from topology import TopologyAnalyzer, STATIC_FULL_MAP
from belief_state import BeliefState


def rollout_ghost(
    ghost_pos: Tuple,
    ghost_move: Tuple,
    belief: BeliefState,
    memory_map: np.ndarray,
    topo: TopologyAnalyzer,
    horizon: int = 5,
    n_rollouts: int = 10,
    pacman_speed: int = 2,
) -> float:
    """Simulate n_rollouts from ghost_pos after ghost_move.

    Returns average survival score.
    """
    total_score = 0.0

    for _ in range(n_rollouts):
        score = _single_rollout_ghost(
            ghost_pos, ghost_move, belief, memory_map, topo,
            horizon, pacman_speed
        )
        total_score += score

    return total_score / max(1, n_rollouts)


def _single_rollout_ghost(
    ghost_pos: Tuple,
    ghost_move: Tuple,
    belief: BeliefState,
    memory_map: np.ndarray,
    topo: TopologyAnalyzer,
    horizon: int,
    pacman_speed: int,
) -> float:
    """Single rollout simulation for Ghost."""
    g_pos = _apply(ghost_pos, ghost_move)
    if not _valid(g_pos, memory_map):
        return -10000.0

    # Sample Pacman position from belief
    prob_cells = belief.highest_prob_cells(top_k=5)
    if prob_cells:
        probs = [p for _, p in prob_cells]
        total = sum(probs)
        if total > 0:
            r = random.random() * total
            cumsum = 0.0
            p_pos = prob_cells[0][0]
            for cell, prob in prob_cells:
                cumsum += prob
                if r <= cumsum:
                    p_pos = cell
                    break
        else:
            p_pos = belief.threat_center()
    else:
        p_pos = belief.threat_center()

    score = 0.0
    survived = True

    for step in range(horizon):
        # Ghost moves: prefer moves away from Pacman
        g_legal = _legal(g_pos, memory_map)
        if not g_legal:
            survived = False
            break

        best_g_move = g_legal[0]
        best_dist = -1
        for m in g_legal:
            nxt = _apply(g_pos, m)
            d = _manhattan(nxt, p_pos)
            if d > best_dist:
                best_dist = d
                best_g_move = m
        g_pos = _apply(g_pos, best_g_move)

        # Pacman moves: toward Ghost (heuristic)
        p_legal = _legal(p_pos, memory_map)
        if p_legal:
            best_p_move = p_legal[0]
            best_p_dist = float("inf")
            for m in p_legal:
                nxt = _apply(p_pos, m)
                d = _manhattan(nxt, g_pos)
                if d < best_p_dist:
                    best_p_dist = d
                    best_p_move = m
            p_pos = _apply(p_pos, best_p_move)

            # Pacman speed
            for _ in range(1, pacman_speed):
                nxt = _apply(p_pos, best_p_move)
                if not _valid(nxt, memory_map):
                    break
                p_pos = nxt

        # Check capture
        if _manhattan(g_pos, p_pos) < 2:
            survived = False
            score -= 5000.0
            break

        # Step score
        score += 10.0
        if topo.ready:
            if g_pos in topo.junctions:
                score += 20.0
            if g_pos in topo.loops:
                score += 30.0
            if g_pos in topo.dead_ends:
                score -= 100.0

    if survived:
        score += 100.0

    return score


def rollout_pacman(
    pacman_pos: Tuple,
    pacman_move: Tuple,
    enemy_pos: Optional[Tuple],
    belief: BeliefState,
    memory_map: np.ndarray,
    topo: TopologyAnalyzer,
    horizon: int = 5,
    n_rollouts: int = 10,
    pacman_speed: int = 2,
) -> float:
    """Simulate n_rollouts from pacman_pos after pacman_move.

    Returns average capture score.
    """
    total_score = 0.0

    for _ in range(n_rollouts):
        score = _single_rollout_pacman(
            pacman_pos, pacman_move, enemy_pos, belief, memory_map, topo,
            horizon, pacman_speed
        )
        total_score += score

    return total_score / max(1, n_rollouts)


def _single_rollout_pacman(
    pacman_pos: Tuple,
    pacman_move: Tuple,
    enemy_pos: Optional[Tuple],
    belief: BeliefState,
    memory_map: np.ndarray,
    topo: TopologyAnalyzer,
    horizon: int,
    pacman_speed: int,
) -> float:
    """Single rollout simulation for Pacman."""
    p_pos = _apply(pacman_pos, pacman_move)
    if not _valid(p_pos, memory_map):
        return -10000.0

    # Ghost position
    if enemy_pos is not None:
        g_pos = enemy_pos
    else:
        prob_cells = belief.highest_prob_cells(top_k=3)
        if prob_cells:
            g_pos = prob_cells[0][0]
        else:
            g_pos = belief.threat_center()

    score = 0.0
    captured = False

    for step in range(horizon):
        # Check capture
        if _manhattan(p_pos, g_pos) < 2:
            captured = True
            score += 5000.0
            break

        # Pacman moves toward Ghost
        p_legal = _legal(p_pos, memory_map)
        if p_legal:
            best_p_move = p_legal[0]
            best_dist = float("inf")
            for m in p_legal:
                nxt = _apply(p_pos, m)
                d = _manhattan(nxt, g_pos)
                if d < best_dist:
                    best_dist = d
                    best_p_move = m
            p_pos = _apply(p_pos, best_p_move)

            # Speed
            for _ in range(1, pacman_speed):
                nxt = _apply(p_pos, best_p_move)
                if not _valid(nxt, memory_map):
                    break
                p_pos = nxt

        # Ghost moves away from Pacman
        g_legal = _legal(g_pos, memory_map)
        if g_legal:
            best_g_move = g_legal[0]
            best_dist = -1
            for m in g_legal:
                nxt = _apply(g_pos, m)
                d = _manhattan(nxt, p_pos)
                if d > best_dist:
                    best_dist = d
                    best_g_move = m
            g_pos = _apply(g_pos, best_g_move)

        # Step score
        score -= 1.0
        dist = _manhattan(p_pos, g_pos)
        score += max(0, 20 - dist)

    return score
