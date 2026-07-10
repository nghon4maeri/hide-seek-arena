"""Timed simultaneous-aware maximin search helpers."""

from __future__ import annotations

from time import perf_counter
from typing import Callable

from search_ordering import adaptive_depth_limit, opponent_actions, opponent_branching, own_actions

MAX_STEP_BUDGET = 0.78
MIN_SEARCH_BUDGET = 0.005


class _Timeout(Exception):
    pass


def alpha_beta_search(
    state,
    role: str,
    evaluate_fn: Callable,
    analysis,
    legal_pacman_actions_fn: Callable,
    legal_ghost_actions_fn: Callable,
    simulate_fn: Callable,
    pacman_speed: int = 2,
    time_budget: float = MAX_STEP_BUDGET,
    max_depth: int | None = None,
):
    """Return the best action from fully completed iterative-deepening searches."""
    role = _normalize_role(role)
    if state.ghost is None:
        return _fallback_action(role)
    actions = own_actions(state, role, analysis, legal_pacman_actions_fn, legal_ghost_actions_fn, simulate_fn, pacman_speed)
    if not actions:
        return _fallback_action(role)

    best_action = actions[0]
    deadline = perf_counter() + _safe_budget(time_budget)
    depth_limit = max_depth or adaptive_depth_limit(
        state,
        analysis,
        len(actions),
        opponent_branching(state, role, analysis, legal_pacman_actions_fn, legal_ghost_actions_fn, pacman_speed),
    )
    for depth in range(1, depth_limit + 1):
        try:
            best_action, _ = _root_search(
                state,
                role,
                depth,
                evaluate_fn,
                analysis,
                legal_pacman_actions_fn,
                legal_ghost_actions_fn,
                simulate_fn,
                pacman_speed,
                deadline,
                {},
            )
        except _Timeout:
            break
    return best_action


def one_ply_search(
    state,
    role: str,
    evaluate_fn: Callable,
    analysis,
    legal_pacman_actions_fn: Callable,
    legal_ghost_actions_fn: Callable,
    simulate_fn: Callable,
    pacman_speed: int = 2,
):
    return alpha_beta_search(
        state,
        role,
        evaluate_fn,
        analysis,
        legal_pacman_actions_fn,
        legal_ghost_actions_fn,
        simulate_fn,
        pacman_speed,
        time_budget=0.05,
        max_depth=1,
    )


def _root_search(state, role, depth, evaluate_fn, analysis, legal_pacman_actions_fn, legal_ghost_actions_fn, simulate_fn, speed, deadline, cache):
    best_action = _fallback_action(role)
    best_score = float("-inf")
    alpha = float("-inf")
    for action in own_actions(state, role, analysis, legal_pacman_actions_fn, legal_ghost_actions_fn, simulate_fn, speed):
        score = _worst_reply(
            state,
            role,
            action,
            depth,
            evaluate_fn,
            analysis,
            legal_pacman_actions_fn,
            legal_ghost_actions_fn,
            simulate_fn,
            speed,
            deadline,
            cache,
            alpha,
            float("inf"),
        )
        if score > best_score:
            best_action, best_score = action, score
        alpha = max(alpha, best_score)
    return best_action, best_score


def _maximin(state, role, depth, evaluate_fn, analysis, legal_pacman_actions_fn, legal_ghost_actions_fn, simulate_fn, speed, deadline, cache, alpha, beta):
    _check_time(deadline)
    if depth <= 0:
        return float(evaluate_fn(state, analysis))
    key = (role, state.pacman, state.ghost, state.step_number, depth)
    if key in cache:
        return cache[key]

    value = float("-inf")
    exact = True
    for action in own_actions(state, role, analysis, legal_pacman_actions_fn, legal_ghost_actions_fn, simulate_fn, speed):
        score = _worst_reply(
            state,
            role,
            action,
            depth,
            evaluate_fn,
            analysis,
            legal_pacman_actions_fn,
            legal_ghost_actions_fn,
            simulate_fn,
            speed,
            deadline,
            cache,
            alpha,
            beta,
        )
        value = max(value, score)
        alpha = max(alpha, value)
        if alpha >= beta:
            exact = False
            break
    if exact:
        cache[key] = value
    return value


def _worst_reply(state, role, action, depth, evaluate_fn, analysis, legal_pacman_actions_fn, legal_ghost_actions_fn, simulate_fn, speed, deadline, cache, alpha, beta):
    worst = float("inf")
    for reply in opponent_actions(state, role, analysis, legal_pacman_actions_fn, legal_ghost_actions_fn, simulate_fn, speed):
        score = _score_after_turn(
            state,
            role,
            action,
            reply,
            depth - 1,
            evaluate_fn,
            analysis,
            legal_pacman_actions_fn,
            legal_ghost_actions_fn,
            simulate_fn,
            speed,
            deadline,
            cache,
            alpha,
            min(beta, worst),
        )
        worst = min(worst, score)
        if worst <= alpha:
            break
    return worst


def _score_after_turn(state, role, action, reply, depth, evaluate_fn, analysis, legal_pacman_actions_fn, legal_ghost_actions_fn, simulate_fn, speed, deadline, cache, alpha, beta):
    pac_action, ghost_action = (action, reply) if role == "seek" else (reply, action)
    new_pacman, new_ghost, captured = simulate_fn(state.pacman, state.ghost, pac_action, ghost_action, analysis)
    next_state = type(state)(new_pacman, new_ghost, state.step_number + 1)
    if captured or depth <= 0:
        return float(evaluate_fn(next_state, analysis))
    return _maximin(
        next_state,
        role,
        depth,
        evaluate_fn,
        analysis,
        legal_pacman_actions_fn,
        legal_ghost_actions_fn,
        simulate_fn,
        speed,
        deadline,
        cache,
        alpha,
        beta,
    )


def _safe_budget(time_budget):
    try:
        budget = float(time_budget)
    except (TypeError, ValueError):
        budget = MAX_STEP_BUDGET
    return min(MAX_STEP_BUDGET, max(MIN_SEARCH_BUDGET, budget))


def _check_time(deadline):
    if perf_counter() >= deadline:
        raise _Timeout


def _normalize_role(role: str):
    normalized = role.lower()
    if normalized in {"seek", "pacman"}:
        return "seek"
    if normalized in {"hide", "ghost"}:
        return "hide"
    raise ValueError(f"Unknown role: {role}")


def _fallback_action(role: str):
    from environment import Move

    return (Move.STAY, 1) if role.lower() in {"seek", "pacman"} else Move.STAY
