"""Runtime hybrid Seek policy: herding exploit + minimax safety leash.

A pure minimax seek catches every ghost at the game value (~11) but no faster —
it plays optimally vs a worst-case evader, so a NAIVE fleer that walks itself into
a trap (greedy / max-distance hides) still takes ~12 instead of the ~6 a cornering
seek achieves. Capture speed is the tournament tie-break, so that gap costs rank.

To catch a predictable fleer fast we must take moves that are NOT minimax-optimal
vs a worst-case ghost (no faster capture exists inside the optimal set) — we bet on
the ghost's sub-optimality. That bet is made SAFE by two guarantees:

  1. Worst-case slack: a move is allowed only if its guaranteed capture (vs optimal
     ghost, from the seek table) is within SLACK steps of the fastest. Bounds how
     much theoretical worst-case we trade for the herding bet.
  2. Step leash: once the step count is high, play the strict table move, which
     catches any ghost in <= game-value (~30) more steps — so the win is reached
     well under the cap even if the herding bet never pays off.

The herding objective itself is a depth-limited search where the ghost is forced
to its greedy-flee reaction (an accurate model of naive hides) and the pac
minimizes capture steps — this finds the move sequence that drives such a ghost
into a dead-end fast. Against a strong (non-greedy) ghost the model is wrong, but
the slack + leash keep capture safe and the win guaranteed.
"""

from __future__ import annotations

import os
from time import perf_counter
from typing import Dict, List, Optional, Tuple

from environment import Move
from game_core import (
    MapAnalysis,
    best_greedy_ghost_move,
    legal_ghost_actions,
    legal_pacman_actions,
    simulate,
)
from seek_logic import SEEK_TABLE_INF, lookup as seek_table_lookup, seek_table_value

Cell = Tuple[int, int]
PacAction = Tuple[Move, int]

# Worst-case capture slack (steps) traded for the herding bet. 0 = pure minimax.
# 1 is the empirical sweet spot: enough room to corner a naive fleer fast, but not
# enough for a STRONG ghost to stall capture past the game value (2+ regresses it).
# Made safe by the step leash below, so it can never cost the win.
SEEK_SLACK = float(os.environ.get("SEEK_SLACK", "1"))

# Beyond this step the seek plays the strict table move — guarantees capture in
# <= game-value more steps, so the win lands well under the 200-step cap even if
# the herding bet stalls against a ghost that resists it.
SEEK_LEASH_STEP = int(os.environ.get("SEEK_LEASH", "140"))

# Herding rollout depth (pac plies) and time budget. The table value at the leaf
# makes shallow search effective; the deadline guards the 1 s arena limit.
_HERD_DEPTH = int(os.environ.get("SEEK_HERD_DEPTH", "6"))
_HERD_BUDGET = 0.15

_OFF_TABLE_VALUE = 50.0


def _leaf(pac: Cell, ghost: Cell, map_state) -> float:
    v = seek_table_value(pac, ghost, map_state)
    if v is None:
        return _OFF_TABLE_VALUE
    if v >= SEEK_TABLE_INF:
        return float(SEEK_TABLE_INF)
    return float(v)


def _herd_steps(pac: Cell, ghost: Cell, depth: int, analysis: MapAnalysis,
                map_state, speed: int, deadline: float,
                memo: Dict) -> float:
    """Predicted steps for the pac to catch a GREEDY-flee ghost from this state.

    Pac minimizes; the ghost is forced to its greedy-flee reaction (deterministic),
    so this is a single-agent herding search, not a minimax. Lower = caught sooner.
    """
    key = (pac, ghost, depth)
    cached = memo.get(key)
    if cached is not None:
        return cached
    if depth <= 0 or perf_counter() >= deadline:
        # Fall back to the optimal-pursuit estimate from the table.
        return _leaf(pac, ghost, map_state)
    g_move = best_greedy_ghost_move(ghost, pac, map_state)  # reacts to old pac
    best = float("inf")
    for pa in legal_pacman_actions(pac, analysis, speed):
        new_pac, new_ghost, captured = simulate(pac, ghost, pa, g_move, analysis)
        if captured:
            best = 1.0
            break
        steps = 1.0 + _herd_steps(new_pac, new_ghost, depth - 1, analysis,
                                  map_state, speed, deadline, memo)
        if steps < best:
            best = steps
    memo[key] = best
    return best


def choose_hybrid_seek_action(
    map_state,
    pac: Cell,
    ghost: Optional[Cell],
    pacman_speed: int = 2,
    step_number: int = 0,
) -> Optional[PacAction]:
    """Return the hybrid Seek (Move, steps), or None to let the caller fall back.

    None on: ghost unknown, table unavailable for the map, or any internal error.
    Never raises.
    """
    if ghost is None:
        return None
    try:
        v_now = seek_table_value(pac, ghost, map_state)
        if v_now is None or v_now >= SEEK_TABLE_INF:
            return None  # off-table — defer to existing fallbacks

        # Leash: late in the game, play the guaranteed-capture table move.
        if step_number >= SEEK_LEASH_STEP:
            return seek_table_lookup(pac, ghost, map_state)

        analysis = MapAnalysis(map_state, pac)
        pac_actions: List[PacAction] = list(legal_pacman_actions(pac, analysis, pacman_speed))
        if not pac_actions:
            return None
        ghost_replies = list(legal_ghost_actions(ghost, analysis))
        if Move.STAY not in ghost_replies:
            ghost_replies.append(Move.STAY)

        # Worst-case guaranteed capture (vs optimal ghost) for each pac move.
        worst: Dict[int, float] = {}
        for i, pa in enumerate(pac_actions):
            w = 0.0
            for gr in ghost_replies:
                new_pac, new_ghost, captured = simulate(pac, ghost, pa, gr, analysis)
                step_cost = 1.0 if captured else 1.0 + _leaf(new_pac, new_ghost, map_state)
                if step_cost > w:
                    w = step_cost
            worst[i] = w
        best_floor = min(worst.values())

        # Candidates: never trade more than SLACK worst-case steps for the bet.
        candidates = [i for i, pa in enumerate(pac_actions)
                      if worst[i] <= best_floor + SEEK_SLACK]

        # Herding objective: among safe candidates, the move that drives a greedy
        # fleer into capture fastest. Tie-break toward lower worst-case.
        deadline = perf_counter() + _HERD_BUDGET
        memo: Dict = {}
        best_i, best_e, best_w = candidates[0], float("inf"), float("inf")
        for i in candidates:
            pa = pac_actions[i]
            g_move = best_greedy_ghost_move(ghost, pac, map_state)
            new_pac, new_ghost, captured = simulate(pac, ghost, pa, g_move, analysis)
            if captured:
                e = 1.0
            else:
                e = 1.0 + _herd_steps(new_pac, new_ghost, _HERD_DEPTH - 1, analysis,
                                      map_state, pacman_speed, deadline, memo)
            if e < best_e or (e == best_e and worst[i] < best_w):
                best_i, best_e, best_w = i, e, worst[i]
        return pac_actions[best_i]
    except Exception:
        return None
