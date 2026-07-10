"""Runtime hybrid Hide policy: safe floor + flee tie-break.

Problem with the pure offline table: it is the ghost's best-response to OUR seek
only. It survives the game-value vs strong seekers but does not run up the clock
vs weak seekers (it sits on positionally-optimal cells the strong seeker would
punish, which a weak seeker never reaches).

Design:
  - Floor: depth-1 maximin where the leaf is the offline table VALUE (an accurate
    survival-vs-optimal-seek estimate). This guarantees we never pick a move whose
    worst-case survival is worse than the table's — strong seekers still get held
    to the game value.
  - Exploit: among moves whose worst-case survival is within TOLERANCE of the best,
    pick the one that maximizes a flee score (graph distance to seeker, open region,
    exits, avoid dead-ends). Against a weak seeker that does not pursue optimally,
    fleeing the open way extends real survival far beyond the worst-case bound.

The table VALUE leaf keeps the search shallow (depth-1) yet deep in effect, so the
per-step cost is well under the 1 s arena timeout.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

from environment import Move
from evaluation_features import (
    dead_end_depth,
    exit_count,
    graph_distance,
    region_size,
    repeat_penalty,
)
from game_core import (
    GameState,
    MapAnalysis,
    legal_ghost_actions,
    legal_pacman_actions,
    simulate,
)
from hide_logic import HIDE_TABLE_INF, hide_table_value

Cell = Tuple[int, int]

# Allowed worst-case survival sacrifice (steps) to chase a better flee line.
# 0 = pure safe (≈ table). 1-2 trades a little theoretical worst case for large
# real gains vs weak seekers. Tuned empirically against the seeker field.
SURVIVAL_TOLERANCE = float(os.environ.get("HIDE_TOL", "1"))

# Flee score weights (only ever break ties among equally-safe moves).
_W_DISTANCE = 10.0
_W_EXITS = 6.0
_W_REGION = 0.4
_W_DEAD_END = 12.0
_W_REPEAT = 4.0

# Leaf value used when the table has no entry (off-table) but the ghost is not
# captured — large so off-table survival is preferred over known short lines.
_OFF_TABLE_VALUE = 50.0


def _flee_score(ghost: Cell, pac: Cell, analysis: MapAnalysis,
                history: Iterable[Cell]) -> float:
    return (
        _W_DISTANCE * graph_distance(analysis, ghost, pac)
        + _W_EXITS * exit_count(analysis, ghost)
        + _W_REGION * region_size(analysis, ghost)
        - _W_DEAD_END * dead_end_depth(analysis, ghost)
        - _W_REPEAT * repeat_penalty(ghost, tuple(history))
    )


def _leaf_survival(pac: Cell, ghost: Cell, map_state) -> float:
    """Accurate survival estimate for a turn-start state via the table value."""
    v = hide_table_value(pac, ghost, map_state)
    if v is None:
        return _OFF_TABLE_VALUE
    if v >= HIDE_TABLE_INF:
        return float(HIDE_TABLE_INF)
    return float(v)


def choose_hybrid_hide_move(
    map_state,
    ghost: Cell,
    pac: Optional[Cell],
    history: Optional[Iterable[Cell]] = None,
    pacman_speed: int = 2,
) -> Optional[Move]:
    """Return the hybrid Hide Move, or None to let the caller fall back.

    None on: no threat known, table unavailable for the map, or any internal error.
    Never raises.
    """
    if pac is None:
        return None
    try:
        hist = tuple(history or ())
        analysis = MapAnalysis(map_state, ghost)
        # Table must be valid for this map; otherwise defer to existing fallbacks.
        if hide_table_value(pac, ghost, map_state) is None:
            return None

        ghost_moves: List[Move] = list(legal_ghost_actions(ghost, analysis))
        if Move.STAY not in ghost_moves:
            ghost_moves.append(Move.STAY)
        pac_replies = list(legal_pacman_actions(pac, analysis, pacman_speed))

        scored = []  # (worst_case_survival, flee_score, landing, move)
        for mg in ghost_moves:
            worst = float("inf")
            landing = ghost
            for pa in pac_replies:
                new_pac, new_ghost, captured = simulate(pac, ghost, pa, mg, analysis)
                landing = new_ghost
                if captured:
                    worst = 0.0
                    break
                val = 1.0 + _leaf_survival(new_pac, new_ghost, map_state)
                if val < worst:
                    worst = val
            flee = _flee_score(landing, pac, analysis, hist)
            scored.append((worst, flee, mg))

        if not scored:
            return None

        best_worst = max(s[0] for s in scored)
        # Among moves within tolerance of the safe floor, maximize flee.
        candidates = [s for s in scored if s[0] >= best_worst - SURVIVAL_TOLERANCE]
        candidates.sort(key=lambda s: s[1], reverse=True)
        return candidates[0][2]
    except Exception:
        return None
