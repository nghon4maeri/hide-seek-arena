"""Hide-side action selection and evaluation using iterative-deepening alpha-beta search."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from environment import Move
from evaluation_features import (
    articulation_pressure,
    capture_turn_estimate,
    dead_end_depth,
    escape_space,
    exit_count,
    graph_distance,
    region_size,
    repeat_penalty,
)
from game_core import GameState, MapAnalysis, search_best_action

Cell = Tuple[int, int]

# ---------------------------------------------------------------------------
# Offline table (loaded lazily at first use)
# ---------------------------------------------------------------------------

# Ghost action decode: must match solve_hide_table.py encoding
# index 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
_GHOST_ACTION_DECODE = [Move.STAY, Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

_HIDE_TABLE_PATH = Path(__file__).parent / "hide_table.npz"

# Module-level cache — None until loaded, False if load failed or map mismatch
_hide_table_state: Optional[Dict] = None
_hide_table_active: Optional[bool] = None


def _map_hash(map_state: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(map_state).tobytes()).hexdigest()[:16]


def _load_hide_table() -> bool:
    """Load hide_table.npz once. Returns True if table is ready."""
    global _hide_table_state, _hide_table_active
    if _hide_table_active is not None:
        return bool(_hide_table_active)

    if not _HIDE_TABLE_PATH.exists():
        _hide_table_active = False
        return False

    try:
        data = np.load(_HIDE_TABLE_PATH, allow_pickle=False)
        cells_arr = data["cells"]          # (n, 2) int16
        best_action = data["best_action"]  # (n, n) uint8
        value = data["value"]              # (n, n) int32
        stored_hash = data["map_hash"].tobytes().decode("ascii")

        cells = [tuple(int(v) for v in row) for row in cells_arr]
        cell_to_idx: Dict[Tuple[int, int], int] = {c: i for i, c in enumerate(cells)}

        _hide_table_state = {
            "stored_hash": stored_hash,
            "cell_to_idx": cell_to_idx,
            "best_action": best_action,
            "value": value,
        }
        _hide_table_active = True
        return True
    except Exception:
        _hide_table_active = False
        return False


def hide_table_lookup(pac: Cell, ghost: Cell, map_state: np.ndarray) -> Optional[Move]:
    """Return Ghost's best Move from precomputed table, or None on any miss.

    None causes the caller to fall through to online alpha-beta search.
    Never raises.
    """
    global _hide_table_active, _hide_table_state
    try:
        if not _load_hide_table():
            return None

        ts = _hide_table_state
        # Verify map hash once — disable table permanently if map differs
        if "map_verified" not in ts:
            if _map_hash(map_state) != ts["stored_hash"]:
                _hide_table_active = False
                return None
            ts["map_verified"] = True

        if not _hide_table_active:
            return None

        c2i = ts["cell_to_idx"]
        pi = c2i.get(tuple(pac))
        gi = c2i.get(tuple(ghost))
        if pi is None or gi is None:
            return None

        # INF sentinel — Ghost evades forever, no table guidance needed
        if int(ts["value"][pi, gi]) >= 10**6:
            return None

        enc = int(ts["best_action"][pi, gi])
        if enc < 0 or enc >= len(_GHOST_ACTION_DECODE):
            return None

        return _GHOST_ACTION_DECODE[enc]
    except Exception:
        return None


# Survival value vs optimal seek: used as an accurate leaf evaluation by the
# runtime hybrid search. INF (>=1e6) means the ghost evades forever from here.
HIDE_TABLE_INF = 10**6


def hide_table_value(pac: Cell, ghost: Cell, map_state: np.ndarray) -> Optional[int]:
    """Return precomputed survival steps for (pac, ghost), or None on any miss.

    The value is the number of steps the ghost survives against optimal seek play
    from this turn-start state. Higher = safer. Returns HIDE_TABLE_INF for
    evade-forever states. Never raises.
    """
    global _hide_table_active, _hide_table_state
    try:
        if not _load_hide_table():
            return None
        ts = _hide_table_state
        if "map_verified" not in ts:
            if _map_hash(map_state) != ts["stored_hash"]:
                _hide_table_active = False
                return None
            ts["map_verified"] = True
        if not _hide_table_active:
            return None
        c2i = ts["cell_to_idx"]
        pi = c2i.get(tuple(pac))
        gi = c2i.get(tuple(ghost))
        if pi is None or gi is None:
            return None
        return int(ts["value"][pi, gi])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Search parameters
# ---------------------------------------------------------------------------

# Full step budget — leaves headroom under the arena's 1 s SIGALRM hard kill.
SEARCH_TIME_BUDGET = 0.78

CAPTURED_PENALTY = 10_000.0
# CAPTURE_TURNS_WEIGHT was 180 — the high value caused a STAY bias: staying at the current
# position showed more apparent capture_turns (static distance / pac_speed) than moving,
# because moving reduces graph_dist to the CURRENT pac pos. Low CW lets DISTANCE_WEIGHT
# dominate so the ghost actually runs away instead of freezing at junctions.
CAPTURE_TURNS_WEIGHT = 30.0
# DISTANCE_WEIGHT raised from 8 → 30 to be the primary escape signal.
DISTANCE_WEIGHT = 30.0
EXIT_WEIGHT = 42.0
REGION_WEIGHT = 0.45
# ESCAPE_SPACE_WEIGHT set to 0: junction cells have more escape space than corridor cells,
# which created a STAY-at-junction bias that overrode the distance signal.
ESCAPE_SPACE_WEIGHT = 0.0
DEAD_END_WEIGHT = 60.0
ARTICULATION_WEIGHT = 35.0
REPEAT_WEIGHT = 18.0


def choose_hide_action(
    map_state: np.ndarray,
    my_position: Cell,
    enemy_position: Optional[Cell],
    recent_positions: Optional[Iterable[Cell]] = None,
) -> Move:
    history = tuple(recent_positions or ())
    if enemy_position is not None:
        table_action = hide_table_lookup(tuple(enemy_position), tuple(my_position), map_state)
        if table_action is not None:
            return table_action

    analysis = MapAnalysis(map_state, my_position)
    if enemy_position is None:
        return _best_no_threat_move(tuple(my_position), analysis, history)
    state = GameState(tuple(enemy_position), tuple(my_position))
    return search_best_action(
        state,
        "hide",
        lambda next_state, next_analysis: evaluate_hide(next_state, next_analysis, history),
        SEARCH_TIME_BUDGET,
        analysis,
    )


def evaluate_hide(state: GameState, analysis, recent_positions: Iterable[Cell] = ()) -> float:
    if state.ghost is None:
        return 0.0
    capture_turns = capture_turn_estimate(analysis, state.pacman, state.ghost)
    if capture_turns == 0:
        return -CAPTURED_PENALTY - state.step_number
    distance = graph_distance(analysis, state.ghost, state.pacman)
    exits = exit_count(analysis, state.ghost)
    region = region_size(analysis, state.ghost)
    escape = escape_space(analysis, state.ghost, state.pacman, depth=3)
    dead_depth = dead_end_depth(analysis, state.ghost)
    articulation = articulation_pressure(analysis, state.ghost, state.pacman)
    repeat = repeat_penalty(state.ghost, recent_positions)
    danger_scale = max(0.0, 4.0 - min(float(capture_turns), 4.0))
    return (
        CAPTURE_TURNS_WEIGHT * min(float(capture_turns), 20.0)
        + DISTANCE_WEIGHT * distance
        + EXIT_WEIGHT * exits
        + REGION_WEIGHT * region
        + ESCAPE_SPACE_WEIGHT * escape
        - DEAD_END_WEIGHT * dead_depth * (1.0 + danger_scale)
        - ARTICULATION_WEIGHT * articulation * (1.0 + danger_scale)
        - REPEAT_WEIGHT * repeat
    )


def _best_no_threat_move(pos: Cell, analysis, recent_positions: Iterable[Cell]) -> Move:
    best_move = Move.STAY
    best_score = _no_threat_score(pos, analysis, recent_positions)
    for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
        neighbor = _next_cell(pos, move)
        if not analysis.is_walkable(neighbor):
            continue
        score = _no_threat_score(neighbor, analysis, recent_positions)
        if score > best_score:
            best_move = move
            best_score = score
    return best_move


def _no_threat_score(pos: Cell, analysis, recent_positions: Iterable[Cell]) -> float:
    return (
        EXIT_WEIGHT * exit_count(analysis, pos)
        + REGION_WEIGHT * region_size(analysis, pos)
        + ESCAPE_SPACE_WEIGHT * escape_space(analysis, pos, None, depth=3)
        - DEAD_END_WEIGHT * dead_end_depth(analysis, pos)
        - REPEAT_WEIGHT * repeat_penalty(pos, recent_positions)
    )


def _next_cell(cell: Cell, move: Move) -> Cell:
    row, col = cell
    dr, dc = move.value
    return row + dr, col + dc
