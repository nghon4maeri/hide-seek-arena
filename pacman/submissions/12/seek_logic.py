"""Seek-side action selection: offline table lookup (primary) + alpha-beta fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from environment import Move
from evaluation_features import capture_turn_estimate, dead_end_depth, exit_count, region_size
from game_core import GameState, MapAnalysis, search_best_action

Cell = Tuple[int, int]

# Full step budget — leaves headroom under the arena's 1 s SIGALRM hard kill.
SEARCH_TIME_BUDGET = 0.78

CAPTURE_BONUS = 10_000.0
CAPTURE_TURNS_WEIGHT = 220.0
DISTANCE_WEIGHT = 18.0
GHOST_MOBILITY_WEIGHT = 10.0
GHOST_REGION_WEIGHT = 0.35
DEAD_END_TRAP_WEIGHT = 24.0
ARTICULATION_TRAP_WEIGHT = 12.0

# ---------------------------------------------------------------------------
# Offline table (loaded lazily at first use)
# ---------------------------------------------------------------------------

# Decoded action list matching the solver's encoding:
#   0=STAY, 1=UP1, 2=DOWN1, 3=LEFT1, 4=RIGHT1, 5=UP2, 6=DOWN2, 7=LEFT2, 8=RIGHT2
_PAC_ACTION_DECODE = [
    (Move.STAY, 1),
    (Move.UP, 1), (Move.DOWN, 1), (Move.LEFT, 1), (Move.RIGHT, 1),
    (Move.UP, 2), (Move.DOWN, 2), (Move.LEFT, 2), (Move.RIGHT, 2),
]

_TABLE_PATH = Path(__file__).parent / "seek_table.npz"

# Module-level cache — None until first call, False if load failed / map mismatch.
_table_state: Optional[Dict] = None
_table_active: Optional[bool] = None   # None=not loaded, True=ready, False=disabled


def _load_table() -> bool:
    """Load seek_table.npz once.  Returns True if table is ready to use."""
    global _table_state, _table_active
    if _table_active is not None:
        return _table_active

    if not _TABLE_PATH.exists():
        _table_active = False
        return False

    try:
        data = np.load(_TABLE_PATH, allow_pickle=False)
        cells_arr = data["cells"]          # (n, 2) int16
        best_action = data["best_action"]  # (n, n) uint8
        value = data["value"]              # (n, n) int32
        map_hash_bytes = data["map_hash"]  # uint8 array of ascii chars

        stored_hash = map_hash_bytes.tobytes().decode("ascii")
        cells = [tuple(int(v) for v in row) for row in cells_arr]
        cell_to_idx = {c: i for i, c in enumerate(cells)}

        _table_state = {
            "stored_hash": stored_hash,
            "cells": cells,
            "cell_to_idx": cell_to_idx,
            "best_action": best_action,
            "value": value,
        }
        # Mark active temporarily; map check happens on first real call
        _table_active = True
        return True
    except Exception:
        _table_active = False
        return False


def _map_hash(map_state: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(map_state).tobytes()).hexdigest()[:16]


def lookup(
    pac: Cell,
    ghost: Cell,
    map_state: np.ndarray,
) -> Optional[Tuple[Move, int]]:
    """Return optimal Seek action from precomputed table, or None on any miss.

    None causes the caller to fall through to alpha-beta search.
    This function must never raise.
    """
    global _table_active, _table_state
    try:
        if not _load_table():
            return None

        ts = _table_state
        # Map-match guard: compare hash once per map (first non-None call).
        # If map differs from solved map, permanently disable the table so
        # all subsequent steps fall through to the alpha-beta core.
        if "map_verified" not in ts:
            if _map_hash(map_state) != ts["stored_hash"]:
                _table_active = False
                return None
            ts["map_verified"] = True

        if not _table_active:
            return None

        c2i = ts["cell_to_idx"]
        pi = c2i.get(tuple(pac))
        gi = c2i.get(tuple(ghost))
        if pi is None or gi is None:
            return None

        # INF sentinel from solver (0x0F4240 = 1_000_000)
        if ts["value"][pi, gi] >= 10**6:
            return None

        enc = int(ts["best_action"][pi, gi])
        if enc < 0 or enc >= len(_PAC_ACTION_DECODE):
            return None

        return _PAC_ACTION_DECODE[enc]
    except Exception:
        return None


# Capture-steps value vs optimal ghost: used as an accurate leaf evaluation by
# the runtime hybrid seek. Lower = faster guaranteed capture. INF (>=1e6) means
# the ghost evades forever from here (out of the solved component).
SEEK_TABLE_INF = 10**6


def seek_table_value(pac: Cell, ghost: Cell, map_state: np.ndarray) -> Optional[int]:
    """Return precomputed capture steps for (pac, ghost), or None on any miss.

    Steps until the seek catches an optimally-evading ghost from this turn-start
    state. Lower = the seek closes faster. Returns SEEK_TABLE_INF for
    evade-forever states. Never raises.
    """
    global _table_active, _table_state
    try:
        if not _load_table():
            return None
        ts = _table_state
        if "map_verified" not in ts:
            if _map_hash(map_state) != ts["stored_hash"]:
                _table_active = False
                return None
            ts["map_verified"] = True
        if not _table_active:
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
# Alpha-beta search entry (used when table is disabled / misses)
# ---------------------------------------------------------------------------

def choose_seek_action(
    map_state: np.ndarray,
    my_position: Cell,
    enemy_position: Optional[Cell],
    pacman_speed: int,
):
    analysis = MapAnalysis(map_state, my_position)
    state = GameState(tuple(my_position), None if enemy_position is None else tuple(enemy_position))
    return search_best_action(state, "seek", evaluate_seek, SEARCH_TIME_BUDGET, analysis, pacman_speed)


def evaluate_seek(state: GameState, analysis) -> float:
    if state.ghost is None:
        return 0.0
    distance = analysis.dist(state.pacman, state.ghost)
    capture_turns = capture_turn_estimate(analysis, state.pacman, state.ghost)
    if capture_turns == 0:
        return CAPTURE_BONUS - state.step_number
    ghost_exits = exit_count(analysis, state.ghost)
    trap_depth = dead_end_depth(analysis, state.ghost)
    region_pressure = max(0, 24 - region_size(analysis, state.ghost))
    articulation_bonus = ARTICULATION_TRAP_WEIGHT if state.ghost in analysis.articulation_points else 0.0
    return (
        -CAPTURE_TURNS_WEIGHT * capture_turns
        - DISTANCE_WEIGHT * distance
        - GHOST_MOBILITY_WEIGHT * ghost_exits
        + DEAD_END_TRAP_WEIGHT * trap_depth
        + GHOST_REGION_WEIGHT * region_pressure
        + articulation_bonus
    )
