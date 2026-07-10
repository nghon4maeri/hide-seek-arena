"""Submission entry point for both HideSeek roles."""

from __future__ import annotations

import os
from pathlib import Path
import sys

_SRC_PATH = Path(__file__).resolve().parents[2] / "arena-source" / "pacman" / "pacman" / "src"
if _SRC_PATH.exists() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

from game_core import (
    GameState,
    MapAnalysis,
    best_greedy_ghost_move,
    best_greedy_pacman_action,
    get_map_analysis,
    validate_action,
)
from hide_logic import choose_hide_action, evaluate_hide, hide_table_lookup
from hybrid_hide import choose_hybrid_hide_move
from hybrid_seek import choose_hybrid_seek_action
from seek_logic import choose_seek_action, evaluate_seek, lookup as seek_table_lookup

# Module-level flag — never print during arena matches.
DEBUG = False

# Budget for the emergency one-ply fallback (leaves ≥0.15 s of headroom).
_ONE_PLY_BUDGET = 0.05


def _offline_table_seek(map_state, my_position, target, speed):
    """Offline table lookup for Seek role.

    Returns the precomputed optimal (Move, steps) action when both positions
    are in the solved component and the map matches the solved map.
    Returns None on any miss — caller falls through to alpha-beta.
    """
    if target is None:
        return None
    return seek_table_lookup(tuple(my_position), tuple(target), map_state)


def _offline_table_hide(map_state, my_position, threat, recent_positions):
    """Offline Hide table lookup — returns precomputed best Ghost Move or None.

    Table was solved assuming Seek plays its own table-optimal policy.
    Miss conditions: threat unknown, positions off-map, map differs from solved map.
    Falls through to alpha-beta on None.
    """
    if threat is None:
        return None
    return hide_table_lookup(tuple(threat), tuple(my_position), map_state)


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.last_known_enemy_pos = None

    def step(self, map_state, my_position, enemy_position, step_number):
        # Update last-known enemy position while it is observable.
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        target = enemy_position or self.last_known_enemy_pos

        speed = self.pacman_speed
        action = None
        analysis = None

        # --- Layer 0: herding exploit (drives naive fleers into a trap fast) ---
        # A minimax worst-case slack (small) keeps capture vs STRONG ghosts at the
        # game value (no regression) while letting weak/predictable fleers be cornered
        # in far fewer steps for the capture tie-break; a step leash guarantees the
        # catch lands well under the step cap regardless.
        # SEEK_NO_HYBRID disables it for A/B comparison against the raw table.
        try:
            if not os.environ.get("SEEK_NO_HYBRID"):
                action = choose_hybrid_seek_action(
                    map_state, tuple(my_position),
                    None if target is None else tuple(target),
                    speed, step_number,
                )
        except Exception:
            action = None

        # --- Layer 1: offline table lookup (fallback when hybrid defers/misses) ---
        if action is None:
            try:
                action = _offline_table_seek(map_state, my_position, target, speed)
            except Exception:
                action = None

        # --- Layer 2: iterative-deepening alpha-beta search ---
        if action is None:
            try:
                analysis = get_map_analysis(map_state, my_position)
                action = choose_seek_action(map_state, my_position, target, speed)
            except Exception:
                action = None

        # --- Layer 3: one-ply search with tight budget ---
        if action is None:
            try:
                if analysis is None:
                    analysis = get_map_analysis(map_state, my_position)
                state = GameState(tuple(my_position), None if target is None else tuple(target))
                from game_core import search_best_action
                action = search_best_action(
                    state, "seek", evaluate_seek, _ONE_PLY_BUDGET, analysis, speed
                )
            except Exception:
                action = None

        # --- Layer 4: greedy best-first move (APSP-guided, never raises) ---
        if action is None:
            try:
                action = best_greedy_pacman_action(
                    tuple(my_position), target, map_state, speed
                )
            except Exception:
                action = None

        # --- Layer 5: validate / clamp → STAY if everything else failed ---
        # This block must NEVER call anything that can raise.
        try:
            action = validate_action(action, "seek", my_position, analysis, speed)
        except Exception:
            action = (Move.STAY, 1)

        return action


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None
        self.recent_positions: list = []

    def step(self, map_state, my_position, enemy_position, step_number):
        # Maintain position history and last-known enemy.
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        self.recent_positions.append(tuple(my_position))
        self.recent_positions = self.recent_positions[-8:]
        threat = enemy_position or self.last_known_enemy_pos

        action = None
        analysis = None

        # --- Layer 0: hybrid runtime policy (safe table-value floor + flee tie-break) ---
        # Holds strong seekers to the game value like the table, but runs up the
        # clock vs weak seekers the table would not exploit. Primary Hide policy.
        # HIDE_NO_HYBRID disables it for A/B comparison against the raw table.
        try:
            if not os.environ.get("HIDE_NO_HYBRID"):
                action = choose_hybrid_hide_move(
                    map_state, tuple(my_position),
                    None if threat is None else tuple(threat),
                    self.recent_positions,
                )
        except Exception:
            action = None

        # --- Layer 1: offline table lookup (fallback when hybrid defers/misses) ---
        if action is None:
            try:
                action = _offline_table_hide(
                    map_state, my_position, threat, self.recent_positions
                )
            except Exception:
                action = None

        # --- Layer 2: iterative-deepening alpha-beta search ---
        if action is None:
            try:
                analysis = get_map_analysis(map_state, my_position)
                action = choose_hide_action(
                    map_state, my_position, threat, self.recent_positions
                )
            except Exception:
                action = None

        # --- Layer 3: one-ply search with tight budget ---
        if action is None:
            try:
                if analysis is None:
                    analysis = get_map_analysis(map_state, my_position)
                pac = tuple(threat) if threat is not None else tuple(my_position)
                state = GameState(pac, tuple(my_position))
                from game_core import search_best_action
                action = search_best_action(
                    state, "hide", evaluate_hide, _ONE_PLY_BUDGET, analysis
                )
            except Exception:
                action = None

        # --- Layer 4: greedy distance-maximising move (never raises) ---
        if action is None:
            try:
                action = best_greedy_ghost_move(
                    tuple(my_position), threat, map_state
                )
            except Exception:
                action = None

        # --- Layer 5: validate / clamp → STAY if everything else failed ---
        # This block must NEVER call anything that can raise.
        try:
            action = validate_action(action, "hide", my_position, analysis)
        except Exception:
            action = Move.STAY

        return action
