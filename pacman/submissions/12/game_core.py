"""Core helpers shared by the HideSeek agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import numpy as np

_SRC_PATH = Path(__file__).resolve().parents[2] / "arena-source" / "pacman" / "pacman" / "src"
if _SRC_PATH.exists() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from environment import Move
from map_analysis import CARDINAL_MOVES, MapAnalysis as _MapAnalysis
from search_core import alpha_beta_search

Cell = Tuple[int, int]
_ACTIVE_ANALYSIS: Optional["MapAnalysis"] = None

# Module-level debug flag — set True only during local development.
DEBUG = False


@dataclass(frozen=True)
class GameState:
    pacman: Cell
    ghost: Optional[Cell]
    step_number: int = 0


class MapAnalysis(_MapAnalysis):
    """Map analysis wrapper that registers the active map for simulation."""

    def __init__(self, map_state: np.ndarray, anchor: Optional[Cell] = None):
        super().__init__(map_state, anchor)
        _set_active_analysis(self)


def get_map_analysis(map_state: np.ndarray, anchor: Optional[Cell] = None) -> MapAnalysis:
    return MapAnalysis(map_state, anchor)


def legal_pacman_actions(pos: Cell, analysis: MapAnalysis, speed: int = 2):
    actions = [(Move.STAY, 1)]
    for move in CARDINAL_MOVES:
        if _apply_single_step(pos, move, analysis) == pos:
            continue
        for steps in range(1, max(1, int(speed)) + 1):
            actions.append((move, steps))
    return tuple(actions)


def legal_ghost_actions(pos: Cell, analysis: MapAnalysis):
    moves = [Move.STAY]
    for move in CARDINAL_MOVES:
        if _apply_single_step(pos, move, analysis) != pos:
            moves.append(move)
    return tuple(moves)


def simulate(pac: Cell, ghost: Cell, pac_act, ghost_act, analysis: Optional[MapAnalysis] = None):
    """Apply one arena turn; callers may pass analysis or initialize MapAnalysis first."""
    analysis = analysis or _require_active_analysis()
    pac_move, pac_steps = _normalize_pacman_action(pac_act)
    ghost_move = ghost_act if isinstance(ghost_act, Move) else Move.STAY
    new_pac = _apply_pacman_move(tuple(pac), pac_move, pac_steps, analysis)
    new_ghost = _apply_single_step(tuple(ghost), ghost_move, analysis)
    captured = _manhattan(new_pac, new_ghost) < 2
    return new_pac, new_ghost, captured


def search_best_action(
    state: GameState,
    role: str,
    evaluate_fn,
    time_budget: float,
    analysis: Optional[MapAnalysis] = None,
    pacman_speed: int = 2,
):
    analysis = analysis or _require_active_analysis()
    return alpha_beta_search(
        state,
        role,
        evaluate_fn,
        analysis,
        legal_pacman_actions,
        legal_ghost_actions,
        simulate,
        pacman_speed,
        time_budget,
    )


def validate_action(action, role: str, pos: Cell, analysis: Optional["MapAnalysis"], pacman_speed: int = 2):
    """Coerce *action* into the valid return type for *role*.

    Never raises — returns a guaranteed-safe STAY if anything is wrong.

    Seek (Pacman): must return (Move, steps) with steps clamped to [1, pacman_speed].
    Hide (Ghost): must return a Move enum value.

    Wall-collision is handled by the arena (ghost walks into wall → stays); we only
    filter out non-Move values and out-of-range steps counts.
    """
    is_seek = role in ("seek", "pacman")
    try:
        if is_seek:
            # Accept bare Move (treat as steps=1) or (Move, steps) tuple.
            if isinstance(action, Move):
                return action, 1
            if (
                isinstance(action, tuple)
                and len(action) == 2
                and isinstance(action[0], Move)
            ):
                move = action[0]
                try:
                    steps = int(action[1])
                except (TypeError, ValueError):
                    steps = 1
                # Clamp steps to [1, pacman_speed].
                speed = max(1, int(pacman_speed))
                steps = max(1, min(steps, speed))
                return move, steps
            # Unknown shape — fall through to STAY.
        else:
            if isinstance(action, Move):
                return action
            # Unknown shape — fall through to STAY.
    except Exception:
        pass

    # Unconditional safe default — never calls anything that can raise.
    return (Move.STAY, 1) if is_seek else Move.STAY


def best_greedy_pacman_action(pos: Cell, target: Optional[Cell], map_state: np.ndarray, pacman_speed: int = 2):
    moves = _ordered_moves_towards(pos, target) if target else list(CARDINAL_MOVES)
    moves.extend(move for move in CARDINAL_MOVES if move not in moves)
    for move in moves:
        steps = _max_valid_steps(pos, move, map_state, pacman_speed)
        if steps > 0:
            return move, steps
    return Move.STAY, 1


def best_greedy_ghost_move(pos: Cell, threat: Optional[Cell], map_state: np.ndarray) -> Move:
    best_move = Move.STAY
    best_score = _manhattan(pos, threat) if threat else 0
    for move in CARDINAL_MOVES:
        next_pos = _apply_raw_single_step(pos, move, map_state)
        if next_pos == pos:
            continue
        score = _manhattan(next_pos, threat) if threat else _valid_neighbor_count(next_pos, map_state)
        if score > best_score:
            best_move = move
            best_score = score
    return best_move


def _set_active_analysis(analysis: MapAnalysis) -> None:
    global _ACTIVE_ANALYSIS
    _ACTIVE_ANALYSIS = analysis


def _require_active_analysis() -> MapAnalysis:
    if _ACTIVE_ANALYSIS is None:
        raise RuntimeError("MapAnalysis must be created before simulate()")
    return _ACTIVE_ANALYSIS


def _normalize_pacman_action(action) -> Tuple[Move, int]:
    if isinstance(action, Move):
        return action, 1
    if isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], Move):
        move, steps = action
        try:
            normalized_steps = int(steps)
        except (TypeError, ValueError):
            normalized_steps = 1
        return move, min(max(1, normalized_steps), 2)
    return Move.STAY, 1


def _apply_pacman_move(pos: Cell, move: Move, steps: int, analysis: MapAnalysis) -> Cell:
    if move == Move.STAY:
        return pos
    current = pos
    for _ in range(steps):
        candidate = _apply_single_step(current, move, analysis)
        if candidate == current:
            break
        current = candidate
    return current


def _apply_single_step(pos: Cell, move: Move, analysis: MapAnalysis) -> Cell:
    candidate = _next_cell(pos, move)
    return candidate if analysis.is_walkable(candidate) else pos


def _ordered_moves_towards(pos: Cell, target: Cell) -> List[Move]:
    row_diff = target[0] - pos[0]
    col_diff = target[1] - pos[1]
    row_move = Move.DOWN if row_diff > 0 else Move.UP if row_diff < 0 else None
    col_move = Move.RIGHT if col_diff > 0 else Move.LEFT if col_diff < 0 else None
    first, second = (row_move, col_move) if abs(row_diff) >= abs(col_diff) else (col_move, row_move)
    return [move for move in (first, second) if move is not None]


def _max_valid_steps(pos: Cell, move: Move, map_state: np.ndarray, max_steps: int) -> int:
    steps = 0
    current = pos
    for _ in range(max(1, int(max_steps))):
        next_pos = _apply_raw_single_step(current, move, map_state)
        if next_pos == current:
            break
        current = next_pos
        steps += 1
    return steps


def _apply_raw_single_step(pos: Cell, move: Move, map_state: np.ndarray) -> Cell:
    next_pos = _next_cell(pos, move)
    return next_pos if _is_empty(next_pos, map_state) else pos


def _valid_neighbor_count(pos: Cell, map_state: np.ndarray) -> int:
    return sum(_apply_raw_single_step(pos, move, map_state) != pos for move in CARDINAL_MOVES)


def _is_empty(pos: Cell, map_state: np.ndarray) -> bool:
    row, col = pos
    return 0 <= row < map_state.shape[0] and 0 <= col < map_state.shape[1] and map_state[row, col] == 0


def _next_cell(cell: Cell, move: Move) -> Cell:
    row, col = cell
    dr, dc = move.value
    return row + dr, col + dc


def _manhattan(a: Cell, b: Optional[Cell]) -> int:
    return 0 if b is None else abs(a[0] - b[0]) + abs(a[1] - b[1])
