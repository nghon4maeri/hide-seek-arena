from __future__ import annotations

from typing import Iterable, List

from .constants import ACTIONS, MOVE_ACTIONS
from .types import Action, Grid, Position


def in_bounds(grid: Grid, p: Position) -> bool:
    return 0 <= p[0] < len(grid) and 0 <= p[1] < len(grid[0])


def is_open(grid: Grid, p: Position) -> bool:
    return in_bounds(grid, p) and grid[p[0]][p[1]] == 0


def apply_action(grid: Grid, p: Position, action: Action) -> Position:
    q = (p[0] + action.dr, p[1] + action.dc)
    return q if is_open(grid, q) else p


def legal_actions(grid: Grid, p: Position, include_stay: bool = True) -> List[Action]:
    candidates = ACTIONS if include_stay else MOVE_ACTIONS
    result = []
    for action in candidates:
        q = (p[0] + action.dr, p[1] + action.dc)
        if action.name == "STAY" or is_open(grid, q):
            result.append(action)
    return result


def neighbors(grid: Grid, p: Position) -> Iterable[Position]:
    for action in MOVE_ACTIONS:
        q = (p[0] + action.dr, p[1] + action.dc)
        if is_open(grid, q):
            yield q

