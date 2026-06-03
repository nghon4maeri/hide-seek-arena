from __future__ import annotations

from typing import Sequence

from .types import Grid, GridKey, Position

try:
    import numpy as np
except Exception:
    np = None


def as_grid(map_state) -> Grid:
    if np is not None and hasattr(map_state, "tolist"):
        map_state = map_state.tolist()
    return [[int(cell) for cell in row] for row in map_state]


def as_position(value) -> Position:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return int(value[0]), int(value[1])


def grid_key(grid: Sequence[Sequence[int]]) -> GridKey:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    flat = tuple(int(grid[r][c]) for r in range(rows) for c in range(cols))
    return rows, cols, flat


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

