"""Small evaluation features shared by Seek and Hide."""

from __future__ import annotations

from collections import deque
from math import ceil, isinf
from typing import Iterable, Optional, Tuple

Cell = Tuple[int, int]


def manhattan(a: Cell, b: Optional[Cell]) -> int:
    return 0 if b is None else abs(a[0] - b[0]) + abs(a[1] - b[1])


def graph_distance(analysis, a: Cell, b: Optional[Cell]) -> float:
    return float("inf") if b is None else analysis.dist(a, b)


def capture_turn_estimate(analysis, pacman: Cell, ghost: Optional[Cell], pacman_speed: int = 2) -> float:
    distance = graph_distance(analysis, pacman, ghost)
    if isinf(distance):
        return float("inf")
    if manhattan(pacman, ghost) < 2:
        return 0.0
    return ceil(max(0.0, distance - 1.0) / max(1, pacman_speed))


def exit_count(analysis, cell: Cell) -> int:
    return len(analysis.neighbors(cell))


def region_size(analysis, cell: Cell) -> int:
    return analysis.region_size(cell)


def dead_end_depth(analysis, cell: Cell) -> int:
    return analysis.dead_end_depth.get(cell, 0)


def escape_space(analysis, start: Cell, blocked_by: Optional[Cell], depth: int = 3) -> int:
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        cell, steps = queue.popleft()
        if steps >= depth:
            continue
        for neighbor in analysis.neighbors(cell):
            if neighbor == blocked_by or neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, steps + 1))
    return len(seen)


def articulation_pressure(analysis, cell: Cell, enemy: Optional[Cell]) -> float:
    if cell in analysis.articulation_points:
        return 1.0
    if enemy is None:
        return 0.0
    return 0.5 if any(neighbor in analysis.articulation_points for neighbor in analysis.neighbors(cell)) else 0.0


def repeat_penalty(cell: Cell, recent_positions: Iterable[Cell]) -> float:
    recent = list(recent_positions)[-6:]
    if not recent:
        return 0.0
    penalty = 0.0
    for index, previous in enumerate(reversed(recent), start=1):
        if previous == cell:
            penalty += 1.0 / index
    return penalty
