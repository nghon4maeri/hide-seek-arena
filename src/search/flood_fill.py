from __future__ import annotations

from collections import deque
from math import ceil
from typing import List

from src.core.types import Position


def safe_reachable_cells(search, pacman: Position, ghost: Position, max_depth: int = 16, ghost_speed: int = 2) -> List[Position]:
    ghost_dist = search.bfs_distance_map(ghost)
    seen = {pacman}
    q = deque([(pacman, 0)])
    cells: List[Position] = []

    while q:
        p, t = q.popleft()
        gd = ghost_dist[p[0]][p[1]]
        ghost_catch_turn = 0 if gd < 2 else ceil((gd - 1) / max(1, ghost_speed))
        if t >= ghost_catch_turn:
            continue
        cells.append(p)
        if t >= max_depth:
            continue
        for n in search.neighbors(p):
            if n not in seen:
                seen.add(n)
                q.append((n, t + 1))
    return cells


def safe_reachable_area(search, pacman: Position, ghost: Position, max_depth: int = 16, ghost_speed: int = 2) -> int:
    return len(safe_reachable_cells(search, pacman, ghost, max_depth, ghost_speed))


def reachable_area(search, start: Position, max_depth: int = 18) -> int:
    seen = {start}
    q = deque([(start, 0)])
    count = 0
    while q:
        p, d = q.popleft()
        count += 1
        if d >= max_depth:
            continue
        for n in search.neighbors(p):
            if n not in seen:
                seen.add(n)
                q.append((n, d + 1))
    return count

