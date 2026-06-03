from __future__ import annotations

from typing import List

from src.core.constants import INF
from src.core.types import Position


def dead_end_cells(search) -> List[Position]:
    dead = search.distance_to_dead_end_map()
    return [p for p in search.passable if dead[p[0]][p[1]] == 0]


def danger_cells(search, ghost: Position, radius: int = 5) -> List[Position]:
    dist = search.bfs_distance_map(ghost)
    return [p for p in search.passable if dist[p[0]][p[1]] <= radius]


def interception_score(search, pacman: Position, ghost: Position) -> float:
    ghost_dist = search.bfs_distance_map(ghost)
    pac_dist = search.bfs_distance_map(pacman)
    score = 0.0
    for cell in search.passable:
        pd = pac_dist[cell[0]][cell[1]]
        gd = ghost_dist[cell[0]][cell[1]]
        if pd <= 5 and gd <= pd + 1:
            score += 1.0 / (1.0 + gd)
    return score


def cutoff_score(search, pacman: Position, ghost: Position) -> float:
    deg = search.degree_map()
    ghost_dist = search.bfs_distance_map(ghost)
    pac_dist = search.bfs_distance_map(pacman)
    score = 0.0
    for cell in search.passable:
        d = deg[cell[0]][cell[1]]
        if d <= 2 and pac_dist[cell[0]][cell[1]] <= 6:
            gd = ghost_dist[cell[0]][cell[1]]
            pd = pac_dist[cell[0]][cell[1]]
            if gd <= pd + 2:
                score += (3 - min(d, 2)) + 1.0 / (1 + gd)
    return score

