from __future__ import annotations

from src.core.constants import GHOST_SPEED
from src.core.map_utils import manhattan
from src.core.types import Position


def evaluate_hide(search, pacman: Position, ghost: Position, step: int, ghost_speed: int = GHOST_SPEED) -> float:
    dist = search.distance(pacman, ghost)
    if manhattan(pacman, ghost) < 2:
        return -100000 + step

    degree = search.degree_map()[pacman[0]][pacman[1]]
    dead_dist = search.distance_to_dead_end_map()[pacman[0]][pacman[1]]
    safe_area = search.safe_reachable_area(pacman, ghost, max_depth=15, ghost_speed=ghost_speed)
    local_area = search.reachable_area(pacman, max_depth=7)

    dead_end_risk = max(0, 8 - dead_dist) * max(0, 9 - dist)
    trap_probability = max(0, 10 - safe_area) * max(0, 8 - dist)
    corridor_risk = 1 if degree <= 2 else 0

    return (
        22.0 * min(dist, 20)
        + 7.5 * safe_area
        + 5.0 * degree
        + 1.2 * local_area
        + 2.5 * min(dead_dist, 12)
        - 13.0 * dead_end_risk
        - 18.0 * trap_probability
        - 22.0 * corridor_risk * max(0, 7 - dist)
        + 0.05 * step
    )

