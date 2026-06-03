from __future__ import annotations

from src.core.map_utils import manhattan
from src.core.types import Position
from .features import cutoff_score, interception_score


def evaluate_seek(search, pacman: Position, ghost: Position, step: int) -> float:
    dist = search.distance(ghost, pacman)
    if manhattan(pacman, ghost) < 2:
        return 100000 - 40 * step

    degree = search.degree_map()[pacman[0]][pacman[1]]
    dead_dist = search.distance_to_dead_end_map()[pacman[0]][pacman[1]]
    pac_area = search.safe_reachable_area(pacman, ghost, max_depth=14, ghost_speed=2)
    trap_potential = max(0, 4 - degree) * 12 + max(0, 7 - dead_dist) * 5

    return (
        -30.0 * dist
        + 14.0 * trap_potential
        + 10.0 * interception_score(search, pacman, ghost)
        + 8.0 * cutoff_score(search, pacman, ghost)
        - 3.0 * pac_area
        - 0.05 * step
    )

