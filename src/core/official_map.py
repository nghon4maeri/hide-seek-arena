from __future__ import annotations

from typing import List, Tuple

from src.core.types import Grid, Position


OFFICIAL_MAP_TEXT = """
#####################
#.........#.........#
#.###.###.#.###.###.#
#.###.###.#.###.###.#
#...................#
#.###.#.#####.#.###.#
#.....#...#...#.....#
#####.###.#.###.#####
#####.#...G...#.#####
#####.#.##.##.#.#####
#.......#...#.......#
#####.#.#####.#.#####
#####.#.......#.#####
#####.#.#####.#.#####
#.........#.........#
#.###.###.#.###.###.#
#...#.....P.....#...#
###.#.#.#####.#.#.###
#.....#...#...#.....#
#.#######.#.#######.#
#...................#
#####################
""".strip()


def parse_official_map(text: str = OFFICIAL_MAP_TEXT) -> Tuple[Grid, Position, Position]:
    rows = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("official map text is empty")

    height = len(rows)
    width = len(rows[0])
    if height != 22:
        raise ValueError(f"official map height must be 22 rows, got {height}")
    if width != 21:
        raise ValueError(f"official map width must be 21 columns, got {width}")
    if any(len(row) != width for row in rows):
        widths = [len(row) for row in rows]
        raise ValueError(f"official map rows must have consistent width: {widths}")

    grid: Grid = []
    pacman_positions: List[Position] = []
    ghost_positions: List[Position] = []

    for r, row in enumerate(rows):
        grid_row: List[int] = []
        for c, char in enumerate(row):
            if char == "#":
                grid_row.append(1)
            elif char == ".":
                grid_row.append(0)
            elif char == "P":
                grid_row.append(0)
                pacman_positions.append((r, c))
            elif char == "G":
                grid_row.append(0)
                ghost_positions.append((r, c))
            else:
                raise ValueError(f"unsupported official map character {char!r} at {(r, c)}")
        grid.append(grid_row)

    if len(pacman_positions) != 1:
        raise ValueError(f"official map must contain exactly one P, got {len(pacman_positions)}")
    if len(ghost_positions) != 1:
        raise ValueError(f"official map must contain exactly one G, got {len(ghost_positions)}")

    return grid, pacman_positions[0], ghost_positions[0]


OFFICIAL_MAP_GRID, PACMAN_START, GHOST_START = parse_official_map()
OFFICIAL_MAP_HEIGHT = len(OFFICIAL_MAP_GRID)
OFFICIAL_MAP_WIDTH = len(OFFICIAL_MAP_GRID[0])

