"""Heuristic functions and scoring utilities.

Shared by PacmanAgent and GhostAgent. All functions are stateless.
"""

from typing import Dict, List, Optional, Set, Tuple

MOVE_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def shape(ms):
    if hasattr(ms, "shape"):
        return int(ms.shape[0]), int(ms.shape[1])
    return len(ms), len(ms[0]) if ms else 0


def cell(ms, r: int, c: int) -> int:
    return int(ms[r, c]) if hasattr(ms, "shape") else int(ms[r][c])


def is_valid(pos: Tuple[int, int], ms) -> bool:
    r, c = pos
    h, w = shape(ms)
    if r < 0 or r >= h or c < 0 or c >= w:
        return False
    return cell(ms, r, c) != 1


def apply_move(pos: Tuple[int, int], move) -> Tuple[int, int]:
    if hasattr(move, "value"):
        dr, dc = move.value
    else:
        dr, dc = move
    return pos[0] + dr, pos[1] + dc


def get_neighbors(pos: Tuple[int, int], ms) -> List[Tuple[Tuple[int, int], object]]:
    from environment import Move

    delta_to_move = {
        (-1, 0): Move.UP,
        (1, 0): Move.DOWN,
        (0, -1): Move.LEFT,
        (0, 1): Move.RIGHT,
    }
    neighbors = []
    for dr, dc in MOVE_DELTAS:
        nxt = (pos[0] + dr, pos[1] + dc)
        if is_valid(nxt, ms):
            neighbors.append((nxt, delta_to_move[(dr, dc)]))
    return neighbors


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def cell_exits(pos: Tuple[int, int], ms) -> int:
    r, c = pos
    h, w = shape(ms)
    exits = 0
    if r > 0 and cell(ms, r - 1, c) != 1:
        exits += 1
    if r < h - 1 and cell(ms, r + 1, c) != 1:
        exits += 1
    if c > 0 and cell(ms, r, c - 1) != 1:
        exits += 1
    if c < w - 1 and cell(ms, r, c + 1) != 1:
        exits += 1
    return exits


PHASE_WEIGHTS = {
    "early": {"dist": 25.0, "flood": 1.0, "junction": 15.0, "dead_end": -150.0, "corridor": -30.0},
    "mid":   {"dist": 20.0, "flood": 0.8, "junction": 12.0, "dead_end": -120.0, "corridor": -25.0},
    "late":  {"dist": 30.0, "flood": 1.2, "junction": 10.0, "dead_end": -200.0, "corridor": -40.0},
}


def game_phase(step_number: int) -> str:
    if step_number <= 60:
        return "early"
    if step_number <= 140:
        return "mid"
    return "late"


def score_ghost_position(
    pos: Tuple[int, int],
    pacman_pos: Tuple[int, int],
    pd: Dict[Tuple, int],
    flood_safe_count: int,
    topo: Optional[Dict],
    phase: str = "mid",
) -> float:
    w = PHASE_WEIGHTS.get(phase, PHASE_WEIGHTS["mid"])
    bfs_dist = pd.get(pos, manhattan(pos, pacman_pos))
    score = w["dist"] * bfs_dist + w["flood"] * flood_safe_count

    if topo is not None:
        exits = topo["degree"].get(pos, 2)
        if exits >= 3:
            score += w["junction"]
        elif exits <= 1:
            dd = topo["dead_end_depth"].get(pos, 5)
            score += w["dead_end"] * min(dd, 5)
        elif exits == 2 and bfs_dist <= 6:
            score += w["corridor"]

    return score


def score_pacman_position(
    pos: Tuple[int, int],
    target: Tuple[int, int],
    ms,
    visited: Set[Tuple],
) -> float:
    dist = manhattan(pos, target)
    unvisited_bonus = 3.0 if pos not in visited else 0.0
    junction_bonus = 1.0 if cell_exits(pos, ms) >= 3 else 0.0
    return -dist + unvisited_bonus + junction_bonus


def trace_to_junction(
    start: Tuple[int, int],
    dr: int,
    dc: int,
    ms,
    junctions: set,
    max_steps: int = 12,
) -> Tuple[int, int]:
    cur = start
    for _ in range(max_steps):
        nxt = (cur[0] + dr, cur[1] + dc)
        if not is_valid(nxt, ms):
            break
        cur = nxt
        if cur in junctions:
            past = (cur[0] + dr, cur[1] + dc)
            if is_valid(past, ms):
                return past
            return cur
    return cur


def compute_safe_ratio(
    gpos: Tuple[int, int], ppos: Tuple[int, int],
    ms, topo: Optional[Dict] = None, max_cells: int = 30,
) -> float:
    """Compute ratio of cells ghost reaches before speed-2 Pacman.

    Returns value in [0, 1]. Higher = safer.
    """
    from search import bfs_distance
    gd = bfs_distance(ms, gpos, max_dist=20)
    pd = bfs_distance(ms, ppos, max_dist=20)

    safe = 0
    total = 0
    for cell_pos, g_val in gd.items():
        if total >= max_cells:
            break
        p_val = pd.get(cell_pos, 999)
        total += 1
        if g_val < (p_val + 1) // 2:
            safe += 1

    if total == 0:
        return 0.0
    return safe / total
