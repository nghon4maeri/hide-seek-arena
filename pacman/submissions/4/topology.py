"""Static map topology analysis.

Called once during agent __init__ to classify every open cell and
build distance-to-junction maps.  All lookups are O(1) during step().
"""

from collections import deque
from typing import Dict, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MOVE_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # UP, DOWN, LEFT, RIGHT


def analyze_map(map_state) -> Dict:
    """Build topology data for *map_state*.

    Returns a dict with keys:
        junctions        – set of (r,c) with >= 3 open exits
        dead_ends        – set of (r,c) with <= 1 open exit
        dead_end_depth   – dict {(r,c): steps to nearest junction}
        junction_distance – dict {(r,c): BFS steps to nearest junction}
        degree           – dict {(r,c): number of open cardinal exits}
    """
    height, width = _shape(map_state)

    # ---- 1. Classify every open cell by degree ---------------------------
    degree: Dict[Tuple, int] = {}
    junctions: Set[Tuple] = set()
    dead_ends: Set[Tuple] = set()
    open_cells: Set[Tuple] = set()

    for r in range(height):
        for c in range(width):
            if _cell(map_state, r, c) == 1:
                continue
            pos = (r, c)
            open_cells.add(pos)
            exits = _count_exits(map_state, pos)
            degree[pos] = exits
            if exits >= 3:
                junctions.add(pos)
            elif exits <= 1:
                dead_ends.add(pos)

    # ---- 2. Dead-end depth: walk from each dead-end toward junction ------
    dead_end_depth: Dict[Tuple, int] = {}
    for de in dead_ends:
        cur = de
        depth = 0
        visited = {de}
        while cur not in junctions:
            found_next = False
            for dr, dc in MOVE_DELTAS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt not in visited and _valid_pos(nxt, map_state):
                    visited.add(nxt)
                    cur = nxt
                    depth += 1
                    found_next = True
                    break
            if not found_next or depth > 20:
                break
        dead_end_depth[de] = min(depth, 10)

    # ---- 3. Junction-distance BFS ----------------------------------------
    jd_map: Dict[Tuple, int] = {}
    queue: deque = deque()
    for j in junctions:
        jd_map[j] = 0
        queue.append(j)
    while queue:
        cur = queue.popleft()
        for dr, dc in MOVE_DELTAS:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt not in jd_map and _valid_pos(nxt, map_state):
                jd_map[nxt] = jd_map[cur] + 1
                queue.append(nxt)

    # ---- 4. Loop detection: find core, find cycles, compute loop distance --
    core = _compute_core(open_cells, map_state)
    loop_set = _find_loops(core, map_state)
    loop_dist_map: Dict[Tuple, int] = {}
    lq: deque = deque()
    for lpos in loop_set:
        loop_dist_map[lpos] = 0
        lq.append(lpos)
    while lq:
        cur = lq.popleft()
        for dr, dc in MOVE_DELTAS:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt not in loop_dist_map and _valid_pos(nxt, map_state):
                loop_dist_map[nxt] = loop_dist_map[cur] + 1
                lq.append(nxt)

    return {
        "junctions": junctions,
        "dead_ends": dead_ends,
        "dead_end_depth": dead_end_depth,
        "junction_distance": jd_map,
        "degree": degree,
        "core": core,
        "loop_set": loop_set,
        "loop_distance": loop_dist_map,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shape(ms):
    """Return (height, width) for numpy array or nested list."""
    if hasattr(ms, "shape"):
        return int(ms.shape[0]), int(ms.shape[1])
    return len(ms), len(ms[0]) if ms else 0


def _cell(ms, r: int, c: int) -> int:
    """Return cell value (0=empty, 1=wall, -1=unseen)."""
    return int(ms[r, c]) if hasattr(ms, "shape") else int(ms[r][c])


def _valid_pos(pos: Tuple[int, int], ms) -> bool:
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1


def _count_exits(ms, pos: Tuple[int, int]) -> int:
    """Number of cardinal exits from *pos*."""
    r, c = pos
    exits = 0
    if r > 0 and _cell(ms, r - 1, c) != 1:
        exits += 1
    if r < _shape(ms)[0] - 1 and _cell(ms, r + 1, c) != 1:
        exits += 1
    if c > 0 and _cell(ms, r, c - 1) != 1:
        exits += 1
    if c < _shape(ms)[1] - 1 and _cell(ms, r, c + 1) != 1:
        exits += 1
    return exits


def _neighbors(pos: Tuple[int, int], ms) -> Set[Tuple[int, int]]:
    r, c = pos
    result = set()
    for dr, dc in MOVE_DELTAS:
        nxt = (r + dr, c + dc)
        if _valid_pos(nxt, ms):
            result.add(nxt)
    return result


def _compute_core(open_cells: Set[Tuple], ms) -> Set[Tuple]:
    """Iteratively trim leaves (degree <= 1) — remaining cells form the core."""
    active = set(open_cells)
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for cell in active:
            deg = sum(1 for nxt in _neighbors(cell, ms) if nxt in active)
            if deg <= 1:
                to_remove.add(cell)
        if to_remove:
            active -= to_remove
            changed = True
    return active


def _find_loops(core: Set[Tuple], ms) -> Set[Tuple]:
    """Find the largest cycle in the core via DFS back-edge detection."""
    if not core:
        return set()
    cycles: list = []
    vis: Set[Tuple] = set()
    parent: Dict[Tuple, Optional[Tuple]] = {}

    for start in list(core):
        if start in vis:
            continue
        stack = [(start, None, iter(_neighbors(start, ms)))]
        vis.add(start)
        parent[start] = None
        while stack:
            u, p, it = stack[-1]
            try:
                v = next(it)
            except StopIteration:
                stack.pop()
                continue
            if v not in core:
                continue
            if v not in vis:
                vis.add(v)
                parent[v] = u
                stack.append((v, u, iter(_neighbors(v, ms))))
            elif v != p:
                cycle = [v]
                cur = u
                while cur is not None and cur != v:
                    cycle.append(cur)
                    cur = parent.get(cur)
                if cur == v and len(cycle) >= 4:
                    cycles.append(cycle)

    if cycles:
        return set(max(cycles, key=len))
    return set()
