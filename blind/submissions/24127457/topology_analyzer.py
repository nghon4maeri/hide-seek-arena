"""topology_analyzer.py — Core grid utilities and static map topology analysis.

Provides shared grid operations, pathfinding (BFS, A*), and the
TopologyAnalyzer that classifies cells into junctions, corridors,
dead-ends, loops, and core regions on the static full map.
"""

import heapq
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ===================================================================
# Constants
# ===================================================================
MOVE_ORDER = (
    (-1, 0),  # UP
    (1, 0),   # DOWN
    (0, -1),  # LEFT
    (0, 1),   # RIGHT
)

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_STATIC_LAYOUT = [
    "#####################",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...................#",
    "#.###.#.#####.#.###.#",
    "#.....#...#...#.....#",
    "#####.###.#.###.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#.........#.........#",
    "#####.#.#####.#.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...#.........#...#..",
    "###.#.#.#####.#.#.###",
    "#.....#...#...#.....#",
    "#.#######.#.#######.#",
    "#...................#",
    "#####################",
]

STATIC_FULL_MAP = np.array(
    [[1 if c == '#' else 0 for c in row] for row in _STATIC_LAYOUT],
    dtype=int,
)
STATIC_FULL_MAP[9, 10] = 0
STATIC_FULL_MAP[15, 10] = 0


# ===================================================================
# Grid utilities
# ===================================================================
def _shape(ms):
    if hasattr(ms, "shape"):
        return int(ms.shape[0]), int(ms.shape[1])
    return len(ms), len(ms[0]) if ms else (0, 0)


def _cell(ms, r, c):
    return int(ms[r, c]) if hasattr(ms, "shape") else int(ms[r][c])


def _apply(pos, delta):
    return (pos[0] + delta[0], pos[1] + delta[1])


def _apply_dir(pos, dr, dc):
    return (pos[0] + dr, pos[1] + dc)


def _valid(pos, ms):
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1


def _legal(pos, ms):
    return [m for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _cell_exits(pos, ms):
    return sum(1 for m in MOVE_ORDER if _valid(_apply(pos, m), ms))


def _dir_from_delta(dr, dc):
    if dr == -1:
        return 0
    if dr == 1:
        return 1
    if dc == -1:
        return 2
    if dc == 1:
        return 3
    return 4


def _random_valid_move(map_state, my_pos):
    H, W = map_state.shape
    r, c = my_pos
    candidates = [
        (dr, dc) for dr, dc in DIRS
        if 0 <= r + dr < H and 0 <= c + dc < W and map_state[r + dr, c + dc] == 0
    ]
    if not candidates:
        return (0, 0)
    return random.choice(candidates)


import random


# ===================================================================
# BFS distance map (capped)
# ===================================================================
def bfs_dist(ms, start, max_dist=40):
    if not _valid(start, ms):
        return {start: 0}
    d = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        if d[cur] >= max_dist:
            continue
        for delta in MOVE_ORDER:
            nxt = _apply(cur, delta)
            if nxt not in d and _valid(nxt, ms):
                d[nxt] = d[cur] + 1
                q.append(nxt)
    return d


# ===================================================================
# A* search (returns list of deltas)
# ===================================================================
def astar(ms, start, goal):
    if goal is None or not _valid(start, ms) or not _valid(goal, ms):
        return []
    if start == goal:
        return []
    open_set = [(0, 0, start)]
    came_from: Dict[Tuple, Tuple[Tuple, Tuple]] = {}
    g_score = {start: 0}
    closed: Set[Tuple] = set()
    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            path = []
            while current != start:
                prev, delta = came_from[current]
                path.append(delta)
                current = prev
            path.reverse()
            return path
        for delta in MOVE_ORDER:
            nxt = _apply(current, delta)
            if not _valid(nxt, ms) or nxt in closed:
                continue
            ng = g + 1
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng
                came_from[nxt] = (current, delta)
                heapq.heappush(open_set, (ng + _manhattan(nxt, goal), ng, nxt))
    return []


# ===================================================================
# Topology Analyzer
# ===================================================================
class TopologyAnalyzer:
    """Analyzes a static map to classify cells by topological role.

    Attributes:
        dead_ends: cells with <= 1 exit
        junctions: cells with >= 3 exits
        core: cells remaining after iterative peeling (2-core)
        corridor_cells: cells on corridors leading from dead-ends to junctions
        loops: junctions that are also in the 2-core
        junction_dist: BFS distance from each cell to nearest junction
    """

    def __init__(self):
        self.dead_ends: Set[Tuple] = set()
        self.junctions: Set[Tuple] = set()
        self.core: Set[Tuple] = set()
        self.junction_dist: Dict[Tuple, int] = {}
        self.corridor_cells: Set[Tuple] = set()
        self.loops: Set[Tuple] = set()
        self._weight_cache: Dict[Tuple, float] = {}
        self.ready = False

    def analyze(self, ms):
        if self.ready:
            return
        H, W = _shape(ms)
        all_free = {(r, c) for r in range(H) for c in range(W) if _cell(ms, r, c) != 1}

        deg = {}
        for p in all_free:
            deg[p] = _cell_exits(p, ms)
            if deg[p] >= 3:
                self.junctions.add(p)
            elif deg[p] <= 1:
                self.dead_ends.add(p)

        dead_seeds = set(self.dead_ends)
        for seed in dead_seeds:
            cur, prev = seed, None
            while True:
                self.corridor_cells.add(cur)
                if cur in self.junctions:
                    break
                nxts = [
                    _apply(cur, m)
                    for m in MOVE_ORDER
                    if _valid(_apply(cur, m), ms) and _apply(cur, m) != prev
                ]
                if not nxts:
                    break
                nxt = nxts[0]
                if nxt in self.corridor_cells:
                    break
                prev, cur = cur, nxt

        active = set(all_free)
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for cell in active:
                na = sum(
                    1 for m in MOVE_ORDER
                    if _valid(_apply(cell, m), ms) and _apply(cell, m) in active
                )
                if na <= 1:
                    to_remove.add(cell)
            if to_remove:
                active -= to_remove
                changed = True
        self.core = active

        self.loops = {p for p in all_free if deg.get(p, 0) >= 3 and p in self.core}

        self.junction_dist = {p: 10**9 for p in all_free}
        q = deque()
        for p in (self.junctions if self.junctions else self.core):
            self.junction_dist[p] = 0
            q.append(p)
        while q:
            cur = q.popleft()
            for m in MOVE_ORDER:
                nxt = _apply(cur, m)
                if _valid(nxt, ms) and self.junction_dist.get(nxt, 10**9) > self.junction_dist[cur] + 1:
                    self.junction_dist[nxt] = self.junction_dist[cur] + 1
                    q.append(nxt)

        self._weight_cache.clear()
        self.ready = True

    def _get_topological_weight(self, cell, ms):
        if cell in self._weight_cache:
            return self._weight_cache[cell]
        w = 1.0
        if cell in self.junctions:
            w = 4.0
        elif cell in self.core and cell not in self.corridor_cells:
            ex = _cell_exits(cell, ms)
            w = 3.0 if ex >= 3 else 2.0 if ex == 2 else 1.0
        elif cell in self.corridor_cells:
            w = 0.5
        elif cell in self.dead_ends:
            w = 0.1
        self._weight_cache[cell] = w
        return w
