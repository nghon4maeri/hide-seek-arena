"""pathfinding.py — Grid utilities, BFS, A*, capture_eta, safe_area.

Ported from Lab 1 Top-1 submission:
  - capture_eta: speed-2-aware turns for Pacman to reach target
  - safe_area: floodfill of cells Ghost reaches before Pacman
  - LRU cache for performance
"""

import heapq
import time
from collections import deque, OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CAPTURE_DISTANCE = 2
INF_DIST = 10**9


def _shape(ms):
    if hasattr(ms, "shape"):
        return int(ms.shape[0]), int(ms.shape[1])
    return len(ms), len(ms[0]) if ms else (0, 0)


def _cell(ms, r, c):
    return int(ms[r, c]) if hasattr(ms, "shape") else int(ms[r][c])


def _apply(pos, delta):
    return (pos[0] + delta[0], pos[1] + delta[1])


def _valid(pos, ms):
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1


def _legal(pos, ms):
    return [d for d in DIRS if _valid(_apply(pos, d), ms)]


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _cell_exits(pos, ms):
    return sum(1 for d in DIRS if _valid(_apply(pos, d), ms))


def bfs_dist(ms, start, max_dist=40):
    """BFS from start, returns dict of {cell: distance}."""
    if not _valid(start, ms):
        return {start: 0}
    d = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        if d[cur] >= max_dist:
            continue
        for delta in DIRS:
            nxt = _apply(cur, delta)
            if nxt not in d and _valid(nxt, ms):
                d[nxt] = d[cur] + 1
                q.append(nxt)
    return d


def astar(ms, start, goal):
    """A* search, returns list of positions along path."""
    if goal is None or not _valid(start, ms) or not _valid(goal, ms):
        return []
    if start == goal:
        return []
    open_set = [(0, 0, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()
    while open_set:
        _, g_val, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        for dr, dc in DIRS:
            nxt = (current[0] + dr, current[1] + dc)
            if not _valid(nxt, ms) or nxt in closed:
                continue
            ng = g_val + 1
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng
                came_from[nxt] = current
                heapq.heappush(open_set, (ng + _manhattan(nxt, goal), ng, nxt))
    return []


def astar_deltas(ms, start, goal):
    """A* returning delta tuples instead of positions."""
    if goal is None or not _valid(start, ms) or not _valid(goal, ms):
        return []
    if start == goal:
        return []
    open_set = [(0, 0, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()
    while open_set:
        _, g_val, current = heapq.heappop(open_set)
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
        for delta in DIRS:
            nxt = _apply(current, delta)
            if not _valid(nxt, ms) or nxt in closed:
                continue
            ng = g_val + 1
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng
                came_from[nxt] = (current, delta)
                heapq.heappush(open_set, (ng + _manhattan(nxt, goal), ng, nxt))
    return []


def _neighbors(pos, ms):
    """Return list of valid neighbor positions."""
    r, c = pos
    result = []
    for dr, dc in DIRS:
        nxt = (r + dr, c + dc)
        if _valid(nxt, ms):
            result.append(nxt)
    return result


# ===================================================================
# LRU Cache (from Lab 1)
# ===================================================================
class LruDictCache:
    """Small fixed-size dict cache, cleared each step."""

    def __init__(self, maxsize=4000):
        self._d = {}
        self._max = maxsize

    def get(self, k, default=None):
        return self._d.get(k, default)

    def put(self, k, v):
        self._d[k] = v
        if len(self._d) > self._max:
            self._d.clear()

    def clear(self):
        self._d.clear()


# ===================================================================
# Capture ETA — speed-2-aware turns for Pacman to reach target
# ===================================================================
def capture_eta(ms, pac, target, pacman_speed=2):
    """Calculate turns for Pacman (speed=N) to reach target.

    Pacman moves up to pacman_speed cells in a straight line per turn.
    Returns estimated number of turns.
    """
    md = _manhattan(pac, target)
    if md < CAPTURE_DISTANCE:
        return 0

    # Fast path: approximate for long distances
    if md > 12:
        return max(1, (md + 1) // pacman_speed)

    path = astar(ms, pac, target)
    if not path:
        return INF_DIST

    pos = pac
    idx = 0
    turns = 0

    while idx < len(path):
        first = path[idx]
        dr, dc = first[0] - pos[0], first[1] - pos[1]
        used = 0

        while idx < len(path) and used < pacman_speed:
            nxt = path[idx]
            if (nxt[0] - pos[0], nxt[1] - pos[1]) != (dr, dc):
                break
            pos = nxt
            idx += 1
            used += 1

            if _manhattan(pos, target) < CAPTURE_DISTANCE:
                turns += 1
                return turns

        turns += 1
        if turns > 80:
            break

    return turns


# ===================================================================
# Safe Area — floodfill cells Ghost reaches BEFORE Pacman
# ===================================================================
def safe_area(ms, ghost, pac_positions, topo=None, pacman_speed=2, max_depth=12):
    """Floodfill from ghost: count cells where ghost ETA < pacman ETA.

    Args:
        ms: static map (STATIC_FULL_MAP)
        ghost: Ghost position (r, c)
        pac_positions: list of possible Pacman positions
        topo: TopologyAnalyzer for degree info
        pacman_speed: Pacman speed
        max_depth: max floodfill depth

    Returns float: weighted safe area score
    """
    total = 0.0
    q = deque([ghost])
    seen = {ghost: 0}

    while q:
        cur = q.popleft()
        gd = seen[cur]
        if gd > max_depth:
            continue

        # Compute minimum capture ETA from any pacman position
        pac_eta = min(
            (capture_eta(ms, p, cur, pacman_speed) for p in pac_positions),
            default=INF_DIST
        )
        margin = pac_eta - gd

        if margin > 0:
            deg = topo.degree.get(cur, 2) if topo and hasattr(topo, 'degree') else 2
            total += 1.0 + min(3, margin) * 0.35 + deg * 0.08

            for nxt in _neighbors(cur, ms):
                if nxt not in seen:
                    seen[nxt] = gd + 1
                    q.append(nxt)

    return total


# ===================================================================
# Follow path with speed (Pacman speed-2 simulation)
# ===================================================================
def follow_path_with_speed(start, path, speed=2):
    """Simulate Pacman following a path with speed > 1."""
    if not path:
        return start

    first = path[0]
    dr, dc = first[0] - start[0], first[1] - start[1]
    pos = start
    used = 0

    for nxt in path:
        if used >= speed:
            break
        if (nxt[0] - pos[0], nxt[1] - pos[1]) != (dr, dc):
            break
        pos = nxt
        used += 1

    return pos


# ===================================================================
# Greedy best-first next position
# ===================================================================
def greedy_best_next(start, target, ms):
    """Return next position toward target (best immediate move)."""
    moves = _neighbors(start, ms)
    if not moves:
        return start
    return min(moves, key=lambda p: _manhattan(p, target))
