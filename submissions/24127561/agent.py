"""Pacman/Seeker Agent — SOTA A* + Choke-point Interception.

Student: 24127561
Role:   Pacman/Seeker Engineer

Algorithm core:
  1. A* with heapq              — O(N log N) shortest path
  2. Junction scouting           — scan 5-7 cells around Ghost for choke points
  3. Choke-point interception    — intercept Ghost at junctions, not chase its tail
  4. Target locking (3 turns)    — prevent oscillation when Ghost feints
  5. Multi-step exploitation     — extract max straight steps from A* path
  6. Path extrapolation          — predict Ghost heading from movement history

Constraints:
  - step() < 1.0s
  - Return Move or (Move, steps), steps ∈ [1, pacman_speed]
  - No L-shaped turns in a single multi-step return
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path
from typing import List, Optional, Tuple

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

# ===================================================================
# Constants
# ===================================================================
MOVE_ORDER: Tuple[Move, ...] = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
CHOKE_SCOUT_DIST = 7     # How far ahead to scan for junctions
LOCK_DURATION = 3        # Turns to hold a choke-point lock
A_STAR_PHASE_END = 10    # Switch from pure A* to interception after this step


# ===================================================================
# Grid utilities (self-contained)
# ===================================================================

def _shape(ms):
    if hasattr(ms, "shape"):
        return int(ms.shape[0]), int(ms.shape[1])
    return len(ms), len(ms[0]) if ms else 0


def _cell(ms, r, c):
    return int(ms[r, c]) if hasattr(ms, "shape") else int(ms[r][c])


def _apply(pos, move):
    return (pos[0] + move.value[0], pos[1] + move.value[1])


def _valid(pos, ms):
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1


def _legal(pos, ms):
    return [m for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _cell_exits(pos, ms):
    """Count how many valid exits a cell has (degree)."""
    return sum(1 for m in MOVE_ORDER if _valid(_apply(pos, m), ms))


# ===================================================================
# A* Search — heapq-optimised O(N log N)
# ===================================================================

def astar(ms, start, goal):
    """A* shortest path returning list of Move from start to goal."""
    if goal is None or not _valid(start, ms) or not _valid(goal, ms):
        return []
    if start == goal:
        return []

    open_set = [(0, 0, start)]  # (f, g, pos)
    came_from = {}
    g_score = {start: 0}
    closed = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = []
            while current != start:
                prev, move = came_from[current]
                path.append(move)
                current = prev
            path.reverse()
            return path

        for move in MOVE_ORDER:
            nxt = _apply(current, move)
            if not _valid(nxt, ms) or nxt in closed:
                continue
            ng = g + 1
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng
                came_from[nxt] = (current, move)
                heapq.heappush(open_set, (ng + _manhattan(nxt, goal), ng, nxt))
    return []


# ===================================================================
# BFS distance map (for junction distance checks)
# ===================================================================

def bfs_dist(ms, start, max_dist=30):
    """BFS distance map from start, up to max_dist steps."""
    if not _valid(start, ms):
        return {}
    dist = {start: 0}
    q = [start]
    for cur in q:
        if dist[cur] >= max_dist:
            continue
        for m in MOVE_ORDER:
            nxt = _apply(cur, m)
            if nxt not in dist and _valid(nxt, ms):
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


# ===================================================================
# PacmanAgent — SOTA Seeker
# ===================================================================

class PacmanAgent(BasePacmanAgent):
    """SOTA Pacman: A* + junction interception + target locking + multi-step."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self._enemy: Optional[Tuple[int, int]] = None
        self._prev_enemy: Optional[Tuple[int, int]] = None
        self._locked_target: Optional[Tuple[int, int]] = None
        self._lock_remaining: int = 0
        self._visited = set()

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        me = tuple(int(v) for v in my_position)
        self._visited.add(me)

        # Track enemy movement for heading calculation
        if enemy_position is not None:
            self._prev_enemy = self._enemy
            self._enemy = tuple(int(v) for v in enemy_position)

        ghost = self._enemy
        if ghost is None:
            return self._explore(me, map_state)

        # Compute ghost heading (for interception)
        heading = None
        if self._prev_enemy is not None:
            dr = ghost[0] - self._prev_enemy[0]
            dc = ghost[1] - self._prev_enemy[1]
            if dr != 0 or dc != 0:
                heading = (dr, dc)

        # ================================================================
        # Phase 1 (early): Pure A* pursuit — close distance fast
        # ================================================================
        if step_number <= A_STAR_PHASE_END:
            self._locked_target = None
            self._lock_remaining = 0
            path = astar(map_state, me, ghost)
            if path:
                return self._path_to_action(me, path, map_state)
            return (Move.STAY, 1)

        # ================================================================
        # Phase 2 (mid-late): Choke-point interception
        # ================================================================
        target = ghost

        # Compute direct BFS distance to ghost
        dist_to_ghost = bfs_dist(map_state, me, max_dist=10).get(ghost, _manhattan(me, ghost))

        # If Ghost is very close, skip choke logic — pursue directly
        if dist_to_ghost <= 3:
            self._locked_target = None
            self._lock_remaining = 0
            path = astar(map_state, me, ghost)
            if path:
                return self._path_to_action(me, path, map_state)
            return (Move.STAY, 1)

        # Maintain target lock
        if self._lock_remaining > 0 and self._locked_target is not None:
            if me == self._locked_target:
                self._locked_target = None
                self._lock_remaining = 0
            else:
                # Check if lock is still relevant: is Ghost heading toward locked target?
                lock_dist_to_ghost = bfs_dist(map_state, self._locked_target, max_dist=10).get(ghost, 999)
                if lock_dist_to_ghost > dist_to_ghost + 5:
                    # Ghost has moved away from the locked target — release
                    self._locked_target = None
                    self._lock_remaining = 0
                else:
                    target = self._locked_target
                    self._lock_remaining -= 1

        if self._lock_remaining <= 0 and heading is not None:
            # Scout for a new choke point
            choke = self._scout_choke(ghost, heading, map_state, me)
            if choke is not None and choke != ghost:
                # Only lock if the choke point is closer than ghost
                choke_dist = bfs_dist(map_state, me, max_dist=10).get(choke, 999)
                if choke_dist <= dist_to_ghost + 3:
                    self._locked_target = choke
                    self._lock_remaining = LOCK_DURATION
                    target = choke

        # Pathfind to selected target
        path = astar(map_state, me, target)
        if not path and target != ghost:
            # Choke point unreachable, fall back to direct pursuit
            self._locked_target = None
            self._lock_remaining = 0
            path = astar(map_state, me, ghost)

        if path:
            return self._path_to_action(me, path, map_state)
        return (Move.STAY, 1)

    # ------------------------------------------------------------------
    # Choke-point scouting
    # ------------------------------------------------------------------
    def _scout_choke(self, ghost_pos, heading, ms, my_pos):
        """Scan forward along Ghost's heading for intercept-able junctions.

        Returns a junction cell that Pacman can reach BEFORE Ghost,
        or None if no advantageous choke point exists.
        """
        dr, dc = heading
        pac_dist = bfs_dist(ms, my_pos, max_dist=25)
        ghost_dist = bfs_dist(ms, ghost_pos, max_dist=25)

        cur = ghost_pos
        for step_ahead in range(1, CHOKE_SCOUT_DIST + 1):
            nxt = (cur[0] + dr, cur[1] + dc)
            if not _valid(nxt, ms):
                # Wall ahead — ghost must turn, choke at current position
                if _cell_exits(cur, ms) <= 2:
                    return cur
                break

            exits = _cell_exits(nxt, ms)

            # Is this a junction (>= 3 exits)?
            if exits >= 3:
                p_to = pac_dist.get(nxt, 999)
                g_to = ghost_dist.get(nxt, 999)
                # Pacman speed-2: Pacman covers distance in ceil(p_to/2) turns
                # Ghost covers distance in g_to turns (or step_ahead from current)
                # Intercept if Pacman can reach in <= Ghost's arrival time
                pac_turns = (p_to + 1) // 2
                if pac_turns <= step_ahead + 1:  # +1 margin for lock
                    return nxt

            # Dead-end entrance (exits == 2 but on a corridor branch)
            if exits == 2 and step_ahead >= 3:
                # Check if this corridor is a trap for Ghost
                p_to = pac_dist.get(nxt, 999)
                g_to = ghost_dist.get(nxt, 999)
                pac_turns = (p_to + 1) // 2
                if pac_turns <= step_ahead:
                    return nxt

            cur = nxt

        return None

    # ------------------------------------------------------------------
    # Convert A* path → (Move, steps) action
    # ------------------------------------------------------------------
    def _path_to_action(self, me, path, ms):
        """Extract first move from A* path, pack consecutive same-direction
        steps up to pacman_speed (straight-line constraint)."""
        first_move = path[0]

        # Count desired consecutive same-direction steps from path
        desired = 1
        for i in range(1, min(len(path), self.pacman_speed)):
            if path[i] == first_move:
                desired += 1
            else:
                break

        # Walk up to `desired` steps, stopping if wall encountered
        steps = 0
        cur = me
        for _ in range(min(self.pacman_speed, desired)):
            nxt = _apply(cur, first_move)
            if not _valid(nxt, ms):
                break
            steps += 1
            cur = nxt

        return (first_move, max(1, steps))

    # ------------------------------------------------------------------
    # Exploration (enemy not visible — fog of war fallback)
    # ------------------------------------------------------------------
    def _explore(self, me, ms):
        candidates = _legal(me, ms)
        if not candidates:
            return (Move.STAY, 1)

        def score(m):
            nxt = _apply(me, m)
            unvisited_bonus = 5 if nxt not in self._visited else 0
            return unvisited_bonus + _cell_exits(nxt, ms)

        best = max(candidates, key=score)
        steps = 0
        cur = me
        for _ in range(self.pacman_speed):
            nxt = _apply(cur, best)
            if not _valid(nxt, ms):
                break
            steps += 1
            cur = nxt
        return (best, max(1, steps))


# ===================================================================
# GhostAgent — minimal placeholder (not primary deliverable)
# ===================================================================

class GhostAgent(BaseGhostAgent):
    """Minimal Ghost for sandbox — Seeker is the primary role for 24127561."""

    def step(self, map_state, my_position, enemy_position, step_number):
        return Move.STAY
