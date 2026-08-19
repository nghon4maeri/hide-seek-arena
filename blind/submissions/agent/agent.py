
import random, sys, time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple
import numpy as np

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

from pathfinding import (
    DIRS, _cell, _valid, _apply, _manhattan, _cell_exits, _legal, _neighbors,
    bfs_dist, astar, capture_eta, safe_area, INF_DIST, CAPTURE_DISTANCE,
)
from topology import TopologyAnalyzer, STATIC_FULL_MAP
from belief_state import BeliefState

STEP_TIME_LIMIT = 0.85
MOVE = {(-1,0):Move.UP,(1,0):Move.DOWN,(0,-1):Move.LEFT,(0,1):Move.RIGHT,(0,0):Move.STAY}

# ===================================================================
# PACMAN — Optimized with Interception + Speed Packing
# ===================================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.memory_map = None
        self.belief = BeliefState(21,21)
        self.topo = TopologyAnalyzer()
        self.last_seen = None
        self._last_seen_step = 0
        self._enemy_dir = None
        self._dir_streak = 0
        self._visited = {}
        self._step_t0 = 0.0

    def _update_memory(self, ms):
        if self.memory_map is None: self.memory_map = np.full_like(ms, -1, dtype=int)
        v = (ms != -1); self.memory_map[v] = ms[v]

    def _ensure_topo(self):
        if not self.topo.ready and self.memory_map is not None:
            k = (self.memory_map != -1)
            if k.sum() > self.memory_map.size*0.3:
                t = self.memory_map.copy(); t[t==-1]=0; self.topo.analyze(t)

    def _time_ok(self): return (time.time()-self._step_t0) < STEP_TIME_LIMIT

    def _delta_to_move(self, delta): return MOVE.get(delta, Move.STAY)

    def _pack_speed(self, path, my_pos):
        """Pack consecutive same-direction steps up to self.speed."""
        if not path: return (Move.STAY, 1)
        first = path[0]
        delta = (first[0]-my_pos[0], first[1]-my_pos[1])
        move = MOVE.get(delta, Move.STAY)
        steps = 1; cur = first
        for nxt in path[1:]:
            nd = (nxt[0]-cur[0], nxt[1]-cur[1])
            if nd == delta and steps < self.speed:
                steps += 1; cur = nxt
            else: break
        return (move, steps)

    def _track_enemy(self, enemy_pos):
        if self.last_seen is None: return
        dr, dc = enemy_pos[0]-self.last_seen[0], enemy_pos[1]-self.last_seen[1]
        nd = (dr, dc)
        if nd == self._enemy_dir and (dr!=0 or dc!=0): self._dir_streak += 1
        else: self._enemy_dir = nd; self._dir_streak = 1 if (dr!=0 or dc!=0) else 0

    def _intercept(self, ms, enemy_pos):
        """Find junction to intercept Ghost along its movement direction."""
        if self._dir_streak < 2 or self._enemy_dir is None: return None
        dr, dc = self._enemy_dir
        er, ec = enemy_pos
        H, W = 21, 21
        best, best_s = None, float("-inf")
        for i in range(2, 6):
            nr, nc = er+dr*i, ec+dc*i
            if not (0<=nr<H and 0<=nc<W): break
            if _cell(ms, nr, nc)==1: break
            nxt = (nr, nc)
            exits = _cell_exits(nxt, ms)
            sc = exits*200
            if self.topo.ready and nxt in self.topo.junctions: sc += 500
            if sc > best_s: best_s, best = sc, nxt
        return best

    def _explore(self, me):
        H, W = self.memory_map.shape
        best, best_s = None, float("-inf")
        for r in range(H):
            for c in range(W):
                if self.memory_map[r,c] != 0: continue
                has_fog = any(0<=r+dr<H and 0<=c+dc<W and self.memory_map[r+dr,c+dc]==-1 for dr,dc in DIRS)
                if not has_fog: continue
                d = _manhattan((r,c), me)
                prob = self.belief.prob_at((r,c))
                sc = prob*500 - d
                if self.topo.ready: sc += self.topo.weight((r,c), self.memory_map)*20
                if sc > best_s: best_s, best = sc, (r,c)
        if best:
            path = astar(self.memory_map, me, best)
            if path: return self._pack_speed(path, me)
        moves = _legal(me, self.memory_map)
        if moves:
            d = min(moves, key=lambda d: self._visited.get(_apply(me, d), 0))
            return MOVE.get(d, Move.STAY)
        return Move.STAY

    def step(self, map_state, my_position, enemy_position, step_number):
        self._step_t0 = time.time()
        self._update_memory(map_state)
        me = tuple(my_position)
        self._ensure_topo()
        self._visited[me] = self._visited.get(me, 0)+1

        enemy = None
        if enemy_position is not None:
            enemy = tuple(int(v) for v in enemy_position)
            self._track_enemy(enemy)
            self.last_seen = enemy
            self._last_seen_step = step_number
        self.belief.update(me, enemy, self.memory_map, enemy_speed=2)

        if self._visited.get(me,0) >= 5:
            self._visited.clear()
            m = _legal(me, self.memory_map)
            if m: return MOVE.get(random.choice(m), Move.STAY)

        if self.topo.ready and me in self.topo.dead_ends:
            m = _legal(me, self.memory_map)
            if m: return MOVE.get(max(m, key=lambda d: self.topo.weight(_apply(me,d),self.memory_map)), Move.STAY)

        if enemy is not None:
            intercept = self._intercept(self.memory_map, enemy)
            if intercept:
                path = astar(self.memory_map, me, intercept)
                if path: return self._pack_speed(path, me)
            path = astar(self.memory_map, me, enemy)
            if path: return self._pack_speed(path, me)

        since = step_number - self._last_seen_step if self._last_seen_step>0 else 999
        if since <= 15 and self.last_seen is not None:
            if me == self.last_seen: self.last_seen = None
            else:
                path = astar(self.memory_map, me, self.last_seen)
                if path: return self._pack_speed(path, me)

        return self._explore(me)
