"""
agent.py — Final Submission for Blind Adversary (Lab 2)
Optimized: capture_eta + safe_area + interception + speed packing
"""

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
# GHOST — Optimized with capture_eta + safe_area
# ===================================================================
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_map = None
        self.belief = BeliefState(21,21)
        self.topo = TopologyAnalyzer()
        self.last_seen = None
        self._hist = deque(maxlen=30)
        self._visited = {}
        self._step_t0 = 0.0
        self._enemy_dir = (0,0)

    def _update_memory(self, ms):
        if self.memory_map is None: self.memory_map = np.full_like(ms, -1, dtype=int)
        v = (ms != -1); self.memory_map[v] = ms[v]

    def _ensure_topo(self):
        if not self.topo.ready: self.topo.analyze(STATIC_FULL_MAP)

    def _time_ok(self): return (time.time()-self._step_t0) < STEP_TIME_LIMIT

    def _legal_deltas(self, me): return _legal(me, STATIC_FULL_MAP)

    def _is_trap(self, cell):
        if cell in self.topo.dead_ends: return True
        vis = {cell}; q = deque([(cell,0)])
        while q:
            cur,d = q.popleft()
            if d >= 6: return False
            for m in DIRS:
                nxt = _apply(cur,m)
                if nxt in vis or not _valid(nxt, STATIC_FULL_MAP): continue
                vis.add(nxt)
                if nxt in self.topo.junctions or (nxt in self.topo.core and nxt not in self.topo.corridor_cells):
                    return False
                q.append((nxt, d+1))
        return True

    def _escape_margin(self, me, enemy):
        gd = bfs_dist(STATIC_FULL_MAP, me, max_dist=6)
        pd = bfs_dist(STATIC_FULL_MAP, enemy, max_dist=12)
        return len(gd)/max(1,len(pd))

    def _score(self, cell, enemy_positions, use_safe_area=True):
        """Optimized scoring: capture_eta instead of Manhattan, safe_area floodfill."""
        s = 0.0
        # --- capture_eta: speed-2-aware, more accurate than Manhattan ---
        worst_eta = min((capture_eta(STATIC_FULL_MAP, p, cell, 2) for p in enemy_positions[:3]), default=INF_DIST)
        worst_dist = min((_manhattan(cell, p) for p in enemy_positions[:3]), default=INF_DIST)

        if worst_dist < 2: s -= 15000.0
        else: s += worst_dist * 100.0  # reduced weight since capture_eta is primary

        if worst_eta <= 0: s -= 80000.0
        elif worst_eta <= 2: s -= 10000.0
        else: s += min(worst_eta, 10) * 600.0

        # --- Topology ---
        if self.topo.ready:
            exits = _cell_exits(cell, STATIC_FULL_MAP)
            s += exits * 150.0
            if cell in self.topo.loops: s += 1000.0
            if cell in self.topo.junctions: s += 600.0
            if cell in self.topo.core: s += 250.0
            if cell in self.topo.dead_ends: s -= 6000.0 + self.topo.dead_end_depth.get(cell,1)*300
            if cell in self.topo.corridor_cells: s -= 1000.0
            jd = self.topo.junction_dist.get(cell, 99)
            s += max(0, 4-jd)*100.0
            ld = self.topo.loop_dist.get(cell, 99)
            s += max(0, 8-ld)*80.0

        # --- Danger time ---
        danger = self.belief.danger_at(cell)
        s += danger * 40.0

        # --- Safe area (capture-ETA floodfill) ---
        if use_safe_area:
            area = safe_area(STATIC_FULL_MAP, cell, tuple(enemy_positions[:3]), self.topo, 2, 10)
            s += min(area, 60.0) * 80.0
            if area < 8: s -= (8-area) * 400.0

        # --- History anti-oscillation ---
        if cell in self._hist: s -= 2500.0
        if len(self._hist)>=2 and cell==self._hist[-2]: s -= 5000.0

        return s

    def _panic_move(self, me, enemy):
        deltas = self._legal_deltas(me)
        if not deltas: return (0,0)
        best_d, best_s = deltas[0], float("-inf")
        for gd in deltas:
            g1 = _apply(me, gd)
            if _manhattan(g1, enemy) < 2: continue
            p_deltas = _legal(enemy, STATIC_FULL_MAP)
            worst = float("inf")
            for pd in p_deltas:
                p1 = _apply(enemy, pd)
                p2 = _apply(p1, pd) if _valid(p1, STATIC_FULL_MAP) else p1
                g_d2 = _legal(g1, STATIC_FULL_MAP)
                best_g = float("-inf")
                for gd2 in g_d2:
                    g2 = _apply(g1, gd2)
                    if _manhattan(g2, p2) < 2: continue
                    sc = _manhattan(g2, p2)*500 + _cell_exits(g2, STATIC_FULL_MAP)*200
                    if g2 in self.topo.junctions: sc += 800
                    if g2 in self.topo.loops: sc += 1200
                    if self._is_trap(g2): sc -= 6000
                    if g2 in self._hist: sc -= 1200
                    if sc > best_g: best_g = sc
                worst = min(worst, best_g)
            if worst > best_s: best_s, best_d = worst, gd
        return best_d

    def _evasion_move(self, me, enemy):
        deltas = self._legal_deltas(me)
        if not deltas: return (0,0)
        best_d, best_s = deltas[0], float("-inf")
        enemy_positions = [enemy]
        for gd in deltas:
            g1 = _apply(me, gd)
            if _manhattan(g1, enemy) < 2: continue
            future = float("inf")
            for pd in _legal(enemy, STATIC_FULL_MAP):
                p1 = _apply(enemy, pd)
                p2 = _apply(p1, pd) if _valid(p1, STATIC_FULL_MAP) else p1
                future = min(future, _manhattan(g1, p2))
            s = future*500.0 + _cell_exits(g1, STATIC_FULL_MAP)*200.0
            if g1 in self.topo.junctions: s += 500
            if g1 in self.topo.loops: s += 1000
            if self._is_trap(g1): s -= 12000.0
            if g1 in self._hist: s -= 2500.0
            s += self._score(g1, enemy_positions, True)*0.3
            if s > best_s: best_s, best_d = s, gd
        return best_d

    def _fortress_move(self, me, enemy_estimate):
        deltas = self._legal_deltas(me)
        if not deltas: return (0,0)
        best_d, best_s = deltas[0], float("-inf")
        for gd in deltas:
            g1 = _apply(me, gd)
            d = _manhattan(g1, enemy_estimate)
            s = d*1000.0 + _cell_exits(g1, STATIC_FULL_MAP)*200
            if g1 in self.topo.loops: s += 1200
            if g1 in self.topo.junctions: s += 800
            if g1 in self.topo.dead_ends: s -= 6000
            if g1 in self.topo.corridor_cells: s -= 1200
            jd = self.topo.junction_dist.get(g1, 99)
            s += max(0, 5-jd)*120
            ld = self.topo.loop_dist.get(g1, 99)
            s += max(0, 8-ld)*60
            if g1 in self._hist: s -= 3500
            if s > best_s: best_s, best_d = s, gd
        return best_d

    def _explore_move(self, me):
        moves = _neighbors(me, STATIC_FULL_MAP)
        if not moves: return (0,0)
        threat = self.belief.threat_center()
        # --- FLEE: if recently saw enemy, aggressively move away ---
        flee_weight = 0.0
        if self.belief.steps_since_seen <= 8 and self.last_seen is not None:
            flee_weight = 300.0 / max(1, self.belief.steps_since_seen)

        gd = bfs_dist(STATIC_FULL_MAP, me, max_dist=6)
        best_c, best_s = me, float("-inf")
        for cell, dist in gd.items():
            if cell == me: continue
            s = _cell_exits(cell, STATIC_FULL_MAP)*120
            s += max(0, 6-dist)*400
            if cell in self.topo.junctions: s += 700
            if cell in self.topo.loops: s += 1000
            if cell in self.topo.core: s += 350
            if cell in self.topo.dead_ends: s -= 2000
            s += _manhattan(cell, threat)*35
            # Flee bonus: move away from last seen enemy position
            if flee_weight > 0 and self.last_seen is not None:
                s += _manhattan(cell, self.last_seen) * flee_weight
            jd = self.topo.junction_dist.get(cell, 99)
            s += max(0, 6-jd)*120
            if cell in self._hist: s -= 600
            if s > best_s: best_s, best_c = s, cell
        if best_c != me:
            path = astar(STATIC_FULL_MAP, me, best_c)
            if path and len(path)>=1 and path[0] in moves:
                return (path[0][0]-me[0], path[0][1]-me[1])
        best_n = max(moves, key=lambda n: (
            _cell_exits(n, STATIC_FULL_MAP)*100
            +(600 if n in self.topo.junctions else 0)
            +(900 if n in self.topo.loops else 0)
            -(_manhattan(n, threat)<3)*500
            +(flee_weight*_manhattan(n, self.last_seen) if flee_weight>0 and self.last_seen else 0)
            -(4000 if self._is_trap(n) else 0)
            -(1000 if n in self._hist else 0)
        ))
        return (best_n[0]-me[0], best_n[1]-me[1])

    def step(self, map_state, my_position, enemy_position, step_number):
        self._step_t0 = time.time()
        self._update_memory(map_state)
        me = tuple(my_position)
        self._ensure_topo()
        self._hist.append(me)
        self._visited[me] = self._visited.get(me, 0)+1

        enemy = None
        if enemy_position is not None:
            enemy = tuple(int(v) for v in enemy_position)
            self.last_seen = enemy
            if hasattr(self,'_last_pp') and self._last_pp is not None:
                dr, dc = enemy[0]-self._last_pp[0], enemy[1]-self._last_pp[1]
                self._enemy_dir = (dr, dc)
            self._last_pp = enemy
        self.belief.update(me, enemy, self.memory_map, enemy_speed=2)

        deltas = self._legal_deltas(me)
        if not deltas: return Move.STAY

        if self._visited.get(me,0) >= 5:
            self._visited.clear(); self._hist.clear()
            return MOVE.get(random.choice(deltas), Move.STAY)

        if enemy is not None:
            if self._is_trap(me):
                safe = [d for d in deltas if not self._is_trap(_apply(me,d))]
                if safe: return MOVE.get(max(safe, key=lambda d: _cell_exits(_apply(me,d),STATIC_FULL_MAP)), Move.STAY)
            margin = self._escape_margin(me, enemy)
            if margin < 0.4:
                safe = [d for d in deltas if not self._is_trap(_apply(me,d))]
                if safe: return MOVE.get(max(safe, key=lambda d: _cell_exits(_apply(me,d),STATIC_FULL_MAP)), Move.STAY)

            dist = _manhattan(me, enemy)
            positions = [enemy]  # also include belief estimates for robustness
            for cell, prob in self.belief.highest_prob_cells(3):
                if prob > 0.01 and cell != enemy: positions.append(cell)

            if dist < 4: return MOVE.get(self._panic_move(me, enemy), Move.STAY)
            elif dist < 8: return MOVE.get(self._evasion_move(me, enemy), Move.STAY)
            else: return MOVE.get(self._fortress_move(me, self.belief.threat_center()), Move.STAY)

        return MOVE.get(self._explore_move(me), Move.STAY)


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
