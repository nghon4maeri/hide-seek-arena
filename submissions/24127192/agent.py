"""Ghost/Hider Agent sandbox for student 24127192.

Role: Ghost/Hider Engineer
Focus: Survival strategy using US-L* + Exact Opponent Modeling Deep Search.
"""

from __future__ import annotations

import sys
import time
import heapq
from functools import lru_cache
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

MOVE_ORDER: Tuple[Move, ...] = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
TIME_BUDGET = 0.95

def _shape(ms):
    if hasattr(ms, "shape"): return int(ms.shape[0]), int(ms.shape[1])
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

class Topo:
    def __init__(self):
        self.deb = set()
        self.ok = False
        
    def init(self, ms):
        if self.ok: return
        h, w = _shape(ms)
        de = set()
        jn = set()
        for r in range(h):
            for c in range(w):
                if _cell(ms, r, c) == 1: continue
                p = (r, c)
                deg = len(_legal(p, ms))
                if deg <= 1: de.add(p)
                if deg >= 3: jn.add(p)
                
        for de_cell in de:
            cur = de_cell
            vis = {cur}
            while cur not in jn:
                self.deb.add(cur)
                found = False
                for m in MOVE_ORDER:
                    nxt = _apply(cur, m)
                    if nxt not in vis and _valid(nxt, ms):
                        vis.add(nxt)
                        cur = nxt
                        found = True
                        break
                if not found: break
            self.deb.add(cur)
        self.ok = True

class USLStar:
    def __init__(self):
        self._permanent = {}
        
    def get_state(self, ms, gp, pp, last_gp):
        dx = gp[0] - pp[0]
        dy = gp[1] - pp[1]
        hdr = gp[0] - last_gp[0] if last_gp else 0
        hdc = gp[1] - last_gp[1] if last_gp else 0
        pac_moves = [m for m in MOVE_ORDER if _valid(_apply(pp, m), ms)]
        n = len(pac_moves)
        if n <= 1: geo = 0
        elif n >= 3: geo = 3
        else:
            m1, m2 = pac_moves[0], pac_moves[1]
            if m1.value[0]+m2.value[0]==0 and m1.value[1]+m2.value[1]==0: geo = 1
            else: geo = 2
        walls = sum((1<<i) for i, m in enumerate(MOVE_ORDER) if not _valid(_apply(pp, m), ms))
        return (dx, dy, hdr, hdc, geo, walls)
        
    def observe(self, state, action):
        self._permanent[state] = action
        
    def predict(self, ms, pp, gp, last_gp, fallback_fn):
        state = self.get_state(ms, gp, pp, last_gp)
        if state in self._permanent:
            dr, dc = self._permanent[state]
            pred = (pp[0] + dr, pp[1] + dc)
            if _valid(pred, ms): return pred
        return fallback_fn(pp, gp, last_gp)

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tp = Topo()
        self._core = set()
        self._loop_set = set()
        self._usl = USLStar()
        self._hist = deque(maxlen=50)
        self._last_pp = None
        self._predicted_pp = None
        
    def _compute_core(self, ms):
        h, w = len(ms), len(ms[0])
        active = set()
        for r in range(h):
            for c in range(w):
                if ms[r][c] != 1:
                    active.add((r, c))
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for cell in active:
                deg = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                          if (cell[0]+dr, cell[1]+dc) in active)
                if deg <= 1: to_remove.add(cell)
            if to_remove:
                active -= to_remove
                changed = True
        self._core = active

    def _dfs_find_loops(self, ms):
        self._loop_set = set()
        if not self._core: return
        cycles = []
        vis = set()
        parent = {}
        for start in list(self._core):
            if start in vis: continue
            stack = [(start, None)]
            vis.add(start)
            parent[start] = None
            while stack:
                u, p = stack[-1]
                found = False
                for m in MOVE_ORDER:
                    v = _apply(u, m)
                    if v not in self._core: continue
                    if v not in vis:
                        vis.add(v)
                        parent[v] = u
                        stack.append((v, u))
                        found = True
                        break
                    elif v != p:
                        cycle = [v]
                        curr = u
                        while curr is not None and curr != v:
                            cycle.append(curr)
                            curr = parent.get(curr)
                        if curr == v:
                            cycle.append(v)
                            if len(cycle) >= 4: cycles.append(cycle)
                if not found:
                    stack.pop()
        if cycles:
            self._loop_set = set(max(cycles, key=len))

    @lru_cache(maxsize=100000)
    def _cached_astar(self, pp, target):
        if pp == target: return ()
        rows, cols = len(self._static_map), len(self._static_map[0])
        def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
        open_set = []
        heapq.heappush(open_set, (heuristic(pp, target), pp))
        closed_set = set()
        came_from = {}
        g_score = {pp: 0}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while open_set:
            _, current = heapq.heappop(open_set)
            if current in closed_set: continue
            closed_set.add(current)
            if current == target:
                path = []
                while current != pp:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return tuple(path)
            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols and self._static_map[nr][nc] == 0:
                    neighbor = (nr, nc)
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, target)
                        heapq.heappush(open_set, (f_score, neighbor))
        return ()

    def _predict_astar_seeker(self, pp, gp, last_gp):
        ms = self._static_map
        target = gp
        if last_gp is not None:
            pr = gp[0] + (gp[0] - last_gp[0])
            pc = gp[1] + (gp[1] - last_gp[1])
            if 0 <= pr < len(ms) and 0 <= pc < len(ms[0]) and ms[pr][pc] == 0:
                target = (pr, pc)
        if pp == target: return pp
        path = self._cached_astar(pp, target)
        if not path: return pp
        first = path[0]
        dr, dc = first[0] - pp[0], first[1] - pp[1]
        steps = 1
        current = first
        for nxt in path[1:]:
            ndr, ndc = nxt[0] - current[0], nxt[1] - current[1]
            if ndr == dr and ndc == dc and steps < 2:
                steps += 1
                current = nxt
            else: break
        return current

    def _eval(self, gp, pp):
        path = self._cached_astar(gp, pp)
        true_dist = len(path) if path else 99
        score = true_dist * 100
        if gp in self._tp.deb: score -= 5000
        if gp in self._core: score += 1000
        if gp in self._loop_set: score += 500
        return score

    def _iterative_deep_search(self, ms, gp, pp, last_gp, depth, max_depth, t0):
        if time.time() - t0 > TIME_BUDGET * 0.85: return None, None
        if depth == max_depth: return self._eval(gp, pp), None
        
        np_pos = self._usl.predict(ms, pp, gp, last_gp, self._predict_astar_seeker)
        best_score = float('-inf')
        best_move = None
        
        cands = _legal(gp, ms)
        cand_scores = [(_manhattan(_apply(gp, m), np_pos), m) for m in cands]
        cand_scores.sort(key=lambda x: x[0], reverse=True)
        
        for _, m in cand_scores:
            ng = _apply(gp, m)
            if _manhattan(ng, np_pos) < 2:
                score = -100000 + (depth + 1) * 1000
            else:
                score, _ = self._iterative_deep_search(ms, ng, np_pos, gp, depth + 1, max_depth, t0)
                if score is None: return None, None
            if score > best_score:
                best_score = score
                best_move = m
        return best_score, best_move

    def step(self, map_state, my_position, enemy_position, step_number):
        t0 = time.time()
        me = tuple(my_position)
        ms = map_state

        if not hasattr(self, '_static_map'):
            self._static_map = tuple(tuple(int(x) for x in row) for row in ms)
            self._tp.init(ms)
            self._compute_core(self._static_map)
            self._dfs_find_loops(self._static_map)

        last_gp = self._hist[-1] if self._hist else None
        
        if enemy_position is not None:
            pac = tuple(enemy_position)
            if self._last_pp is not None and self._predicted_pp is not None:
                actual_action = (pac[0] - self._last_pp[0], pac[1] - self._last_pp[1])
                last_state = self._usl.get_state(ms, last_gp, self._last_pp, self._hist[-2] if len(self._hist) >= 2 else None)
                if pac != self._predicted_pp:
                    self._usl.observe(last_state, actual_action)
                else:
                    self._usl.observe(last_state, actual_action)

            self._predicted_pp = self._usl.predict(ms, pac, me, last_gp, self._predict_astar_seeker)
            self._last_pp = pac
        else:
            pac = None

        cands = _legal(me, ms)
        if not cands: return Move.STAY
        
        self._hist.append(me)
        
        if pac is None: return cands[0]

        best_move_overall = cands[0]
        # iterative deepening
        for depth in range(2, 20, 2):
            score, move = self._iterative_deep_search(ms, me, pac, last_gp, 0, depth, t0)
            if score is None: break # Timeout
            if move is not None: best_move_overall = move
            if score < -50000: break # Inevitable death
            
        return best_move_overall

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def step(self, map_state, my_position, enemy_position, step_number):
        me = tuple(my_position)
        ms = map_state
        cands = _legal(me, ms)
        if not cands: return Move.STAY
        if enemy_position is None: return cands[0]
        
        pac = tuple(enemy_position)
        def bfs(start):
            d = {start: 0}
            q = deque([start])
            while q:
                cur = q.popleft()
                for m in MOVE_ORDER:
                    nxt = _apply(cur, m)
                    if _valid(nxt, ms) and nxt not in d:
                        d[nxt] = d[cur] + 1
                        q.append(nxt)
            return d
            
        enemy_d = bfs(pac)
        best_m = cands[0]
        best_d = float('inf')
        for m in cands:
            nxt = _apply(me, m)
            d = enemy_d.get(nxt, float('inf'))
            if d < best_d:
                best_d = d
                best_m = m
        return best_m