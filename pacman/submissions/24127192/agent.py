"""Ghost/Hider Agent sandbox for student 24127192.

Role: Ghost/Hider Engineer
Focus: Survival strategy, safe-area reasoning, dead-end avoidance, and
       hide heuristics.

Strategy overview
-----------------
GhostAgent (primary deliverable):
    Multi-layer evaluation combining:
    - BFS distance maximisation with memoised distance maps
    - Safe territory analysis (Voronoi with Pacman speed-2 adjustment)
    - Dead-end/corridor/junction topology with branch depth tracking
    - Depth-limited Minimax with Alpha-Beta pruning & iterative deepening
    - Strategic flee target selection (safest reachable hub)
    - Greedy heuristic move ordering for alpha-beta efficiency
    - Anti-oscillation with pattern detection
    - Pacman speed-2 straight-line simulation in adversarial search
    - Adaptive phase-based evaluation weights (early/mid/late)

PacmanAgent (secondary):
    BFS shortest-path chase with Pacman speed exploitation.

This file is self-contained and does NOT modify anything under pacman/src/.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Framework imports
# ---------------------------------------------------------------------------
SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOVE_ORDER: Tuple[Move, ...] = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
TIME_BUDGET = 0.75
MAX_DEPTH = 6
MIN_DEPTH = 2


# ===================================================================
# Core utility
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


# ===================================================================
# BFS with cache
# ===================================================================

class BFS:
    __slots__ = ('_c', '_maxsize')
    def __init__(self, maxsize=128):
        self._c: Dict[Tuple[int,int], Dict[Tuple[int,int],int]] = {}
        self._maxsize = maxsize

    def dist(self, ms, s):
        if s in self._c:
            # Move to end (most recently used) to act as LRU
            d = self._c.pop(s)
            self._c[s] = d
            return d
        d = self._run(ms, s)
        self._c[s] = d
        if len(self._c) > self._maxsize:
            # Remove oldest (first item in dict)
            oldest = next(iter(self._c))
            del self._c[oldest]
        return d

    @staticmethod
    def _run(ms, s):
        if s is None or not _valid(s, ms):
            return {}
        d = {s: 0}
        q = deque([s])
        while q:
            cur = q.popleft()
            for m in MOVE_ORDER:
                nxt = _apply(cur, m)
                if nxt not in d and _valid(nxt, ms):
                    d[nxt] = d[cur] + 1
                    q.append(nxt)
        return d

def _bfs_path(ms, s, g):
    if g is None or not _valid(s, ms) or not _valid(g, ms):
        return []
    if s == g:
        return [Move.STAY]
    parent = {s: (None, None)}
    q = deque([s])
    while q:
        cur = q.popleft()
        if cur == g:
            break
        for m in MOVE_ORDER:
            nxt = _apply(cur, m)
            if nxt not in parent and _valid(nxt, ms):
                parent[nxt] = (cur, m)
                q.append(nxt)
    if g not in parent:
        return []
    path = []
    cur = g
    while cur != s:
        prev, mv = parent[cur]
        path.append(mv)
        cur = prev
    path.reverse()
    return path

def _flood(ms, s, maxd=10):
    if not _valid(s, ms):
        return 0
    seen = {s}
    q = deque([(s, 0)])
    while q:
        cur, d = q.popleft()
        if d >= maxd:
            continue
        for m in MOVE_ORDER:
            nxt = _apply(cur, m)
            if nxt not in seen and _valid(nxt, ms):
                seen.add(nxt)
                q.append((nxt, d + 1))
    return len(seen)


# ===================================================================
# Topology
# ===================================================================

class Topo:
    def __init__(self):
        self.br: Dict[Tuple,int] = {}
        self.de: Set[Tuple] = set()
        self.co: Set[Tuple] = set()
        self.jn: Set[Tuple] = set()
        self.deb: Set[Tuple] = set()
        self.deb_d: Dict[Tuple,int] = {}
        self.jd: Dict[Tuple,int] = {}
        self.hub: Dict[Tuple,float] = {}
        self.adj: Dict[Tuple, List[Tuple[Move, Tuple]]] = {}
        self.ok = False

    def init(self, ms):
        if self.ok:
            return
        h, w = _shape(ms)
        for r in range(h):
            for c in range(w):
                if _cell(ms, r, c) == 1:
                    continue
                p = (r, c)
                deg = len(_legal(p, ms))
                self.br[p] = deg
                if deg <= 1:
                    self.de.add(p)
                elif deg == 2:
                    nb = _legal(p, ms)
                    d1, d2 = nb[0].value, nb[1].value
                    if d1[0]+d2[0] == 0 and d1[1]+d2[1] == 0:
                        self.co.add(p)
                if deg >= 3:
                    self.jn.add(p)

                # Precompute valid moves
                valid_moves = []
                for m in MOVE_ORDER:
                    nxt = _apply(p, m)
                    if _valid(nxt, ms):
                        valid_moves.append((m, nxt))
                self.adj[p] = valid_moves

        # Multi-source BFS from junctions
        q = deque()
        for j in self.jn:
            self.jd[j] = 0
            q.append(j)
        while q:
            cur = q.popleft()
            for m in MOVE_ORDER:
                nxt = _apply(cur, m)
                if nxt not in self.jd and _valid(nxt, ms):
                    self.jd[nxt] = self.jd[cur] + 1
                    q.append(nxt)

        # Dead-end branches
        for de_cell in self.de:
            branch = []
            cur = de_cell
            vis = {cur}
            while cur not in self.jn:
                branch.append(cur)
                found = False
                for m in MOVE_ORDER:
                    nxt = _apply(cur, m)
                    if nxt not in vis and _valid(nxt, ms):
                        vis.add(nxt)
                        cur = nxt
                        found = True
                        break
                if not found:
                    break
            if cur in self.jn:
                total = len(branch)
                for i, cell in enumerate(branch):
                    self.deb.add(cell)
                    self.deb_d[cell] = total - i
            else:
                for cell in branch:
                    self.deb.add(cell)
                    self.deb_d[cell] = 999

        # Hub connectivity scores
        for j in self.jn:
            sc = 0.0
            d = BFS._run(ms, j)
            for j2 in self.jn:
                if j2 == j:
                    continue
                dd = d.get(j2, 99)
                if dd <= 6:
                    sc += (7.0 - dd) / 7.0
            self.hub[j] = sc

        self.ok = True


# ===================================================================
# Pacman speed-2 reach
# ===================================================================

def _pac_reach(p, ms):
    """All positions Pacman can reach in 1 turn (speed 2, straight only)."""
    r = {p}
    for m in MOVE_ORDER:
        p1 = _apply(p, m)
        if _valid(p1, ms):
            r.add(p1)
            p2 = _apply(p1, m)
            if _valid(p2, ms):
                r.add(p2)
    return list(r)


# ===================================================================
# GhostAgent
# ===================================================================

class GhostAgent(BaseGhostAgent):
    """Advanced Ghost (Hider) with iterative-deepening minimax + alpha-beta."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._enemy: Optional[Tuple[int,int]] = None
        self._hist = deque(maxlen=30)
        self._bfs = BFS()
        self._tp = Topo()
        self._nopen = 0
        self._flee_target: Optional[Tuple[int,int]] = None

    def step(self, map_state, my_position, enemy_position, step_number):
        t0 = time.time()
        me = tuple(my_position)

        if not self._tp.ok:
            self._tp.init(map_state)
            h, w = _shape(map_state)
            self._nopen = sum(1 for r in range(h) for c in range(w)
                              if _cell(map_state, r, c) != 1)

        if enemy_position is not None:
            self._enemy = tuple(enemy_position)

        if not self._hist or self._hist[-1] != me:
            self._hist.append(me)

        cands = _legal(me, map_state)
        if not cands:
            return Move.STAY

        pac = self._enemy
        if pac is None:
            return self._mob_move(me, cands, map_state)

        # Update strategic flee target
        self._update_flee_target(map_state, me, pac)

        return self._decide(map_state, me, pac, cands, step_number, t0)

    def _update_flee_target(self, ms, me, pac):
        pd = self._bfs.dist(ms, pac)
        gd = self._bfs.dist(ms, me)

        best_score = float("-inf")
        best = None

        for j in self._tp.jn:
            g_to_j = gd.get(j, 999)
            p_to_j = pd.get(j, 999)
            p_eff = (p_to_j + 1) // 2  # Pacman speed 2

            # Ghost must arrive safely before Pacman with a margin
            # Since Pacman is speed 2, a 1-turn lead is unsafe at a junction
            if g_to_j >= p_eff - 2:
                continue

            hub_sc = self._tp.hub.get(j, 0)
            # Score: far from Pacman + well-connected + not too far for Ghost
            score = p_to_j * 3 + hub_sc * 8 - g_to_j * 2

            if score > best_score:
                best_score = score
                best = j

        if best is None:
            # Fallback: farthest safe cell
            best_d = 0
            for cell, g in gd.items():
                p = pd.get(cell, 999)
                p_eff = (p + 1) // 2
                if g < p_eff and p > best_d:
                    best_d = p
                    best = cell

        self._flee_target = best

    # ------------------------------------------------------------------
    # Decision: iterative deepening minimax with alpha-beta
    # ------------------------------------------------------------------
    def _decide(self, ms, gpos, ppos, cands, step, t0):
        pd = self._bfs.dist(ms, ppos)
        gd = self._bfs.dist(ms, gpos)
        bfs_d = pd.get(gpos, _manhattan(gpos, ppos))
        phase = 'early' if step < 40 else ('mid' if step < 130 else 'late')
        
        ft_dist = self._bfs.dist(ms, self._flee_target) if self._flee_target else {}

        ordered = self._order(ms, gpos, ppos, cands, pd, phase, ft_dist)
        # Restore fast max_d since heuristic prevents horizon traps
        max_d = 8 if pd.get(gpos, 99) <= 3 else (7 if pd.get(gpos, 99) <= 6 else (6 if pd.get(gpos, 99) <= 10 else 5))

        for depth in range(MIN_DEPTH, max_d + 1):
            if time.time() - t0 > TIME_BUDGET * 0.75 and depth > MIN_DEPTH:
                break
            a_val, b_val = float("-inf"), float("inf")
            cb, cv = ordered[0], float("-inf")
            timeout = False
            
            # Precompute cand_nxt for the root node to avoid _apply
            cand_nxt = {}
            for m in ordered:
                for m_adj, nxt in self._tp.adj[gpos]:
                    if m == m_adj:
                        cand_nxt[m] = nxt
                        break

            for m in ordered:
                ng = cand_nxt[m]
                if _manhattan(ng, ppos) < 2:
                    v = -100000.0
                else:
                    v = self._ab(ms, ng, ppos, depth - 1, False,
                                 a_val, b_val, pd, ft_dist, step, phase, t0)
                if v is None:
                    timeout = True
                    break
                if v > cv:
                    cv = v
                    cb = m
                a_val = max(a_val, v)
            if not timeout:
                best = cb
        return best

    # ------------------------------------------------------------------
    # Alpha-Beta search
    # ------------------------------------------------------------------
    def _ab(self, ms, gp, pp, depth, is_g, a, b, pd_orig, ft_dist, step, phase, t0):
        if time.time() - t0 > TIME_BUDGET:
            return None
        if _manhattan(gp, pp) < 2:
            return -100000.0
        if depth <= 0:
            return self._eval(ms, gp, pp, pd_orig, ft_dist, step, phase)

        if is_g:
            v = float("-inf")
            # Ghost moves 1 step, use precomputed adj
            for _, ng in self._tp.adj[gp]:
                r = self._ab(ms, ng, pp, depth - 1, False,
                             a, b, pd_orig, ft_dist, step, phase, t0)
                if r is None:
                    return None
                if r > v:
                    v = r
                a = max(a, v)
                if b <= a:
                    break
            return v
        if not is_g:
            v = float("inf")
            # Pacman moves 2 steps, use _pac_reach
            for np_ in _pac_reach(pp, ms):
                r = self._ab(ms, gp, np_, depth - 1, True,
                             a, b, pd_orig, ft_dist, step, phase, t0)
                if r is None:
                    return None
                if r < v:
                    v = r
                b = min(b, v)
                if b <= a:
                    break
            return v

    # ------------------------------------------------------------------
    # Evaluation function
    # ------------------------------------------------------------------
    def _eval(self, ms, gp, pp, pd_orig, ft_dist, step, phase):
        pd = pd_orig
        gd = self._bfs.dist(ms, gp)

        # 1. BFS distance from Pacman
        bfs_d = pd.get(gp, _manhattan(gp, pp) + 30)

        # 2. Safe territory (speed-2 adjusted Voronoi)
        safe_t = 0
        for c, g_val in gd.items():
            p_val = pd.get(c, 9999)
            if g_val < (p_val + 1) // 2:
                safe_t += 1

        # 3. Mobility
        if not hasattr(self, '_flood_cache'):
            self._flood_cache = {}
        f_key = (gp, 8)
        if f_key not in self._flood_cache:
            self._flood_cache[f_key] = _flood(ms, gp, 8)
        mob = self._flood_cache[f_key]

        # 4. Branching factor
        branch = self._tp.br.get(gp, 0)

        # 5. Dead-end penalty
        de_pen = self._de_pen(gp, bfs_d, pd)

        # 6. Corridor penalty
        co_pen = 0.0
        if gp in self._tp.co:
            if bfs_d <= 4:
                co_pen = 1.0
            elif bfs_d <= 8:
                co_pen = 0.5

        # 7. Oscillation penalty
        osc = self._osc(gp)

        # 8. Escape routes (speed-2 aware)
        esc = 0
        for m in MOVE_ORDER:
            nxt = _apply(gp, m)
            g_val = gd.get(nxt, 9999)
            if g_val >= 9999:
                continue
            p_val = pd.get(nxt, 9999)
            if g_val < (p_val + 1) // 2:
                esc += 1

        # 9. Junction proximity
        j_bonus = max(0, 5 - self._tp.jd.get(gp, 10))

        # 10. Hub connectivity in neighbourhood
        hub = 0.0
        for j, sc in self._tp.hub.items():
            d = gd.get(j, 99)
            if d <= 4:
                hub += sc * (5 - d) / 5.0

        # 11. Strategic target distance bonus
        tgt_bonus = 0.0
        if self._flee_target is not None and ft_dist:
            td = ft_dist.get(gp, 99)
            if td < 99:
                tgt_bonus = max(0, 15 - td)

        # 12. Control ratio
        ctrl = safe_t / max(1, self._nopen)

        # Weights
        w = self._w(phase, bfs_d)
        return (
            w['d'] * bfs_d
            + w['s'] * safe_t
            + w['m'] * mob
            + w['b'] * branch
            + w['e'] * esc
            + w['j'] * j_bonus
            + w['h'] * hub
            + w['c'] * ctrl
            + w.get('t', 0) * tgt_bonus
            - w['de'] * de_pen
            - w['co'] * co_pen
            - w['o'] * osc
        )

    def _de_pen(self, gp, bfs_d, pd):
        """Dead-end branch and corridor threat penalty."""
        jd_val = self._tp.jd.get(gp, 0)
        p_val = pd.get(gp, 99)
        p_eff = (p_val + 1) // 2

        if gp in self._tp.deb:
            bd = self._tp.deb_d.get(gp, 0)
            if p_eff <= jd_val + 2:
                return 100.0  # FATAL TRAP: severely penalize
            elif bfs_d < bd + 3:
                return 0.8
            else:
                return 0.5
        
        if gp in self._tp.co:
            # If Pacman can outrun Ghost inside the corridor before junction
            if p_val <= jd_val + 1:
                return 50.0  # FATAL CORRIDOR
            if p_val <= jd_val + 3:
                return 0.8

        return 0.0

    # ------------------------------------------------------------------
    # Phase weights
    # ------------------------------------------------------------------
    def _w(self, phase, bfs_d):
        if phase == 'early':
            w = dict(d=15, s=0.1, m=2, b=5, e=1, j=5, h=0.2, c=1, t=10,
                     de=80, co=30, o=1)
        elif phase == 'mid':
            w = dict(d=12, s=0.2, m=2.5, b=5, e=1.5, j=6, h=0.3, c=1.5, t=7,
                     de=90, co=35, o=1.5)
        else:
            w = dict(d=10, s=0.3, m=3, b=6, e=2, j=7, h=0.5, c=2, t=5,
                     de=100, co=40, o=2)

        if bfs_d <= 3:
            w['de'] *= 3
            w['co'] *= 2.5
            w['d'] *= 2
            w['e'] *= 2
            w['o'] *= 0.3
        elif bfs_d <= 6:
            w['de'] *= 1.8
            w['co'] *= 1.5
            w['d'] *= 1.3
        return w

    # ------------------------------------------------------------------
    # Move ordering for alpha-beta
    # ------------------------------------------------------------------
    def _order(self, ms, gp, pp, cands, pd, phase, ft_dist):
        """Sort moves by quick heuristic — best first for alpha-beta."""
        # Retrieve precomputed next positions for candidates
        cand_nxt = {}
        for m in cands:
            for m_adj, nxt in self._tp.adj[gp]:
                if m == m_adj:
                    cand_nxt[m] = nxt
                    break

        def qs(m):
            nxt = cand_nxt[m]
            if _manhattan(nxt, pp) < 2:
                return -999999.0

            # BFS distance from Pacman to next position
            d = pd.get(nxt, _manhattan(nxt, pp) + 30)
            
            # Dead-end penalty
            de = self._de_pen(nxt, d, pd)
            
            # Distance from flee target (if any)
            td = ft_dist.get(nxt, 0) if ft_dist else 0
            
            return d * 100 - de * 500 - td

        return sorted(cands, key=qs, reverse=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _mob_move(self, pos, cands, ms):
        return max(cands, key=lambda m: _flood(ms, _apply(pos, m), 8))

    def _osc(self, pos):
        if not self._hist:
            return 0.0
        hl = list(self._hist)
        c = hl[-8:].count(pos)
        # Detect 2-cell oscillation
        if len(hl) >= 4 and len(set(hl[-4:])) <= 2:
            c += 4
        # Detect 3-cell oscillation
        if len(hl) >= 6 and len(set(hl[-6:])) <= 3:
            c += 2
        return float(c)


# ===================================================================
# PacmanAgent
# ===================================================================

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self._enemy = None
        self._vis: Set[Tuple] = set()

    def step(self, map_state, my_position, enemy_position, step_number):
        me = tuple(my_position)
        self._vis.add(me)
        if enemy_position is not None:
            self._enemy = tuple(enemy_position)
        t = self._enemy
        if t is not None:
            path = _bfs_path(map_state, me, t)
            if path:
                return path[0]
        cands = _legal(me, map_state)
        if not cands:
            return Move.STAY
        return max(cands, key=lambda m: (5.0 if _apply(me, m) not in self._vis else 0.0)
                                         + _flood(map_state, _apply(me, m), 4))
