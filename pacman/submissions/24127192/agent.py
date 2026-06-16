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
    __slots__ = ('_c',)
    def __init__(self):
        self._c: Dict[Tuple[int,int], Dict[Tuple[int,int],int]] = {}

    def dist(self, ms, s):
        if s in self._c:
            return self._c[s]
        d = self._run(ms, s)
        self._c[s] = d
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

    # ------------------------------------------------------------------
    # Strategic flee target: pick safest reachable hub far from Pacman
    # ------------------------------------------------------------------
    def _update_flee_target(self, ms, me, pac):
        pd = self._bfs.dist(ms, pac)
        gd = self._bfs.dist(ms, me)

        best_score = float("-inf")
        best = None

        for j in self._tp.jn:
            g_to_j = gd.get(j, 999)
            p_to_j = pd.get(j, 999)
            p_eff = (p_to_j + 1) // 2  # Pacman speed 2

            # Ghost must arrive before Pacman
            if g_to_j >= p_eff:
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
        best = ordered[0]

        max_d = 6 if bfs_d <= 3 else (5 if bfs_d <= 6 else (4 if bfs_d <= 10 else 3))

        for depth in range(MIN_DEPTH, max_d + 1):
            if time.time() - t0 > TIME_BUDGET * 0.75 and depth > MIN_DEPTH:
                break
            a_val, b_val = float("-inf"), float("inf")
            cb, cv = ordered[0], float("-inf")
            timeout = False
            for m in ordered:
                ng = _apply(gpos, m)
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
            moves = _legal(gp, ms)
            if not moves:
                moves = [Move.STAY]
            for m in moves:
                r = self._ab(ms, _apply(gp, m), pp, depth - 1, False,
                             a, b, pd_orig, ft_dist, step, phase, t0)
                if r is None:
                    return None
                if r > v:
                    v = r
                a = max(a, v)
                if b <= a:
                    break
            return v
        else:
            v = float("inf")
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

        # 1. BFS distance from Pacman
        bfs_d = pd.get(gp, _manhattan(gp, pp) + 30)

        # 2. Mobility
        mob = _flood(ms, gp, 8)

        # 3. Branching factor
        branch = self._tp.br.get(gp, 0)

        # 4. Dead-end penalty
        de_pen = self._de_pen(gp, bfs_d, pd)

        # 5. Corridor penalty
        co_pen = 0.0
        if gp in self._tp.co:
            if bfs_d <= 4:
                co_pen = 1.0
            elif bfs_d <= 8:
                co_pen = 0.5

        # 6. Junction proximity
        j_bonus = max(0, 5 - self._tp.jd.get(gp, 10))

        # 7. Strategic target distance bonus
        tgt_bonus = 0.0
        if self._flee_target is not None and ft_dist:
            td = ft_dist.get(gp, 99)
            if td < 99:
                tgt_bonus = max(0, 15 - td)

        # Weights
        w = self._w(phase, bfs_d)
        return (
            w['d'] * bfs_d
            + w['m'] * mob
            + w['b'] * branch
            + w['j'] * j_bonus
            + w.get('t', 0) * tgt_bonus
            - w['de'] * de_pen
            - w['co'] * co_pen
        )

    def _de_pen(self, gp, bfs_d, pd):
        """Dead-end branch penalty."""
        if gp not in self._tp.deb:
            return 0.0
        bd = self._tp.deb_d.get(gp, 0)
        jd_val = self._tp.jd.get(gp, 0)
        p_val = pd.get(gp, 99)
        p_eff = (p_val + 1) // 2
        if p_eff <= jd_val + 1:
            return 1.0
        if bfs_d < bd + 3:
            return 0.8
        if bfs_d < bd + 6:
            return 0.5
        return 0.15

    # ------------------------------------------------------------------
    # Phase weights
    # ------------------------------------------------------------------
    def _w(self, phase, bfs_d):
        if phase == 'early':
            w = dict(d=15, s=3, m=2, b=5, e=10, j=5, h=3, c=6, t=10,
                     de=70, co=30, o=12)
        elif phase == 'mid':
            w = dict(d=12, s=4, m=2.5, b=5, e=12, j=6, h=4, c=8, t=7,
                     de=80, co=35, o=15)
        else:
            w = dict(d=10, s=5, m=3, b=6, e=15, j=7, h=5, c=10, t=5,
                     de=90, co=40, o=18)

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
        ft = self._flee_target

        def qs(m):
            nxt = _apply(gp, m)
            if _manhattan(nxt, pp) < 2:
                return -999999.0

            # BFS distance from Pacman to next position
            d = pd.get(nxt, _manhattan(nxt, pp) + 30)
            
            # Branch factor
            br = self._tp.br.get(nxt, 0)

            # Dead-end penalty (very important)
            de = 0.0
            if nxt in self._tp.de:
                de = -150.0
            elif nxt in self._tp.deb:
                bd = self._tp.deb_d.get(nxt, 0)
                jd_val = self._tp.jd.get(nxt, 0)
                pd_nxt = pd.get(nxt, 99)
                pd_eff = (pd_nxt + 1) // 2
                if pd_eff <= jd_val + 1:
                    de = -200.0  # trap!
                elif d < bd + 3:
                    de = -120.0
                else:
                    de = -40.0

            # Corridor penalty
            co = -30.0 if (nxt in self._tp.co and d < 6) else 0.0

            # Oscillation penalty
            osc = -15.0 * self._osc(nxt)

            # Flee target bonus (using pre-computed BFS from target)
            tgt = 0.0
            if ft and ft_dist:
                cur_d = ft_dist.get(gp, 99)
                nxt_d = ft_dist.get(nxt, 99)
                if nxt_d < cur_d:
                    tgt = 15.0

            return 15.0 * d + 5.0 * br + de + co + osc + tgt

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
