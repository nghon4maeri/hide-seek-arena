"""Ghost/Hider Agent — A* Strategic Evasion + Tactical Minimax.

Student: 24127192
Role:   Ghost/Hider Engineer

Algorithm core:
  1. Strategic (Pacman far):  A* to the farthest cell Ghost can reach before Pacman
  2. Tactical (Pacman close): Light minimax for obstacle avoidance + max distance
  3. Map topology:             Dead-end/corridor avoidance, junction/loop preference
  4. Anti-oscillation:         Hard ban on revisiting cells within 6-step window
  5. Direction commitment:     Persist escape direction, never reverse

Constraints:
  - step() < 1.0s
  - Return Move ONLY (UP/DOWN/LEFT/RIGHT/STAY)
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

TIME_BUDGET = 0.90
A_STAR_THRESHOLD = 4   # Use A* when BFS > this, minimax when closer
MINIMAX_DEPTH = 8       # Depth for close-range minimax
HISTORY_BAN = 6         # Steps to ban revisiting


# ===================================================================
# Grid utilities
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
    return sum(1 for m in MOVE_ORDER if _valid(_apply(pos, m), ms))


# ===================================================================
# BFS distance map (cached)
# ===================================================================

class BFS:
    def __init__(self, maxsize=64):
        self._cache: Dict[Tuple, Dict[Tuple, int]] = {}
        self._maxsize = maxsize

    def dist(self, ms, start):
        if start in self._cache:
            d = self._cache.pop(start)
            self._cache[start] = d
            return d
        d = self._compute(ms, start)
        self._cache[start] = d
        if len(self._cache) > self._maxsize:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return d

    @staticmethod
    def _compute(ms, start):
        if not _valid(start, ms):
            return {}
        d = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for m in MOVE_ORDER:
                nxt = _apply(cur, m)
                if nxt not in d and _valid(nxt, ms):
                    d[nxt] = d[cur] + 1
                    q.append(nxt)
        return d


# ===================================================================
# A* pathfinding
# ===================================================================

def astar(ms, start, goal):
    if goal is None or not _valid(start, ms) or not _valid(goal, ms):
        return []
    if start == goal:
        return []

    import heapq
    open_set = [(0, 0, start)]
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
# Pacman speed-2 reachable set (pessimistic model)
# ===================================================================

def pacman_reach_2(pos, ms):
    reach = {pos}
    for m1 in MOVE_ORDER:
        p1 = _apply(pos, m1)
        if not _valid(p1, ms):
            continue
        reach.add(p1)
        for m2 in MOVE_ORDER:
            p2 = _apply(p1, m2)
            if _valid(p2, ms):
                reach.add(p2)
    return list(reach)


# ===================================================================
# GhostAgent — A*-Strategic + Minimax-Tactical
# ===================================================================

class GhostAgent(BaseGhostAgent):
    """SOTA Ghost: A* to safest cell + minimax evasion when cornered."""

    # Class-level topology cache — keyed by map signature
    _topology_cache: Dict[int, Dict] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._enemy: Optional[Tuple[int, int]] = None
        self._bfs = BFS()
        self._history: List[Tuple[int, int]] = []
        self._step_count: int = 0
        self._flee_target: Optional[Tuple[int, int]] = None
        self._open_cells: int = 0
        self._dead_ends: Set[Tuple] = set()
        self._initialized: bool = False
        self.floodfill_rerank_used: int = 0
        self.floodfill_rerank_skipped: int = 0
        self.opening_escape_used: int = 0
        self.opening_escape_skipped: int = 0
        self.phase_strategy_used: int = 0
        self.phase_strategy_skipped: int = 0
        self.topology_cache_hits: int = 0
        self.topology_cache_misses: int = 0
        self._last_move: Optional[Move] = None
        self._topo: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        t0 = time.time()
        me = tuple(int(v) for v in my_position)
        self._step_count = step_number

        # One-time map analysis
        if not self._initialized:
            self._analyze_map(map_state)
            self._initialized = True

        # Track enemy
        if enemy_position is not None:
            self._enemy = tuple(int(v) for v in enemy_position)

        # Track history
        if not self._history or self._history[-1] != me:
            self._history.append(me)

        # Legal moves
        candidates = _legal(me, map_state)
        if not candidates:
            return Move.STAY

        pac = self._enemy
        if pac is None:
            return self._explore(me, candidates, map_state)

        # Compute distances
        pd = self._bfs.dist(map_state, pac)
        gd = self._bfs.dist(map_state, me)
        bfs_d = pd.get(me, _manhattan(me, pac))

            # Filter out recently visited cells (hard anti-oscillation)
        safe_candidates = self._filter_oscillation(me, candidates)

        # ---- Build candidate move via existing pipeline ----
        chosen_move = None
        OPENING_WINDOW = 10

        # Opening escape: compute but do NOT return early; let phase cross-check it
        if step_number <= OPENING_WINDOW:
            chosen_move = self._opening_escape_move(map_state, me, pac, pd, candidates)
            if chosen_move is not None:
                self.opening_escape_used += 1
            else:
                self.opening_escape_skipped += 1

        # Strategic: A* to farthest safe cell
        if chosen_move is None and bfs_d > A_STAR_THRESHOLD:
            chosen_move = self._strategic_move(map_state, me, pac, pd, gd, safe_candidates)
            if chosen_move is None:
                chosen_move = self._floodfill_move(map_state, me, pac, pd, safe_candidates)

        # Tactical: minimax evasion
        if chosen_move is None:
            chosen_move = self._tactical_move(map_state, me, pac, pd, gd, safe_candidates, t0)

        # ---- Phase strategy: topology-aware cross-check ----
        if chosen_move is not None:
            topo = self._ensure_topology(map_state)
            phase_move, phase_score, chosen_phase_score = self._phase_scoring_move(
                map_state, me, pac, pd, gd, candidates, step_number,
                topo, chosen_move
            )

            if phase_move is not None and phase_move != chosen_move:
                chosen_nxt = _apply(me, chosen_move)
                chosen_exits = _cell_exits(chosen_nxt, map_state)
                chosen_bfs = pd.get(chosen_nxt, _manhattan(chosen_nxt, pac))
                is_dangerous = (chosen_exits <= 1 or chosen_bfs <= 3)

                PHASE_THRESHOLD = 20.0

                if (step_number <= 60
                        or is_dangerous
                        or phase_score > chosen_phase_score + PHASE_THRESHOLD):
                    self.phase_strategy_used += 1
                    self._last_move = phase_move
                    return phase_move
                else:
                    self.phase_strategy_skipped += 1
            else:
                self.phase_strategy_skipped += 1

        # ---- Rerank: floodfill safety tie-breaker ----
        return self._rerank_with_floodfill(
            map_state, me, pac, pd, candidates, chosen_move
        )

    # ------------------------------------------------------------------
    # Opening escape (step ≤ 10) — high-weight safety scorer
    # ------------------------------------------------------------------
    def _opening_escape_move(self, ms, me, pac, pd, candidates):
        """Conservative opening strategy for the critical first 10 steps.

        Uses higher weights than the normal scorer because early survival
        is disproportionately important.
        """
        if not candidates:
            return None

        best_move = None
        best_score = float("-inf")

        for m in candidates:
            nxt = _apply(me, m)
            if _manhattan(nxt, pac) < 2:
                continue

            bfs_dist = pd.get(nxt, _manhattan(nxt, pac))

            # Safe floodfill — cells Ghost reaches before speed‑2 Pacman
            gd = self._bfs.dist(ms, nxt)
            safe_count = 0
            for _cell_key, g_val in gd.items():
                if g_val > 15:
                    continue
                p_val = pd.get(_cell_key, 999)
                if g_val < (p_val + 1) // 2:
                    safe_count += 1

            exits = _cell_exits(nxt, ms)
            junction_bonus = 1 if exits >= 3 else 0
            dead_end_penalty = 1 if exits <= 1 else 0
            close_corridor_penalty = 1 if (exits == 2 and bfs_dist <= 6) else 0

            score = (
                100 * bfs_dist
                + 2 * safe_count
                + 40 * junction_bonus
                - 120 * dead_end_penalty
                - 60 * close_corridor_penalty
            )

            # Mild anti-backtrack: penalise reversing last move direction
            if self._last_move is not None:
                _reverse = {
                    Move.UP: Move.DOWN,
                    Move.DOWN: Move.UP,
                    Move.LEFT: Move.RIGHT,
                    Move.RIGHT: Move.LEFT,
                }
                if _reverse.get(m) == self._last_move:
                    score -= 30

            if score > best_score:
                best_score = score
                best_move = m

        if best_move is not None:
            self._last_move = best_move
        return best_move

    # ------------------------------------------------------------------
    # Phase-based topology-aware scoring
    # ------------------------------------------------------------------
    def _phase_scoring_move(self, ms, me, pac, pd, gd, candidates, step_number,
                            topo, chosen_move):
        """Score every legal move with phase-appropriate weights + topology data.

        Returns (best_phase_move, phase_score, chosen_score).
        """
        if not candidates:
            return None, float("-inf"), float("-inf")

        # Phase weights
        if step_number <= 60:
            w_dist, w_flood, w_junc = 120, 3, 60
            w_de, w_corr, w_back = -150, -80, -40
        elif step_number <= 140:
            w_dist, w_flood, w_junc = 90, 2, 45
            w_de, w_corr, w_back = -120, -50, -25
        else:
            w_dist, w_flood, w_junc = 150, 4, 30
            w_de, w_corr, w_back = -200, -100, -10

        _reverse = {
            Move.UP: Move.DOWN, Move.DOWN: Move.UP,
            Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT,
        }

        junctions = topo["junctions"]
        de_depth = topo["dead_end_depth"]
        jd_map = topo["junction_distance"]

        best_move = None
        best_score = float("-inf")
        chosen_score = float("-inf")

        for m in candidates:
            nxt = _apply(me, m)
            if _manhattan(nxt, pac) < 2:
                if m == chosen_move:
                    chosen_score = float("-inf")
                continue

            bfs_dist = pd.get(nxt, _manhattan(nxt, pac))

            # Safe floodfill count from candidate destination
            gd_local = self._bfs.dist(ms, nxt)
            safe_count = 0
            for cell, g_val in gd_local.items():
                if g_val > 15:
                    continue
                p_val = pd.get(cell, 999)
                if g_val < (p_val + 1) // 2:
                    safe_count += 1

            exits = _cell_exits(nxt, ms)

            # --- Topology-aware penalties (differentiates from opening escape) ---
            # Dead-end: penalty proportional to depth (capped at 5)
            if exits <= 1:
                dd = de_depth.get(nxt, 5)
                dead_end_penalty = min(dd, 5)
            else:
                dead_end_penalty = 0

            # Junction bonus
            junction_bonus = 1 if exits >= 3 else 0

            # Corridor risk: check if Pacman is close to either corridor exit
            corridor_risk = 0
            if exits == 2:
                nb_cells = [_apply(nxt, mv) for mv in MOVE_ORDER
                            if _valid(_apply(nxt, mv), ms)]
                for nb in nb_cells:
                    nb_exits = _cell_exits(nb, ms)
                    if nb_exits >= 3:
                        nb_pd = pd.get(nb, 999)
                        if nb_pd <= bfs_dist + 3:
                            corridor_risk += 1
                if bfs_dist <= 6 and corridor_risk == 0:
                    corridor_risk = 1

            # Junction proximity bonus: closer to junction = more escape options
            jd = jd_map.get(nxt, 10)
            junction_prox_bonus = max(0, 5 - jd)  # 5 for junction, 0 for >=5 away

            score = (
                w_dist * bfs_dist
                + w_flood * safe_count
                + w_junc * junction_bonus
                + w_de * dead_end_penalty
                + w_corr * corridor_risk
                + w_junc * junction_prox_bonus * 0.5  # scaled junction proximity
            )

            # Backtrack penalty
            if self._last_move is not None:
                if _reverse.get(m) == self._last_move:
                    score += w_back

            if m == chosen_move:
                chosen_score = score

            if score > best_score:
                best_score = score
                best_move = m

        return best_move, best_score, chosen_score

    # ------------------------------------------------------------------
    # Strategic: identify farthest safe cell and A* to it
    # ------------------------------------------------------------------
    def _strategic_move(self, ms, me, pac, pd, gd, candidates):
        # Find the farthest cell Ghost can reach before Pacman
        best_cell = None
        best_dist = 0

        for cell, g_val in gd.items():
            if g_val < 3 or g_val > 18:
                continue
            p_val = pd.get(cell, 999)
            p_turns = (p_val + 1) // 2

            # Must be safe: Ghost arrives before Pacman
            if g_val >= p_turns:
                continue

            # Avoid dead ends
            if _cell_exits(cell, ms) <= 1:
                continue

            # Score: maximize distance from Pacman
            if p_val > best_dist:
                best_dist = p_val
                best_cell = cell

        if best_cell is None:
            return None

        self._flee_target = best_cell

        # A* path to target
        path = astar(ms, me, best_cell)
        if not path:
            return None

        move = path[0]
        if move not in candidates:
            return None

        nxt = _apply(me, move)
        if _manhattan(nxt, pac) < 2:
            return None  # Would move into capture

        return move

    # ------------------------------------------------------------------
    # Floodfill-based safety scoring (when A* strategic fails to find target)
    # ------------------------------------------------------------------
    def _score_position(self, ms, gpos, pac, pd):
        """Score a candidate ghost position. Higher = safer.

        Weights are tuned experimentally for the 21×21 Pacman maze.
        """
        score = 0.0
        bfs_dist = pd.get(gpos, _manhattan(gpos, pac))

        # ---- 1. Distance from Pacman (primary survival signal) ----
        score += 10.0 * bfs_dist

        # ---- 2. Safe reachable area — cells Ghost reaches before Pacman (speed‑2) ----
        gd = self._bfs.dist(ms, gpos)
        safe_count = 0
        for _cell_key, g_val in gd.items():
            if g_val > 15:
                continue
            p_val = pd.get(_cell_key, 999)
            # Ghost arrives strictly before speed‑2 Pacman
            if g_val < (p_val + 1) // 2:
                safe_count += 1
        score += 0.5 * safe_count

        # ---- 3. Topology: reward junctions, penalise dead‑ends and corridors ----
        exits = _cell_exits(gpos, ms)
        if exits >= 3:
            score += 30.0          # junction → more escape options
        elif exits <= 1:
            score -= 50.0          # dead‑end → trap risk
        elif exits == 2 and bfs_dist <= 6:
            score -= 15.0          # corridor near Pacman → risky

        # ---- 4. Pre‑computed dead‑end map penalty ----
        if gpos in self._dead_ends:
            score -= 80.0

        return score

    def _floodfill_move(self, ms, me, pac, pd, candidates):
        """Pick the move whose destination scores highest on safety heuristics.

        Returns None only when every move puts Ghost adjacent to Pacman.
        """
        if not candidates:
            return None

        best_move = None
        best_score = float("-inf")

        for m in candidates:
            nxt = _apply(me, m)
            if _manhattan(nxt, pac) < 2:
                continue  # would step into capture range
            s = self._score_position(ms, nxt, pac, pd)
            if s > best_score:
                best_score = s
                best_move = m

        return best_move

    def _rerank_with_floodfill(self, ms, me, pac, pd, candidates, chosen_move):
        """Rerank the chosen move against floodfill-best among all candidates.

        Returns chosen_move unchanged unless floodfill finds a significantly
        safer alternative OR the chosen destination is dangerous.
        """
        if not candidates or chosen_move is None:
            self.floodfill_rerank_skipped += 1
            return chosen_move

        # Score the currently chosen move's destination
        chosen_nxt = _apply(me, chosen_move)
        chosen_score = self._score_position(ms, chosen_nxt, pac, pd)

        # Find the floodfill-best move among all legal candidates
        best_move = chosen_move
        best_score = chosen_score

        for m in candidates:
            nxt = _apply(me, m)
            if _manhattan(nxt, pac) < 2:
                continue
            s = self._score_position(ms, nxt, pac, pd)
            if s > best_score:
                best_score = s
                best_move = m

        if best_move == chosen_move:
            self.floodfill_rerank_skipped += 1
            return chosen_move

        # --- Override conditions ---
        chosen_exits = _cell_exits(chosen_nxt, ms)
        chosen_bfs = pd.get(chosen_nxt, _manhattan(chosen_nxt, pac))
        is_dangerous = (chosen_exits <= 1 or chosen_bfs <= 3)

        RERANK_THRESHOLD = 30.0

        if best_score > chosen_score + RERANK_THRESHOLD or is_dangerous:
            self.floodfill_rerank_used += 1
            return best_move

        self.floodfill_rerank_skipped += 1
        return chosen_move

    # ------------------------------------------------------------------
    # Tactical: minimax alpha-beta for close-range evasion
    # ------------------------------------------------------------------
    def _tactical_move(self, ms, me, pac, pd, gd, candidates, t0):
        if not candidates:
            return Move.STAY

        depth = MINIMAX_DEPTH
        alpha, beta = float("-inf"), float("inf")
        best_move = candidates[0]
        best_val = float("-inf")

        cand_next = {m: _apply(me, m) for m in candidates}

        # Order moves: farther from Pacman first
        ordered = sorted(candidates,
                         key=lambda m: pd.get(cand_next[m], _manhattan(cand_next[m], pac) + 50),
                         reverse=True)

        for m in ordered:
            ng = cand_next[m]

            if _manhattan(ng, pac) < 2:
                val = -100000.0
            else:
                val = self._minimax(ms, ng, pac, depth - 1, False, alpha, beta, pd, t0)

            if val is None:
                break

            if val > best_val:
                best_val = val
                best_move = m
            alpha = max(alpha, val)

        return best_move

    # ------------------------------------------------------------------
    # Minimax: Ghost maximizes, Pacman minimizes
    # ------------------------------------------------------------------
    def _minimax(self, ms, gpos, ppos, depth, is_ghost, alpha, beta, pd_orig, t0):
        if time.time() - t0 > TIME_BUDGET:
            return None

        md = _manhattan(gpos, ppos)
        if md < 2:
            return -100000.0 + depth * 100

        if depth <= 0:
            # Simple evaluation: maximize distance from Pacman
            bfs_d = pd_orig.get(gpos, md + 50)
            # Penalize dead ends
            if _cell_exits(gpos, ms) <= 1:
                bfs_d -= 50
            # Penalize recent positions
            if gpos in self._history[-6:]:
                bfs_d -= 20 * list(self._history[-6:]).count(gpos)
            return bfs_d * 10.0

        if is_ghost:
            value = float("-inf")
            moves = _legal(gpos, ms)
            # Sort: farther from Pacman first
            moves.sort(key=lambda m: pd_orig.get(
                _apply(gpos, m), _manhattan(_apply(gpos, m), ppos) + 50
            ), reverse=True)

            for m in moves:
                ng = _apply(gpos, m)
                if _manhattan(ng, ppos) < 2:
                    v = -100000.0
                else:
                    v = self._minimax(ms, ng, ppos, depth - 1, False,
                                      alpha, beta, pd_orig, t0)
                if v is None:
                    return None
                value = max(value, v)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value

        else:
            value = float("inf")
            for np_ in pacman_reach_2(ppos, ms):
                v = self._minimax(ms, gpos, np_, depth - 1, True,
                                  alpha, beta, pd_orig, t0)
                if v is None:
                    return None
                value = min(value, v)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    # ------------------------------------------------------------------
    # Anti-oscillation: filter out recently visited cells
    # ------------------------------------------------------------------
    def _filter_oscillation(self, me, candidates):
        """Remove moves that go to recently visited cells."""
        if len(self._history) < 2:
            return candidates

        recent = self._history[-HISTORY_BAN:] if len(self._history) >= HISTORY_BAN else self._history
        recent_set = set(recent)

        filtered = []
        for m in candidates:
            nxt = _apply(me, m)
            # Always allow if it's the only safe option
            if nxt not in recent_set:
                filtered.append(m)

        # If all moves are filtered (cornered), allow all to avoid being trapped
        if not filtered:
            return candidates

        return filtered

    # ------------------------------------------------------------------
    # One-time map analysis
    # ------------------------------------------------------------------
    def _analyze_map(self, ms):
        h, w = _shape(ms)
        self._open_cells = sum(
            1 for r in range(h) for c in range(w) if _cell(ms, r, c) != 1
        )
        for r in range(h):
            for c in range(w):
                if _cell(ms, r, c) == 1:
                    continue
                if _cell_exits((r, c), ms) <= 1:
                    self._dead_ends.add((r, c))

    # ------------------------------------------------------------------
    # Static topology precompute (lazy, once per map signature)
    # ------------------------------------------------------------------
    @staticmethod
    def _map_signature(ms):
        h, w = _shape(ms)
        rows = tuple(
            tuple(int(_cell(ms, r, c)) for c in range(w))
            for r in range(h)
        )
        return hash(rows)

    def _ensure_topology(self, ms):
        sig = self._map_signature(ms)
        if sig in self._topology_cache:
            self.topology_cache_hits += 1
            return self._topology_cache[sig]

        self.topology_cache_misses += 1
        h, w = _shape(ms)

        degree = {}
        junctions = set()
        corridors = set()
        dead_ends = set()
        open_cells = set()

        for r in range(h):
            for c in range(w):
                if _cell(ms, r, c) == 1:
                    continue
                pos = (r, c)
                exits = _cell_exits(pos, ms)
                degree[pos] = exits
                if exits >= 3:
                    junctions.add(pos)
                elif exits == 2:
                    corridors.add(pos)
                else:
                    dead_ends.add(pos)
                open_cells.add(pos)

        # Dead-end depth: walk from each dead-end toward nearest junction
        depth_map = {}
        for de in dead_ends:
            cur = de
            depth = 0
            visited = {de}
            while cur not in junctions:
                found_next = False
                for m in MOVE_ORDER:
                    nxt = _apply(cur, m)
                    if nxt not in visited and _valid(nxt, ms):
                        visited.add(nxt)
                        cur = nxt
                        depth += 1
                        found_next = True
                        break
                if not found_next or depth > 20:
                    break
            depth_map[de] = depth

        # Junction-distance map: BFS distance from each cell to nearest junction
        jd_map = {}
        q = deque()
        for j in junctions:
            jd_map[j] = 0
            q.append(j)
        while q:
            cur = q.popleft()
            for m in MOVE_ORDER:
                nxt = _apply(cur, m)
                if nxt not in jd_map and _valid(nxt, ms):
                    jd_map[nxt] = jd_map[cur] + 1
                    q.append(nxt)

        topo = {
            "degree": degree,
            "junctions": junctions,
            "corridors": corridors,
            "dead_ends": dead_ends,
            "open_cells": open_cells,
            "dead_end_depth": depth_map,
            "junction_distance": jd_map,
        }
        self._topology_cache[sig] = topo
        return topo

    # ------------------------------------------------------------------
    # Exploration fallback
    # ------------------------------------------------------------------
    def _explore(self, pos, candidates, ms):
        best = candidates[0]
        best_flood = 0
        for m in candidates:
            nxt = _apply(pos, m)
            seen = {nxt}
            q = deque([nxt])
            while q:
                cur = q.popleft()
                for mv in MOVE_ORDER:
                    neighbor = _apply(cur, mv)
                    if neighbor not in seen and _valid(neighbor, ms):
                        seen.add(neighbor)
                        q.append(neighbor)
                        if len(seen) > 50:
                            break
                if len(seen) > 50:
                    break
            if len(seen) > best_flood:
                best_flood = len(seen)
                best = m
        return best


# ===================================================================
# PacmanAgent — minimal placeholder (not primary deliverable)
# ===================================================================

class PacmanAgent(BasePacmanAgent):
    """Minimal Pacman for sandbox — Hider is the primary role for 24127192."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

    def step(self, map_state, my_position, enemy_position, step_number):
        return (Move.STAY, 1)
