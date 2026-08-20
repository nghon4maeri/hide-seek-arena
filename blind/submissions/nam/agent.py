"""agent.py — Champion submission for Blind Adversary (Lab 2).

PacmanAgent (Seeker)
  Visible ghost : Alpha-Beta minimax (depth 5, speed-2 model) +
                  guaranteed-capture shortcut
  Lost ghost    : Bayesian belief-guided chase (argmax prob - travel),
                  then systematic sweep biased toward junctions
GhostAgent (Hider)
  Visible pacman: Alpha-Beta minimax (panic/evasion), line-of-sight
                  breaking moves to shake off the pursuer
  Hidden pacman : Expectiminimax over the belief state + Monte Carlo
                  rollouts; prefers loop-rich core, camps when safe

Both roles know the full wall layout from step 1 (walls are always
visible in this game) and keep a Bayesian belief over the enemy with a
start-region prior (Ghost starts top band, Pacman bottom band).
"""

import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

from cch_core import (Grid, PacmanModel, astar_path, pack_path, safe_area_count,
                      CAPTURE_DISTANCE, INF, visible_cells_of, cross_visible)
from cch_topology import Topology
from cch_belief import EnemyBelief, ghost_reach_fn, pacman_reach_fn
from cch_search import (PacmanSearcher, GhostSearcher, expectiminimax_ghost,
                        mc_rollout_ghost, ghost_eval, TimeBudget, WIN)

MOVE = {(-1, 0): Move.UP, (1, 0): Move.DOWN, (0, -1): Move.LEFT,
        (0, 1): Move.RIGHT, (0, 0): Move.STAY}

STEP_TIME_LIMIT = 0.85
OBS_RADIUS = 5


def _cross_visible(p, c, radius=OBS_RADIUS):
    """True if p sees c along a straight cross-ray (walls block)."""
    pr, pc = p
    cr, cc = c
    if pr == cr:
        lo, hi = min(pc, cc), max(pc, cc)
        return hi - lo <= radius
    if pc == cc:
        lo, hi = min(pr, cr), max(pr, cr)
        return hi - lo <= radius
    return False


class _Base:
    def _init_static(self, obs):
        """Build the static grid once: wall cells are always visible."""
        arr = np.asarray(obs)
        static = (arr == 1).astype(np.int8)
        self.grid = Grid(static)
        self.topo = Topology(self.grid)

    def _visible_empty(self, obs):
        arr = np.asarray(obs)
        return [tuple(p) for p in np.argwhere(arr == 0)]

    def _top_band(self):
        return [c for c in self.grid.free if c[0] < self.grid.h * 0.4]

    def _bottom_band(self):
        return [c for c in self.grid.free if c[0] > self.grid.h * 0.6]

    def _time_ok(self):
        return (time.time() - self._step_t0) < STEP_TIME_LIMIT


# =====================================================================
# PACMAN — Seeker
# =====================================================================
class PacmanAgent(_Base, BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.pac_model = PacmanModel(self.speed)
        self.grid = None
        self.topo = None
        self.belief = None
        self.searcher = None
        self.visited = {}
        self.target = None
        self._step_t0 = 0.0

    def _ensure(self, obs):
        if self.grid is None:
            self._init_static(obs)
            self.belief = EnemyBelief(self.grid, ghost_reach_fn(self.grid))
            self.searcher = PacmanSearcher(self.grid, self.topo, self.pac_model)
            # Coverage sweep precomputation: which cells does each
            # position reveal, and which cells are known empty.
            self.vis_sets = {cell: visible_cells_of(self.grid, cell)
                             for cell in self.grid.free}
            self.exposure = {cell: len(s) for cell, s in self.vis_sets.items()}
            center = ((self.grid.h - 1) / 2.0, (self.grid.w - 1) / 2.0)
            self.edge_dist = {cell: abs(cell[0] - center[0]) + abs(cell[1] - center[1])
                              for cell in self.grid.free}
            self.cleared = set()
            self.ever_seen = False

    def _update_cleared(self, obs, me, step_number):
        for cell in self._visible_empty(obs):
            self.cleared.add(cell)
        # Ghost just escaped our sight: everything within ~12 steps of the
        # last seen spot is no longer guaranteed empty (ghost moved since).
        if (self.ever_seen and self.belief.steps_since_seen == 1
                and self.belief.last_seen is not None):
            for cell in list(self.cleared):
                if self.grid.dist_between(cell, self.belief.last_seen) <= 12:
                    self.cleared.discard(cell)

    def _act_to_move(self, action):
        delta, steps = action
        return (MOVE.get(delta, Move.STAY), max(1, steps))

    def _target_score(self, cell, me, step, chase_mode):
        if chase_mode:
            return self.belief.prob_at(cell) * 3000.0 - self.grid.dist_between(me, cell) * 8.0
        # Coverage sweep: greedy information gain, biased toward hideouts
        # (far from center, low exposure) where camping ghosts sit.
        gain = len(self.vis_sets[cell] - self.cleared)
        age = step - self.visited.get(cell, -9999)
        if age < 0:
            age = 0
        age = min(age, 300)
        frontier_adj = sum(1 for n in self.grid.neighbors(cell)
                           if n in self.cleared)
        hideout = max(0.0, 6.0 - self.exposure[cell])
        return (gain * 12.0
                + self.belief.prob_at(cell) * 2000.0
                + age * 0.5
                + self.edge_dist[cell] * 0.4
                + frontier_adj * 6.0
                + hideout * 3.0
                - self.grid.dist_between(me, cell) * 4.0)

    def _pick_target(self, me, step, chase_mode):
        """Persistent target with hysteresis to avoid oscillation."""
        g = self.grid
        if self.target is not None:
            t = self.target
            if t == me:
                self.target = None
            elif self.visited.get(t, -9999) >= step - 3:
                self.target = None
            elif g.walkable(t):
                return t
        best, best_sc = None, float("-inf")
        for cell in g.free:
            sc = self._target_score(cell, me, step, chase_mode)
            if sc > best_sc:
                best_sc, best = sc, cell
        if best is not None:
            if self.target is not None:
                old_sc = self._target_score(self.target, me, step, chase_mode)
                if best_sc <= old_sc + 60.0:
                    return self.target
            self.target = best
        return best

    def _guaranteed_capture(self, me, ghost):
        """If some action captures the ghost no matter where it moves."""
        for action in self.pac_model.legal_actions(self.grid, me):
            npac = self.pac_model.apply(self.grid, me, action[0], action[1])
            worst = -1
            for m in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                ng = (ghost[0] + m[0], ghost[1] + m[1])
                if self.grid.walkable(ng):
                    worst = max(worst, self.grid.dist_between(npac, ng))
                else:
                    worst = max(worst, self.grid.dist_between(npac, ghost))
            if worst < CAPTURE_DISTANCE:
                return action
        return None

    def _safe_fallback(self, me):
        if self.grid is None:
            return (Move.STAY, 1)
        moves = [(n[0] - me[0], n[1] - me[1]) for n in self.grid.neighbors(me)]
        if moves:
            return (MOVE.get(moves[0], Move.STAY), 1)
        return (Move.STAY, 1)

    def step(self, map_state, my_position, enemy_position, step_number):
        self._step_t0 = time.time()
        try:
            return self._step_impl(map_state, my_position, enemy_position, step_number)
        except Exception:
            return self._safe_fallback(tuple(my_position))

    def _step_impl(self, map_state, my_position, enemy_position, step_number):
        self._ensure(map_state)
        me = tuple(int(v) for v in my_position)
        self.visited[me] = step_number

        enemy = None
        if enemy_position is not None:
            enemy = tuple(int(v) for v in enemy_position)
            self.ever_seen = True
        self.belief.update(me, enemy, self._visible_empty(map_state), step_number,
                           flee_center=me)
        self._update_cleared(map_state, me, step_number)

        # --- Anti-stuck detection ---------------------------------------
        stuck = False
        if step_number > 6:
            recent = [c for c, s in self.visited.items() if step_number - s <= 6]
            if len(recent) <= 3:
                stuck = True

        # --- Enemy visible: adversarial search (iterative deepening) ----
        if enemy is not None:
            forced = self._guaranteed_capture(me, enemy)
            if forced is not None:
                return self._act_to_move(forced)
            best_action = None
            for depth in (3, 4, 5, 6, 7, 9):
                action, score = self.searcher.search(me, enemy, depth=depth,
                                                     time_limit=0.35)
                if action is not None and self.searcher.budget.ok():
                    best_action = action
                if score >= WIN or not self._time_ok():
                    break
            if best_action is None:
                best_action = ((0, 0), 0)
            return self._act_to_move(best_action)

        # --- Hidden: belief-guided chase, then systematic sweep ---------
        chase_mode = (self.belief.steps_since_seen <= 12
                      or self.belief.prob.max() >= 0.03)
        target = self._pick_target(me, step_number, chase_mode)

        if target is not None and target != me:
            path = astar_path(self.grid, me, target)
            if path:
                delta, steps = pack_path(path, me, self.speed)
                nxt = (me[0] + delta[0] * steps, me[1] + delta[1] * steps)
                if not (stuck and self.visited.get(nxt, -99) >= step_number - 6):
                    return (MOVE.get(delta, Move.STAY), max(1, steps))
        # Anti-stuck: move to the least recently visited neighbour.
        neigh = self.grid.neighbors(me)
        if neigh:
            best = min(neigh, key=lambda n: self.visited.get(n, -9999))
            return (MOVE.get((best[0] - me[0], best[1] - me[1]), Move.STAY), 1)
        return self._safe_fallback(me)


# =====================================================================
# GHOST — Hider
# =====================================================================
class GhostAgent(_Base, BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pac_model = PacmanModel(2)
        self.grid = None
        self.topo = None
        self.belief = None
        self.searcher = None
        self.hist = deque(maxlen=24)
        self.visited = {}
        self.visit_count = {}
        self.exposure = {}
        self.corners = set()
        self.adj_walls = {}
        self.hide_target = None
        self.target_gen = 0
        self.camp_turns = 0
        self._step_t0 = 0.0

    def _ensure(self, obs):
        if self.grid is None:
            self._init_static(obs)
            self.belief = EnemyBelief(self.grid,
                                      pacman_reach_fn(self.grid, self.pac_model),
                                      prior_cells=self._bottom_band())
            self.searcher = GhostSearcher(self.grid, self.topo, self.pac_model)
            for cell in self.grid.free:
                self.exposure[cell] = len(visible_cells_of(self.grid, cell))
                nb = self.grid.neighbors(cell)
                if len(nb) == 2:
                    v1 = (nb[0][0] - cell[0], nb[0][1] - cell[1])
                    v2 = (nb[1][0] - cell[0], nb[1][1] - cell[1])
                    if v1[0] * v2[0] + v1[1] * v2[1] == 0:
                        self.corners.add(cell)
                walls = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if not self.grid.walkable((cell[0] + dr, cell[1] + dc)):
                        walls += 1
                self.adj_walls[cell] = walls

    def _good_camp(self, cell):
        deg = self.grid.exits(cell)
        return (deg >= 2 and self.exposure.get(cell, 99) <= 5
                and cell in self.corners)

    def _pick_hide_target(self, me, step):
        """Team-6-style hide scoring: low exposure, corner-ish, near walls."""
        g = self.grid
        center = ((g.h - 1) / 2.0, (g.w - 1) / 2.0)
        from_last = None
        if self.belief.last_seen is not None and g.walkable(self.belief.last_seen):
            from_last = self.belief.last_seen
        candidates = []
        for cell in g.free:
            deg = g.exits(cell)
            if deg <= 1:
                sc = -900.0
            elif deg == 2:
                sc = 210.0 if cell in self.corners else 50.0
            elif deg == 3:
                sc = 260.0
            else:
                sc = 290.0
            sc -= 11.0 * self.exposure.get(cell, 99)
            sc += 6.0 * (abs(cell[0] - center[0]) + abs(cell[1] - center[1]))
            sc += 20.0 * min(self.adj_walls.get(cell, 0), 2)
            sc -= 32.0 * self.visit_count.get(cell, 0)
            if cell in self.hist:
                sc -= 120.0
            if from_last is not None:
                stale = max(0.2, 1.0 - self.belief.steps_since_seen / 20.0)
                sc += 20.0 * stale * g.dist_between(cell, from_last)
            candidates.append((sc, cell))
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top = min(5, len(candidates))
        idx = (step + self.target_gen * 3 + me[0] * 31 + me[1] * 17) % top
        return candidates[idx][1]

    def _hide_move(self, me, step):
        """Move toward the hide target (team-6 greedy scoring)."""
        if self.hide_target is None:
            self.hide_target = self._pick_hide_target(me, step)
            self.target_gen += 1
            self.camp_turns = 0
        target = self.hide_target
        if target is None or target == me:
            return (0, 0)
        best_move, best_score = (0, 0), float("-inf")
        for m in self._moves(me):
            nxt = (me[0] + m[0], me[1] + m[1])
            sc = -120.0 * self.grid.dist_between(nxt, target)
            sc += 20.0 * self.grid.exits(nxt)
            sc -= 4.0 * self.exposure.get(nxt, 99)
            sc -= 25.0 * self.visit_count.get(nxt, 0)
            if nxt in self.hist:
                sc -= 80.0
            if sc > best_score:
                best_score, best_move = sc, m
        return best_move

    def _moves(self, me):
        return [(n[0] - me[0], n[1] - me[1]) for n in self.grid.neighbors(me)]

    def _safe_fallback(self, me):
        if self.grid is None:
            return Move.STAY
        moves = self._moves(me)
        if moves:
            return MOVE.get(moves[0], Move.STAY)
        return Move.STAY

    def step(self, map_state, my_position, enemy_position, step_number):
        self._step_t0 = time.time()
        try:
            return self._step_impl(map_state, my_position, enemy_position, step_number)
        except Exception:
            return self._safe_fallback(tuple(my_position))

    def _hidden_from(self, cell, pac_positions):
        """True if cell is invisible to every pacman position."""
        for p in pac_positions:
            if _cross_visible(p, cell):
                return False
        return True

    def _step_impl(self, map_state, my_position, enemy_position, step_number):
        self._ensure(map_state)
        me = tuple(int(v) for v in my_position)
        self.hist.append(me)
        self.visited[me] = step_number

        enemy = None
        if enemy_position is not None:
            enemy = tuple(int(v) for v in enemy_position)
        self.belief.update(me, enemy, self._visible_empty(map_state), step_number)

        moves = self._moves(me)
        if not moves:
            return Move.STAY

        # --- Survival gate: leave dead ends / corridors ------------------
        if (me in self.topo.dead_ends or me in self.topo.corridors):
            escape = max(moves, key=lambda d: self.topo.weight_of(
                (me[0] + d[0], me[1] + d[1])))
            return MOVE.get(escape, Move.STAY)

        # --- Pacman visible: adversarial search + LOS breaking ----------
        if enemy is not None:
            d = self.grid.dist_between(me, enemy)
            eta = self.pac_model.eta(self.grid, enemy, me)

            if eta <= 1 or d < 3:
                depth = 6            # panic
            elif d < 9 or eta <= 4:
                depth = 5            # evasion
            else:
                depth = 4            # comfortable

            move = self.searcher.search(me, enemy, depth=depth)

            # Prefer shaking off line-of-sight when not in immediate danger.
            if d >= 3:
                pac_endpoints = [self.pac_model.apply(self.grid, enemy, a[0], a[1])
                                 for a in self.pac_model.legal_actions(self.grid, enemy)]
                hidden_moves = []
                for m in moves:
                    nxt = (me[0] + m[0], me[1] + m[1])
                    if (self.grid.dist_between(nxt, enemy) >= CAPTURE_DISTANCE
                            and self._hidden_from(nxt, pac_endpoints)):
                        hidden_moves.append(m)
                if hidden_moves:
                    move = max(hidden_moves, key=lambda m: (
                        self.pac_model.eta(self.grid, enemy, (me[0] + m[0], me[1] + m[1]))
                        if eta > 3 else
                        self.grid.dist_between((me[0] + m[0], me[1] + m[1]), enemy)))

            nxt = (me[0] + move[0], me[1] + move[1])
            if self.grid.dist_between(nxt, enemy) < CAPTURE_DISTANCE:
                safe = [m for m in moves
                        if self.grid.dist_between((me[0] + m[0], me[1] + m[1]),
                                                  enemy) >= CAPTURE_DISTANCE]
                if safe:
                    move = max(safe, key=lambda m: self.grid.dist_between(
                        (me[0] + m[0], me[1] + m[1]), enemy))
            return MOVE.get(move, Move.STAY)

        # --- Pacman hidden: hide-and-camp when safe, flee when near ------
        samples = self.belief.threat_samples(8)
        total_w = sum(w for _, w in samples) or 1.0
        pac_cells = [p for p, _ in samples]
        # A flat belief means we have no idea where the pacman is: treat
        # it as far (hide mode).  A concentrated belief gives a real eta.
        if self.belief.prob.max() < 0.05:
            min_eta = INF
        else:
            min_eta = min(self.pac_model.eta(self.grid, p, me) for p in pac_cells)
        self.visit_count[me] = self.visit_count.get(me, 0) + 1

        # HIDE MODE (inherited from team 6): camp hard at a good spot
        # once the pacman has been out of sight long enough.  Relocate
        # periodically so a converging seeker belief never pins us down.
        if min_eta > 4:
            if (self.belief.steps_since_seen > 6 and self._good_camp(me)
                    and self.camp_turns <= 14):
                self.hide_target = me
                self.camp_turns += 1
                return Move.STAY
            if self.camp_turns > 14:
                self.hide_target = None
                self.target_gen += 1
                self.camp_turns = 0
            move = self._hide_move(me, step_number)
            if move == (0, 0):
                # At a mediocre hide target: wait briefly, then re-pick.
                self.camp_turns += 1
                if (self._good_camp(me)
                        or self.camp_turns <= 6
                        or self.belief.steps_since_seen <= 6):
                    return Move.STAY
                self.hide_target = None
                self.camp_turns = 0
                move = self._hide_move(me, step_number)
            return MOVE.get(move, Move.STAY)

        # FLEE MODE: expectiminimax + Monte Carlo validation ---------------
        prev_cell = self.hist[-2] if len(self.hist) >= 2 else None
        candidates = moves + [(0, 0)]
        best_move, best_score = moves[0], float("-inf")
        for m in candidates:
            nxt = (me[0] + m[0], me[1] + m[1])
            expected = 0.0
            for pac_pos, w in samples:
                actions = self.pac_model.legal_actions(self.grid, pac_pos)
                worst = float("inf")
                for action in actions:
                    npac = self.pac_model.apply(self.grid, pac_pos, action[0], action[1])
                    val = ghost_eval(self.grid, self.topo, self.pac_model, nxt, npac)
                    if val < worst:
                        worst = val
                if not actions:
                    worst = ghost_eval(self.grid, self.topo, self.pac_model, nxt, pac_pos)
                expected += (w / total_w) * worst
            if self._hidden_from(nxt, pac_cells):
                expected += 600.0            # staying unseen is valuable
            if nxt in self.hist:
                expected -= 900.0
            if nxt == prev_cell:
                expected -= 3500.0           # hard reversal penalty
            if expected > best_score:
                best_score, best_move = expected, m
        move = best_move

        # Monte Carlo validation of the top candidates.
        if self._time_ok():
            scored = []
            for m in moves:
                nxt = (me[0] + m[0], me[1] + m[1])
                scored.append((ghost_eval(self.grid, self.topo, self.pac_model,
                                          nxt, self.belief.threat_center()), m))
            scored.sort(reverse=True)
            top_cand = [m for _, m in scored[:2]]
            if top_cand and move in top_cand:
                mc = mc_rollout_ghost(self.grid, self.topo, self.pac_model,
                                      me, top_cand, self.belief,
                                      rollouts=8, horizon=6)
                best_mc = max(mc, key=lambda mm: mc[mm])
                if best_mc != move and mc[best_mc] > mc[move] + 200.0:
                    move = best_mc

        return MOVE.get(move, Move.STAY)
