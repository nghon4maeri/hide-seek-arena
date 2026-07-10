"""PacmanAgent — Blind Seeker (Adapted from Lab 1).

Student: 24127561
Role:   Seek Agent Engineer

Lab 2 changes (Blind/Partial Observability):
- Maintains self.memory_map to accumulate observations across steps
- Handles enemy_position = None (enemy not visible)
- A* pathfinding runs on memory_map (optimistic: treats -1 as traversable)
- Frontier-based exploration when enemy is lost
- Belief-state tracking: probability distribution over the ghost's likely
  position while it is unseen, propagated step by step with a learned
  transition model
- Opponent modeling: learns the ghost's turning / persistence habits from
  observed moves and uses them both to propagate the belief state and to
  bias interception targets

v2 changes (faster capture):
- Predictive / time-matched interception while blind: instead of chasing
  the current most-likely ghost cell, projects the belief forward to the
  turn Pacman would actually arrive and targets where the ghost is likely
  to BE THEN (fixed-point refinement over ETA).
- Fixed an opponent-model learning bug: after a multi-turn blind gap, the
  raw delta between last-seen and reacquired position no longer
  corresponds to a single move, so it is no longer fed into the learned
  transition model (this was silently corrupting the model before).
"""

from __future__ import annotations

import sys
import math
import heapq
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

import numpy as np

# ===================================================================
# Constants
# ===================================================================
MOVE_ORDER = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)

CHOKE_SCOUT_DIST = 7
LOCK_DURATION = 3
A_STAR_PHASE_END = 10

# Belief-state tuning
BELIEF_ENTROPY_THRESHOLD = 3.5   # above this, belief is "too spread out" -> explore instead
BELIEF_MAX_SUPPORT = 200         # keep the belief dict sparse/bounded
BELIEF_PRUNE_EPS = 1e-4
STAY_WEIGHT = 0.15               # small prior weight for the ghost "staying put"
PERSISTENCE_WEIGHT = 2.0         # default bias towards continuing straight (used before
                                  # enough data has been learned for the opponent model)

# Predictive interception tuning
INTERCEPT_MAX_ITERS = 4          # fixed-point refinement steps for ETA <-> target
INTERCEPT_LOOKAHEAD_CAP = 12     # cap on how many turns of belief we project forward


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
    """Valid if within bounds and NOT a wall (1). -1 (unseen) is considered traversable (optimistic)."""
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1


def _known_empty(pos, ms):
    """Strict check: only cells known to be empty (0)."""
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) == 0


def _legal(pos, ms):
    return [m for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _cell_exits(pos, ms):
    return sum(1 for m in MOVE_ORDER if _valid(_apply(pos, m), ms))


def _dir_to_move(delta) -> Optional[Move]:
    """Map a unit (dr, dc) delta to the corresponding Move, if any."""
    for m in MOVE_ORDER:
        if m.value == delta:
            return m
    return None


# ===================================================================
# A* Search (on memory map)
# ===================================================================
def astar(ms, start, goal):
    if goal is None or not _valid(start, ms) or not _valid(goal, ms):
        return []
    if start == goal:
        return []

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
# PacmanAgent — Blind Seeker
# ===================================================================
class PacmanAgent(BasePacmanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))

        # === Blind mode state ===
        self.memory_map: Optional[np.ndarray] = None
        self.last_seen_enemy: Optional[Tuple[int, int]] = None

        # Ghost direction tracking (used for interception while visible)
        self._enemy_direction = None
        self._direction_streak = 0

        # Number of consecutive steps the ghost has been unseen. Used to
        # detect "reacquired after a gap" so we don't feed a multi-turn
        # delta into the single-move opponent model.
        self._blind_steps = 0

        # --- Belief-state tracking (Lab 2) ---
        # Sparse probability distribution over the ghost's current cell,
        # maintained only while the ghost is *not* visible.
        self.belief: Optional[Dict[Tuple[int, int], float]] = None

        # --- Opponent modeling (Lab 2) ---
        # Learns P(next_move | prev_move) from observed ghost movement so the
        # belief propagation and interception logic can favor the ghost's
        # actual habits (e.g. "keeps going straight at junctions" or
        # "reverses when cornered") instead of assuming uniform randomness.
        self.ghost_move_counts: Dict[Optional[Move], Dict[Move, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._prev_enemy_move: Optional[Move] = None

        # Path cache
        self._cached_target = None
        self._cached_path: List = []
        self._cached_my_pos = None

        # Feature gates
        self.enable_interception = True

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        # Update accumulated memory map
        self._update_memory(map_state)

        target = None

        if enemy_position is not None:
            enemy_position = tuple(enemy_position)
            reacquired_after_gap = self._blind_steps > 0
            self._update_ghost_tracking(enemy_position, reacquired_after_gap)
            self._blind_steps = 0
            # Ghost is visible again -> belief state is no longer needed
            # until we lose track of it once more.
            self.belief = None

            # Interception planning
            if self.enable_interception and self._direction_streak >= 2:
                inter_target = self._compute_interception_target(
                    self.memory_map, enemy_position, my_position
                )
                if inter_target is not None:
                    path_to_inter = astar(self.memory_map, my_position, inter_target)
                    path_to_direct = astar(self.memory_map, my_position, enemy_position)
                    dist_inter_to_ghost = (
                        abs(inter_target[0] - enemy_position[0])
                        + abs(inter_target[1] - enemy_position[1])
                    )
                    if path_to_inter and (
                        not path_to_direct
                        or len(path_to_inter) <= len(path_to_direct)
                        or dist_inter_to_ghost <= 2
                    ):
                        target = inter_target

            if target is None:
                target = enemy_position
            self.last_seen_enemy = enemy_position
        else:
            # Enemy not visible — fall back to belief-state prediction
            self._blind_steps += 1
            target = self._resolve_blind_target(my_position)
            if target is None or target == "__EXPLORE__":
                return self._explore(my_position)

        if my_position == target:
            return (Move.STAY, 1)

        # Path caching
        cache_valid = (
            self._cached_target == target
            and self._cached_my_pos == my_position
            and self._cached_path
        )
        if cache_valid:
            path = self._cached_path
        else:
            path = astar(self.memory_map, my_position, target)
            self._cached_target = target
            self._cached_path = path
            self._cached_my_pos = my_position

        if not path:
            return self._explore(my_position)

        result = self._path_to_move(path, my_position)
        if isinstance(result, tuple):
            consumed = result[1]
            mv = result[0]
        else:
            consumed = 1
            mv = result
        self._cached_path = path[consumed:]
        exp_pos = self._advance_position(my_position, mv, consumed)
        self._cached_my_pos = exp_pos
        return result

    # ------------------------------------------------------------------
    # Memory map
    # ------------------------------------------------------------------
    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = (map_state != -1)
        self.memory_map[visible_mask] = map_state[visible_mask]

    # ------------------------------------------------------------------
    # Ghost direction tracking + opponent-model learning
    # ------------------------------------------------------------------
    def _update_ghost_tracking(self, enemy_pos, reacquired_after_gap: bool = False):
        if self.last_seen_enemy is None:
            return

        if reacquired_after_gap:
            # We lost the ghost for one or more turns. The raw delta from
            # last_seen_enemy to enemy_pos may span several hidden moves,
            # so it can't be trusted as a single (prev_move -> move)
            # transition. Feeding it into the opponent model would corrupt
            # the learned probabilities. Reset direction tracking instead
            # and simply resume learning from the *next* visible step.
            self._enemy_direction = None
            self._direction_streak = 0
            self._prev_enemy_move = None
            return

        dr = enemy_pos[0] - self.last_seen_enemy[0]
        dc = enemy_pos[1] - self.last_seen_enemy[1]
        new_dir = (dr, dc)

        # --- Opponent modeling: learn prev_move -> next_move transitions ---
        # Only learn from single-cell steps; if the ghost is faster than one
        # cell/turn (or we skipped a frame) we simply don't have a clean
        # move to attribute, so we skip learning for that step.
        move = _dir_to_move(new_dir) if new_dir != (0, 0) else None
        if move is not None:
            self.ghost_move_counts[self._prev_enemy_move][move] += 1
            self._prev_enemy_move = move
        elif new_dir == (0, 0):
            # Ghost stayed in place
            self.ghost_move_counts[self._prev_enemy_move][None] += 1

        if new_dir == self._enemy_direction and (dr != 0 or dc != 0):
            self._direction_streak += 1
        else:
            self._enemy_direction = new_dir
            self._direction_streak = 1 if (dr != 0 or dc != 0) else 0

    def _learned_transition_probs(self, prev_move: Optional[Move]) -> Optional[Dict[Optional[Move], float]]:
        """Return P(next_move | prev_move) learned from observation, or None
        if we don't have enough data yet for this prev_move."""
        counts = self.ghost_move_counts.get(prev_move)
        if not counts:
            return None
        total = sum(counts.values())
        if total < 2:  # not enough evidence yet, let caller fall back to a heuristic
            return None
        return {m: c / total for m, c in counts.items()}

    # ------------------------------------------------------------------
    # Belief-state tracking (probability distribution over ghost position)
    # ------------------------------------------------------------------
    def _resolve_blind_target(self, my_position):
        """Update/propagate the belief state while the ghost is unseen and
        decide whether to chase a predicted cell or fall back to frontier
        exploration. Returns a target cell, None, or the sentinel
        "__EXPLORE__"."""
        if self.last_seen_enemy is None:
            return None

        if self.belief is None:
            # Just lost sight of the ghost: seed the belief at its last
            # known position.
            self.belief = {self.last_seen_enemy: 1.0}
        else:
            self._propagate_belief()

        if not self.belief:
            # Belief collapsed to nothing (e.g. fully boxed in) — give up
            # the chase and go exploring.
            self.last_seen_enemy = None
            return "__EXPLORE__"

        predicted = self._most_likely_ghost_pos()
        entropy = self._belief_entropy()

        if my_position == self.last_seen_enemy and (
            predicted is None or entropy > BELIEF_ENTROPY_THRESHOLD
        ):
            # We reached the spot the ghost was last seen at and still have
            # no confident prediction — stop chasing a ghost, go explore.
            self.last_seen_enemy = None
            self.belief = None
            return "__EXPLORE__"

        if entropy > BELIEF_ENTROPY_THRESHOLD:
            # Belief too spread out to commit to a single cell: explore,
            # but bias the frontier choice towards the most likely region.
            self._explore_belief_hint = predicted
            return "__EXPLORE__"

        # Predictive / time-matched interception: don't chase where the
        # ghost is NOW, chase where it's likely to be by the time we
        # actually arrive.
        intercept = self._predictive_intercept_target(my_position)
        return intercept if intercept is not None else predicted

    def _propagate_belief_dict(self, belief: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """Pure one-step belief propagation. Does not mutate self.belief —
        used both for the live belief update and for projecting the belief
        forward hypothetically (predictive interception)."""
        if not belief:
            return {}
        ms = self.memory_map
        new_belief: Dict[Tuple[int, int], float] = defaultdict(float)

        learned = self._learned_transition_probs(self._prev_enemy_move)

        for pos, prob in belief.items():
            legal = _legal(pos, ms)
            if not legal:
                # Ghost can't move from here (fully walled/unseen) — mass stays.
                new_belief[pos] += prob
                continue

            weights: Dict[Move, float] = {}
            for m in legal:
                if learned is not None:
                    weights[m] = learned.get(m, 0.05)
                else:
                    # Fallback heuristic: prefer continuing the last known
                    # direction (persistence), spread the rest uniformly.
                    if self._enemy_direction is not None and m.value == self._enemy_direction:
                        weights[m] = PERSISTENCE_WEIGHT
                    else:
                        weights[m] = 1.0

            stay_w = (learned.get(None, STAY_WEIGHT) if learned is not None else STAY_WEIGHT)
            total_w = sum(weights.values()) + stay_w
            if total_w <= 0:
                total_w = 1.0

            new_belief[pos] += prob * (stay_w / total_w)
            for m, w in weights.items():
                nxt = _apply(pos, m)
                new_belief[nxt] += prob * (w / total_w)

        total = sum(new_belief.values())
        if total <= 0:
            return {}

        pruned = {p: v / total for p, v in new_belief.items() if v / total > BELIEF_PRUNE_EPS}
        if not pruned:
            pruned = {max(new_belief, key=new_belief.get): 1.0}

        if len(pruned) > BELIEF_MAX_SUPPORT:
            top = sorted(pruned.items(), key=lambda kv: -kv[1])[:BELIEF_MAX_SUPPORT]
            s = sum(v for _, v in top)
            pruned = {p: v / s for p, v in top}

        return pruned

    def _propagate_belief(self):
        self.belief = self._propagate_belief_dict(self.belief)

    def _belief_entropy(self) -> float:
        if not self.belief:
            return 0.0
        return -sum(p * math.log(p + 1e-12) for p in self.belief.values())

    def _most_likely_ghost_pos(self) -> Optional[Tuple[int, int]]:
        if not self.belief:
            return None
        return max(self.belief.items(), key=lambda kv: kv[1])[0]

    def _predictive_intercept_target(self, my_position, max_iters: int = INTERCEPT_MAX_ITERS):
        """Time-matched interception under uncertainty.

        Chasing `_most_likely_ghost_pos()` directly means we're always
        aiming at where the ghost *was* probabilistically, not where it
        will be once we actually get there — for a spread-out, moving
        belief this costs extra steps every turn.

        Instead: guess a target, compute how many turns it'll take Pacman
        to reach it (ETA), project the belief forward that many turns using
        the learned opponent model, then re-pick the best target from the
        *projected* distribution (weighted by probability and by how cheap
        it is to reach). Repeat a few times until the target stabilizes
        (fixed point) or the iteration budget runs out.
        """
        if not self.belief:
            return None

        target = self._most_likely_ghost_pos()
        if target is None:
            return None

        for _ in range(max_iters):
            path = astar(self.memory_map, my_position, target)
            eta = max(1, math.ceil(len(path) / self.pacman_speed)) if path else 1
            eta = min(eta, INTERCEPT_LOOKAHEAD_CAP)

            projected = dict(self.belief)
            for _ in range(eta):
                projected = self._propagate_belief_dict(projected)
                if not projected:
                    break
            if not projected:
                break

            def score(item):
                cell, p = item
                d = _manhattan(my_position, cell)
                return p / (1 + d / self.pacman_speed)

            new_target = max(projected.items(), key=score)[0]
            if new_target == target:
                break
            target = new_target

        return target

    # ------------------------------------------------------------------
    # Interception target (ghost currently visible)
    # ------------------------------------------------------------------
    def _compute_interception_target(self, ms, enemy_pos, my_pos):
        dr, dc = self._enemy_direction
        cur_row, cur_col = enemy_pos

        # Opponent modeling: if we've learned this ghost tends to turn
        # rather than go straight, don't project as far ahead.
        current_move = _dir_to_move((dr, dc))
        learned = self._learned_transition_probs(current_move)
        persistence = learned.get(current_move, 0.5) if learned else 0.6
        max_lookahead = 4 if persistence >= 0.5 else 2

        best = None
        for i in range(1, max_lookahead + 1):
            nr, nc = cur_row + dr * i, cur_col + dc * i
            h, w = _shape(ms)
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if _cell(ms, nr, nc) == 1:
                break
            nxt = (nr, nc)
            exits = _cell_exits(nxt, ms)
            if exits >= 3:
                return nxt
            if exits == 2 and best is None:
                best = nxt
        return best

    # ------------------------------------------------------------------
    # Convert A* path (list of Move) to (Move, steps)
    # ------------------------------------------------------------------
    def _path_to_move(self, path, my_position):
        first_move = path[0]
        move = first_move if isinstance(first_move, Move) else Move.STAY
        steps = 1
        for m in path[1:]:
            if m == move and steps < self.pacman_speed:
                steps += 1
            else:
                break
        return (move, steps)

    def _advance_position(self, pos, move, steps):
        cur = pos
        for _ in range(steps):
            nxt = _apply(cur, move)
            if not _valid(nxt, self.memory_map):
                break
            cur = nxt
        return cur

    # ------------------------------------------------------------------
    # Exploration (frontier-based, with optional belief bias)
    # ------------------------------------------------------------------
    def _explore(self, my_position, belief_hint=None):
        ms = self.memory_map
        if ms is None:
            moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            random.shuffle(moves)
            return (moves[0], 1)

        if belief_hint is None:
            belief_hint = getattr(self, "_explore_belief_hint", None)
        self._explore_belief_hint = None

        # Find nearest frontier cell (known-empty cell adjacent to unknown
        # territory), biasing towards the belief-state prediction when we
        # have one, so exploration still leans toward "where the ghost
        # probably went" instead of ignoring everything we've learned.
        h, w = ms.shape
        target = None
        best_score = float("inf")
        for r in range(h):
            for c in range(w):
                if ms[r, c] != 0:
                    continue
                has_unknown = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and ms[nr, nc] == -1:
                        has_unknown = True
                        break
                if has_unknown:
                    d = abs(r - my_position[0]) + abs(c - my_position[1])
                    score = d
                    if belief_hint is not None:
                        d_hint = abs(r - belief_hint[0]) + abs(c - belief_hint[1])
                        score = 0.5 * d + 0.5 * d_hint
                    if score < best_score:
                        best_score = score
                        target = (r, c)

        if target:
            path = astar(ms, my_position, target)
            if path:
                return self._path_to_move(path, my_position)

        moves = _legal(my_position, ms)
        if moves:
            return (random.choice(moves), 1)
        return (Move.STAY, 1)


# ===================================================================
# GhostAgent — placeholder (not primary deliverable for 24127561)
# ===================================================================
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_map = None
        self.last_seen_enemy = None

    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = (map_state != -1)
        self.memory_map[visible_mask] = map_state[visible_mask]

    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_memory(map_state)
        if enemy_position is not None:
            self.last_seen_enemy = tuple(enemy_position)
        # Ghost returns a bare Move (no tuple), as required.
        return Move.STAY