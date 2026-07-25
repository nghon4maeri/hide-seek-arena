"""PacmanAgent — Blind Seeker + PPO (v3).

Student: 24127561
Role:   Seek Agent Engineer

This file = your v2 agent, unchanged in every branch, PLUS a PPO hook that
only activates in the blind-chase branch (ghost currently unseen) and only
if a trained checkpoint is found. If PPO is unavailable or untrained, the
agent falls back to the exact same belief-state + A* + predictive
interception logic as before — nothing about the visible-ghost behavior
changes.

Why PPO only for the blind branch:
  - When the ghost is visible, direction-tracking + interception already
    computes near-optimal cut-off points geometrically; there's little for
    a learned policy to gain there and much more risk of it being worse
    than exact search.
  - When the ghost is blind, "which cell to chase" is a proxy for "which
    action minimizes expected time-to-capture under uncertainty" — that's
    exactly the kind of sequential decision problem under a stochastic,
    partially-known opponent model that PPO is suited for, and it can
    learn to trade off "commit to the most likely branch" vs "hedge toward
    a chokepoint that covers several branches" better than a fixed
    heuristic target + A*.

Usage:
  agent = PacmanAgent(pacman_speed=2, ppo_checkpoint="ppo_blind.pt")
  If ppo_checkpoint is None or the file doesn't exist, behaves like v2.
"""

from __future__ import annotations

import os
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

try:
    from ppo_ghost_hunter import ActorCritic, encode_state, TORCH_AVAILABLE
    if TORCH_AVAILABLE:
        import torch
except ImportError:
    TORCH_AVAILABLE = False


# ===================================================================
# Constants (unchanged from v2)
# ===================================================================
MOVE_ORDER = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
ACTION_TO_MOVE = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY)

CHOKE_SCOUT_DIST = 7
LOCK_DURATION = 3
A_STAR_PHASE_END = 10

BELIEF_ENTROPY_THRESHOLD = 3.5
BELIEF_MAX_SUPPORT = 200
BELIEF_PRUNE_EPS = 1e-4
STAY_WEIGHT = 0.15
PERSISTENCE_WEIGHT = 2.0

INTERCEPT_MAX_ITERS = 4
INTERCEPT_LOOKAHEAD_CAP = 12

MAP_MAX_DIM_DEFAULT = 30  # used to normalize belief-centroid offset for the PPO state


# ===================================================================
# Grid utilities (unchanged from v2)
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


def _known_empty(pos, ms):
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
    for m in MOVE_ORDER:
        if m.value == delta:
            return m
    return None


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
# PacmanAgent — Blind Seeker + PPO
# ===================================================================
class PacmanAgent(BasePacmanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.map_max_dim = int(kwargs.get("map_max_dim", MAP_MAX_DIM_DEFAULT))

        self.memory_map: Optional[np.ndarray] = None
        self.last_seen_enemy: Optional[Tuple[int, int]] = None

        self._enemy_direction = None
        self._direction_streak = 0
        self._blind_steps = 0

        self.belief: Optional[Dict[Tuple[int, int], float]] = None

        self.ghost_move_counts: Dict[Optional[Move], Dict[Move, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._prev_enemy_move: Optional[Move] = None

        self._cached_target = None
        self._cached_path: List = []
        self._cached_my_pos = None

        self.enable_interception = True

        # --- PPO hook ---
        self.ppo_net = None
        ckpt = kwargs.get("ppo_checkpoint", str(Path(__file__).parent / "ppo_blind.pt"))
        if ckpt and TORCH_AVAILABLE and os.path.exists(ckpt):
            self.ppo_net = ActorCritic()
            self.ppo_net.load_state_dict(torch.load(ckpt, map_location="cpu"))
            self.ppo_net.eval()

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_memory(map_state)

        target = None

        if enemy_position is not None:
            enemy_position = tuple(enemy_position)
            reacquired_after_gap = self._blind_steps > 0
            self._update_ghost_tracking(enemy_position, reacquired_after_gap)
            self._blind_steps = 0
            self.belief = None

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
            # === Blind: update belief first (needed by both PPO & fallback) ===
            self._blind_steps += 1
            self._update_belief_only()

            if self.belief and self.ppo_net is not None:
                move = self._ppo_act(my_position)
                if move is not None:
                    self._cached_path = []  # invalidate A* cache, PPO drives now
                    return (move, 1)
                # else fall through to heuristic target-based path below

            target = self._resolve_blind_target(my_position)
            if target is None or target == "__EXPLORE__":
                return self._explore(my_position)

        if my_position == target:
            return (Move.STAY, 1)

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
    # PPO action selection (blind phase only)
    # ------------------------------------------------------------------
    def _ppo_act(self, my_position) -> Optional[Move]:
        """Ask the trained policy directly for a Move, given the current
        belief + local map. Returns None if PPO can't produce a legal move
        (shouldn't normally happen given action masking), so caller can
        fall back to the heuristic path."""
        state = encode_state(
            self.memory_map, self.belief, my_position, self.pacman_speed, self.map_max_dim
        )
        mask = np.zeros(len(ACTION_TO_MOVE), dtype=bool)
        for i, m in enumerate(ACTION_TO_MOVE[:-1]):
            mask[i] = _valid(_apply(my_position, m), self.memory_map)
        mask[-1] = True  # STAY always legal

        action, _log_prob, _value = self.ppo_net.act(state, action_mask=mask, deterministic=True)
        move = ACTION_TO_MOVE[action]
        if move != Move.STAY and not _valid(_apply(my_position, move), self.memory_map):
            return None
        return move

    def _update_belief_only(self):
        """Belief propagation extracted so it can run before the PPO branch
        as well as before the heuristic fallback branch."""
        if self.last_seen_enemy is None:
            self.belief = None
            return
        if self.belief is None:
            self.belief = {self.last_seen_enemy: 1.0}
        else:
            self.belief = self._propagate_belief_dict(self.belief)

    # ------------------------------------------------------------------
    # Memory map
    # ------------------------------------------------------------------
    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = (map_state != -1)
        self.memory_map[visible_mask] = map_state[visible_mask]

    # ------------------------------------------------------------------
    # Ghost direction tracking + opponent-model learning (unchanged)
    # ------------------------------------------------------------------
    def _update_ghost_tracking(self, enemy_pos, reacquired_after_gap: bool = False):
        if self.last_seen_enemy is None:
            return
        if reacquired_after_gap:
            self._enemy_direction = None
            self._direction_streak = 0
            self._prev_enemy_move = None
            return
        dr = enemy_pos[0] - self.last_seen_enemy[0]
        dc = enemy_pos[1] - self.last_seen_enemy[1]
        new_dir = (dr, dc)
        move = _dir_to_move(new_dir) if new_dir != (0, 0) else None
        if move is not None:
            self.ghost_move_counts[self._prev_enemy_move][move] += 1
            self._prev_enemy_move = move
        elif new_dir == (0, 0):
            self.ghost_move_counts[self._prev_enemy_move][None] += 1
        if new_dir == self._enemy_direction and (dr != 0 or dc != 0):
            self._direction_streak += 1
        else:
            self._enemy_direction = new_dir
            self._direction_streak = 1 if (dr != 0 or dc != 0) else 0

    def _learned_transition_probs(self, prev_move: Optional[Move]) -> Optional[Dict[Optional[Move], float]]:
        counts = self.ghost_move_counts.get(prev_move)
        if not counts:
            return None
        total = sum(counts.values())
        if total < 2:
            return None
        return {m: c / total for m, c in counts.items()}

    # ------------------------------------------------------------------
    # Belief-state tracking (unchanged, used as fallback + by PPO state)
    # ------------------------------------------------------------------
    def _resolve_blind_target(self, my_position):
        if self.last_seen_enemy is None:
            return None
        if not self.belief:
            self.last_seen_enemy = None
            return "__EXPLORE__"

        predicted = self._most_likely_ghost_pos()
        entropy = self._belief_entropy()

        if my_position == self.last_seen_enemy and (
            predicted is None or entropy > BELIEF_ENTROPY_THRESHOLD
        ):
            self.last_seen_enemy = None
            self.belief = None
            return "__EXPLORE__"

        if entropy > BELIEF_ENTROPY_THRESHOLD:
            self._explore_belief_hint = predicted
            return "__EXPLORE__"

        intercept = self._predictive_intercept_target(my_position)
        return intercept if intercept is not None else predicted

    def _propagate_belief_dict(self, belief: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        if not belief:
            return {}
        ms = self.memory_map
        new_belief: Dict[Tuple[int, int], float] = defaultdict(float)
        learned = self._learned_transition_probs(self._prev_enemy_move)
        for pos, prob in belief.items():
            legal = _legal(pos, ms)
            if not legal:
                new_belief[pos] += prob
                continue
            weights: Dict[Move, float] = {}
            for m in legal:
                if learned is not None:
                    weights[m] = learned.get(m, 0.05)
                else:
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

    def _belief_entropy(self) -> float:
        if not self.belief:
            return 0.0
        return -sum(p * math.log(p + 1e-12) for p in self.belief.values())

    def _most_likely_ghost_pos(self) -> Optional[Tuple[int, int]]:
        if not self.belief:
            return None
        return max(self.belief.items(), key=lambda kv: kv[1])[0]

    def _predictive_intercept_target(self, my_position, max_iters: int = INTERCEPT_MAX_ITERS):
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
    # Interception target (ghost currently visible) — unchanged
    # ------------------------------------------------------------------
    def _compute_interception_target(self, ms, enemy_pos, my_pos):
        dr, dc = self._enemy_direction
        cur_row, cur_col = enemy_pos
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
    def _explore(self, my_position, belief_hint=None):
        ms = self.memory_map
        if ms is None:
            moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            random.shuffle(moves)
            return (moves[0], 1)
        if belief_hint is None:
            belief_hint = getattr(self, "_explore_belief_hint", None)
        self._explore_belief_hint = None
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
        return Move.STAY