"""agent.py — Runtime entrypoint for Blind Adversary (Lab 2).

Loaded by arena.py. Implements PacmanAgent (3-Tier Lexicographic)
and GhostAgent (Dynamic Mode Selection) using modular components.
"""

import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

from topology_analyzer import (
    STATIC_FULL_MAP, MOVE_ORDER, DIRS,
    _shape, _cell, _valid, _apply, _manhattan, _cell_exits, _legal,
    _dir_from_delta, _random_valid_move,
    bfs_dist, astar, TopologyAnalyzer,
)
from state_trackers import SpatialHeatmap, BeliefStateTracker
from tactical_engines import IntentTracker, TrapEvaluator, DynamicModeSelector
from network_architect import RecurrentActorCritic, INPUT_CHANNELS, POS_DIM, HIDDEN_SIZE

MODEL_DIR = Path(__file__).resolve().parent
PACMAN_MODEL_PATH = MODEL_DIR / "pacman_model.pth"
GHOST_MODEL_PATH  = MODEL_DIR / "ghost_model.pth"
GHOST_MLP_PATH    = MODEL_DIR / "ghost_mlp.pth"

RL_TIMEOUT = 0.75
STEP_TIME_LIMIT = 0.90

PACMAN_ACTIONS = [
    Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT,
    Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT,
    Move.STAY,
]
PACMAN_STEPS = [1, 1, 1, 1, 2, 2, 2, 2, 1]
GHOST_ACTIONS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]


# ===================================================================
# Observation builder — 6-channel tensor
# ===================================================================
def _get_visible_cells(pos, map_state, H, W):
    visible = {pos}
    r, c = pos
    for dr, dc in DIRS:
        for dist in range(1, 6):
            nr, nc = r + dr * dist, c + dc * dist
            if not (0 <= nr < H and 0 <= nc < W):
                break
            visible.add((nr, nc))
            if map_state[nr, nc] == 1:
                break
    return visible


def _build_obs_tensor(map_state, my_position, enemy_position, model,
                      belief_tracker, topo_analyzer,
                      step_number=1, max_steps=200):
    """Build 6-channel + 7-dim position tensor for RecurrentActorCritic."""
    H, W = map_state.shape
    visible = _get_visible_cells(my_position, map_state, H, W)

    ch_wall   = np.zeros((H, W), dtype=np.float32)
    ch_seen   = np.zeros((H, W), dtype=np.float32)
    ch_fog    = np.zeros((H, W), dtype=np.float32)
    ch_enemy  = np.zeros((H, W), dtype=np.float32)
    ch_belief = np.zeros((H, W), dtype=np.float32)
    ch_topo   = np.zeros((H, W), dtype=np.float32)

    for r in range(H):
        for c in range(W):
            if map_state[r, c] == 1:
                ch_wall[r, c] = 1.0
            elif (r, c) in visible:
                ch_seen[r, c] = 1.0
            else:
                ch_fog[r, c] = 1.0

    vflag = 0.0
    er_n = ec_n = 0.0
    if enemy_position is not None:
        er, ec = enemy_position
        if 0 <= er < H and 0 <= ec < W:
            ch_enemy[er, ec] = 1.0
            vflag = 1.0
            er_n = float(er) / H
            ec_n = float(ec) / W

    if belief_tracker is not None:
        b_total = belief_tracker.belief.sum()
        if b_total > 0:
            ch_belief = (belief_tracker.belief / b_total).astype(np.float32)

    if topo_analyzer is not None and topo_analyzer.ready:
        for r in range(H):
            for c in range(W):
                ch_topo[r, c] = topo_analyzer._get_topological_weight((r, c), map_state)
        mx = ch_topo.max()
        if mx > 0:
            ch_topo /= mx

    threat_level = 0.0
    if belief_tracker is not None:
        tc = belief_tracker.threat_center()
        threat_level = 1.0 - min(1.0, _manhattan(my_position, tc) / max(H, W))

    game_progress = float(step_number) / float(max_steps)

    obs_img = np.stack([ch_wall, ch_seen, ch_fog, ch_enemy,
                        ch_belief, ch_topo], axis=0)
    pos_vec = np.array([
        my_position[0] / H, my_position[1] / W,
        er_n, ec_n, vflag,
        threat_level, game_progress,
    ], dtype=np.float32)

    return (torch.from_numpy(obs_img).unsqueeze(0),
            torch.from_numpy(pos_vec).unsqueeze(0))


def _reset_lstm_state():
    return (torch.zeros(1, 1, HIDDEN_SIZE), torch.zeros(1, 1, HIDDEN_SIZE))


def _get_fov_channels(map_state, my_position):
    H, W = map_state.shape
    visible = _get_visible_cells(my_position, map_state, H, W)
    ch_wall = np.zeros((H, W), dtype=np.int32)
    ch_seen = np.zeros((H, W), dtype=np.int32)
    ch_fog  = np.zeros((H, W), dtype=np.int32)
    for r in range(H):
        for c in range(W):
            if map_state[r, c] == 1:
                ch_wall[r, c] = 1
            elif (r, c) in visible:
                ch_seen[r, c] = 1
            else:
                ch_fog[r, c] = 1
    return ch_wall, ch_seen, ch_fog


# ===================================================================
# Ghost MLP (lightweight move predictor)
# ===================================================================
class GhostMoveMLP(nn.Module):
    def __init__(self, input_dim=30, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 5),
        )

    def forward(self, x):
        return self.net(x)


# ===================================================================
# PACMAN — 3-Tier Lexicographic Architecture
# ===================================================================
class PacmanAgent(BasePacmanAgent):
    """Blind Seeker with 3-Tier Lexicographic decision pipeline.

    Tier 1: Safety & Trap Filter (overrides all below)
    Tier 2: Dynamic Interception & Intent Prediction
    Tier 3: Strategic Exploration & DRQN Fallback
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.device = torch.device('cpu')

        self.model = RecurrentActorCritic(9)
        if PACMAN_MODEL_PATH.exists():
            state = torch.load(str(PACMAN_MODEL_PATH), map_location=self.device,
                               weights_only=True)
            self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.hidden_state = _reset_lstm_state()

        self.memory_map: Optional[np.ndarray] = None
        self.last_seen_enemy: Optional[Tuple] = None
        self._last_seen_step: int = 0
        self._topo = TopologyAnalyzer()
        self._heatmap: Optional[SpatialHeatmap] = None
        self._belief: Optional[BeliefStateTracker] = None
        self._intent_tracker = IntentTracker(max_history=8)
        self._trap_evaluator = TrapEvaluator()

        self._visit_count: Dict[Tuple, int] = {}
        self._oscillation_moves: List[Tuple] = []

    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
            H, W = map_state.shape
            self._heatmap = SpatialHeatmap(H, W)
            self._belief = BeliefStateTracker(H, W)
        visible_mask = (map_state != -1)
        self.memory_map[visible_mask] = map_state[visible_mask]
        visible_cells = set(zip(*np.where(visible_mask)))
        self._heatmap.update(visible_cells)

    def _ensure_topo(self):
        if not self._topo.ready and self.memory_map is not None:
            known = (self.memory_map != -1)
            total = self.memory_map.size
            if known.sum() > total * 0.3:
                tmp = self.memory_map.copy()
                tmp[tmp == -1] = 0
                self._topo.analyze(tmp)

    def _ensure_belief(self, me):
        if self._belief is not None and self.memory_map is not None:
            self._belief.update(me, self.last_seen_enemy, self.memory_map)

    def _path_to_action(self, path, my_pos):
        if not path:
            return (Move.STAY, 1)
        move_delta = path[0]
        move = Move(move_delta)
        steps = 1
        cur = _apply(my_pos, move_delta)
        for m in path[1:]:
            if m == move_delta and steps < self.pacman_speed:
                steps += 1
                cur = _apply(cur, m)
            else:
                break
        return (move, steps) if steps > 1 else move

    def _time_ok(self, started_at: float) -> bool:
        return (time.time() - started_at) < STEP_TIME_LIMIT

    def _break_stuck(self, map_state, me):
        self.hidden_state = _reset_lstm_state()
        self._visit_count.clear()
        self._oscillation_moves.clear()
        return Move(_random_valid_move(map_state, me))

    # ── TIER 1: Safety & Trap Filter ──────────────────────────────
    def _tier1_safety_filter(self, me, ms):
        exits = _cell_exits(me, ms)
        moves = _legal(me, ms)
        if not moves:
            return Move.STAY

        if exits <= 1:
            return Move(max(moves, key=lambda m: (
                self._topo._get_topological_weight(_apply(me, m), ms)
                - (1000 if _apply(me, m) in self._visit_count else 0)
            )))

        safe_moves = [m for m in moves
                      if not self._trap_evaluator.is_dead_end_trap(
                          _apply(me, m), self._topo, ms, lookahead=6)]
        if not safe_moves:
            return Move(max(moves, key=lambda m:
                self._topo._get_topological_weight(_apply(me, m), ms)))

        self._visit_count[me] = self._visit_count.get(me, 0) + 1
        self._oscillation_moves.append(me)
        if len(self._oscillation_moves) > 8:
            self._oscillation_moves = self._oscillation_moves[-8:]

        if self._visit_count.get(me, 0) >= 4:
            return self._break_stuck(ms, me)
        if len(self._oscillation_moves) >= 4:
            last4 = self._oscillation_moves[-4:]
            if len(set(last4)) <= 2:
                return self._break_stuck(ms, me)

        return None

    # ── TIER 2: Dynamic Interception & Intent Prediction ──────────
    def _tier2_pursuit(self, me, enemy, ms, step_num):
        if enemy is not None and self._intent_tracker.velocity_streak >= 2:
            bottleneck = self._intent_tracker.predict_bottleneck(
                self._topo, ms, enemy, depth=5)
            if bottleneck:
                path_to_bn = astar(ms, me, bottleneck)
                enemy_to_bn = _manhattan(enemy, bottleneck)
                if path_to_bn and len(path_to_bn) <= enemy_to_bn + 1:
                    return self._path_to_action(path_to_bn, me)

        if enemy is not None:
            path = astar(ms, me, enemy)
            if path:
                valid = True
                cur = me
                for step_move in path[:6]:
                    nxt = _apply(cur, step_move)
                    if self._trap_evaluator.is_dead_end_trap(
                            nxt, self._topo, ms, lookahead=4):
                        valid = False
                        break
                    cur = nxt
                if valid:
                    return self._path_to_action(path, me)

        steps_since_seen = step_num - self._last_seen_step if self._last_seen_step > 0 else 999
        if steps_since_seen <= 15 and self.last_seen_enemy is not None:
            if self._belief is not None:
                threat_cells = self._belief.highest_threat_cells(top_k=5)
                best_target = None
                best_score = float("-inf")
                for cell, prob in threat_cells:
                    d = _manhattan(me, cell)
                    score = prob * 1000 - d
                    if cell in self._topo.junctions:
                        score += 200
                    if score > best_score:
                        best_score = score
                        best_target = cell
                if best_target:
                    path = astar(ms, me, best_target)
                    if path:
                        return self._path_to_action(path, me)

            path = astar(ms, me, self.last_seen_enemy)
            if path:
                return self._path_to_action(path, me)
            if me == self.last_seen_enemy:
                self.last_seen_enemy = None

        return None

    # ── TIER 3: Strategic Exploration & DRQN Fallback ─────────────
    def _tier3_explore(self, me, ms):
        heat_target = self._heatmap.best_target(
            ms, me, self.last_seen_enemy, self._topo)
        if heat_target:
            path = astar(ms, me, heat_target)
            if path:
                return self._path_to_action(path, me)

        frontier = self._frontier_topo_search(ms, me)
        if frontier:
            path = astar(ms, me, frontier)
            if path:
                return self._path_to_action(path, me)
        return None

    def _frontier_topo_search(self, ms, me):
        H, W = ms.shape
        best, best_score = None, float("-inf")
        for r in range(H):
            for c in range(W):
                if ms[r, c] != 0:
                    continue
                has_fog = any(0 <= r + dr < H and 0 <= c + dc < W
                              and ms[r + dr, c + dc] == -1
                              for dr, dc in DIRS)
                if not has_fog:
                    continue
                d = _manhattan((r, c), me)
                topo_w = self._topo._get_topological_weight((r, c), ms)
                belief_bonus = 0.0
                if self._belief is not None:
                    belief_bonus = float(self._belief.belief[r, c]) * 500
                score = topo_w * 100 - d + belief_bonus
                if score > best_score:
                    best_score = score
                    best = (r, c)
        return best

    def _tier3_rl_fallback(self, map_state, me, t0, step_num, enemy=None):
        obs_t, pos_t = _build_obs_tensor(
            map_state, me, enemy, self.model,
            self._belief, self._topo,
            step_number=step_num)
        with torch.no_grad():
            action, _, _, _, self.hidden_state = \
                self.model.get_action_and_value(
                    obs_t, pos_t, self.hidden_state, deterministic=True)
        if time.time() - t0 > RL_TIMEOUT:
            return Move.STAY
        idx = action.item()
        move = PACMAN_ACTIONS[idx]
        steps = min(PACMAN_STEPS[idx], self.pacman_speed)
        return (move, steps) if steps > 1 else move

    # ── Main step ─────────────────────────────────────────────────
    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_memory(map_state)
        me = tuple(my_position)
        t0 = time.time()
        self._ensure_topo()
        self._ensure_belief(me)
        self._trap_evaluator.reset_cache()

        if enemy_position is not None:
            enemy = tuple(int(v) for v in enemy_position)
            self._intent_tracker.update(enemy)
            self.last_seen_enemy = enemy
            self._last_seen_step = step_number
        else:
            enemy = None

        action = self._tier1_safety_filter(me, self.memory_map)
        if action is not None:
            return action

        action = self._tier2_pursuit(me, enemy, self.memory_map, step_number)
        if action is not None:
            return action

        action = self._tier3_explore(me, self.memory_map)
        if action is not None:
            return action

        return self._tier3_rl_fallback(map_state, me, t0, step_number, enemy)


# ===================================================================
# GHOST — Dynamic Mode Selection Architecture
# ===================================================================
class GhostAgent(BaseGhostAgent):
    """Blind Hider with Dynamic Mode Selection.

    Tier 1: Provable Survival Gate (overrides all below)
    Tier 2: Dynamic Mode Dispatch (Panic / Evasion / Fortress / Exploration)
    Tier 3: DRQN Fallback
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device('cpu')

        self.model = RecurrentActorCritic(5)
        if GHOST_MODEL_PATH.exists():
            state = torch.load(str(GHOST_MODEL_PATH), map_location=self.device,
                               weights_only=True)
            self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.hidden_state = _reset_lstm_state()

        self.mlp_model: Optional[GhostMoveMLP] = None
        if GHOST_MLP_PATH.exists():
            self.mlp_model = GhostMoveMLP()
            self.mlp_model.load_state_dict(
                torch.load(str(GHOST_MLP_PATH), map_location=self.device,
                           weights_only=True))
            self.mlp_model.to(self.device)
            self.mlp_model.eval()

        self.memory_map: Optional[np.ndarray] = None
        self._topo = TopologyAnalyzer()
        self._belief: Optional[BeliefStateTracker] = None
        self._trap_evaluator = TrapEvaluator()
        self._mode_selector = DynamicModeSelector(hysteresis=3)
        self._intent_tracker = IntentTracker(max_history=8)

        self._enemy: Optional[Tuple] = None
        self._last_enemy: Optional[Tuple] = None
        self._history: Deque[Tuple] = deque(maxlen=20)

        self._visit_count: Dict[Tuple, int] = {}
        self._oscillation_moves: List[Tuple] = []
        self._step_t0: float = 0.0

    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
            self._belief = BeliefStateTracker(21, 21)
        visible_mask = (map_state != -1)
        self.memory_map[visible_mask] = map_state[visible_mask]

    def _update_belief_map(self, ghost_pos, enemy_pos):
        if self._belief is not None and self.memory_map is not None:
            self._belief.update(ghost_pos, enemy_pos, self.memory_map)

    def _ensure_topo(self):
        if not self._topo.ready:
            self._topo.analyze(STATIC_FULL_MAP)

    def _anti_stuck_check(self, me):
        self._visit_count[me] = self._visit_count.get(me, 0) + 1
        self._oscillation_moves.append(me)
        if len(self._oscillation_moves) > 8:
            self._oscillation_moves = self._oscillation_moves[-8:]
        if self._visit_count.get(me, 0) >= 4:
            return True
        if len(self._oscillation_moves) >= 4:
            if len(set(self._oscillation_moves[-4:])) <= 2:
                return True
        return False

    def _break_stuck(self, map_state, me):
        self.hidden_state = _reset_lstm_state()
        self._visit_count.clear()
        self._oscillation_moves.clear()
        return Move(_random_valid_move(map_state, me))

    def _time_ok(self) -> bool:
        return (time.time() - self._step_t0) < STEP_TIME_LIMIT

    # ── TIER 1: Provable Survival Gate ────────────────────────────
    def _tier1_survival_gate(self, me, enemy, ms):
        moves = _legal(me, ms)
        if not moves:
            return Move.STAY

        in_dead_end = (me in self._topo.dead_ends or
                       (_cell_exits(me, ms) <= 1 and me in self._topo.corridor_cells))
        if in_dead_end:
            safe = [m for m in moves
                    if _apply(me, m) not in self._topo.dead_ends
                    and not self._trap_evaluator.is_dead_end_trap(
                        _apply(me, m), self._topo, ms, lookahead=8)]
            if safe:
                return Move(max(safe, key=lambda m:
                    self._topo._get_topological_weight(_apply(me, m), ms)))
            return Move(max(moves, key=lambda m:
                self._topo._get_topological_weight(_apply(me, m), ms)))

        if enemy is None:
            return None

        margin = self._trap_evaluator.escape_margin(
            me, enemy, ms, pacman_speed=2, sim_depth=6)
        if margin < 0.5:
            safe = [m for m in moves
                    if not self._trap_evaluator.is_dead_end_trap(
                        _apply(me, m), self._topo, ms, lookahead=8)]
            if safe:
                return Move(max(safe, key=lambda m:
                    (_cell_exits(_apply(me, m), STATIC_FULL_MAP) * 100
                     + (500 if _apply(me, m) in self._topo.junctions else 0)
                     - _manhattan(_apply(me, m), enemy) * 10)))
            return Move(max(moves, key=lambda m:
                _cell_exits(_apply(me, m), STATIC_FULL_MAP)))

        return None

    # ── TIER 2: Dynamic Mode Dispatch ─────────────────────────────
    def _tier2_mode_dispatch(self, me, enemy, ms, step_number):
        dist = _manhattan(me, enemy) if enemy is not None else None
        topo_safety = self._topo._get_topological_weight(me, ms)
        mode = self._mode_selector.select(
            enemy is not None, dist, step_number, 200, topo_safety)

        if mode == DynamicModeSelector.MODE_PANIC:
            return self._mode_panic(me, enemy, ms) if enemy else None
        elif mode == DynamicModeSelector.MODE_EVASION:
            return self._mode_evasion(me, enemy, ms) if enemy else None
        elif mode == DynamicModeSelector.MODE_FORTRESS:
            return self._mode_fortress(me, enemy, ms)
        else:
            return self._mode_exploration(me, ms)

    # ── PANIC MODE: 3-ply Minimax ─────────────────────────────────
    def _mode_panic(self, me, enemy, ms):
        moves = _legal(me, ms)
        if not moves:
            return Move.STAY

        best_move, best_score = moves[0], float("-inf")
        for g_move in moves:
            if not self._time_ok():
                return Move(best_move)
            g1 = _apply(me, g_move)
            if _manhattan(g1, enemy) < 2:
                continue
            p_moves = _legal(enemy, ms)
            worst_after_pacman = float("inf")
            for p_move in p_moves:
                p1 = _apply(enemy, p_move)
                p2 = _apply(p1, p_move) if _valid(p1, ms) else p1
                g_moves_2 = _legal(g1, ms)
                best_g_score = float("-inf")
                for g_move_2 in g_moves_2:
                    g2 = _apply(g1, g_move_2)
                    if _manhattan(g2, p2) < 2:
                        continue
                    score = (_manhattan(g2, p2) * 500
                             + _cell_exits(g2, STATIC_FULL_MAP) * 200
                             + (800 if g2 in self._topo.junctions else 0)
                             - (5000 if self._trap_evaluator.is_dead_end_trap(
                                 g2, self._topo, ms, 6) else 0)
                             - (1000 if g2 in self._history else 0))
                    if score > best_g_score:
                        best_g_score = score
                worst_after_pacman = min(worst_after_pacman, best_g_score)
            if worst_after_pacman > best_score:
                best_score = worst_after_pacman
                best_move = g_move
        return Move(best_move)

    # ── EVASION MODE: Strategic Flee + Future Sim ─────────────────
    def _mode_evasion(self, me, enemy, ms):
        gd = bfs_dist(ms, me, max_dist=20)
        moves = _legal(me, ms)
        if not moves:
            return Move.STAY

        best_move, best_score = moves[0], float("-inf")
        for g_move in moves:
            g1 = _apply(me, g_move)
            if _manhattan(g1, enemy) < 2:
                continue
            p_moves = _legal(enemy, ms)
            future_min_dist = float("inf")
            for p_move in p_moves:
                p1 = _apply(enemy, p_move)
                p2 = _apply(p1, p_move) if _valid(p1, ms) else p1
                future_min_dist = min(future_min_dist, _manhattan(g1, p2))

            score = (future_min_dist * 500.0
                     + gd.get(g1, 99) * 10
                     + _cell_exits(g1, STATIC_FULL_MAP) * 200.0
                     + (500 if g1 in self._topo.junctions else 0)
                     + (300 if g1 in self._topo.core else 0))
            if self._trap_evaluator.is_dead_end_trap(g1, self._topo, ms, 6):
                score -= 10000.0
            if g1 in self._history:
                score -= 2000.0
            if score > best_score:
                best_score = score
                best_move = g_move
        return Move(best_move)

    # ── FORTRESS MODE: Maximize Safe Distance ─────────────────────
    def _mode_fortress(self, me, enemy, ms):
        threat_center = (self._belief.threat_center()
                         if self._belief else (10, 10))
        moves = _legal(me, ms)
        if not moves:
            return Move.STAY

        best_move, best_score = moves[0], float("-inf")
        for m in moves:
            nxt = _apply(me, m)
            d = _manhattan(nxt, threat_center)
            score = (d * 1000.0
                     + (1000 if nxt in self._topo.loops else 0)
                     + (800 if nxt in self._topo.junctions else 0)
                     + _cell_exits(nxt, STATIC_FULL_MAP) * 200
                     + self._topo.junction_dist.get(nxt, 99) * 50)
            if nxt in self._topo.dead_ends:
                score -= 5000.0
            if nxt in self._topo.corridor_cells:
                score -= 1000.0
            if nxt in self._history:
                score -= 3000.0
            if score > best_score:
                best_score = score
                best_move = m
        return Move(best_move)

    # ── EXPLORATION MODE: Belief-Guided Safe Explore ───────────────
    def _mode_exploration(self, me, ms):
        moves = _legal(me, ms)
        if not moves:
            return Move.STAY

        threat = (self._belief.threat_center()
                  if self._belief else (10, 10))
        gd = bfs_dist(ms, me, max_dist=6)
        best_cell, best_score = me, float("-inf")
        for cell in gd:
            if cell == me:
                continue
            exits = _cell_exits(cell, STATIC_FULL_MAP)
            score = exits * 150
            score += max(0, 6 - gd[cell]) * 400
            if cell in self._topo.junctions:
                score += 600
            if cell in self._topo.core:
                score += 300
            if cell in self._topo.loops:
                score += 500
            if cell in self._topo.dead_ends:
                score -= 1500
            score += _manhattan(cell, threat) * 30
            jd = self._topo.junction_dist.get(cell, 99)
            score += max(0, 6 - jd) * 120
            if cell in self._history:
                score -= 500
            if score > best_score:
                best_score = score
                best_cell = cell

        if best_cell != me:
            path = astar(ms, me, best_cell)
            if path and path[0] in moves:
                return Move(path[0])

        return Move(max(moves, key=lambda m: (
            _cell_exits(_apply(me, m), STATIC_FULL_MAP) * 100
            + (300 if _apply(me, m) in self._topo.junctions else 0)
            - (800 if _apply(me, m) in self._topo.dead_ends else 0)
            - (400 if _apply(me, m) in self._history else 0))))

    # ── TIER 3: Enhanced RL Fallback ──────────────────────────────
    def _tier3_rl(self, map_state, me, t0, step_num, enemy=None):
        obs_t, pos_t = _build_obs_tensor(
            map_state, me, enemy, self.model,
            self._belief, self._topo,
            step_number=step_num)
        with torch.no_grad():
            action, _, _, _, self.hidden_state = \
                self.model.get_action_and_value(
                    obs_t, pos_t, self.hidden_state, deterministic=True)
        if time.time() - t0 > RL_TIMEOUT:
            return Move.STAY
        idx = min(action.item(), 4)
        return GHOST_ACTIONS[idx]

    # ── Main step ─────────────────────────────────────────────────
    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_memory(map_state)
        me = tuple(my_position)
        t0 = time.time()
        self._step_t0 = t0
        self._ensure_topo()
        self._trap_evaluator.reset_cache()

        if self._anti_stuck_check(me):
            return self._break_stuck(map_state, me)

        self._history.append(me)

        if enemy_position is not None:
            enemy = tuple(int(v) for v in enemy_position)
            self._intent_tracker.update(enemy)
            self._last_enemy = self._enemy
            self._enemy = enemy
            self._update_belief_map(me, enemy)
        else:
            enemy = None
            self._update_belief_map(me, None)

        action = self._tier1_survival_gate(me, enemy, self.memory_map)
        if action is not None:
            return action

        action = self._tier2_mode_dispatch(me, enemy, self.memory_map, step_number)
        if action is not None:
            return action

        return self._tier3_rl(map_state, me, t0, step_number, enemy)
