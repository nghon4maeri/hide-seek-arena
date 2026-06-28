"""24127457 — Blind Adversary (Lab 2) — V3 Strategic.

Pacman:  A* chase + interception + heatmap search when enemy lost.
Ghost:   Strategic flee (topology-aware) + anti-velocity + lookahead + RL stealth.
"""

import sys
import time
import heapq
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Deque

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import torch
import torch.nn as nn

from network_architect import RecurrentActorCritic

MODEL_DIR = Path(__file__).resolve().parent
PACMAN_MODEL_PATH = MODEL_DIR / "pacman_model.pth"
GHOST_MODEL_PATH  = MODEL_DIR / "ghost_model.pth"
GHOST_MLP_PATH    = MODEL_DIR / "ghost_mlp.pth"

_PACMAN_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
_PACMAN_ACTIONS = [
    _PACMAN_MOVES[0], _PACMAN_MOVES[1], _PACMAN_MOVES[2], _PACMAN_MOVES[3],
    _PACMAN_MOVES[0], _PACMAN_MOVES[1], _PACMAN_MOVES[2], _PACMAN_MOVES[3],
    Move.STAY,
]
_PACMAN_STEPS = [1, 1, 1, 1, 2, 2, 2, 2, 1]
_GHOST_ACTIONS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]

MOVE_ORDER = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
VISIBILITY_RADIUS = 5
HIDDEN_SIZE = 128

# Ghost thresholds
GHOST_FLEE_DIST = 14
GHOST_DANGER_DIST = 6
GHOST_LOOKAHEAD = 4

# Pacman thresholds
PACMAN_LAST_SEEN_TIMEOUT = 20
INTERCEPT_MIN_STREAK = 2
INTERCEPT_LOOKAHEAD = 4

# RL model time budget
RL_TIMEOUT = 0.85

# ===================================================================
# Grid utilities
# ===================================================================
def _shape(ms):
    if hasattr(ms, "shape"): return int(ms.shape[0]), int(ms.shape[1])
    return len(ms), len(ms[0]) if ms else 0

def _cell(ms, r, c):
    return int(ms[r, c]) if hasattr(ms, "shape") else int(ms[r][c])

def _apply(pos, move):
    return (pos[0] + move.value[0], pos[1] + move.value[1])

def _valid(pos, ms):
    r, c = pos; h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1

def _legal(pos, ms):
    return [m for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]

def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _cell_exits(pos, ms):
    return sum(1 for m in MOVE_ORDER if _valid(_apply(pos, m), ms))

# ===================================================================
# BFS distance map (capped to save time)
# ===================================================================
def bfs_dist(ms, start, max_dist=40):
    if not _valid(start, ms): return {start: 0}
    d = {start: 0}; q = deque([start])
    while q:
        cur = q.popleft()
        if d[cur] >= max_dist: continue
        for m in MOVE_ORDER:
            nxt = _apply(cur, m)
            if nxt not in d and _valid(nxt, ms):
                d[nxt] = d[cur] + 1; q.append(nxt)
    return d

# ===================================================================
# A* (returns list of Move)
# ===================================================================
def astar(ms, start, goal):
    if goal is None or not _valid(start, ms) or not _valid(goal, ms): return []
    if start == goal: return []
    open_set = [(0, 0, start)]
    came_from: Dict[Tuple, Tuple[Tuple, Move]] = {}
    g_score = {start: 0}; closed: Set[Tuple] = set()
    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current in closed: continue
        closed.add(current)
        if current == goal:
            path = []
            while current != start:
                prev, move = came_from[current]; path.append(move); current = prev
            path.reverse(); return path
        for move in MOVE_ORDER:
            nxt = _apply(current, move)
            if not _valid(nxt, ms) or nxt in closed: continue
            ng = g + 1
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng; came_from[nxt] = (current, move)
                heapq.heappush(open_set, (ng + _manhattan(nxt, goal), ng, nxt))
    return []

# ===================================================================
# Topology analysis (runs once per map)
# ===================================================================
class Topo:
    def __init__(self):
        self.dead_ends: Set[Tuple] = set()
        self.junctions: Set[Tuple] = set()
        self.core: Set[Tuple] = set()
        self.junction_dist: Dict[Tuple, int] = {}
        self.initialized = False

    def init(self, ms):
        if self.initialized: return
        H, W = _shape(ms)
        deg = {}
        for r in range(H):
            for c in range(W):
                if _cell(ms, r, c) == 1: continue
                p = (r, c)
                deg[p] = _cell_exits(p, ms)
                if deg[p] >= 3: self.junctions.add(p)
                elif deg[p] <= 1: self.dead_ends.add(p)

        # Expand dead-ends to branches
        dead_seeds = set(self.dead_ends)
        for seed in dead_seeds:
            cur, prev = seed, None
            while True:
                self.dead_ends.add(cur)
                if cur in self.junctions: break
                nxts = [x for x in (_apply(cur, m) for m in MOVE_ORDER)
                         if _valid(x, ms) and x != prev]
                if not nxts: break
                nxt = nxts[0]
                if nxt in self.dead_ends: break
                prev, cur = cur, nxt

        # Core: cells with ≥2 connections after iterative pruning
        active = {(r, c) for r in range(H) for c in range(W) if _cell(ms, r, c) != 1}
        changed = True
        while changed:
            changed = False; to_remove = set()
            for cell in active:
                if sum(1 for m in MOVE_ORDER if _valid(_apply(cell, m), ms) and _apply(cell, m) in active) <= 1:
                    to_remove.add(cell)
            if to_remove: active -= to_remove; changed = True
        self.core = active

        # Junction distance BFS
        self.junction_dist = {p: 10**9 for p in {(r, c) for r in range(H) for c in range(W) if _cell(ms, r, c) != 1}}
        q = deque()
        starts = self.junctions or self.core
        for p in starts:
            self.junction_dist[p] = 0; q.append(p)
        while q:
            cur = q.popleft()
            for nxt in (_apply(cur, m) for m in MOVE_ORDER if _valid(_apply(cur, m), ms)):
                if self.junction_dist.get(nxt, 10**9) > self.junction_dist[cur] + 1:
                    self.junction_dist[nxt] = self.junction_dist[cur] + 1; q.append(nxt)

        self.initialized = True


# ===================================================================
# RL observation builder (compatible with V1/V2)
# ===================================================================
def _get_visible_cells(pos, map_state, H, W):
    visible = {pos}; r, c = pos
    for dr, dc in _DIRS:
        for dist in range(1, VISIBILITY_RADIUS + 1):
            nr, nc = r + dr * dist, c + dc * dist
            if not (0 <= nr < H and 0 <= nc < W): break
            visible.add((nr, nc))
            if map_state[nr, nc] == 1: break
    return visible

def _build_obs_tensor(map_state, my_position, enemy_position=None, model=None):
    H, W = map_state.shape
    visible = _get_visible_cells(my_position, map_state, H, W)
    ch_wall = np.zeros((H, W), dtype=np.float32)
    ch_seen = np.zeros((H, W), dtype=np.float32)
    ch_fog  = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            if map_state[r, c] == 1: ch_wall[r, c] = 1.0
            elif (r, c) in visible: ch_seen[r, c] = 1.0
            else: ch_fog[r, c] = 1.0
    use_v2 = (model is not None and hasattr(model, 'INPUT_CHANNELS') and model.INPUT_CHANNELS == 4)
    if use_v2:
        ch_enemy = np.zeros((H, W), dtype=np.float32)
        if enemy_position is not None:
            er, ec = enemy_position
            if 0 <= er < H and 0 <= ec < W: ch_enemy[er, ec] = 1.0
        obs_img = np.stack([ch_wall, ch_seen, ch_fog, ch_enemy], axis=0)
        if enemy_position is not None:
            er, ec = enemy_position
            pos_norm = np.array([my_position[0] / H, my_position[1] / W, er / H, ec / W, 1.0], dtype=np.float32)
        else:
            pos_norm = np.array([my_position[0] / H, my_position[1] / W, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        obs_img = np.stack([ch_wall, ch_seen, ch_fog], axis=0)
        pos_norm = np.array([my_position[0] / H, my_position[1] / W], dtype=np.float32)
    obs_t = torch.from_numpy(obs_img).unsqueeze(0)
    pos_t = torch.from_numpy(pos_norm).unsqueeze(0)
    return obs_t, pos_t

def _get_random_valid_move(map_state, my_position):
    H, W = map_state.shape; r, c = my_position
    candidates = [(dr, dc) for dr, dc in _DIRS
                  if 0 <= r + dr < H and 0 <= c + dc < W and map_state[r + dr, c + dc] == 0]
    if not candidates: return Move.STAY
    dr, dc = random.choice(candidates)
    if (dr, dc) == (-1, 0): return Move.UP
    if (dr, dc) == (1, 0):  return Move.DOWN
    if (dr, dc) == (0, -1): return Move.LEFT
    return Move.RIGHT

def _reset_lstm_state():
    h = torch.zeros(1, 1, HIDDEN_SIZE); c = torch.zeros(1, 1, HIDDEN_SIZE)
    return (h, c)


# ===================================================================
# PACMAN — V3: A* chase + heatmap search
# ===================================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.device = torch.device('cpu')
        self.model = RecurrentActorCritic(9)
        if PACMAN_MODEL_PATH.exists():
            state = torch.load(str(PACMAN_MODEL_PATH), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=True)
        self.model.to(self.device); self.model.eval()
        self.hidden_state = _reset_lstm_state()

        self.memory_map: Optional[np.ndarray] = None
        self.last_seen_enemy: Optional[Tuple] = None
        self._enemy_direction: Optional[Tuple] = None
        self._direction_streak = 0
        self._steps_since_seen = 0
        self._stay_counter = 0
        self._last_pos: Optional[Tuple] = None
        self._stuck_counter = 0

    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        self.memory_map[map_state != -1] = map_state[map_state != -1]

    def _path_to_move(self, path, my_pos):
        if not path: return (Move.STAY, 1)
        move = path[0]; steps = 1; cur = _apply(my_pos, move)
        for m in path[1:]:
            if m == move and steps < self.pacman_speed:
                steps += 1; cur = _apply(cur, m)
            else: break
        return (move, steps)

    def _track_enemy(self, enemy_pos):
        if self.last_seen_enemy is None: return
        dr = enemy_pos[0] - self.last_seen_enemy[0]
        dc = enemy_pos[1] - self.last_seen_enemy[1]
        nd = (dr, dc)
        if nd == self._enemy_direction and (dr != 0 or dc != 0):
            self._direction_streak += 1
        else:
            self._enemy_direction = nd
            self._direction_streak = 1 if (dr != 0 or dc != 0) else 0

    def _intercept_target(self, enemy_pos):
        if self._direction_streak < INTERCEPT_MIN_STREAK or self._enemy_direction is None:
            return None
        dr, dc = self._enemy_direction
        er, ec = enemy_pos
        H, W = _shape(self.memory_map)
        best = None
        for i in range(1, INTERCEPT_LOOKAHEAD + 1):
            nr, nc = er + dr * i, ec + dc * i
            if not (0 <= nr < H and 0 <= nc < W): break
            if _cell(self.memory_map, nr, nc) == 1: break
            exits = _cell_exits((nr, nc), self.memory_map)
            if exits >= 3: return (nr, nc)
            if exits == 2 and best is None: best = (nr, nc)
        return best

    def _frontier_search(self):
        """Find nearest cell on frontier that is closest to last known enemy position."""
        ms = self.memory_map; H, W = ms.shape
        best, best_d = None, float("inf")
        for r in range(H):
            for c in range(W):
                if ms[r, c] != 0: continue
                has_fog = any(0 <= r + dr < H and 0 <= c + dc < W and ms[r + dr, c + dc] == -1
                              for dr, dc in _DIRS)
                if not has_fog: continue
                d = _manhattan((r, c), self.last_seen_enemy) if self.last_seen_enemy else 0
                if d < best_d: best_d = d; best = (r, c)
        return best

    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_memory(map_state)
        me = tuple(my_position)
        t0 = time.time()

        # --- stuck check ---
        if self._last_pos is not None:
            if me == self._last_pos: self._stuck_counter += 1
            else: self._stuck_counter = 0
        self._last_pos = me
        if self._stuck_counter >= 5:
            self._stuck_counter = 0
            return self._safe_random_move(map_state, me)

        # --- enemy VISIBLE: A* chase + interception ---
        if enemy_position is not None:
            enemy = tuple(int(v) for v in enemy_position)
            self._track_enemy(enemy)
            self.last_seen_enemy = enemy
            self._steps_since_seen = 0

            target = self._intercept_target(enemy) or enemy
            path = astar(self.memory_map, me, target)
            if path: return self._path_to_move(path, me)
            path = astar(self.memory_map, me, enemy)
            if path: return self._path_to_move(path, me)

        # --- enemy LOST: search last known + frontier ---
        self._steps_since_seen += 1
        if self._steps_since_seen <= PACMAN_LAST_SEEN_TIMEOUT and self.last_seen_enemy:
            path = astar(self.memory_map, me, self.last_seen_enemy)
            if path: return self._path_to_move(path, me)
            if me == self.last_seen_enemy:
                self.last_seen_enemy = None

        frontier = self._frontier_search()
        if frontier:
            path = astar(self.memory_map, me, frontier)
            if path: return self._path_to_move(path, me)

        return self._rl_fallback(map_state, me, t0)

    def _safe_random_move(self, map_state, me):
        self.hidden_state = _reset_lstm_state()
        move = _get_random_valid_move(map_state, me)
        return (move, 1) if move != Move.STAY else Move.STAY

    def _rl_fallback(self, map_state, me, t0):
        obs_t, pos_t = _build_obs_tensor(map_state, me, None, self.model)
        with torch.no_grad():
            action, _, _, _, self.hidden_state = self.model.get_action_and_value(
                obs_t, pos_t, self.hidden_state, deterministic=True)
        if time.time() - t0 > RL_TIMEOUT: return Move.STAY
        idx = action.item()
        move = _PACMAN_ACTIONS[idx]; steps = min(_PACMAN_STEPS[idx], self.pacman_speed)
        if steps == 1: return move
        return (move, steps)


# ===================================================================
# Ghost MLP — lightweight move predictor (trained on V3 heuristic)
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
# GHOST — V3: Strategic flee + MLP predictor
# ===================================================================
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device('cpu')
        self.model = RecurrentActorCritic(5)
        if GHOST_MODEL_PATH.exists():
            state = torch.load(str(GHOST_MODEL_PATH), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=True)
        self.model.to(self.device); self.model.eval()
        self.hidden_state = _reset_lstm_state()

        # MLP move predictor
        self.mlp_model: Optional[GhostMoveMLP] = None
        if GHOST_MLP_PATH.exists():
            self.mlp_model = GhostMoveMLP()
            self.mlp_model.load_state_dict(torch.load(str(GHOST_MLP_PATH), map_location=self.device, weights_only=True))
            self.mlp_model.to(self.device); self.mlp_model.eval()

        self.memory_map: Optional[np.ndarray] = None
        self._topo = Topo()
        self._enemy: Optional[Tuple] = None
        self._last_enemy: Optional[Tuple] = None
        self._history: Deque[Tuple] = deque(maxlen=20)
        self._stay_counter = 0
        self._last_pos: Optional[Tuple] = None
        self._stuck_counter = 0
        self._initialized = False

    def _mlp_extract_features(self, ghost_pos, enemy_pos):
        """Extract 30-dim features for MLP input."""
        ms = self.memory_map; H, W = _shape(ms)
        feats = [ghost_pos[0]/H, ghost_pos[1]/W, enemy_pos[0]/H, enemy_pos[1]/W, 1.0]
        for move in _GHOST_ACTIONS:
            nxt = _apply(ghost_pos, move)
            if _valid(nxt, ms):
                d = _manhattan(nxt, enemy_pos)
                exits = _cell_exits(nxt, ms)
                is_dead = 1.0 if nxt in self._topo.dead_ends else 0.0
                is_junc = 1.0 if nxt in self._topo.junctions else 0.0
                jd = self._topo.junction_dist.get(nxt, 99)
            else:
                d = 99; exits = 0; is_dead = 0.0; is_junc = 0.0; jd = 99
            feats.extend([min(d, 40) / 40.0, exits / 4.0, is_dead, is_junc, min(jd, 10) / 10.0])
        return np.array(feats, dtype=np.float32)

    def _mlp_move(self, ghost_pos, enemy_pos):
        """Use MLP to predict best move. Returns Move or None on failure."""
        if self.mlp_model is None: return None
        try:
            x = self._mlp_extract_features(ghost_pos, enemy_pos)
            x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.mlp_model(x_t)
                idx = logits.argmax(dim=-1).item()
            return _GHOST_ACTIONS[idx]
        except:
            return None

    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        self.memory_map[map_state != -1] = map_state[map_state != -1]

    def _ensure_topo(self):
        if not self._topo.initialized:
            self._topo.init(self.memory_map)

    def _score_cell(self, cell, pacman_dist_map, ghost_dist_map, pacman_pos):
        ms = self.memory_map
        pac_dist = pacman_dist_map.get(cell, 99)
        ghost_dist = ghost_dist_map.get(cell, 99)
        if ghost_dist < 2: return float("-inf")

        score = 0.0
        # Primary: maximize distance from Pacman
        score += pac_dist * 200.0
        # Secondary: prefer cells reachable before Pacman arrives
        pacman_eta = (pac_dist + 1) // 2  # Pacman speed-2
        margin = pacman_eta - ghost_dist
        score += max(0, margin) * 150.0
        # Topology bonuses
        if cell in self._topo.core: score += 800.0
        if cell in self._topo.junctions: score += 1200.0
        if cell in self._topo.dead_ends: score -= 6000.0
        # Junction proximity
        jd = self._topo.junction_dist.get(cell, 99)
        score += max(0, 4 - jd) * 600.0
        # Mobility
        score += _cell_exits(cell, ms) * 300.0
        # Avoid recent history
        if cell in self._history: score -= 2000.0
        if len(self._history) >= 2 and cell == self._history[-2]: score -= 3000.0
        return score

    def _strategic_flee(self, my_pos, pacman_pos):
        """Find the best cell within reasonable reach, then A* toward it."""
        ms = self.memory_map
        gd = bfs_dist(ms, my_pos, max_dist=20)
        pd = bfs_dist(ms, pacman_pos, max_dist=40)

        best_cell, best_score = my_pos, float("-inf")
        for cell in gd:
            score = self._score_cell(cell, pd, gd, pacman_pos)
            if score > best_score:
                best_score = score; best_cell = cell

        if best_cell == my_pos:
            # No good cell found — pick best immediate move
            moves = _legal(my_pos, ms)
            if not moves: return Move.STAY
            return max(moves, key=lambda m: self._score_cell(
                _apply(my_pos, m), pd, gd, pacman_pos))

        path = astar(ms, my_pos, best_cell)
        if path and path[0] in _legal(my_pos, ms):
            nxt = _apply(my_pos, path[0])
            if _manhattan(nxt, pacman_pos) >= 2:
                return path[0]

        # Path blocked — try greedy best adjacent cell
        moves = _legal(my_pos, ms)
        if not moves: return Move.STAY
        move = max(moves, key=lambda m: self._score_cell(
            _apply(my_pos, m), pd, gd, pacman_pos))
        return move

    def _anti_velocity_move(self, my_pos, pacman_pos):
        """If Pacman has consistent direction, move opposite to it."""
        if self._last_enemy is None: return None
        dr = pacman_pos[0] - self._last_enemy[0]
        dc = pacman_pos[1] - self._last_enemy[1]
        if dr == 0 and dc == 0: return None

        # Try moving in the same direction as Pacman (to stay behind) or opposite
        ms = self.memory_map
        moves = _legal(my_pos, ms)
        if not moves: return None

        # Prefer going opposite to Pacman's direction
        opp_dr, opp_dc = -dr, -dc
        best_move, best_dist = None, -1
        for m in moves:
            nxt = _apply(my_pos, m)
            if _manhattan(nxt, pacman_pos) < 2: continue
            nd = _manhattan(nxt, pacman_pos)
            # Bonus if move direction matches opposite of Pacman
            bonus = 3.0 if (m.value[0] * opp_dr + m.value[1] * opp_dc) > 0 else 0
            if nd + bonus > best_dist:
                best_dist = nd + bonus; best_move = m
        return best_move

    def _minimax_flee(self, my_pos, pacman_pos, depth=GHOST_LOOKAHEAD, t0=None):
        """Simple 1-ply minimax: try all Ghost moves, for each simulate Pacman moving closer."""
        ms = self.memory_map
        pd = bfs_dist(ms, pacman_pos, max_dist=30)
        moves = _legal(my_pos, ms)
        if not moves: return Move.STAY

        best_move, best_score = moves[0], float("-inf")
        for m in moves:
            ng = _apply(my_pos, m)
            if _manhattan(ng, pacman_pos) < 2: continue

            # Simulate Pacman: for each possible Pacman move (speed-1 for simplicity),
            # compute worst-case distance after 1 step
            pacman_moves = _legal(pacman_pos, ms)
            worst_pac_dist = float("inf")
            for pm in pacman_moves:
                # Pacman can move speed-2 straight
                np1 = _apply(pacman_pos, pm)
                np2 = _apply(np1, pm) if _valid(np1, ms) else np1
                d = pd.get(ng, _manhattan(ng, np2))
                worst_pac_dist = min(worst_pac_dist, d)

            score = worst_pac_dist * 100.0
            if m in _legal(my_pos, ms):
                nxt = _apply(my_pos, m)
                score += (_cell_exits(nxt, ms) - 2) * 50.0
                if nxt in self._topo.junctions: score += 300.0
                if nxt in self._topo.dead_ends: score -= 2000.0
            if score > best_score:
                best_score = score; best_move = m

        return best_move

    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_memory(map_state)
        me = tuple(my_position)
        t0 = time.time()

        # --- stuck check ---
        if self._last_pos is not None:
            if me == self._last_pos: self._stuck_counter += 1
            else: self._stuck_counter = 0
        self._last_pos = me
        if self._stuck_counter >= 5:
            self._stuck_counter = 0
            return self._safe_random_move(map_state, me)

        self._history.append(me)

        # --- enemy VISIBLE ---
        if enemy_position is not None:
            enemy = tuple(int(v) for v in enemy_position)
            self._last_enemy = self._enemy
            self._enemy = enemy
            self._ensure_topo()

            dist = _manhattan(me, enemy)

            if dist < GHOST_DANGER_DIST:
                # Danger! Try MLP first (fastest), then minimax
                move = self._mlp_move(me, enemy)
                if move and _manhattan(_apply(me, move), enemy) >= 2:
                    return move
                move = self._minimax_flee(me, enemy, t0=t0)
                if move and _manhattan(_apply(me, move), enemy) >= 2:
                    return move

            if dist < GHOST_FLEE_DIST:
                # Flee zone — try MLP first
                move = self._mlp_move(me, enemy)
                if move and _manhattan(_apply(me, move), enemy) >= 2:
                    return move

                move = self._strategic_flee(me, enemy)
                if _manhattan(_apply(me, move), enemy) >= 2:
                    return move

                # Try anti-velocity
                move = self._anti_velocity_move(me, enemy)
                if move and _manhattan(_apply(me, move), enemy) >= 2:
                    return move

            # Enemy visible but far — RL stealth
            return self._rl_fallback(map_state, me, t0, enemy)

        # --- enemy NOT visible: RL stealth in fog ---
        return self._rl_fallback(map_state, me, t0, None)

    def _safe_random_move(self, map_state, me):
        self.hidden_state = _reset_lstm_state()
        return _get_random_valid_move(map_state, me)

    def _rl_fallback(self, map_state, me, t0, enemy=None):
        obs_t, pos_t = _build_obs_tensor(map_state, me, enemy, self.model)
        with torch.no_grad():
            action, _, _, _, self.hidden_state = self.model.get_action_and_value(
                obs_t, pos_t, self.hidden_state, deterministic=True)
        if time.time() - t0 > RL_TIMEOUT: return Move.STAY
        idx = min(action.item(), 4)
        return _GHOST_ACTIONS[idx]
