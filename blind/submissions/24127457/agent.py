import sys
import time
import random
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import torch
import torch.nn.functional as F

from network_architect import RecurrentActorCritic


MODEL_DIR = Path(__file__).resolve().parent
PACMAN_MODEL_PATH = MODEL_DIR / "pacman_model.pth"
GHOST_MODEL_PATH  = MODEL_DIR / "ghost_model.pth"

# Action mappings (must match training notebook)
# Pacman: 0-3=speed1, 4-7=speed2, 8=STAY
_PACMAN_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
_PACMAN_ACTIONS = [
    _PACMAN_MOVES[0], _PACMAN_MOVES[1], _PACMAN_MOVES[2], _PACMAN_MOVES[3],
    _PACMAN_MOVES[0], _PACMAN_MOVES[1], _PACMAN_MOVES[2], _PACMAN_MOVES[3],
    Move.STAY,
]
_PACMAN_STEPS  = [1, 1, 1, 1, 2, 2, 2, 2, 1]

_GHOST_ACTIONS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]

VISIBILITY_RADIUS = 5
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

HIDDEN_SIZE = 128
STAY_DEADLOCK_THRESHOLD = 3
STUCK_DEADLOCK_THRESHOLD = 5


def _build_obs_tensor(map_state, my_position):
    H, W = map_state.shape
    visible = _get_visible_cells(my_position, map_state, H, W)

    ch_wall = np.zeros((H, W), dtype=np.float32)
    ch_seen = np.zeros((H, W), dtype=np.float32)
    ch_fog  = np.zeros((H, W), dtype=np.float32)

    for r in range(H):
        for c in range(W):
            if map_state[r, c] == 1:
                ch_wall[r, c] = 1.0
            elif (r, c) in visible:
                ch_seen[r, c] = 1.0
            else:
                ch_fog[r, c] = 1.0

    obs_img = np.stack([ch_wall, ch_seen, ch_fog], axis=0)
    pos_norm = np.array([my_position[0] / H, my_position[1] / W], dtype=np.float32)

    obs_t = torch.from_numpy(obs_img).unsqueeze(0)
    pos_t = torch.from_numpy(pos_norm).unsqueeze(0)
    return obs_t, pos_t


def _get_visible_cells(pos, map_state, H, W):
    visible = {pos}
    r, c = pos
    for dr, dc in _DIRS:
        for dist in range(1, VISIBILITY_RADIUS + 1):
            nr, nc = r + dr * dist, c + dc * dist
            if not (0 <= nr < H and 0 <= nc < W):
                break
            visible.add((nr, nc))
            if map_state[nr, nc] == 1:
                break
    return visible


def _get_random_valid_move(map_state, my_position):
    """Return a random valid Move from current position."""
    H, W = map_state.shape
    r, c = my_position
    candidates = []
    for dr, dc in _DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and map_state[nr, nc] == 0:
            candidates.append((dr, dc))
    if not candidates:
        return Move.STAY
    dr, dc = random.choice(candidates)
    if (dr, dc) == (-1, 0):
        return Move.UP
    if (dr, dc) == (1, 0):
        return Move.DOWN
    if (dr, dc) == (0, -1):
        return Move.LEFT
    if (dr, dc) == (0, 1):
        return Move.RIGHT
    return Move.STAY


def _reset_lstm_state():
    """Return zero-initialized LSTM hidden state (h, c)."""
    h = torch.zeros(1, 1, HIDDEN_SIZE)
    c = torch.zeros(1, 1, HIDDEN_SIZE)
    return (h, c)


# ============================================================
#  BASE MIXIN — shared stuck-detection logic
# ============================================================

class _BaseBlindAgent:
    """Mixin providing deadlock-breaking logic for both agents."""

    def _break_deadlock(self, map_state, my_position, is_pacman):
        """Force a random valid move and reset LSTM state."""
        self.hidden_state = _reset_lstm_state()
        self.stay_counter = 0
        self.stuck_counter = 0
        move = _get_random_valid_move(map_state, my_position)
        if is_pacman:
            if move == Move.STAY:
                return Move.STAY
            return (move, 1)
        return move

    def _check_and_update_stuck(self, current_position):
        """Track position changes; return True if stuck break needed."""
        pos_changed = (current_position != self.last_position)
        self.last_position = current_position
        if pos_changed:
            self.stuck_counter = 0
            return False
        self.stuck_counter += 1
        return self.stuck_counter >= STUCK_DEADLOCK_THRESHOLD


# ============================================================
#  PACMAN AGENT
# ============================================================

class PacmanAgent(BasePacmanAgent, _BaseBlindAgent):
    """Blind Seeker — RL inference with LSTM memory."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))

        self.device = torch.device('cpu')
        self.model = RecurrentActorCritic(9)
        if PACMAN_MODEL_PATH.exists():
            state = torch.load(str(PACMAN_MODEL_PATH), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.hidden_state = _reset_lstm_state()
        self.timeout_limit = 0.85
        self.stay_counter = 0
        self.stuck_counter = 0
        self.last_position = None

    def step(self, map_state, my_position, enemy_position, step_number):
        _ = enemy_position
        _unused_step = step_number
        t0 = time.time()

        my_position = tuple(my_position)

        # --- deadlock break guard (STAY-based) ---
        if self.stay_counter >= STAY_DEADLOCK_THRESHOLD:
            return self._break_deadlock(map_state, my_position, is_pacman=True)

        # --- deadlock break guard (position-based) ---
        if self.last_position is not None and self._check_and_update_stuck(my_position):
            return self._break_deadlock(map_state, my_position, is_pacman=True)
        if self.last_position is None:
            self.last_position = my_position

        obs_t, pos_t = _build_obs_tensor(map_state, my_position)

        with torch.no_grad():
            action, _, _, _, self.hidden_state = self.model.get_action_and_value(
                obs_t, pos_t, self.hidden_state, deterministic=True
            )
        action_idx = action.item()

        elapsed = time.time() - t0
        if elapsed > self.timeout_limit:
            return Move.STAY

        move = _PACMAN_ACTIONS[action_idx]
        steps = _PACMAN_STEPS[action_idx]
        steps = min(steps, self.pacman_speed)

        # track consecutive STAYs
        if move == Move.STAY:
            self.stay_counter += 1
        else:
            self.stay_counter = 0

        if steps == 1:
            return move
        return (move, steps)


# ============================================================
#  GHOST AGENT
# ============================================================

class GhostAgent(BaseGhostAgent, _BaseBlindAgent):
    """Blind Hider — RL inference with LSTM memory."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.device = torch.device('cpu')
        self.model = RecurrentActorCritic(5)
        if GHOST_MODEL_PATH.exists():
            state = torch.load(str(GHOST_MODEL_PATH), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.hidden_state = _reset_lstm_state()
        self.timeout_limit = 0.85
        self.stay_counter = 0
        self.stuck_counter = 0
        self.last_position = None

    def step(self, map_state, my_position, enemy_position, step_number):
        _ = enemy_position
        _unused_step = step_number
        t0 = time.time()

        my_position = tuple(my_position)

        # --- deadlock break guard (STAY-based) ---
        if self.stay_counter >= STAY_DEADLOCK_THRESHOLD:
            return self._break_deadlock(map_state, my_position, is_pacman=False)

        # --- deadlock break guard (position-based) ---
        if self.last_position is not None and self._check_and_update_stuck(my_position):
            return self._break_deadlock(map_state, my_position, is_pacman=False)
        if self.last_position is None:
            self.last_position = my_position

        obs_t, pos_t = _build_obs_tensor(map_state, my_position)

        with torch.no_grad():
            action, _, _, _, self.hidden_state = self.model.get_action_and_value(
                obs_t, pos_t, self.hidden_state, deterministic=True
            )
        action_idx = action.item()
        action_idx = min(action_idx, 4)

        elapsed = time.time() - t0
        if elapsed > self.timeout_limit:
            return Move.STAY

        move = _GHOST_ACTIONS[action_idx]

        # track consecutive STAYs
        if move == Move.STAY:
            self.stay_counter += 1
        else:
            self.stay_counter = 0

        return move
