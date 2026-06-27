import sys
import time
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


class PacmanAgent(BasePacmanAgent):
    """Blind Seeker — RL inference with LSTM memory."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))

        self.device = torch.device('cpu')
        self.model = RecurrentActorCritic(9)
        if PACMAN_MODEL_PATH.exists():
            state = torch.load(str(PACMAN_MODEL_PATH), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.hidden_state = None
        self.last_action_time = 0.0
        self.timeout_limit = 0.85

    def step(self, map_state, my_position, enemy_position, step_number):
        t0 = time.time()

        my_position = tuple(my_position)
        obs_t, pos_t = _build_obs_tensor(map_state, my_position)

        with torch.no_grad():
            action, _, _, _, self.hidden_state = self.model.get_action_and_value(
                obs_t, pos_t, self.hidden_state, deterministic=True
            )
        action_idx = action.item()

        elapsed = time.time() - t0
        fallback = elapsed > self.timeout_limit
        if fallback:
            action_idx = 8

        move = _PACMAN_ACTIONS[action_idx]
        steps = _PACMAN_STEPS[action_idx]
        steps = min(steps, self.pacman_speed)

        if fallback:
            return Move.STAY
        if steps == 1:
            return move
        return (move, steps)


class GhostAgent(BaseGhostAgent):
    """Blind Hider — RL inference with LSTM memory."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.device = torch.device('cpu')
        self.model = RecurrentActorCritic(5)
        if GHOST_MODEL_PATH.exists():
            state = torch.load(str(GHOST_MODEL_PATH), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.hidden_state = None
        self.last_action_time = 0.0
        self.timeout_limit = 0.85

    def step(self, map_state, my_position, enemy_position, step_number):
        t0 = time.time()

        my_position = tuple(my_position)
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

        return _GHOST_ACTIONS[action_idx]
