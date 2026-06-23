"""Leader Integration Sandbox — Blind Adversary (Lab 2).

Student: 24127457
Role:   Leader / Integration / Benchmark / Final Submission

This is the integration sandbox for testing both Blind agents together.
The final merged version goes into team_submission/.
"""

import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """Blind Seeker — integration sandbox."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.memory_map = None
        self.last_seen_enemy = None

    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = (map_state != -1)
        self.memory_map[visible_mask] = map_state[visible_mask]

    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_memory(map_state)
        my_position = tuple(my_position)
        if enemy_position is not None:
            self.last_seen_enemy = tuple(enemy_position)
        # TODO: Integrate Seeker algorithm here
        return Move.STAY


class GhostAgent(BaseGhostAgent):
    """Blind Hider — integration sandbox."""

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
        my_position = tuple(my_position)
        if enemy_position is not None:
            self.last_seen_enemy = tuple(enemy_position)
        # TODO: Integrate Hider algorithm here
        return Move.STAY
