"""Leader sandbox agent.

This is a minimal valid submission placeholder. It preserves the official
interface and is intentionally not optimized.
"""

import sys
from pathlib import Path


SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

    def step(self, map_state, my_position, enemy_position, step_number):
        return Move.STAY


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, map_state, my_position, enemy_position, step_number):
        return Move.STAY
