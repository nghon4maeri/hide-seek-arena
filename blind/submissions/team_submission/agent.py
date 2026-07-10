"""
Blind Adversary — Team Submission (Lab 2)

Placeholder. The team lead will merge reviewed implementations here
before the final submission.

Expected agents:
  class PacmanAgent(BasePacmanAgent):   Blind Seeker
  class GhostAgent(BaseGhostAgent):     Blind Hider

Both must handle:
  - map_state with -1 (unseen) cells
  - enemy_position possibly being None
  - maintaining internal memory across steps
"""

import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    """Blind Seeker — TODO: implement."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))

    def step(self, map_state, my_position, enemy_position, step_number):
        return Move.STAY


class GhostAgent(BaseGhostAgent):
    """Blind Hider — TODO: implement."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, map_state, my_position, enemy_position, step_number):
        return Move.STAY
