"""
TEMPLATE for Blind Adversary (Lab 2) — Partial Observability

Copy this file to submissions/<your_student_id>/agent.py and implement
your Blind Hide / Blind Seek agents.

Key differences from Lab 1:
- map_state contains -1 for cells outside your field of view
- enemy_position can be None when the enemy is not visible
- Vision is cross-shaped: up to 5 cells in 4 cardinal directions,
  blocked by walls
- You MUST maintain your own memory/mental map across steps
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
    """Blind Seeker — find and catch the Ghost under limited vision."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        # TODO: Initialize your data structures (memory map, last known enemy, etc.)

    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Decide the next move.

        Args:
            map_state: numpy array (21x21)
                - 0 = visible empty path
                - 1 = wall (always visible)
                - -1 = UNSEEN (outside your field of view)
            my_position: (row, col) tuple — your current position
            enemy_position: (row, col) tuple OR None if not currently visible
            step_number: int, starts at 1

        Returns:
            Move OR (Move, steps) where 1 <= steps <= self.pacman_speed
        """
        my_position = tuple(my_position)

        # TODO: Update your internal memory map

        # TODO: Handle enemy visibility
        if enemy_position is not None:
            # Enemy is visible — plan pursuit
            pass
        else:
            # Enemy is hidden — search / explore / predict
            pass

        # TODO: Implement your search algorithm (A*, BFS, Minimax, etc.)
        return Move.STAY


class GhostAgent(BaseGhostAgent):
    """Blind Hider — survive under limited vision."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize your data structures (memory map, last known enemy, etc.)

    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Decide the next move.

        Args:
            map_state: numpy array (21x21)
                - 0 = visible empty path
                - 1 = wall (always visible)
                - -1 = UNSEEN (outside your field of view)
            my_position: (row, col) tuple — your current position
            enemy_position: (row, col) tuple OR None if not currently visible
            step_number: int, starts at 1

        Returns:
            Move enum (UP, DOWN, LEFT, RIGHT, STAY) — NOT a tuple
        """
        my_position = tuple(my_position)

        # TODO: Update your internal memory map

        # TODO: Handle enemy visibility
        if enemy_position is not None:
            # Enemy is visible — evade!
            pass
        else:
            # Enemy is hidden — explore safely or stay unpredictable
            pass

        # TODO: Implement your evasion algorithm (Minimax, Monte Carlo, etc.)
        return Move.STAY
