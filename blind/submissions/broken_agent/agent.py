"""
A broken agent for testing error logging in blind mode.
"""
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_PATH))

from environment import Move
import random


class PacmanAgent:
    def __init__(self, **kwargs):
        self.step_count = 0

    def step(self, map_state, my_position, enemy_position, step_number):
        self.step_count += 1
        if self.step_count > 5:
            raise RuntimeError(f"Intentional error for testing at step {step_number}")
        return random.choice([Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT])


class GhostAgent:
    def __init__(self, **kwargs):
        self.step_count = 0

    def step(self, map_state, my_position, enemy_position, step_number):
        self.step_count += 1
        if self.step_count > 5:
            raise RuntimeError(f"Intentional error for testing at step {step_number}")
        return random.choice([Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT])
