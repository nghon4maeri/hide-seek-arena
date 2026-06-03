"""Minimax helpers.

The role-specific agents implement their own compact minimax loops because
their evaluation and action ordering are asymmetric.  This module documents the
shared scoring convention and provides a small terminal helper.
"""

from src.core.map_utils import manhattan
from src.core.types import Position


def is_capture(pacman: Position, ghost: Position) -> bool:
    return manhattan(pacman, ghost) < 2

