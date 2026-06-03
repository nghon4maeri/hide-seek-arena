from __future__ import annotations

from .types import Action


INF = 10**8
DEFAULT_TIME_LIMIT = 0.85
DEFAULT_MAX_STEPS = 200
GHOST_SPEED = 2

ACTIONS = (
    Action("UP", -1, 0),
    Action("DOWN", 1, 0),
    Action("LEFT", 0, -1),
    Action("RIGHT", 0, 1),
    Action("STAY", 0, 0),
)

MOVE_ACTIONS = ACTIONS[:4]
ACTION_BY_NAME = {action.name: action for action in ACTIONS}

