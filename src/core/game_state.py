from __future__ import annotations

from .constants import DEFAULT_MAX_STEPS
from .map_utils import as_grid, as_position
from .types import GameState


def parse_state(*args, **kwargs) -> GameState:
    if args and isinstance(args[0], dict):
        data = dict(args[0])
    else:
        data = dict(kwargs)
        names = ["map_state", "my_position", "enemy_position", "step_number"]
        for name, value in zip(names, args):
            data.setdefault(name, value)

    def first_present(*names):
        for name in names:
            if name in data and data[name] is not None:
                return data[name]
        raise KeyError("missing required state field: " + "/".join(names))

    return GameState(
        grid=as_grid(first_present("map_state", "grid", "board")),
        my_position=as_position(first_present("my_position", "position", "pacman_position", "ghost_position")),
        enemy_position=as_position(
            first_present("enemy_position", "opponent_position", "ghost_position", "pacman_position")
        ),
        step_number=int(data.get("step_number", data.get("turn", data.get("step", 0)))),
        max_steps=int(data.get("max_steps", data.get("maximum_steps", DEFAULT_MAX_STEPS))),
    )

