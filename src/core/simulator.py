from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from src.agents import HideAgent, SeekAgent
from src.core.constants import ACTION_BY_NAME, DEFAULT_MAX_STEPS
from src.core.map_utils import manhattan
from src.core.movement import apply_action
from src.core.official_map import GHOST_START, OFFICIAL_MAP_GRID, PACMAN_START
from src.core.types import Grid, Position


def default_map() -> Grid:
    return [row[:] for row in OFFICIAL_MAP_GRID]


class LocalSimulator:
    def __init__(
        self,
        grid: Grid | None = None,
        pacman: Position = PACMAN_START,
        ghost: Position = GHOST_START,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        self.grid = grid if grid is not None else default_map()
        self.start_pacman = pacman
        self.start_ghost = ghost
        self.max_steps = max_steps
        self.hide_agent = HideAgent()
        self.seek_agent = SeekAgent()

    def run(self, debug: bool = True) -> Dict[str, Any]:
        pacman = self.start_pacman
        ghost = self.start_ghost
        frames: List[Dict[str, Any]] = []
        winner = "hide"

        for step in range(self.max_steps):
            hide_result = self.hide_agent.get_action(
                self.grid, pacman, ghost, step, max_steps=self.max_steps, return_trace=debug
            )
            seek_result = self.seek_agent.get_action(
                self.grid, ghost, pacman, step, max_steps=self.max_steps, return_trace=debug
            )
            if debug:
                hide_action, hide_trace = hide_result
                seek_action, seek_trace = seek_result
            else:
                hide_action, seek_action = hide_result, seek_result
                hide_trace = seek_trace = None

            pacman_next = apply_action(self.grid, pacman, ACTION_BY_NAME.get(hide_action, ACTION_BY_NAME["STAY"]))
            ghost_next = apply_action(self.grid, ghost, ACTION_BY_NAME.get(seek_action, ACTION_BY_NAME["STAY"]))

            frame = {
                "step": step,
                "pacman": pacman,
                "ghost": ghost,
                "pacman_next": pacman_next,
                "ghost_next": ghost_next,
                "hide_action": hide_action,
                "seek_action": seek_action,
                "hide_trace": asdict(hide_trace) if hide_trace is not None else None,
                "seek_trace": asdict(seek_trace) if seek_trace is not None else None,
            }
            frames.append(frame)

            pacman, ghost = pacman_next, ghost_next
            if manhattan(pacman, ghost) < 2:
                winner = "seek"
                break

        return {
            "grid": self.grid,
            "start_pacman": self.start_pacman,
            "start_ghost": self.start_ghost,
            "max_steps": self.max_steps,
            "winner": winner,
            "frames": frames,
        }

    def save_replay(self, path: str | Path, debug: bool = True) -> Dict[str, Any]:
        replay = self.run(debug=debug)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(replay, indent=2), encoding="utf-8")
        return replay
