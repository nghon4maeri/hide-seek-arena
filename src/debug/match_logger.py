from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.core.types import Grid, Position


def _pos(value: Position) -> List[int]:
    return [int(value[0]), int(value[1])]


class MatchLogger:
    """In-memory replay logger for local visualization scripts only."""

    def __init__(self, map_grid: Grid, pacman_start: Position, ghost_start: Position):
        self.buffer: List[Dict[str, Any]] = []
        self.map_grid = [row[:] for row in map_grid]
        self.pacman_start = pacman_start
        self.ghost_start = ghost_start

    def log_step(
        self,
        *,
        step_number: int,
        pacman_pos: Position,
        ghost_pos: Position,
        pacman_action: str,
        ghost_action: str,
        manhattan_distance: int,
        pacman: Dict[str, Any],
        ghost: Dict[str, Any],
    ) -> None:
        pacman_payload = {
            "pos": _pos(pacman_pos),
            "action": pacman_action,
            "candidateScores": {name: float(value) for name, value in pacman.get("candidateScores", {}).items()},
            "exploredNodes": [_pos(pos) for pos in pacman.get("exploredNodes", [])],
            "predictedPath": [_pos(pos) for pos in pacman.get("predictedPath", [])],
            "score": float(pacman.get("score", 0.0)),
            "algorithm": str(pacman.get("algorithm", "")),
            "explanation": str(pacman.get("explanation", "")),
        }
        ghost_payload = {
            "pos": _pos(ghost_pos),
            "action": ghost_action,
            "candidateScores": {name: float(value) for name, value in ghost.get("candidateScores", {}).items()},
            "exploredNodes": [_pos(pos) for pos in ghost.get("exploredNodes", [])],
            "predictedPath": [_pos(pos) for pos in ghost.get("predictedPath", [])],
            "score": float(ghost.get("score", 0.0)),
            "algorithm": str(ghost.get("algorithm", "")),
            "explanation": str(ghost.get("explanation", "")),
        }
        self.buffer.append(
            {
                "stepNumber": int(step_number),
                "pacman": pacman_payload,
                "ghost": ghost_payload,
                "manhattanDistance": int(manhattan_distance),
                # Backward-compatible flat fields for older visualizer builds.
                "pacmanPos": _pos(pacman_pos),
                "ghostPos": _pos(ghost_pos),
                "pacmanAction": pacman_action,
                "ghostAction": ghost_action,
                "exploredNodes": pacman_payload["exploredNodes"],
                "predictedPath": pacman_payload["predictedPath"],
                "score": pacman_payload["score"],
                "candidateScores": pacman_payload["candidateScores"],
                "chosenAgent": "hide",
                "algorithm": pacman_payload["algorithm"],
                "explanation": pacman_payload["explanation"],
            }
        )

    def export_json(self, path: str | Path) -> None:
        output = {
            "map": self.map_grid,
            "width": len(self.map_grid[0]) if self.map_grid else 0,
            "height": len(self.map_grid),
            "initial": {
                "pacman": _pos(self.pacman_start),
                "ghost": _pos(self.ghost_start),
            },
            "steps": self.buffer,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
