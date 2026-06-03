from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


Position = Tuple[int, int]
Grid = List[List[int]]
GridKey = Tuple[int, int, Tuple[int, ...]]


@dataclass(frozen=True)
class Action:
    name: str
    dr: int
    dc: int


@dataclass
class GameState:
    grid: Grid
    my_position: Position
    enemy_position: Position
    step_number: int = 0
    max_steps: int = 200


@dataclass
class SearchTrace:
    algorithm: str
    start: Optional[Position] = None
    goal: Optional[Position] = None
    explored_nodes: List[Position] = field(default_factory=list)
    frontier_snapshots: List[List[Position]] = field(default_factory=list)
    final_path: List[Position] = field(default_factory=list)
    candidate_actions: List[str] = field(default_factory=list)
    evaluation_scores: Dict[str, float] = field(default_factory=dict)
    chosen_action: Optional[str] = None
    danger_cells: List[Position] = field(default_factory=list)
    safe_area: List[Position] = field(default_factory=list)
    dead_end_cells: List[Position] = field(default_factory=list)

