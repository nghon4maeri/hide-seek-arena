"""Maze graph analysis for the HideSeek agents."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from environment import Move
from graph_regions import articulation_and_regions, assign_regions

Cell = Tuple[int, int]
CARDINAL_MOVES = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
_MAP_CACHE: Dict[Tuple[Tuple[int, ...], bytes, Cell], "MapAnalysis"] = {}


class MapAnalysis:
    """Parsed, cached graph view of the main or agent-reachable component."""

    def __new__(cls, map_state: np.ndarray, anchor: Optional[Cell] = None):
        normalized_anchor = _normalize_anchor(anchor)
        key = (tuple(map_state.shape), np.ascontiguousarray(map_state).tobytes(), normalized_anchor)
        cached = _MAP_CACHE.get(key)
        if cached is not None:
            return cached
        instance = super().__new__(cls)
        _MAP_CACHE[key] = instance
        return instance

    def __init__(self, map_state: np.ndarray, anchor: Optional[Cell] = None):
        if getattr(self, "_initialized", False):
            return
        self.map_state = np.asarray(map_state)
        self.height, self.width = self.map_state.shape
        self.anchor = _normalize_anchor(anchor)
        self.walkable_cells = _component_from_anchor(self.map_state, self.anchor)
        self.neighbor_map = {
            cell: tuple(_walkable_neighbors(cell, self.walkable_cells))
            for cell in self.walkable_cells
        }
        self._dist = _all_pairs_distances(self.walkable_cells, self.neighbor_map)
        self._cell_class = _classify_cells(self.neighbor_map)
        self.dead_end_depth = _dead_end_depths(self.neighbor_map, self._cell_class)
        self.articulation_points, self._regions = articulation_and_regions(self.neighbor_map)
        self._region_by_cell = assign_regions(self.walkable_cells, self._regions)
        self._initialized = True

    def neighbors(self, cell: Cell) -> Tuple[Cell, ...]:
        return self.neighbor_map.get(tuple(cell), ())

    def is_walkable(self, cell: Cell) -> bool:
        return tuple(cell) in self.walkable_cells

    def dist(self, a: Cell, b: Cell) -> float:
        return self._dist.get(tuple(a), {}).get(tuple(b), float("inf"))

    def cell_class(self, cell: Cell) -> str:
        return self._cell_class.get(tuple(cell), "wall")

    def region_of(self, cell: Cell) -> Optional[int]:
        return self._region_by_cell.get(tuple(cell))

    def region_size(self, cell: Cell) -> int:
        region_id = self.region_of(cell)
        return 0 if region_id is None else len(self._regions[region_id])

    def pacman_reachable(self, pos: Cell, speed: int = 2) -> Set[Cell]:
        reachable = {tuple(pos)}
        for move in CARDINAL_MOVES:
            current = tuple(pos)
            for _ in range(max(1, int(speed))):
                current = _next_cell(current, move)
                if current not in self.walkable_cells:
                    break
                reachable.add(current)
        return reachable


def _normalize_anchor(anchor: Optional[Cell]) -> Cell:
    return (int(anchor[0]), int(anchor[1])) if anchor is not None else (-1, -1)


def _component_from_anchor(map_state: np.ndarray, anchor: Cell) -> Set[Cell]:
    walkable = {tuple(int(v) for v in cell) for cell in np.argwhere(map_state == 0)}
    return _component_from_walkable(anchor, walkable) if anchor in walkable else _largest_component(walkable)


def _largest_component(walkable: Set[Cell]) -> Set[Cell]:
    remaining = set(walkable)
    largest: Set[Cell] = set()
    while remaining:
        component = _component_from_walkable(next(iter(remaining)), walkable)
        remaining.difference_update(component)
        if len(component) > len(largest):
            largest = component
    return largest


def _component_from_walkable(anchor: Cell, walkable: Set[Cell]) -> Set[Cell]:
    seen = {anchor}
    queue = deque([anchor])
    while queue:
        cell = queue.popleft()
        for neighbor in _walkable_neighbors(cell, walkable):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def _walkable_neighbors(cell: Cell, walkable: Set[Cell]) -> Iterable[Cell]:
    for move in CARDINAL_MOVES:
        neighbor = _next_cell(cell, move)
        if neighbor in walkable:
            yield neighbor


def _all_pairs_distances(cells: Set[Cell], neighbors: Dict[Cell, Tuple[Cell, ...]]) -> Dict[Cell, Dict[Cell, int]]:
    return {cell: _bfs_distances(cell, neighbors) for cell in cells}


def _bfs_distances(start: Cell, neighbors: Dict[Cell, Tuple[Cell, ...]]) -> Dict[Cell, int]:
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in neighbors[cell]:
            if neighbor not in dist:
                dist[neighbor] = dist[cell] + 1
                queue.append(neighbor)
    return dist


def _classify_cells(neighbors: Dict[Cell, Tuple[Cell, ...]]) -> Dict[Cell, str]:
    classes = {}
    for cell, adjacent in neighbors.items():
        degree = len(adjacent)
        classes[cell] = "dead_end" if degree <= 1 else "corridor" if degree == 2 else "junction"
    return classes


def _dead_end_depths(neighbors: Dict[Cell, Tuple[Cell, ...]], classes: Dict[Cell, str]) -> Dict[Cell, int]:
    depths = {cell: 0 for cell in neighbors}
    for cell, cell_class in classes.items():
        if cell_class != "dead_end":
            continue
        path = _dead_end_path(cell, neighbors, classes)
        for index, path_cell in enumerate(path):
            depths[path_cell] = max(depths[path_cell], len(path) - index)
    return depths


def _dead_end_path(start: Cell, neighbors: Dict[Cell, Tuple[Cell, ...]], classes: Dict[Cell, str]) -> List[Cell]:
    path = [start]
    previous = None
    current = start
    while classes.get(current) != "junction":
        next_cells = [cell for cell in neighbors[current] if cell != previous]
        if not next_cells:
            break
        previous, current = current, next_cells[0]
        if classes.get(current) == "junction":
            break
        path.append(current)
    return path


def _next_cell(cell: Cell, move: Move) -> Cell:
    row, col = cell
    dr, dc = move.value
    return row + dr, col + dc
