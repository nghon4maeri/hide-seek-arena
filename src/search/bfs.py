from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence

from src.core.constants import INF
from src.core.map_utils import as_grid, as_position, grid_key, manhattan
from src.core.movement import apply_action, is_open, legal_actions, neighbors
from src.core.types import Action, Grid, GridKey, Position, SearchTrace


class SearchToolkit:
    """Cached grid search helper for the static arena map."""

    _instances: Dict[GridKey, "SearchToolkit"] = {}

    def __new__(cls, grid: Sequence[Sequence[int]]):
        key = grid_key(grid)
        instance = cls._instances.get(key)
        if instance is None:
            instance = super().__new__(cls)
            cls._instances[key] = instance
            instance._initialized = False
        return instance

    def __init__(self, grid: Sequence[Sequence[int]]):
        if self._initialized:
            return
        self.grid: Grid = as_grid(grid)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows else 0
        self.passable: List[Position] = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] == 0
        ]
        self._dist_cache: Dict[Position, List[List[int]]] = {}
        self._dead_end_cache: Optional[List[List[int]]] = None
        self._degree_cache: Optional[List[List[int]]] = None
        self._initialized = True

    def is_open(self, p: Position) -> bool:
        return is_open(self.grid, p)

    def step(self, p: Position, action: Action) -> Position:
        return apply_action(self.grid, p, action)

    def legal_actions(self, p: Position, include_stay: bool = True) -> List[Action]:
        return legal_actions(self.grid, p, include_stay)

    def neighbors(self, p: Position) -> Iterable[Position]:
        return neighbors(self.grid, p)

    def bfs_distance_map(self, source: Position, trace: Optional[SearchTrace] = None) -> List[List[int]]:
        source = as_position(source)
        cached = self._dist_cache.get(source)
        if cached is not None and trace is None:
            return cached

        dist = [[INF] * self.cols for _ in range(self.rows)]
        if not self.is_open(source):
            self._dist_cache[source] = dist
            return dist

        q = deque([source])
        dist[source[0]][source[1]] = 0
        if trace is not None:
            trace.algorithm = "BFS"
            trace.start = source

        while q:
            p = q.popleft()
            if trace is not None:
                trace.explored_nodes.append(p)
                if len(trace.frontier_snapshots) < 8:
                    trace.frontier_snapshots.append(list(q)[:64])
            nd = dist[p[0]][p[1]] + 1
            for n in self.neighbors(p):
                if dist[n[0]][n[1]] == INF:
                    dist[n[0]][n[1]] = nd
                    q.append(n)

        if trace is None:
            self._dist_cache[source] = dist
        return dist

    def distance(self, a: Position, b: Position) -> int:
        if not self.is_open(a) or not self.is_open(b):
            return manhattan(a, b)
        d = self.bfs_distance_map(a)[b[0]][b[1]]
        return d if d < INF else manhattan(a, b) + 20

    def degree_map(self) -> List[List[int]]:
        if self._degree_cache is not None:
            return self._degree_cache
        deg = [[0] * self.cols for _ in range(self.rows)]
        for p in self.passable:
            deg[p[0]][p[1]] = sum(1 for _ in self.neighbors(p))
        self._degree_cache = deg
        return deg

    def distance_to_dead_end_map(self) -> List[List[int]]:
        if self._dead_end_cache is not None:
            return self._dead_end_cache
        deg = self.degree_map()
        dist = [[INF] * self.cols for _ in range(self.rows)]
        q = deque()
        for p in self.passable:
            if deg[p[0]][p[1]] <= 1:
                dist[p[0]][p[1]] = 0
                q.append(p)
        while q:
            p = q.popleft()
            nd = dist[p[0]][p[1]] + 1
            for n in self.neighbors(p):
                if dist[n[0]][n[1]] == INF:
                    dist[n[0]][n[1]] = nd
                    q.append(n)
        self._dead_end_cache = dist
        return dist

    def astar_path(self, start: Position, goal: Position, trace: Optional[SearchTrace] = None) -> List[Position]:
        from .astar import astar_path

        return astar_path(self, start, goal, trace)

    def safe_reachable_area(self, pacman: Position, ghost: Position, max_depth: int = 16, ghost_speed: int = 2) -> int:
        from .flood_fill import safe_reachable_area

        return safe_reachable_area(self, pacman, ghost, max_depth, ghost_speed)

    def reachable_area(self, start: Position, max_depth: int = 18) -> int:
        from .flood_fill import reachable_area

        return reachable_area(self, start, max_depth)

    def safe_reachable_cells(
        self, pacman: Position, ghost: Position, max_depth: int = 16, ghost_speed: int = 2
    ) -> List[Position]:
        from .flood_fill import safe_reachable_cells

        return safe_reachable_cells(self, pacman, ghost, max_depth, ghost_speed)

