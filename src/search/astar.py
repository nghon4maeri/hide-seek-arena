from __future__ import annotations

from heapq import heappop, heappush
from typing import Dict, List, Optional, Tuple

from src.core.constants import INF
from src.core.map_utils import manhattan
from src.core.types import Position, SearchTrace


def astar_path(search, start: Position, goal: Position, trace: Optional[SearchTrace] = None) -> List[Position]:
    if trace is not None:
        trace.algorithm = "A*"
        trace.start = start
        trace.goal = goal
    if start == goal or not search.is_open(start) or not search.is_open(goal):
        return []

    frontier: List[Tuple[int, int, Position]] = []
    heappush(frontier, (manhattan(start, goal), 0, start))
    came_from: Dict[Position, Optional[Position]] = {start: None}
    cost: Dict[Position, int] = {start: 0}

    while frontier:
        _, g, current = heappop(frontier)
        if g != cost[current]:
            continue
        if trace is not None:
            trace.explored_nodes.append(current)
            if len(trace.frontier_snapshots) < 8:
                trace.frontier_snapshots.append([item[2] for item in frontier[:64]])
        if current == goal:
            break
        for nxt in search.neighbors(current):
            ng = g + 1
            if ng < cost.get(nxt, INF):
                cost[nxt] = ng
                came_from[nxt] = current
                heappush(frontier, (ng + manhattan(nxt, goal), ng, nxt))

    if goal not in came_from:
        return []

    path = []
    cur: Optional[Position] = goal
    while cur is not None and cur != start:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    if trace is not None:
        trace.final_path = path
    return path

