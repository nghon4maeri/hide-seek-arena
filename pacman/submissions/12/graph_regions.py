"""Tarjan region helpers for maze graphs."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]


def articulation_and_regions(neighbors: Dict[Cell, Tuple[Cell, ...]]) -> Tuple[Set[Cell], List[Set[Cell]]]:
    disc: Dict[Cell, int] = {}
    low: Dict[Cell, int] = {}
    parent: Dict[Cell, Optional[Cell]] = {}
    edge_stack: List[Tuple[Cell, Cell]] = []
    articulation: Set[Cell] = set()
    regions: List[Set[Cell]] = []
    time = 0

    def dfs(cell: Cell) -> None:
        nonlocal time
        time += 1
        disc[cell] = low[cell] = time
        child_count = 0
        for neighbor in neighbors[cell]:
            if neighbor not in disc:
                parent[neighbor] = cell
                child_count += 1
                edge_stack.append((cell, neighbor))
                dfs(neighbor)
                low[cell] = min(low[cell], low[neighbor])
                if _is_split(cell, neighbor, child_count, parent, disc, low):
                    articulation.add(cell)
                if low[neighbor] >= disc[cell]:
                    regions.append(_pop_region(edge_stack, (cell, neighbor)))
            elif neighbor != parent.get(cell) and disc[neighbor] < disc[cell]:
                low[cell] = min(low[cell], disc[neighbor])
                edge_stack.append((cell, neighbor))

    for cell in neighbors:
        if cell in disc:
            continue
        parent[cell] = None
        dfs(cell)
        if edge_stack:
            regions.append(_pop_region(edge_stack, None))
    return articulation, regions


def assign_regions(cells: Set[Cell], regions: List[Set[Cell]]) -> Dict[Cell, int]:
    by_cell: Dict[Cell, int] = {}
    ordered_regions = sorted(enumerate(regions), key=lambda item: len(item[1]), reverse=True)
    for region_id, region in ordered_regions:
        for cell in region:
            by_cell.setdefault(cell, region_id)
    for cell in cells:
        by_cell.setdefault(cell, len(regions))
    return by_cell


def _is_split(
    cell: Cell,
    neighbor: Cell,
    child_count: int,
    parent: Dict[Cell, Optional[Cell]],
    disc: Dict[Cell, int],
    low: Dict[Cell, int],
) -> bool:
    if parent.get(cell) is None:
        return child_count > 1
    return low[neighbor] >= disc[cell]


def _pop_region(edge_stack: List[Tuple[Cell, Cell]], stop_edge: Optional[Tuple[Cell, Cell]]) -> Set[Cell]:
    region: Set[Cell] = set()
    while edge_stack:
        edge = edge_stack.pop()
        region.update(edge)
        if stop_edge is not None and edge == stop_edge:
            break
    return region
