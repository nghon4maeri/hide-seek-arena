from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

from src.core.constants import ACTIONS, INF, MOVE_ACTIONS
from src.core.map_utils import manhattan
from src.core.movement import apply_action, is_open, legal_actions, neighbors
from src.core.types import Action, Grid, Position
from src.evaluation.hide_eval import evaluate_hide
from src.evaluation.seek_eval import evaluate_seek
from src.search.bfs import SearchToolkit

TraceLevel = str


PIPELINE = [
    "legal_move_generation",
    "bfs_distance_map",
    "flood_fill_safe_area",
    "dead_end_analysis",
    "danger_map",
    "candidate_evaluation",
    "minimax_alpha_beta",
]


def pos(value: Position) -> List[int]:
    return [int(value[0]), int(value[1])]


def key(value: Position) -> str:
    return f"{value[0]},{value[1]}"


def bfs_details(grid: Grid, start: Position, goal: Optional[Position] = None) -> Dict[str, Any]:
    q = deque([start])
    visited = {start}
    parent: Dict[Position, Optional[Position]] = {start: None}
    dist = {start: 0}
    explored: List[Position] = []
    frontier_frames: List[List[Position]] = []

    while q:
        current = q.popleft()
        explored.append(current)
        frontier_frames.append(list(q)[:80])
        if goal is not None and current == goal:
            break
        for nxt in neighbors(grid, current):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = current
                dist[nxt] = dist[current] + 1
                q.append(nxt)

    path: List[Position] = []
    if goal is not None and goal in parent:
        cur: Optional[Position] = goal
        while cur is not None and cur != start:
            path.append(cur)
            cur = parent[cur]
        path.reverse()

    return {
        "enabled": True,
        "start": pos(start),
        "goals": [pos(goal)] if goal is not None else [],
        "explored_order": [pos(cell) for cell in explored],
        "frontier_by_frame": [[pos(cell) for cell in frame] for frame in frontier_frames],
        "frontier_snapshots": [[pos(cell) for cell in frame] for frame in frontier_frames],
        "parent_map": {key(cell): key(parent_cell) for cell, parent_cell in parent.items() if parent_cell is not None},
        "distance_map": {key(cell): int(value) for cell, value in dist.items()},
        "final_path": [pos(cell) for cell in path],
    }


def astar_details(grid: Grid, start: Position, goal: Position) -> Dict[str, Any]:
    if not is_open(grid, start) or not is_open(grid, goal):
        return {"enabled": False, "start": pos(start), "goal": pos(goal), "frames": [], "final_path": []}

    frontier: List[Tuple[int, int, Position]] = []
    heappush(frontier, (manhattan(start, goal), 0, start))
    came_from: Dict[Position, Optional[Position]] = {start: None}
    g_score: Dict[Position, int] = {start: 0}
    closed: List[Position] = []
    frames: List[Dict[str, Any]] = []

    while frontier:
        _, g, current = heappop(frontier)
        if g != g_score[current]:
            continue
        closed.append(current)
        frames.append(
            {
                "current": pos(current),
                "open_set": [pos(item[2]) for item in frontier[:80]],
                "closed_set": [pos(cell) for cell in closed],
                "g": {key(cell): value for cell, value in g_score.items()},
                "h": {key(cell): manhattan(cell, goal) for cell in g_score},
                "f": {key(cell): value + manhattan(cell, goal) for cell, value in g_score.items()},
            }
        )
        if current == goal:
            break
        for nxt in neighbors(grid, current):
            ng = g + 1
            if ng < g_score.get(nxt, INF):
                g_score[nxt] = ng
                came_from[nxt] = current
                heappush(frontier, (ng + manhattan(nxt, goal), ng, nxt))

    path: List[Position] = []
    if goal in came_from:
        cur: Optional[Position] = goal
        while cur is not None and cur != start:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()

    return {"enabled": True, "start": pos(start), "goal": pos(goal), "frames": frames, "final_path": [pos(cell) for cell in path]}


def flood_fill_details(search: SearchToolkit, start: Position, enemy: Position, max_depth: int = 16, ghost_speed: int = 2) -> Dict[str, Any]:
    ghost_dist = search.bfs_distance_map(enemy)
    seen = {start}
    q = deque([(start, 0)])
    expansion: List[Position] = []
    reachable: List[Position] = []
    safe: List[Position] = []

    while q:
        current, depth = q.popleft()
        expansion.append(current)
        reachable.append(current)
        gd = ghost_dist[current[0]][current[1]]
        ghost_catch_turn = 0 if gd < 2 else ceil((gd - 1) / max(1, ghost_speed))
        if depth < ghost_catch_turn:
            safe.append(current)
        if depth >= max_depth:
            continue
        for nxt in search.neighbors(current):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, depth + 1))

    return {
        "enabled": True,
        "start": pos(start),
        "expansion_order": [pos(cell) for cell in expansion],
        "reachable_cells": [pos(cell) for cell in reachable],
        "safe_cells": [pos(cell) for cell in safe],
        "reachable_count": len(reachable),
        "safe_count": len(safe),
    }


def danger_details(search: SearchToolkit, enemy: Position, radius: int = 7) -> Dict[str, Any]:
    dist = search.bfs_distance_map(enemy)
    danger_cells: List[Position] = []
    levels: Dict[str, int] = {}
    for cell in search.passable:
        d = dist[cell[0]][cell[1]]
        if d <= radius:
            danger_cells.append(cell)
            levels[key(cell)] = int(max(1, radius + 1 - d))
    return {"enabled": True, "danger_cells": [pos(cell) for cell in danger_cells], "danger_level": levels}


def dead_end_details(search: SearchToolkit) -> Dict[str, Any]:
    deg = search.degree_map()
    dead: List[Position] = []
    corridors: List[Position] = []
    junctions: List[Position] = []
    for cell in search.passable:
        d = deg[cell[0]][cell[1]]
        if d <= 1:
            dead.append(cell)
        elif d == 2:
            corridors.append(cell)
        elif d >= 3:
            junctions.append(cell)
    return {
        "enabled": True,
        "dead_end_cells": [pos(cell) for cell in dead],
        "corridor_cells": [pos(cell) for cell in corridors],
        "junction_cells": [pos(cell) for cell in junctions],
    }


def candidate_details(
    search: SearchToolkit,
    grid: Grid,
    position: Position,
    enemy: Position,
    step_number: int,
    role: str,
    minimax_scores: Dict[str, float],
) -> Dict[str, Any]:
    actions = ACTIONS if role == "hide" else MOVE_ACTIONS
    legal = {action.name for action in legal_actions(grid, position, include_stay=(role == "hide"))}
    candidates: Dict[str, Any] = {}
    ranked: List[Tuple[str, float]] = []
    deg = search.degree_map()
    dead_dist = search.distance_to_dead_end_map()
    enemy_dist = search.bfs_distance_map(enemy)

    for action in actions:
        next_position = apply_action(grid, position, action)
        is_legal_action = action.name in legal
        distance_to_enemy = search.distance(next_position, enemy)
        reachable_area = search.reachable_area(next_position, max_depth=9)
        safe_area = search.safe_reachable_area(next_position, enemy, max_depth=12, ghost_speed=2)
        branching = deg[next_position[0]][next_position[1]]
        dead_penalty = max(0, 8 - dead_dist[next_position[0]][next_position[1]])
        danger_penalty = max(0, 8 - enemy_dist[next_position[0]][next_position[1]])
        trap_penalty = max(0, 12 - safe_area)
        path_score = -distance_to_enemy if role == "seek" else distance_to_enemy
        minimax_score = float(minimax_scores.get(action.name, 0.0))
        total_score = (
            float(evaluate_hide(search, next_position, enemy, step_number) if role == "hide" else evaluate_seek(search, enemy, next_position, step_number))
            if is_legal_action
            else -999999.0
        )
        if minimax_score:
            total_score = minimax_score

        weighted_terms = {
            "distance_term": 22.0 * distance_to_enemy if role == "hide" else -30.0 * distance_to_enemy,
            "area_term": 1.2 * reachable_area,
            "safety_term": 7.5 * safe_area if role == "hide" else -3.0 * safe_area,
            "dead_end_term": -13.0 * dead_penalty,
            "danger_term": -18.0 * danger_penalty,
            "minimax_term": minimax_score,
        }
        reason = (
            f"{action.name} moves to {pos(next_position)} with distance {distance_to_enemy}, "
            f"safe area {safe_area}, branching {branching}, and total score {total_score:.2f}."
        )
        candidates[action.name] = {
            "next_position": pos(next_position),
            "is_legal": is_legal_action,
            "features": {
                "distance_to_enemy": distance_to_enemy,
                "reachable_area": reachable_area,
                "safe_area": safe_area,
                "branching_factor": branching,
                "dead_end_penalty": dead_penalty,
                "danger_penalty": danger_penalty,
                "trap_penalty": trap_penalty,
                "path_score": path_score,
                "minimax_score": minimax_score,
            },
            "weighted_terms": weighted_terms,
            "total_score": total_score,
            "reason": reason,
        }
        ranked.append((action.name, total_score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return {
        "enabled": True,
        "candidates": candidates,
        "ranked_actions": [[name, score] for name, score in ranked],
    }


def minimax_details(
    search: SearchToolkit,
    grid: Grid,
    position: Position,
    enemy: Position,
    role: str,
    chosen_action: str,
    candidate_scores: Dict[str, float],
    max_depth: int = 3,
) -> Dict[str, Any]:
    actions = legal_actions(grid, position, include_stay=(role == "hide"))
    nodes: List[Dict[str, Any]] = []
    leaves: List[Dict[str, Any]] = []
    prunes: List[Dict[str, Any]] = []
    root_id = "0"
    nodes.append(
        {
            "id": root_id,
            "parent": None,
            "depth": 0,
            "player": role,
            "action": None,
            "position": pos(position),
            "enemy_position": pos(enemy),
            "alpha": -999999,
            "beta": 999999,
            "value_before": None,
            "value_after": max(candidate_scores.values()) if candidate_scores else 0,
            "is_pruned": False,
            "children": [],
        }
    )
    alpha = -999999.0
    beta = 999999.0
    node_id = 1
    for index, action in enumerate(actions):
        child_id = str(node_id)
        node_id += 1
        next_position = apply_action(grid, position, action)
        score = float(candidate_scores.get(action.name, 0.0))
        is_pruned = index >= 4 and beta <= alpha
        nodes[0]["children"].append(child_id)
        nodes.append(
            {
                "id": child_id,
                "parent": root_id,
                "depth": 1,
                "player": "seek" if role == "hide" else "hide",
                "action": action.name,
                "position": pos(next_position),
                "enemy_position": pos(enemy),
                "alpha": alpha,
                "beta": beta,
                "value_before": None,
                "value_after": score,
                "is_pruned": is_pruned,
                "children": [],
            }
        )
        leaves.append({"id": child_id, "value": score, "features": {"action": action.name, "next_position": pos(next_position)}})
        alpha = max(alpha, score)
        if is_pruned:
            prunes.append({"node_id": child_id, "depth": 1, "alpha": alpha, "beta": beta, "reason": "alpha >= beta"})

    best_value = float(candidate_scores.get(chosen_action, 0.0))
    return {
        "enabled": True,
        "max_depth": max_depth,
        "root_player": role,
        "nodes": nodes,
        "leaf_nodes": leaves,
        "prune_events": prunes,
        "best_action": chosen_action,
        "best_value": best_value,
    }


def build_full_trace(
    grid: Grid,
    position: Position,
    enemy_position: Position,
    step_number: int,
    role: str,
    chosen_action: str,
    base_trace: Dict[str, Any] | None = None,
    trace_level: TraceLevel = "full",
) -> Dict[str, Any]:
    base_trace = base_trace or {}
    agent_name = "Hide Agent" if role == "hide" else "Seek Agent"
    algorithm_name = "Minimax + Flood Fill" if role == "hide" else "A* + Minimax"
    search = SearchToolkit(grid)
    legal = [action.name for action in legal_actions(grid, position, include_stay=(role == "hide"))]
    scores = base_trace.get("evaluation_scores") or base_trace.get("candidate_scores") or {}
    if trace_level == "summary":
        return {
            "agent_name": agent_name,
            "step_number": step_number,
            "position": pos(position),
            "enemy_position": pos(enemy_position),
            "legal_actions": legal,
            "chosen_action": chosen_action,
            "algorithm_pipeline": ["legal_move_generation", "candidate_evaluation", "final_decision"],
            "algorithm_name": algorithm_name,
            "candidate_scores": scores,
            "candidate_actions": list(scores) or legal,
            "explanation": explain_choice(agent_name, chosen_action, scores, {}),
        }

    bfs = bfs_details(grid, position, enemy_position)
    astar = astar_details(grid, position, enemy_position)
    flood = flood_fill_details(search, position, enemy_position)
    danger = danger_details(search, enemy_position)
    dead = dead_end_details(search)
    candidates = candidate_details(search, grid, position, enemy_position, step_number, role, scores)
    minimax = minimax_details(search, grid, position, enemy_position, role, chosen_action, scores)
    return {
        "agent_name": agent_name,
        "step_number": step_number,
        "position": pos(position),
        "enemy_position": pos(enemy_position),
        "legal_actions": legal,
        "chosen_action": chosen_action,
        "algorithm_name": algorithm_name,
        "algorithm_pipeline": PIPELINE + ["final_decision"],
        "explanation": explain_choice(agent_name, chosen_action, scores, candidates),
        "bfs": bfs,
        "astar": astar,
        "flood_fill": flood,
        "danger_map": danger,
        "dead_end_analysis": dead,
        "candidate_evaluation": candidates,
        "minimax": minimax if trace_level == "full" else {"enabled": False, "nodes": [], "leaf_nodes": [], "prune_events": []},
        "candidate_actions": list(candidates["candidates"]),
        "candidate_scores": {name: data["total_score"] for name, data in candidates["candidates"].items()},
        "danger_cells": danger["danger_cells"],
        "dead_end_cells": dead["dead_end_cells"],
    }


def explain_choice(agent_name: str, action: str, scores: Dict[str, float], candidates: Dict[str, Any]) -> str:
    if not action:
        return f"{agent_name} has no selected action."
    if action == "STAY" and agent_name == "Hide Agent":
        return "STAY was chosen because all movement options are more dangerous or reduce safe reachable area."
    ranked = candidates.get("ranked_actions") if candidates else None
    if ranked:
        rank = next((index + 1 for index, item in enumerate(ranked) if item[0] == action), 1)
        total = len(ranked)
        details = candidates["candidates"].get(action, {})
        features = details.get("features", {})
        return (
            f"{agent_name} selected {action}. This action ranks {rank}/{total} because it gives "
            f"distance {features.get('distance_to_enemy', 'n/a')}, safe area {features.get('safe_area', 'n/a')}, "
            f"branching factor {features.get('branching_factor', 'n/a')}, and minimax score "
            f"{features.get('minimax_score', 0):.2f}."
        )
    if scores:
        best = max(scores.items(), key=lambda item: item[1])[0]
        return f"{agent_name} selected {action}; {best} has the strongest available score in the summary trace."
    return f"{agent_name} selected {action} using the configured classical search pipeline."

