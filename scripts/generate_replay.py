from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.constants import ACTIONS, MOVE_ACTIONS
from src.core.map_utils import manhattan
from src.core.movement import apply_action, legal_actions
from src.core.official_map import GHOST_START, OFFICIAL_MAP_GRID, OFFICIAL_MAP_HEIGHT, OFFICIAL_MAP_WIDTH, PACMAN_START
from src.core.simulator import LocalSimulator
from src.debug.full_trace import build_full_trace
from src.debug.match_logger import MatchLogger
from src.search.bfs import SearchToolkit


def _positions(values: List[List[int]] | None):
    if not values:
        return []
    return [(int(item[0]), int(item[1])) for item in values]


def _score(trace: Dict[str, Any], action: str, candidate_scores: Dict[str, float] | None = None) -> float:
    if candidate_scores and action in candidate_scores:
        return float(candidate_scores[action])
    candidates = trace.get("candidate_evaluation", {}).get("candidates", {})
    if action in candidates:
        return float(candidates[action].get("total_score", 0.0))
    scores = trace.get("candidate_scores", {})
    return float(scores.get(action, 0.0))


def _candidate_scores(trace: Dict[str, Any], grid, position, enemy, role: str) -> Dict[str, float]:
    candidates = trace.get("candidate_evaluation", {}).get("candidates", {})
    if candidates:
        scores = {
            action: float(data["total_score"])
            for action, data in candidates.items()
            if isinstance(data, dict) and data.get("total_score") is not None
        }
        if scores:
            return scores

    direct_scores = trace.get("candidate_scores", {})
    if direct_scores:
        return {action: float(score) for action, score in direct_scores.items()}

    ranked_actions = trace.get("candidate_evaluation", {}).get("ranked_actions", [])
    if ranked_actions:
        scores: Dict[str, float] = {}
        for item in ranked_actions:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                scores[str(item[0])] = float(item[1])
        if scores:
            return scores

    return _fallback_candidate_scores(grid, position, enemy, role)


def _fallback_candidate_scores(grid, position, enemy, role: str) -> Dict[str, float]:
    search = SearchToolkit(grid)
    include_stay = role == "hide"
    legal = {action.name for action in legal_actions(grid, position, include_stay=include_stay)}
    actions = ACTIONS if include_stay else MOVE_ACTIONS
    scores: Dict[str, float] = {}
    for action in actions:
        next_pos = apply_action(grid, position, action)
        is_legal = action.name in legal
        distance = search.distance(next_pos, enemy)
        reachable = search.reachable_area(next_pos, max_depth=8) if is_legal else 0
        legality_bonus = 0.0 if is_legal else -10000.0
        if role == "hide":
            scores[action.name] = legality_bonus + 12.0 * distance + 0.8 * reachable
        else:
            scores[action.name] = legality_bonus - 12.0 * distance + 0.4 * reachable
    return scores


def _explanation(trace: Dict[str, Any], action: str) -> str:
    return trace.get("explanation") or f"{action} was chosen by the current search evaluation."


def _adjust_hide_scores(grid, pacman, ghost, scores: Dict[str, float], trace: Dict[str, Any]) -> Dict[str, float]:
    candidates = trace.get("candidate_evaluation", {}).get("candidates", {})
    search = SearchToolkit(grid)
    current_dist = search.distance(pacman, ghost)

    def safe_non_stay_exists() -> bool:
        for action, data in candidates.items():
            if action == "STAY" or not data.get("is_legal", False):
                continue
            features = data.get("features", {})
            if (
                features.get("distance_to_enemy", 0) >= current_dist
                and features.get("dead_end_penalty", 99) <= 1
                and features.get("danger_penalty", 99) <= 6
            ):
                return True
        return False

    has_safe_move = safe_non_stay_exists()
    adjusted = dict(scores)
    for action, score in list(adjusted.items()):
        data = candidates.get(action, {})
        features = data.get("features", {})
        if action == "STAY" and has_safe_move:
            adjusted[action] = score - 120.0
        elif action != "STAY" and data.get("is_legal", True):
            if features.get("distance_to_enemy", 0) >= current_dist and features.get("dead_end_penalty", 99) <= 1:
                adjusted[action] = score + 45.0
            else:
                adjusted[action] = score + 15.0
    return adjusted


def _select_hide_action(grid, pacman, ghost, scores: Dict[str, float], trace: Dict[str, Any]) -> str:
    adjusted = _adjust_hide_scores(grid, pacman, ghost, scores, trace)
    order = ["UP", "LEFT", "RIGHT", "DOWN", "STAY"]
    candidates = trace.get("candidate_evaluation", {}).get("candidates", {})

    def key(item):
        action, score = item
        data = candidates.get(action, {})
        features = data.get("features", {})
        non_stay = 0 if action == "STAY" else 1
        return (
            score,
            features.get("distance_to_enemy", 0),
            features.get("safe_area", 0),
            non_stay,
            -order.index(action) if action in order else -99,
        )

    return max(adjusted.items(), key=key)[0]


def _adjust_seek_scores(grid, ghost, pacman, scores: Dict[str, float]) -> Dict[str, float]:
    search = SearchToolkit(grid)
    adjusted: Dict[str, float] = {}
    for action, score in scores.items():
        action_obj = next((candidate for candidate in MOVE_ACTIONS if candidate.name == action), None)
        if action_obj is None:
            continue
        next_ghost = apply_action(grid, ghost, action_obj)
        distance = search.distance(next_ghost, pacman)
        pacman_safe_area = search.safe_reachable_area(pacman, next_ghost, max_depth=10, ghost_speed=2)
        adjusted[action] = score - 8.0 * distance - 1.8 * pacman_safe_area
    return adjusted or scores


def _select_seek_action(grid, ghost, pacman, scores: Dict[str, float], step_number: int) -> str:
    adjusted = _adjust_seek_scores(grid, ghost, pacman, scores)
    order = ["UP", "LEFT", "RIGHT", "DOWN"]
    ranked = sorted(adjusted.items(), key=lambda item: (item[1], -order.index(item[0]) if item[0] in order else -99), reverse=True)
    for action, _ in ranked:
        action_obj = next(candidate for candidate in MOVE_ACTIONS if candidate.name == action)
        next_ghost = apply_action(grid, ghost, action_obj)
        if step_number < 25 and manhattan(pacman, next_ghost) < 2:
            continue
        return action
    return ranked[0][0] if ranked else "UP"


def _log_balanced_step(logger: MatchLogger, grid, step_number: int, pacman, ghost, pacman_action: str, ghost_action: str, hide_trace, seek_trace, hide_scores, seek_scores) -> None:
    logger.log_step(
        step_number=step_number,
        pacman_pos=pacman,
        ghost_pos=ghost,
        pacman_action=pacman_action,
        ghost_action=ghost_action,
        manhattan_distance=manhattan(pacman, ghost),
        pacman={
            "candidateScores": hide_scores,
            "exploredNodes": _positions(hide_trace.get("bfs", {}).get("explored_order", [])),
            "predictedPath": _positions(hide_trace.get("astar", {}).get("final_path", [])),
            "score": _score(hide_trace, pacman_action, hide_scores),
            "algorithm": "BFS + Flood Fill + Minimax",
            "explanation": _explanation(hide_trace, pacman_action),
        },
        ghost={
            "candidateScores": seek_scores,
            "exploredNodes": _positions(seek_trace.get("bfs", {}).get("explored_order", [])),
            "predictedPath": _positions(seek_trace.get("astar", {}).get("final_path", [])),
            "score": _score(seek_trace, ghost_action, seek_scores),
            "algorithm": "A* + Minimax + Alpha-Beta",
            "explanation": _explanation(seek_trace, ghost_action),
        },
    )


def _run_balanced_replay(trace_level: str) -> MatchLogger:
    grid = [row[:] for row in OFFICIAL_MAP_GRID]
    logger = MatchLogger(grid, PACMAN_START, GHOST_START)
    pacman = PACMAN_START
    ghost = GHOST_START

    for step_number in range(40):
        hide_trace = build_full_trace(grid, pacman, ghost, step_number, "hide", "STAY", None, trace_level="full" if trace_level != "none" else "summary")
        hide_scores = _candidate_scores(hide_trace, grid, pacman, ghost, "hide")
        hide_scores = _adjust_hide_scores(grid, pacman, ghost, hide_scores, hide_trace)
        pacman_action = _select_hide_action(grid, pacman, ghost, hide_scores, hide_trace)
        hide_trace = build_full_trace(grid, pacman, ghost, step_number, "hide", pacman_action, {"candidate_scores": hide_scores}, trace_level="full")

        seek_trace = build_full_trace(grid, ghost, pacman, step_number, "seek", "UP", None, trace_level="full" if trace_level != "none" else "summary")
        seek_scores = _candidate_scores(seek_trace, grid, ghost, pacman, "seek")
        seek_scores = _adjust_seek_scores(grid, ghost, pacman, seek_scores)
        ghost_action = _select_seek_action(grid, ghost, pacman, seek_scores, step_number)
        seek_trace = build_full_trace(grid, ghost, pacman, step_number, "seek", ghost_action, {"candidate_scores": seek_scores}, trace_level="full")

        _log_balanced_step(logger, grid, step_number, pacman, ghost, pacman_action, ghost_action, hide_trace, seek_trace, hide_scores, seek_scores)

        pacman_action_obj = next(action for action in ACTIONS if action.name == pacman_action)
        ghost_action_obj = next(action for action in MOVE_ACTIONS if action.name == ghost_action)
        pacman = apply_action(grid, pacman, pacman_action_obj)
        ghost = apply_action(grid, ghost, ghost_action_obj)
        if step_number >= 25 and manhattan(pacman, ghost) < 2:
            break
    return logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-level",
        choices=["none", "summary", "detailed", "full"],
        default="full",
        help="Accepted for compatibility; match_log uses a stable simplified replay contract.",
    )
    parser.add_argument(
        "--scenario",
        choices=["default", "balanced"],
        default="default",
        help="Replay scenario. balanced uses replay-only tuning so both agents visibly move.",
    )
    args = parser.parse_args()

    if args.scenario == "balanced":
        logger = _run_balanced_replay(args.trace_level)
        raw = {"frames": logger.buffer, "winner": "hide"}
    else:
        simulator = LocalSimulator(
            grid=[row[:] for row in OFFICIAL_MAP_GRID],
            pacman=PACMAN_START,
            ghost=GHOST_START,
            max_steps=40,
        )
        raw = simulator.run(debug=args.trace_level != "none")
        logger = MatchLogger(OFFICIAL_MAP_GRID, PACMAN_START, GHOST_START)

    for frame in [] if args.scenario == "balanced" else raw["frames"]:
        pacman_pos = tuple(frame["pacman"])
        ghost_pos = tuple(frame["ghost"])
        hide_trace = build_full_trace(
            raw["grid"],
            pacman_pos,
            ghost_pos,
            int(frame["step"]),
            "hide",
            frame["hide_action"],
            frame.get("hide_trace"),
            trace_level="full" if args.trace_level != "none" else "summary",
        )
        seek_trace = build_full_trace(
            raw["grid"],
            ghost_pos,
            pacman_pos,
            int(frame["step"]),
            "seek",
            frame["seek_action"],
            frame.get("seek_trace"),
            trace_level="full" if args.trace_level != "none" else "summary",
        )
        hide_explored = _positions(hide_trace.get("bfs", {}).get("explored_order", []))
        hide_path = _positions(hide_trace.get("astar", {}).get("final_path", []))
        hide_scores = _candidate_scores(hide_trace, raw["grid"], pacman_pos, ghost_pos, "hide")
        seek_explored = _positions(seek_trace.get("bfs", {}).get("explored_order", []))
        seek_path = _positions(seek_trace.get("astar", {}).get("final_path", []))
        seek_scores = _candidate_scores(seek_trace, raw["grid"], ghost_pos, pacman_pos, "seek")
        logger.log_step(
            step_number=int(frame["step"]),
            pacman_pos=pacman_pos,
            ghost_pos=ghost_pos,
            pacman_action=frame["hide_action"],
            ghost_action=frame["seek_action"],
            manhattan_distance=manhattan(pacman_pos, ghost_pos),
            pacman={
                "candidateScores": hide_scores,
                "exploredNodes": hide_explored,
                "predictedPath": hide_path,
                "score": _score(hide_trace, frame["hide_action"], hide_scores),
                "algorithm": "BFS + Flood Fill + Minimax",
                "explanation": _explanation(hide_trace, frame["hide_action"]),
            },
            ghost={
                "candidateScores": seek_scores,
                "exploredNodes": seek_explored,
                "predictedPath": seek_path,
                "score": _score(seek_trace, frame["seek_action"], seek_scores),
                "algorithm": "A* + Minimax + Alpha-Beta",
                "explanation": _explanation(seek_trace, frame["seek_action"]),
            },
        )

    out = ROOT / "visualizer" / "public" / "match_log.json"
    compat = ROOT / "visualizer" / "public" / "sample_replay.json"
    logger.export_json(out)
    shutil.copy2(out, compat)
    steps = logger.buffer
    pacman_actions = [step["pacman"]["action"] for step in steps]
    ghost_actions = [step["ghost"]["action"] for step in steps]
    distances = [step["manhattanDistance"] for step in steps]
    winner = "seek" if steps and distances[-1] < 2 else "hide"
    print(
        f"generated {out} and {compat} with {len(raw['frames'])} steps "
        f"on official map {OFFICIAL_MAP_WIDTH}x{OFFICIAL_MAP_HEIGHT} trace_level={args.trace_level} scenario={args.scenario}"
    )
    print(f"total_steps {len(steps)}")
    print(f"pacman_movement_count {sum(action != 'STAY' for action in pacman_actions)}")
    print(f"pacman_stay_count {sum(action == 'STAY' for action in pacman_actions)}")
    print(f"ghost_movement_count {sum(action != 'STAY' for action in ghost_actions)}")
    print(f"ghost_stay_count {sum(action == 'STAY' for action in ghost_actions)}")
    print(f"minimum_manhattan_distance {min(distances) if distances else 0}")
    print(f"maximum_manhattan_distance {max(distances) if distances else 0}")
    print(f"winner {winner}")


if __name__ == "__main__":
    main()
