from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.simulator import LocalSimulator
from src.core.official_map import (
    GHOST_START,
    OFFICIAL_MAP_GRID,
    OFFICIAL_MAP_HEIGHT,
    OFFICIAL_MAP_WIDTH,
    PACMAN_START,
)
from src.debug.full_trace import build_full_trace


def pos_list(value) -> List[int]:
    return [int(value[0]), int(value[1])]


def positions(values) -> List[List[int]]:
    if not values:
        return []
    return [pos_list(value) for value in values]


def snapshots(values) -> List[List[List[int]]]:
    if not values:
        return []
    return [positions(snapshot) for snapshot in values]


def explanation(agent_name: str, action: str, scores: Dict[str, float]) -> str:
    if not action:
        return f"{agent_name} has no selected action in this frame."
    if not scores:
        return f"{agent_name} selected {action} using the available search policy."
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best = ordered[0][0]
    if best == action:
        return f"{action} has the strongest evaluated score after heuristic ordering and adversarial lookahead."
    return f"{action} was selected after time-bounded search; {best} had the highest immediate heuristic score."


def convert_trace(raw: Dict[str, Any] | None, agent_name: str, algorithm_name: str, action: str) -> Dict[str, Any]:
    raw = raw or {}
    scores = raw.get("evaluation_scores") or raw.get("candidate_scores") or {}
    candidate_actions = raw.get("candidate_actions") or list(scores)
    final_path = positions(raw.get("final_path"))
    explored = positions(raw.get("explored_nodes"))
    safe_area = positions(raw.get("safe_area"))
    danger = positions(raw.get("danger_cells"))
    dead_ends = positions(raw.get("dead_end_cells"))
    return {
        "agent_name": agent_name,
        "algorithm_name": algorithm_name,
        "candidate_actions": candidate_actions,
        "candidate_scores": scores,
        "chosen_action": raw.get("chosen_action") or action,
        "explanation": explanation(agent_name, action, scores),
        "bfs": {
            "explored_order": explored,
            "frontier_snapshots": snapshots(raw.get("frontier_snapshots")),
            "final_path": positions(raw.get("bfs", {}).get("final_path") if isinstance(raw.get("bfs"), dict) else []),
        },
        "astar": {
            "open_set": positions(raw.get("astar", {}).get("open_set") if isinstance(raw.get("astar"), dict) else []),
            "closed_set": explored,
            "final_path": final_path,
        },
        "flood_fill": {
            "reachable_cells": safe_area,
            "safe_cells": safe_area,
        },
        "minimax": {
            "simulated_positions": [],
            "leaf_scores": scores,
            "pruned_branches": [],
        },
        "danger_cells": danger,
        "dead_end_cells": dead_ends,
    }


def convert_replay(raw: Dict[str, Any]) -> Dict[str, Any]:
    frames = raw["frames"]
    steps = []
    for index, frame in enumerate(frames):
        is_last = index == len(frames) - 1
        status = "running"
        if is_last and raw.get("winner") == "seek":
            status = "seek_win"
        elif is_last and raw.get("winner") == "hide":
            status = "hide_win"

        hide_action = frame["hide_action"]
        seek_action = frame["seek_action"]
        steps.append(
            {
                "step": int(frame["step"]),
                "status": status,
                "hide": {
                    "position": pos_list(frame["pacman"]),
                    "action": hide_action,
                    "trace": convert_trace(frame.get("hide_trace"), "Hide Agent", "Minimax + Flood Fill", hide_action),
                },
                "seek": {
                    "position": pos_list(frame["ghost"]),
                    "action": seek_action,
                    "trace": convert_trace(frame.get("seek_trace"), "Seek Agent", "A* + Minimax", seek_action),
                },
            }
        )
    return {
        "map": raw["grid"],
        "width": len(raw["grid"][0]),
        "height": len(raw["grid"]),
        "initial": {
            "pacman": pos_list(raw["start_pacman"]),
            "ghost": pos_list(raw["start_ghost"]),
        },
        "legend": {
            "wall": 1,
            "empty": 0,
        },
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-level",
        choices=["none", "summary", "detailed", "full"],
        default="full",
        help="Trace detail level for generated visualizer replay.",
    )
    args = parser.parse_args()
    simulator = LocalSimulator(
        grid=[row[:] for row in OFFICIAL_MAP_GRID],
        pacman=PACMAN_START,
        ghost=GHOST_START,
        max_steps=40,
    )
    raw = simulator.run(debug=args.trace_level != "none")
    replay = convert_replay(raw)
    if args.trace_level != "none":
        for frame, step in zip(raw["frames"], replay["steps"]):
            hide_trace = build_full_trace(
                raw["grid"],
                tuple(frame["pacman"]),
                tuple(frame["ghost"]),
                int(frame["step"]),
                "hide",
                frame["hide_action"],
                frame.get("hide_trace"),
                trace_level=args.trace_level,
            )
            seek_trace = build_full_trace(
                raw["grid"],
                tuple(frame["ghost"]),
                tuple(frame["pacman"]),
                int(frame["step"]),
                "seek",
                frame["seek_action"],
                frame.get("seek_trace"),
                trace_level=args.trace_level,
            )
            step["hide"]["trace"] = hide_trace
            step["seek"]["trace"] = seek_trace
    else:
        for step in replay["steps"]:
            step["hide"]["trace"] = {}
            step["seek"]["trace"] = {}
    out = ROOT / "visualizer" / "public" / "sample_replay.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    print(
        f"generated {out} with {len(replay['steps'])} steps trace_level={args.trace_level} "
        f"on official map {OFFICIAL_MAP_WIDTH}x{OFFICIAL_MAP_HEIGHT} "
        f"pacman={PACMAN_START} ghost={GHOST_START}"
    )


if __name__ == "__main__":
    main()
