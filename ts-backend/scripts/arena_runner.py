"""
arena_runner.py — JSON Lines bridge for TS backend.
Wraps pacman/src/arena.py or blind/src/arena.py, outputs per-step JSON Lines.
DO NOT modify pacman/ or blind/ directories.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PAC_SRC = str(ROOT_DIR / "pacman" / "src")
BLIND_SRC = str(ROOT_DIR / "blind" / "src")


def _json_line(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


def run_lab1(args):
    sys.path.insert(0, PAC_SRC)
    from environment import Environment, Move

    submissions_dir = str(ROOT_DIR / "pacman" / "submissions")
    # Must import agent_loader AFTER path is set
    sys.path.insert(0, str(Path(submissions_dir).resolve()))
    from agent_loader import AgentLoader, AgentLoadError

    env = Environment(
        max_steps=args.max_steps,
        deterministic_starts=not args.random_spawn,
        capture_distance_threshold=args.capture_distance,
        pacman_speed=args.pacman_speed,
    )
    loader = AgentLoader(submissions_dir=submissions_dir)

    # Load agents
    pacman_agent = loader.load_agent(args.seek, "pacman", init_kwargs={"pacman_speed": args.pacman_speed})
    ghost_agent = loader.load_agent(args.hide, "ghost")

    map_state, pacman_pos, ghost_pos = env.reset()

    grid = env.map.tolist()
    _json_line({
        "type": "init",
        "grid": grid,
        "width": env.width,
        "height": env.height,
        "pacmanStart": list(pacman_pos),
        "ghostStart": list(ghost_pos),
        "config": {
            "labId": "lab1",
            "maxSteps": args.max_steps,
            "captureDistance": args.capture_distance,
            "pacmanSpeed": args.pacman_speed,
        },
    })

    step = 0
    game_over = False
    result = ""

    while not game_over:
        step += 1

        try:
            raw_pm = pacman_agent.step(map_state.copy(), tuple(pacman_pos), tuple(ghost_pos), step)
            pm = loader.validate_agent_move(raw_pm, "pacman", args.seek, args.pacman_speed)
            pacman_move, pacman_steps = pm
        except Exception as e:
            game_over = True
            result = "ghost_wins"
            _json_line({"type": "error", "agent": "pacman", "step": step, "message": str(e)})
            break

        try:
            raw_gm = ghost_agent.step(map_state.copy(), tuple(ghost_pos), tuple(pacman_pos), step)
            gm = loader.validate_agent_move(raw_gm, "ghost", args.hide)
        except Exception as e:
            game_over = True
            result = "pacman_wins"
            _json_line({"type": "error", "agent": "ghost", "step": step, "message": str(e)})
            break

        # Execute step in environment
        game_over, result, new_state = env.step(pm, gm)
        map_state, pacman_pos, ghost_pos = new_state

        dist = abs(pacman_pos[0] - ghost_pos[0]) + abs(pacman_pos[1] - ghost_pos[1])
        status = "running"
        if result == "pacman_wins":
            status = "pacman_wins"
        elif result == "ghost_wins":
            status = "ghost_wins"

        _json_line({
            "type": "step",
            "stepNumber": step,
            "pacmanPos": list(pacman_pos),
            "ghostPos": list(ghost_pos),
            "pacmanAction": str(pacman_move),
            "pacmanSteps": pacman_steps,
            "ghostAction": str(gm),
            "manhattanDistance": dist,
            "status": status,
        })

    _json_line({
        "type": "end",
        "winner": result if result else "ghost_wins",
        "totalSteps": step,
    })
    return 0


def run_lab2(args):
    sys.path.insert(0, BLIND_SRC)
    from environment import Environment, Move

    submissions_dir = str(ROOT_DIR / "blind" / "submissions")
    sys.path.insert(0, str(Path(submissions_dir).resolve()))
    from agent_loader import AgentLoader, AgentLoadError

    env = Environment(
        max_steps=args.max_steps,
        deterministic_starts=not args.random_spawn,
        capture_distance_threshold=args.capture_distance,
        pacman_speed=args.pacman_speed,
    )
    loader = AgentLoader(submissions_dir=submissions_dir)

    pacman_obs = max(1, int(args.pacman_obs_radius))
    ghost_obs = max(1, int(args.ghost_obs_radius))

    pacman_agent = loader.load_agent(args.seek, "pacman", init_kwargs={"pacman_speed": args.pacman_speed})
    ghost_agent = loader.load_agent(args.hide, "ghost")

    map_state, pacman_pos, ghost_pos = env.reset()

    grid = env.map.tolist()
    _json_line({
        "type": "init",
        "grid": grid,
        "width": env.width,
        "height": env.height,
        "pacmanStart": list(pacman_pos),
        "ghostStart": list(ghost_pos),
        "config": {
            "labId": "lab2",
            "maxSteps": args.max_steps,
            "captureDistance": args.capture_distance,
            "pacmanSpeed": args.pacman_speed,
            "pacmanObsRadius": pacman_obs,
            "ghostObsRadius": ghost_obs,
        },
    })

    step = 0
    game_over = False
    result = ""

    while not game_over:
        step += 1

        # Lab 2: cross-shaped vision
        pacman_obs, pacman_my_pos, pacman_vis_enemy = env.get_observation(
            "pacman", pacman_obs, ghost_obs
        )
        ghost_obs_map, ghost_my_pos, ghost_vis_enemy = env.get_observation(
            "ghost", pacman_obs, ghost_obs
        )

        try:
            raw_pm = pacman_agent.step(pacman_obs, pacman_my_pos, pacman_vis_enemy, step)
            pm = loader.validate_agent_move(raw_pm, "pacman", args.seek, args.pacman_speed)
            pacman_move, pacman_steps = pm
        except Exception as e:
            game_over = True
            result = "ghost_wins"
            _json_line({"type": "error", "agent": "pacman", "step": step, "message": str(e)})
            break

        try:
            raw_gm = ghost_agent.step(ghost_obs_map, ghost_my_pos, ghost_vis_enemy, step)
            gm = loader.validate_agent_move(raw_gm, "ghost", args.hide)
        except Exception as e:
            game_over = True
            result = "pacman_wins"
            _json_line({"type": "error", "agent": "ghost", "step": step, "message": str(e)})
            break

        game_over, result, new_state = env.step(pm, gm)
        map_state, pacman_pos, ghost_pos = new_state

        dist = abs(pacman_pos[0] - ghost_pos[0]) + abs(pacman_pos[1] - ghost_pos[1])
        status = "running"
        if result == "pacman_wins":
            status = "pacman_wins"
        elif result == "ghost_wins":
            status = "ghost_wins"

        _json_line({
            "type": "step",
            "stepNumber": step,
            "pacmanPos": list(pacman_pos),
            "ghostPos": list(ghost_pos),
            "pacmanAction": str(pacman_move),
            "pacmanSteps": pacman_steps,
            "ghostAction": str(gm),
            "manhattanDistance": dist,
            "status": status,
        })

    _json_line({
        "type": "end",
        "winner": result if result else "ghost_wins",
        "totalSteps": step,
    })
    return 0


def main():
    parser = argparse.ArgumentParser(description="Arena JSON Lines runner")
    parser.add_argument("--lab", required=True, choices=["lab1", "lab2"])
    parser.add_argument("--seek", required=True)
    parser.add_argument("--hide", required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--capture-distance", type=int, default=2)
    parser.add_argument("--pacman-speed", type=int, default=2)
    parser.add_argument("--pacman-obs-radius", type=int, default=5)
    parser.add_argument("--ghost-obs-radius", type=int, default=5)
    parser.add_argument("--random-spawn", action="store_true")
    args = parser.parse_args()

    try:
        if args.lab == "lab1":
            return run_lab1(args)
        else:
            return run_lab2(args)
    except Exception as e:
        _json_line({"type": "error", "message": str(e), "phase": "init"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
