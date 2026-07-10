"""Benchmark runner for Blind Adversary submissions.

Runs multiple matches with partial observability enabled by default:
  --pacman-obs-radius 5 --ghost-obs-radius 5

Results are printed per game for win/loss analysis.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BLIND_ROOT = Path(__file__).resolve().parents[1]
ARENA = BLIND_ROOT / "src" / "arena.py"


def run_game(seek: str, hide: str, max_steps: int,
             pacman_obs: int = 5, ghost_obs: int = 5) -> int:
    command = [
        sys.executable,
        str(ARENA),
        "--seek", seek,
        "--hide", hide,
        "--max-steps", str(max_steps),
        "--no-viz",
        "--pacman-obs-radius", str(pacman_obs),
        "--ghost-obs-radius", str(ghost_obs),
        "--submissions-dir", str(BLIND_ROOT / "submissions"),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(command, cwd=BLIND_ROOT / "src", env=env, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark matches for Blind Adversary."
    )
    parser.add_argument("--seek", default="team_submission",
                        help="Pacman/seeker submission ID.")
    parser.add_argument("--hide", default="example_student",
                        help="Ghost/hider submission ID.")
    parser.add_argument("--games", type=int, default=10,
                        help="Number of repeated games.")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Maximum steps per game.")
    parser.add_argument("--pacman-obs", type=int, default=5,
                        help="Pacman observation radius (default: 5).")
    parser.add_argument("--ghost-obs", type=int, default=5,
                        help="Ghost observation radius (default: 5).")
    args = parser.parse_args()

    failures = 0
    for index in range(1, args.games + 1):
        code = run_game(args.seek, args.hide, args.max_steps,
                        args.pacman_obs, args.ghost_obs)
        print(f"game={index} seek={args.seek} hide={args.hide} return_code={code}")
        if code != 0:
            failures += 1

    print(f"summary games={args.games} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
