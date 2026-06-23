"""Placeholder benchmark runner for team submissions.

TODO:
- Parse arena output into win/loss counts.
- Record average capture or survival steps.
- Append results to docs/benchmark_report.md after leader review.

This script intentionally does not modify agent code.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PACMAN_ROOT = Path(__file__).resolve().parents[1]
ARENA = PACMAN_ROOT / "src" / "arena.py"


def run_game(seek: str, hide: str, max_steps: int) -> int:
    command = [
        sys.executable,
        str(ARENA),
        "--seek",
        seek,
        "--hide",
        hide,
        "--max-steps",
        str(max_steps),
        "--no-viz",
        "--submissions-dir",
        str(PACMAN_ROOT / "submissions"),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(command, cwd=PACMAN_ROOT / "src", env=env, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight benchmark matches.")
    parser.add_argument("--seek", default="team_submission", help="Pacman/seeker submission ID.")
    parser.add_argument("--hide", default="example_student", help="Ghost/hider submission ID.")
    parser.add_argument("--games", type=int, default=1, help="Number of repeated games.")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum steps per game.")
    args = parser.parse_args()

    failures = 0
    for index in range(1, args.games + 1):
        code = run_game(args.seek, args.hide, args.max_steps)
        print(f"game={index} seek={args.seek} hide={args.hide} return_code={code}")
        if code != 0:
            failures += 1

    print(f"summary games={args.games} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
