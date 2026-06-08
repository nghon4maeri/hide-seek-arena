"""Quick workspace smoke test.

Runs a short arena match using the final team submission against the bundled
example submission. This does not grade gameplay quality.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PACMAN_ROOT = Path(__file__).resolve().parents[1]
ARENA = PACMAN_ROOT / "src" / "arena.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a short Pacman workspace smoke test.")
    parser.add_argument("--seek", default="team_submission", help="Pacman/seeker submission ID.")
    parser.add_argument("--hide", default="example_student", help="Ghost/hider submission ID.")
    parser.add_argument("--max-steps", type=int, default=5, help="Short smoke-test step limit.")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(ARENA),
        "--seek",
        args.seek,
        "--hide",
        args.hide,
        "--max-steps",
        str(args.max_steps),
        "--no-viz",
        "--submissions-dir",
        str(PACMAN_ROOT / "submissions"),
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(command, cwd=PACMAN_ROOT / "src", env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
