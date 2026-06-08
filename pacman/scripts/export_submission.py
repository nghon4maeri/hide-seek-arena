"""Export a submission folder into a zip-ready directory.

This helper copies `submissions/team_submission/` by default into
`dist/team_submission/`. It does not upload or submit anything.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PACMAN_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy a Pacman submission into a zip-ready folder.")
    parser.add_argument("student_id", nargs="?", default="team_submission", help="Submission folder to export.")
    parser.add_argument("--output", default="dist/team_submission", help="Output folder under pacman/.")
    parser.add_argument("--force", action="store_true", help="Overwrite output folder if it exists.")
    args = parser.parse_args()

    source = PACMAN_ROOT / "submissions" / args.student_id
    output = PACMAN_ROOT / args.output

    if not source.is_dir():
        raise SystemExit(f"submission folder not found: {source}")
    if not (source / "agent.py").is_file():
        raise SystemExit(f"submission is missing agent.py: {source}")
    if output.exists():
        if not args.force:
            raise SystemExit(f"output already exists: {output}; use --force to overwrite")
        shutil.rmtree(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)
    print(f"copied {source} -> {output}")
    print("zip the exported folder according to the course submission instructions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
