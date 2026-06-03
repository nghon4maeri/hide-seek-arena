from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"
RUNTIME_FILES = {
    "src": ["__init__.py"],
    "src/agents": ["__init__.py", "base_agent.py", "hide_agent.py", "seek_agent.py"],
    "src/core": ["__init__.py", "constants.py", "game_state.py", "map_utils.py", "movement.py", "types.py"],
    "src/search": ["__init__.py", "alpha_beta.py", "astar.py", "bfs.py", "flood_fill.py", "minimax.py"],
    "src/evaluation": ["__init__.py", "features.py", "hide_eval.py", "seek_eval.py"],
}


def copy_runtime_file(relative_dir: str, filename: str) -> None:
    src = ROOT / relative_dir / filename
    dst_dir = OUT / relative_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / filename)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    shutil.copy2(ROOT / "agent.py", OUT / "agent.py")
    for relative_dir, filenames in RUNTIME_FILES.items():
        for filename in filenames:
            copy_runtime_file(relative_dir, filename)
    print(f"created {OUT}")


if __name__ == "__main__":
    main()
