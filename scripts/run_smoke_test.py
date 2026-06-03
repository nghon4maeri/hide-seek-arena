from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import HideAgent, SeekAgent, get_action, step
from src.core.official_map import GHOST_START, OFFICIAL_MAP_GRID, OFFICIAL_MAP_HEIGHT, OFFICIAL_MAP_WIDTH, PACMAN_START
from src.core.simulator import LocalSimulator


def main() -> None:
    grid = [row[:] for row in OFFICIAL_MAP_GRID]
    state = {"map_state": grid, "my_position": PACMAN_START, "enemy_position": GHOST_START, "step_number": 0}
    hide = HideAgent().get_action(state)
    seek = SeekAgent().get_action(map_state=grid, my_position=GHOST_START, enemy_position=PACMAN_START, step_number=0)
    generic = get_action(map_state=grid, my_position=PACMAN_START, enemy_position=GHOST_START, step_number=0)
    compat = step(grid, PACMAN_START, GHOST_START, 0)
    valid = {"UP", "DOWN", "LEFT", "RIGHT", "STAY"}
    assert hide in valid
    assert seek in valid
    assert generic in valid
    assert compat in valid

    replay = LocalSimulator(max_steps=8).save_replay(ROOT / "replays" / "smoke_replay.json", debug=True)
    assert replay["frames"]
    print(
        json.dumps(
            {
                "map_size": [OFFICIAL_MAP_HEIGHT, OFFICIAL_MAP_WIDTH],
                "pacman_start": list(PACMAN_START),
                "ghost_start": list(GHOST_START),
                "hide_action": hide,
                "seek_action": seek,
                "replay_frames": len(replay["frames"]),
                "winner": replay["winner"],
            }
        )
    )


if __name__ == "__main__":
    main()
