from __future__ import annotations

import json
from pathlib import Path

from .visualizer import ArenaVisualizer


def load_replay(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def view_replay(path: str | Path, interactive: bool = True) -> str:
    visualizer = ArenaVisualizer(load_replay(path))
    if interactive:
        visualizer.run()
    return visualizer.render_headless_summary()

