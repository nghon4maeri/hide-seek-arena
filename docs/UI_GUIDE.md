# UI Guide

The visualizer is optional and is not imported by the tournament `agent.py`.

Headless smoke check:

```bash
python scripts/run_visualizer.py
```

Interactive mode:

```bash
python scripts/run_visualizer.py --interactive
```

Controls:

- `SPACE`: pause or resume.
- `N`: step one frame.
- `R`: restart replay.
- `1`: toggle BFS explored nodes.
- `2`: toggle A* path.
- `3`: toggle danger, safe, and dead-end overlays.
- `4`: toggle minimax candidate display.
- `ESC`: quit.

Replay files are JSON. The smoke test writes `replays/smoke_replay.json`.

