# Handoff — Ghost-192++ Project

## Current Best Ghost

Ghost 192 (`24127192`) remains the strongest ghost agent, with average survival of **13.5 steps** against Pacman 561.

Ghost 457 (`24127457`) averages **12.8 steps** against Pacman 561.

**Survival gain: -0.7 steps** — Ghost 192 is still better.

## Best Modification Attempted

Stronger closeness penalties (`md <= 4: -(5-md)*12.0`) + loop proximity bonus. Achieved +0.7 in 15-game sample but regressed to -0.7 in 25-game sample.

## Accepted Changes

None. All changes failed to produce consistent survival improvement.

## Current Bottleneck

Ghost 457's evaluation function is fundamentally simpler than Ghost 192's:

| Component | Ghost 192 | Ghost 457 |
|-----------|-----------|-----------|
| Distance penalty | Exponential (-22k at d=1) | Linear (-12 at md=4) |
| ETA penalty | Exponential (-36k at eta=1) | None |
| Safe area | Capture-ETA floodfill | BFS-based floodfill |
| Pacman prediction | Multi-hypothesis (7 models + US-L*) | Single (current position) |
| Anti-velocity | Explicit scoring (+620/+760) | Weak leaf penalty |
| Pacman influence | Decaying position history | None |
| Search depth | 2-18 (up to 17 levels) | 6-8 |
| Move ordering | Cell safety + anti-velocity | BFS distance + topology |

## Next Experiment

Port Ghost 192's `_cell_safety_score` to Ghost 457's search evaluator. This is the single most impactful change because it drives both move ordering and leaf evaluation.

Key numbers to port:
- `d <= 0: -100000 * weight`
- `d == 1: -22000 * weight`  
- `d == 2: -4500 * weight`
- `eta == 0: -120000 * weight`
- `eta == 1: -36000 * weight`
- `eta == 2: -9000 * weight`
- `min(worst, 18) * 900`
- `min(worst_eta, 10) * 1200`
- `safe_area: min(area, 80) * 125`
- `area < 10: -(10 - area) * 760`
- core: +2400, loop: +1600, junction: +650, dead_end: -6500
- degree: * 260
- pacman_influence: decaying from last 12 positions

## Required Files

```
pacman/submissions/24127457/agent.py      — GhostAgent (search-first ID alpha-beta)
pacman/submissions/24127457/search.py     — GhostSearchEvaluator + fast leaf eval
pacman/submissions/24127457/topology.py   — Topology with loop detection
pacman/submissions/24127457/heuristic.py  — Utility functions
```

## To Continue

1. Add `_capture_eta` computation (A* + speed-2 simulation) to GhostSearchEvaluator
2. Replace `leaf_score` with a port of Ghost 192's `_cell_safety_score`
3. Add multi-hypothesis Pacman position prediction
4. Add US-L* style opponent learning
5. Benchmark: `python bench.py` from `pacman/submissions/24127457/`

## Benchmark Command

```bash
cd pacman/src
python arena.py --seek 24127561 --hide 24127457 --no-viz --max-steps 200 --start-mode stochastic
```
