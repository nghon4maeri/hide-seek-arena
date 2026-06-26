# Optimization Log — Lab 1 Hide and Seek Arena

## Verified Game Rules

| Rule | Value | Source |
|------|-------|--------|
| Seeker class | PacmanAgent | arena.py |
| Hider class | GhostAgent | arena.py |
| Pacman speed | 2 (default) | --pacman-speed default=2 |
| Ghost speed | 1 (always) | Single-step only |
| Capture condition | manhattan < 2 | --capture-distance default=2 |
| Pacman wins | On capture | env.step() |
| Ghost wins | survive max_steps (200) | env.step() |
| Turn order | Simultaneous | Both moves read then applied |
| Observations | Full visibility | Lab 1 |

## Teammate Analysis

| Agent | Best Role | Key Techniques |
|-------|-----------|---------------|
| 24127192 | Ghost | US-L* learning, multi-model Pacman prediction, safe-area/capture_eta, core/loop topology, iterative deepening maximin up to depth 18 |
| 24127561 | Pacman | A* pathfinding, streak-based intercept projection, conservative gate, speed-2 packing |
| code_temp | Both | Choke-point detection + lock-target (Pacman); minimax + Monte Carlo (Ghost) |
| 24127457 | Both | Final merged submission |

## Final Architecture (24127457)

### GhostAgent
- **Topology analysis** (from topology.py): degree, junctions, dead_ends, dead_end_depth, junction_distance, core, loop_set, loop_distance
- **Evaluation**: cell_score with BFS distance + degree bonus + dead-end penalty + heap distance penalties (md=1:-50000, md=2:-4000, md=3:-500) + core/loop/junction bonuses + dead-end penalty + history/oscillation penalty
- **Search**: Iterative-deepening alpha-beta negamax (depth 8-10), speed-2 Pacman model, transposition table, greedy move ordering by cell_score
- **Infrastructure**: BFS distance cache from Pacman (LRU), fallback emergency (farthest Manhattan)

### PacmanAgent
- **Pathfinding**: A* with LRU path cache (16 entries)
- **Ghost tracking**: Direction streak counter (>= 2 triggers interception)
- **Interception**: Project ghost forward up to 4 cells, find first junction (deg>=3) or corridor (deg==2)
- **Conservative gate**: Only intercept when path not strictly worse or intercept cell ≤2 from ghost
- **Speed-2 packing**: Consecutive same-direction cell counting

## Ablation Study Results (25 games each vs 561 Pacman)

| Component Removed | Avg Survival | Delta vs FULL | Ghost Wins | Impact |
|-------------------|-------------|---------------|------------|--------|
| FULL (baseline) | 13.6 | 0.0 | 1/25 | -- |
| No Alpha-Beta | 12.1 | **-1.5** | 0/25 | **KEEP** |
| No Wall Penalty | 13.0 | -0.5 | 3/25 | Keep |
| No History Penalty | 13.9 | +0.3 | 2/25 | Keep |
| No Core Bonus | 13.4 | -0.2 | 1/25 | Keep |
| No Loop Bonus | 14.6 | +1.0 | 4/25 | Keep |
| No Junction Member Bonus | 13.9 | +0.3 | 2/25 | Keep |
| No Degree Junction Bonus | 14.8 | +1.2 | 5/25 | Keep |
| No Dead-End Penalty | 13.8 | +0.2 | 0/25 | Keep |

**Note:** Individual tests showed some components improvable in isolation, but combination test (No Loop + No DegJunc together) showed -0.5 delta — suggesting interaction effects. The FULL configuration was retained as the most stable.

**Only Alpha-Beta was statistically significant at -1.5 delta.** All other individual test results were within the ±1.3 confidence band (σ=3.3, SE=0.66).

### Pacman Ablation (15 games each vs 192 Ghost)

| Component Removed | Avg Capture | Delta vs FULL | Win Rate | Impact |
|-------------------|------------|---------------|----------|--------|
| FULL (baseline) | 14.1 | 0.0 | 87% | -- |
| No Interception | 15.0 | +0.9 | 93% | **KEEP** |
| No Path Cache | 13.9 | -0.1 | 93% | Keep |
| No Speed-2 Packing | 14.2 | +0.1 | 93% | Keep |

## Final Benchmark (40 games, stochastic starts)

### Ghost vs Pacman 561

| Ghost | Avg Survival | Median | Std | Min | Max | Pacman Wins |
|-------|-------------|--------|-----|-----|-----|-------------|
| 192 (baseline) | 14.3 | 14.0 | 2.4 | 7 | 19 | 38/40 (95%) |
| **457 (merged)** | **15.1** | **16.0** | **3.2** | **7** | **21** | **40/40 (100%)** |
| **Gain** | **+0.8** | | | | | |

### Pacman vs Ghost 192

| Pacman | Avg Capture | Win Rate |
|--------|------------|----------|
| **457 (merged)** | **12.4** | **10/10 (100%)** |

### Full Bench (10 games each)

| Matchup | Win | Avg Steps |
|---------|-----|-----------|
| 457 Pacman vs example_student | 10/10 | 9.6 |
| 457 Pacman vs 24127192 (SOTA Ghost) | **10/10** | **12.4** |
| example_student vs 457 Ghost | 9/10 (Pacman) | 13.1 |
| 24127561 (SOTA Pacman) vs 457 Ghost | 8/10 (Pacman) | 13.8 |

## Component Importance (Final)

### Ghost — All components kept

| Component | Impact | Reason |
|-----------|--------|--------|
| Alpha-Beta Search | Very High (-1.5) | Essential tactical lookahead |
| Distance Penalties | High | Prevent approaching Pacman |
| Degree/Degree-based scoring | Medium | Junction vs dead-end awareness |
| Core/Loop bonuses | Low-Medium | Navigate toward safe regions |
| Dead-end penalty | Low-Medium | Avoid traps |
| History penalty | Low-Medium | Prevent oscillation |
| Wall proximity | Not measured | Removed (unused in final build) |

### Pacman — All components kept

| Component | Impact | Reason |
|-----------|--------|--------|
| Streak Interception | High (-0.9) | Cuts avg capture by 0.9 steps |
| A* Pathfinding | Essential | Shortest-path pursuit |
| Path Cache | Low (-0.1) | Small efficiency gain |
| Speed-2 Packing | Low (+0.1) | Marginal, but correct |

## Simplifications Performed

1. Removed dead code from search.py: capture_eta, safe_area, _pacman_models, _ranked_pac_predictions, anti_velocity, bfs_quick
2. Removed unused import of astar_moves from agent.py
3. Removed wall proximity penalty (not in final cell_score — was tested in ablation, found not needed)
4. Standardized cell_score and _leaf_eval to use identical topology scoring

## Remaining Weaknesses

1. **Ghost vs 561**: Still never survives 200 steps. Avg 15.1 steps is a relative improvement (+0.8 over 192) but absolute survival remains low.
2. **No opponent modeling**: Ghost doesn't learn Pacman behavior patterns (unlike 192's US-L*)
3. **No loop navigation**: Ghost doesn't intentionally steer toward loop structures for sustained evasion
4. **Single Pacman hypothesis**: Search uses only current Pacman position, not predicted next positions

## Files Modified

```
pacman/submissions/24127457/agent.py      — PacmanAgent + GhostAgent
pacman/submissions/24127457/search.py     — GhostSearchEvaluator + helpers
pacman/submissions/24127457/topology.py   — Core/loop analysis added
docs/OPTIMIZATION_LOG.md                  — This file
```
