# 24127457 — Leader Sandbox Architecture

## Module Structure

```
24127457/
├── agent.py          # PacmanAgent + GhostAgent (entry point)
├── search.py         # A*, BFS, flood fill algorithms
├── topology.py       # Static map analysis
├── heuristic.py      # Scoring functions and distance utilities
├── cache.py          # LRU path and distance cache
├── RESEARCH.md       # Framework + teammate research
├── README.md         # This file
└── NOTES.md          # Development notes
```

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                      agent.py                            │
│  ┌─────────────────┐   ┌─────────────────┐              │
│  │  PacmanAgent    │   │  GhostAgent     │              │
│  │                 │   │                 │              │
│  │  step()         │   │  step()         │              │
│  │   ├─ cache      │   │   ├─ topology   │              │
│  │   ├─ search     │   │   ├─ search     │              │
│  │   └─ heuristic  │   │   └─ heuristic  │              │
│  └────────┬────────┘   └────────┬────────┘              │
│           │                     │                        │
│           ▼                     ▼                        │
│  ┌─────────────────────────────────────────┐            │
│  │          Shared Modules                  │            │
│  │  search.py │ topology.py │ heuristic.py │ cache.py   │
│  └─────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────┘
```

## PacmanAgent Design

### Decision Pipeline

```
step(map_state, my_position, enemy_position, step_number)
    │
    ├─ 1. Update memory (last seen enemy, visited set)
    │
    ├─ 2. Determine target
    │   ├─ enemy visible      → predict next position (simple direction projection)
    │   ├─ enemy not visible  → use last_known_enemy_pos
    │   └─ no last known      → explore (maximize unvisited area)
    │
    ├─ 3. Check path cache
    │   ├─ cache hit + valid  → reuse cached path
    │   └─ cache miss         → compute A* to target
    │
    ├─ 4. Pack multi-step moves
    │   └─ Collapse consecutive same-direction steps up to pacman_speed
    │
    └─ 5. Validate and return (Move, steps)
```

### Algorithms

| Component | Algorithm | Complexity |
|-----------|-----------|------------|
| Pathfinding | A\* with Manhattan heuristic | O(N log N), N ≈ 250 |
| Target prediction | 1-step direction extrapolation | O(1) |
| Path caching | LRU cache, invalidated on target shift > 2 | O(1) lookup |
| Multi-step packing | Greedy forward scan | O(speed) |
| Exploration | Maximize unvisited + junction bias | O(neighbors) |

### Key Design Choices

- **Simple prediction** over complex interception: predict Ghost pos + direction, not junction projection
- **Path cache with soft invalidation**: reuse path until Ghost deviates > 2 cells
- **Speed-aware packing**: walk path forward, pack consecutive same-direction cells
- **No minimax in Pacman**: keeps step() fast, avoids complexity

## GhostAgent Design

### Decision Pipeline

```
step(map_state, my_position, enemy_position, step_number)
    │
    ├─ 1. One-time init: topology analysis (junctions, dead-ends)
    │
    ├─ 2. Compute BFS distance map from Pacman
    │
    ├─ 3. Determine game phase
    │   ├─ step ≤ 60   → early (conservative)
    │   ├─ step ≤ 140  → mid (balanced)
    │   └─ step > 140  → late (survival)
    │
    ├─ 4. Filter candidates
    │   ├─ Remove moves into capture range (dist < 2)
    │   └─ Apply soft anti-oscillation penalty
    │
    ├─ 5. Score each candidate
    │   ├─ BFS distance from Pacman (weighted by phase)
    │   ├─ Floodfill safe area count
    │   ├─ Junction proximity bonus
    │   ├─ Dead-end depth penalty
    │   └─ Corridor risk penalty
    │
    ├─ 6. If Pacman close (dist ≤ 5)
    │   └─ Shallow minimax (depth 4) to tie-break top candidates
    │
    └─ 7. Return best Move
```

### Algorithms

| Component | Algorithm | Complexity |
|-----------|-----------|------------|
| Distance map | BFS from Pacman (full map) | O(N), N ≈ 250 |
| Safety scoring | Floodfill from candidate, compare arrival times | O(N) per candidate |
| Topology | One-time static analysis in `__init__` | O(N), once |
| Phase strategy | Weighted linear score | O(1) |
| Close-range evasion | Minimax alpha-beta, depth 4 | O(b^d), b≈3, d=4 → 81 leaves |

### Phase Weights

| Weight | Early (≤60) | Mid (61-140) | Late (141-200) |
|--------|------------|--------------|----------------|
| Distance from Pacman | 10.0 | 8.0 | 12.0 |
| Safe floodfill area | 0.5 | 0.3 | 0.4 |
| Junction proximity | 5.0 | 4.0 | 3.0 |
| Dead-end penalty | -80 | -60 | -100 |
| Corridor risk penalty | -15 | -10 | -20 |

### Key Design Choices

- **Soft anti-oscillation** over hard ban: penalize recent positions, don't exclude them
- **Shallow minimax** only when cornered: avoids timeout risk
- **Phase-based weights**: strategy adapts as game progresses
- **No opponent modeling**: simpler, less fragile than DFA learning

## Shared Modules

### search.py

```python
astar(map_state, start, goal) → list of (row, col) positions
bfs_distance(map_state, start, max_dist=999) → dict {(r,c): distance}
bfs_path(map_state, start, goal) → list of Move
floodfill_count(map_state, start, max_depth=15) → int
```

### topology.py

```python
analyze_map(map_state) → {
    "junctions": set of (r,c),
    "dead_ends": set of (r,c),
    "dead_end_depth": dict {(r,c): depth},
    "junction_distance": dict {(r,c): dist_to_nearest_junction},
    "degree": dict {(r,c): exit_count},
}
```

### heuristic.py

```python
manhattan(a, b) → int
cell_exits(pos, map_state) → int
is_valid(pos, map_state) → bool
get_neighbors(pos, map_state) → [(pos, move), ...]
score_ghost_position(pos, pacman_pos, pd, ms, topo, phase) → float
score_pacman_position(pos, target, ms) → float
```

### cache.py

```python
class LRUCache:
    get(key) → value or None
    put(key, value)
    clear()
```

## Complexity Budget

| Operation | Complexity | Per Step? | Within 1s? |
|-----------|-----------|-----------|------------|
| A\* pathfinding | O(N log N), N≈250 | Pacman only | Yes (< 10ms) |
| BFS distance map | O(N), N≈250 | Ghost only | Yes (< 5ms) |
| Topology analysis | O(N), N≈250 | Once in init | Yes (init only) |
| Floodfill per candidate | O(N), N≈250 | Ghost × 2-4 | Yes (< 20ms) |
| Minimax depth 4 | O(3^4) = 81 leaves | Ghost (rare) | Yes (< 50ms) |
| Multi-step packing | O(speed), max 2 | Pacman only | Yes (< 1ms) |

**Total per step: < 100ms**, well within 1.0s budget.

## TEMPLATE Compatibility Checklist

- [x] Same import structure (SRC_PATH, Move, BasePacmanAgent, BaseGhostAgent)
- [x] Class names: `PacmanAgent`, `GhostAgent`
- [x] Method signatures: `__init__(self, **kwargs)`, `step(self, map_state, my_position, enemy_position, step_number)`
- [x] Pacman returns: `Move` enum or `(Move, steps)` tuple
- [x] Ghost returns: `Move` enum only
- [x] No framework modifications
- [x] No external dependencies beyond numpy + stdlib
- [x] All helper modules in `24127457/`
