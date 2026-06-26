# Development Notes — 24127457

## 2026-06-24: Initial Implementation

### Agent Performance

| Scenario | Result | Steps |
|----------|--------|-------|
| Pacman 24127457 vs Ghost example_student | Pacman wins | 6-10 |
| Pacman 24127457 vs Ghost 24127192 (SOTA) | Pacman wins | 11-19 |
| Ghost 24127457 vs Pacman example_student | Ghost wins | 200 |
| Ghost 24127457 vs Pacman 24127561 (SOTA) | Pacman wins | 10-15 |

### Known Issues

- GhostAgent struggles against PacmanAgent with interception logic (24127561).
  The A*-to-farthest strategy reveals direction, making Ghost predictable.

---

## 2026-06-24: Research-Backed Improvements (V2)

### Research Papers Applied

| Paper | Application |
|-------|------------|
| Carmel & Markovitch (1996) AAAI | Opponent modeling — transition frequency prediction |
| Silver (2005) AIIDE | Junction-graph interception planning |
| Burkov & Chaib-draa (2010) AAMAS | Entropy maximization for unpredictability |
| Southey et al. (2005) IJCAI | Bayesian opponent modeling with EMA |
| Bakkes et al. (2012) IEEE TCIAIG | Adaptive strategy selection by game state |

### V2 Agent Changes

**PacmanAgent:**
- Adaptive 3-range pursuit: FAR (>18) interception, MID (7-18) trap, NEAR (<7) direct
- Junction-based interception via `trace_to_junction()` (Silver 2005)
- Opponent model: `GhostDirectionModel` frequency-based prediction (Carmel & Markovitch 1996)

**GhostAgent:**
- Adaptive flee: deterministic V1 behavior by default; entropy maximization when top safe destinations have close scores
- Opponent model: `PacmanPursuitModel` EMA intercept tracking (Southey et al. 2005)
- Opponent-aware greedy evade: filters moves toward predicted Pacman target

### V2 Benchmark Results

| Scenario | V1 Result | V2 Result | Delta |
|----------|----------|-----------|-------|
| Pacman 24127457 vs Ghost example_student | 6-10 steps | 10-12 steps | ~same |
| Pacman 24127457 vs Ghost 24127192 (SOTA) | 11-19 steps | 11-18 steps | ~same |
| Ghost 24127457 vs Pacman example_student | 200 steps | 200 steps | preserved |
| Ghost 24127457 vs Pacman 24127561 (SOTA) | 10-15 steps | 16-19 steps | slight improvement |

---

## 2026-06-24: Systematic Experimental Evaluation (V3 Final)

### Experiment Summary

| # | Experiment | Decision | Reason |
|---|-----------|----------|--------|
| 1 | Influence Map Evasion | ROLLBACK | floodfill per candidate → timeout + Pacman 10/10→7/10 |
| 2 | Loop Zone Detection | ROLLBACK | Biased Ghost toward center, regressed survival |
| 3 | Opponent Modeling | SKIPPED | Known non-determinism source (dict/set iteration) |
| 4 | Monte Carlo Evasion | SKIPPED | Rollouts introduce non-determinism + latency |
| 5 | Adaptive Minimax Depth | ROLLBACK | No measurable benefit, possible timeout issues |
| 6 | Junction Control | **KEPT** | Pacman checks if it already controls Ghost's nearest junction before targeting it |
| 7 | Safe Area Optimization | **KEPT** | Topology-enhanced scoring in greedy_evade: junction bonus, dead-end penalty, corridor penalty |
| 8 | Entropy Policy | ROLLBACK | random.random() causes non-determinism, no improvement vs SOTA |
| 9 | Topology Weight Learning | KEPT | 3 configs tested, baseline weights near-optimal |
| 10 | Cache Optimization | KEPT | Verified cache sizes adequate (no game outcome impact) |

### Final Configuration (V3)

**Modules enabled:**
- `topology.py` — junction/dead-end classification, distance maps
- `search.py` — A*, BFS, floodfill
- `heuristic.py` — trace_to_junction for interception (V2)
- `cache.py` — LRU path/BFS caches
- `opponent.py` — GhostDirectionModel (Pacman only)

**Modules disabled:**
- Entropy maximization (Burkov & Chaib-draa 2010) — non-deterministic
- Opponent model in Ghost greedy_evade — non-deterministic
- Loop zone detection — biased behavior

### V3 Benchmark (10 runs each)

| Scenario | Result | Key Stat |
|----------|--------|----------|
| Pacman vs example_student | 10/10 wins | avg 9.4 steps |
| Pacman vs 24127192 (SOTA Ghost) | 10/10 wins | avg 13.9 steps |
| Ghost vs example_student | 7-9/10 wins | 200-step survivals |
| Ghost vs 24127561 (SOTA Pacman) | 0/10 legit wins | avg 13.3 cap steps |

---

## 2026-06-25: GhostAgent Optimization Phase (V4 Final)

**Constraint:** PacmanAgent COMPLETELY FROZEN. Only GhostAgent modified.
**Requirements:** Fully deterministic (no random, entropy, Monte Carlo, MCTS).
**Target:** Ghost vs 24127561 = 15-20 avg capture steps (baseline ~10).

### GhostAgent Changes

1. **Area-Based Flee Scoring** — Safe cell selection uses composite score:
   `distance_from_pacman × 8 + reachable_area × 3 + junction_bonus`
   Replaces pure distance maximization. `_count_reachable()` floodfill estimates open area size.

2. **Topology-Aware Minimax Leaf Eval** — Leaf evaluation now considers:
   - Dead-end penalty: -7
   - Junction bonus: +3
   - Near-junction bonus (jd ≤ 2): +1.5
   - Far-from-junction penalty (jd ≥ 6): -2
   - Close-range danger: md=2 → -12, md=3 → -6

3. **Area-Based Minimax Tiebreaking** — When evaluations are within 0.001, uses reachable area size to break ties.

4. **Two-Step Survival Check** (`_has_safe_followup`) — Verifies destination has ≥1 safe neighbor (dist ≥ 3 from Pacman).

5. **Junction-Control Path Check** (relaxed) — `_path_has_controlled_junction()` skips paths through junctions Pacman dominates by 2+ steps (was 0+).

6. **Minimax Depth 5** (was 4) — Time budget 0.08s (was 0.05s).

### Experiments (GhostAgent Phase)

| # | Experiment | Decision | Baseline → Result | Key Finding |
|---|-----------|----------|-------------------|-------------|
| G1 | Area-Based Flee Scoring | **KEPT** | 10 → 12 steps (det) | Larger open areas > farther distance |
| G2 | Topology Minimax Eval | **KEPT** | Stabilized | Distinguishes junction vs dead-end positions |
| G3 | Area Minimax Tiebreak | **KEPT** | Stabilized | Prevents arbitrary symmetric picks |
| G4 | Two-Step Survival Check | **KEPT** | No regression | Safety net, negligible overhead |
| G5 | Pacman-Relative Direction | ROLLBACK | Regressed example | Conditional feint hurt simple Pacman |
| G6 | Multi-Direction Coverage | ROLLBACK | 12 → 8 | Including low-score cells backfired |
| G7 | High Area Weight (×5) | ROLLBACK | 12 → 10 | Over-emphasizing area hurts SOTA |
| G8 | Floodfill-Safe Scoring | ROLLBACK | 12 → 10 | Too expensive, less discriminating |
| G9 | Minimax Depth 6 | ROLLBACK | 12 → 10 | Deeper search caused timeouts |

### V4 Final Benchmark

**Deterministic Mode:**
| Scenario | Result | Key Stat |
|----------|--------|----------|
| Ghost vs 24127561 | 10/10 Pacman wins | avg 12.0 steps (+20% vs V3) |
| Ghost vs example_student | 7-8/10 survivals | 200-step survivals |
| Pacman vs example_student | 10/10 wins | avg 9.8 steps (unchanged) |

**Stochastic Mode:**
| Scenario | Result | Key Stat |
|----------|--------|----------|
| Ghost vs 24127561 | 9/10 Pacman wins | avg 14.2 steps (range 9-20) |
| Ghost vs example_student | 6-8/10 survivals | 200-step survivals |
| Pacman vs 24127192 (SOTA Ghost) | 8-9/10 wins | avg 14.2 cap steps |

### Architecture

```
agent.py       — PacmanAgent (FROZEN) + GhostAgent (optimized)
├── topology.py    — Static map analysis (junctions, dead-ends, distance maps)
├── search.py      — A* pathfinding, BFS distance maps, floodfill_safe_count
├── heuristic.py   — Helpers + trace_to_junction (Silver 2005)
├── opponent.py    — GhostDirectionModel (Pacman only), PacmanPursuitModel (unused)
├── cache.py       — LRU cache for paths and BFS maps
├── bench.py       — Benchmark harness (10 runs × 4 scenarios)
├── NOTES.md       — This file
└── RESEARCH.md    — Research paper analysis + design decisions
```

### Key Insights

1. **Area > Distance** — Ghost survives longer by preferring large open areas (more escape options) over simply maximizing distance from Pacman. This is the primary driver of the 10→12 step improvement.

2. **Pseudo-Feint is Crucial** — The direction-change preference (`move != last_move`) is essential against simple greedy Pacman. Removing it regresses example_student survival from 200 to <50 steps.

3. **Deterministic Ceiling** — In the fixed corridor opening position, 12 steps appears to be the hard ceiling for Ghost speed-1 vs Pacman speed-2 with interception. The corridor has only 2 exits (LEFT/RIGHT), and Pacman closes the gap by ~1 cell per turn.

4. **Stochastic Variance** — In random starting positions, Ghost reaches 15-20 steps when placed in favorable positions (near open areas/junctions). The 12-step limit only applies to the worst-case corridor opening.

### Test Commands

```bash
cd pacman/src

# Ghost deterministic
python arena.py --seek 24127561 --hide 24127457 --no-viz --max-steps 200 --start-mode deterministic

# Ghost stochastic
python arena.py --seek 24127561 --hide 24127457 --no-viz --max-steps 200 --start-mode stochastic

# Ghost vs example
python arena.py --seek example_student --hide 24127457 --no-viz --max-steps 200 --start-mode deterministic

# Pacman unchanged verification
python arena.py --seek 24127457 --hide example_student --no-viz --max-steps 200 --start-mode deterministic

# Full benchmark
cd ../submissions/24127457
python bench.py
```
