# Research Findings — Ghost (Hider) 24127192

## Paper 1: Carmel & Markovitch (1996) — Opponent Modeling in Multi-Agent Systems

**Key idea:** Model opponent as a DFA (finite automaton). Learn opponent's strategy from
observed input/output behavior. Predict their next move, play optimally against that model.
When prediction fails, update the model.

**Application to Ghost:**
- Model Pacman's chase pattern: "direct A* chase", "predictive chase", "random exploration"
- If Pacman does linear prediction (current_pos + velocity), Ghost can zigzag to break it
- Counter-strategy: when Pacman predicts next_position, move perpendicular instead

## Paper 2: Teng (2026) — Reverse Engineering Pac-Man Ghost AI

**Key idea:** Blinky's greedy chase appears strategic purely due to maze constraints.
Movement that seems "smart" is often just maze topology + simple rules.

**Application to Ghost:**
- Exploit maze topology better than Pacman can: precompute all corridors, dead-ends, loops
- Do NOT enter corridors where Pacman can reach the exit before Ghost
- Prefer loops where Ghost can cycle indefinitely if Pacman approaches

## Paper 3: Yannakakis & Hallam (2004) — Evolving Opponents for Interesting Games

**Key idea:** The most interesting opponents are NEITHER too easy NOR too hard:
- Interest I = γT + δS + εE[Hn]
  - T: difference between avg and max kill time (higher = more interesting)
  - S: variance in kill time (more variance = more interesting)
  - Hn: normalized entropy of cell visits (more diverse movement = more interesting)

**Application to Ghost:**
- Ghost should NOT always take the "mathematically optimal" escape — that makes behavior
  predictable and creates either "too easy to catch" or "too hard to catch" outcomes
- Add controlled randomness among top-N safe moves → increases entropy Hn
- Vary evasion strategy based on step_number: early game = conservative, late game = riskier
- Track cell visit frequency, prefer less-visited cells among equally-safe options

---

## Proposed Improvements for GhostAgent

### 1. Opponent Modeling + Prediction Breaking (from Carmel-Markovitch)
- Track Pacman's last-K positions and detect chase pattern
- If Pacman consistently predicts Ghost's position via linear extrapolation:
  - When Pacman is 3-5 tiles away, make a perpendicular move to break the prediction
- Maintain confidence score in opponent model; reset when behavior changes

### 2. Precomputed Maze Topology (from Teng)
- Precompute once in __init__:
  - Junction graph (nodes = junctions with ≥3 exits, edges = corridor length between them)
  - Dead-end depth map (how deep into a dead-end each cell is)
  - Loop zones (connected components with ≥2 paths between any two cells)
- Use these for O(1) lookup during step() instead of recomputing

### 3. Safe Flood Fill with Speed-Weighted Reachability
- Current: uses `gd.get(cell, ...)` for simple BFS distance
- Improve: Ghost reaches cell in `gd[cell]` steps, Pacman in `ceil(pd[cell]/2)` steps
- A cell is SAFE only if `gd[cell] + MARGIN < ceil(pd[cell]/2)`
- Prefer cells that are safe AND maximize Pacman's travel distance

### 4. Diversity-Driven Move Selection (from Yannakakis)
- Among top-3 safe moves, apply weighted random selection
- Weight = 0.5 × distance_score + 0.3 × cell_visit_entropy + 0.2 × random
- Track global cell visit counts (reset each game) — prefer less-visited cells
- This increases Hn (entropy), making Ghost less predictable

### 5. Dynamic Strategy Switching
- Phase 1 (step 1-50): Conservative — maximize distance, avoid all risks
- Phase 2 (step 51-150): Balanced — mix distance + entropy
- Phase 3 (step 151+): Survival — tight minimax, maximum depth
- Also switch if Pacman changes behavior (detected via opponent model)

### 6. Improved Anti-Oscillation
- Current: hard ban on 6 recent cells (can force bad moves)
- Improve: soft penalty proportional to recency, trade off against safety score
- Allow revisiting if the alternative is certain capture
