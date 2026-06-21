# Research Findings — Pacman (Seeker) 24127561

## Paper 1: Carmel & Markovitch (1996) — Opponent Modeling in Multi-Agent Systems

**Key idea:** Model opponent as a DFA (finite automaton). Learn opponent's strategy from
observed input/output behavior. Predict their next move, play optimally against that model.
When prediction fails, update the model.

**Application to Pacman:**
- Observe Ghost's moves over time, build a pattern predictor
- If Ghost always flees to distant cells, Pacman can predict that target and intercept
- When Ghost changes behavior (e.g. from strategic to tactical), detect the switch and adapt

## Paper 2: Teng (2026) — Reverse Engineering Pac-Man Ghost AI

**Key idea:** Blinky (red ghost) uses greedy Manhattan-distance minimization. The pursuit
appears strategic purely from maze constraints interacting with a simple local rule.

**Application to Pacman:**
- A* already beats greedy chase — but the insight is: maze topology does the heavy lifting
- Precompute corridor/chokepoint map — knowing where to intercept is more important than
  the chase algorithm itself
- Use interception: aim for the junction Ghost must pass through, not Ghost's current tile

## Paper 3: Yannakakis & Hallam (2004) — Evolving Opponents for Interesting Games

**Key idea:** Optimal predators make the game boring. Interesting opponents generate
appropriate challenge level + behavioral diversity + entropy in movement patterns.

**Application to Pacman (inverse):**
- Ghost will try to be unpredictable → don't chase linearly, use interception
- Ghost's safe-area heuristic gives hints about its next target — attack the safe area
- The measure I = γT + δS + εE[Hn] — understanding this helps predict Ghost's design goals

---

## Proposed Improvements for PacmanAgent

### 1. Opponent Modeling (from Carmel-Markovitch)
- Track Ghost's last-N positions as a movement history
- Detect patterns: "always runs to top-right", "prefers junctions", "oscillates in corridors"
- If Ghost is using A*-to-farthest-cell, predict which cell that is and intercept

### 2. Interception Planning (from Teng)
- Precompute junction graph of the maze (junctions + corridors connecting them)
- Instead of A* to Ghost's current position, A* to the junction Ghost is heading toward
- BFS distance map from Ghost: find cells where Pacman can arrive before Ghost

### 3. Trap Pressure (from Yannakakis — inverse application)
- For each candidate Pacman move, compute: how much does Ghost's "safe flood fill" shrink?
- Prefer moves that maximize trap pressure (reduce Ghost's escape options fastest)
- Combine with A*: when close, switch from chase to zone-control

### 4. Path Caching
- Cache the A* path; only replan when Ghost deviates significantly (>2 tiles from expected)
- Reuse cached path segments when possible

### 5. Exploration Strategy (Fog of War fallback)
- When enemy is unseen, target last-known-position (already implemented)
- Add: visit high-degree cells (junctions) first for maximum visibility coverage
- Mark visited regions and prefer unexplored areas
