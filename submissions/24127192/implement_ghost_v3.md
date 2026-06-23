# Ghost Agent v3: Advanced Counter-A* Survival Strategy

## Problem Analysis

The current ghost agent survives only **8 steps** consistently against Tony's A* Pacman (24127561). The core issues:

### Why the Ghost Fails

1. **A* with velocity prediction is devastating**: Tony's Pacman uses `A*(current_pos → predicted_ghost_future_pos)` with speed-2 straight-line moves. This means Pacman effectively moves 2× faster on corridors.

2. **Current strategy is too reactive**: The ghost evaluates moves based on BFS distance but doesn't account for **time-to-intercept** — Pacman at speed-2 closes distance much faster than the ghost's scoring anticipates.

3. **Default map geometry is hostile**: Ghost starts at `(9,10)`, Pacman at `(15,10)` — only 6 Manhattan apart. With speed-2, Pacman can reach Ghost in ~3 turns on straight paths. The ghost needs to immediately flee to a structurally advantageous area.

4. **Kiting scores are dominated by simple BFS distance**: The `p_dist * 1000` term overwhelms nuanced structural bonuses, leading to greedy "run away from Pacman" behavior that gets cornered.

5. **US-L* DFA learning is useless in early game**: DFA has no data in first few steps, so fallback to A* simulation provides no tactical advantage.

### Tony's A* Pacman Strategy (from [agent.py](file:///Users/mac/Documents/AI/Hide-Seek/pacman/submissions/24127561/agent.py))

```
1. Predict ghost's future position: ghost_pos + (ghost_pos - last_ghost_pos)
2. A* pathfind to predicted position
3. Take speed-2 steps along A* path (2 cells if straight line)
```

**Key weakness**: Pacman uses **velocity extrapolation** to predict ghost. If ghost changes direction frequently, prediction is wrong and Pacman overshoots.

## Proposed Strategy: **Hamiltonian Cycle Runner + Influence Map + Temporal Safety Analysis**

### Core Idea

Instead of greedily maximizing BFS distance (which A* easily counters), the ghost should:

1. **Run on pre-computed safe circuits** — cyclic paths through the 2-core where ghost always has ≥2 exits, making it impossible for a single pursuer to corner it.
2. **Use Temporal Safety Analysis (TSA)** — compute "time-to-intercept" for each candidate move considering Pacman's speed-2 advantage, not just BFS distance.
3. **Exploit Pacman's velocity prediction** — deliberately change direction at **corners/junctions** to make Pacman's velocity extrapolation predict the wrong target.
4. **Influence Map anti-pursuit** — maintain a decaying heat map of Pacman's recent positions/predicted trajectories to avoid running into Pacman's "sweep" pattern.
5. **Voronoi territory control** — choose moves that maximize the ghost's "safe territory" (cells closer to ghost than to Pacman at speed-2).

### Algorithm Details

#### 1. Temporal Safety Analysis (TSA)
For each candidate move `m` to position `nxt`:
```
ghost_time_to_pos = 1  (ghost always takes 1 step)
pacman_effective_dist = BFS(pacman, nxt)
pacman_time_to_pos = pacman_effective_dist / effective_speed(path)
  where effective_speed = 2 for straight segments, 1 for corners/turns
safety_window = pacman_time_to_pos - ghost_time_to_pos
```
A position is only safe if `safety_window ≥ 2` (need margin for ghost's next escape).

#### 2. Direction-Change Exploitation (Anti-Velocity-Prediction)
Tony predicts `ghost_future = ghost + velocity`. If ghost changes direction at junctions:
- Pacman's target is wrong
- A* path to wrong target wastes Pacman's speed-2 advantage
- Ghost gains 1-2 extra turns of survival

Strategy: Prefer moves that **change direction** from previous step, especially at junctions.

#### 3. Safe Circuit Running
Pre-compute the largest Hamiltonian-like circuit through the 2-core. Ghost runs along this circuit, always moving away from the nearest Pacman approach vector. The circuit guarantees:
- Never entering dead ends
- Always having ≥ 2 escape routes
- Predictable path for ghost but hard for A* to cut off

#### 4. Influence Map
Track Pacman's trajectory and create a heat map:
```
influence[cell] = sum(decay^t for each past Pacman position within distance d)
```
Ghost avoids high-influence areas.

#### 5. Multi-step Lookahead with Pacman Speed Model
Instead of simple minimax, use a **game tree search** that accurately models:
- Pacman moves 2 on straights, 1 on turns
- Ghost moves 1 per turn
- Pacman uses velocity prediction for targeting

## Proposed Changes

### [MODIFY] [agent.py](file:///Users/mac/Documents/AI/Hide-Seek/pacman/submissions/24127192/agent.py)

Major restructuring of the `GhostAgent` class:

1. **Replace kiting_and_baiting with TSA-based escape** — new `_temporal_escape()` method
2. **Add Influence Map** — new `_influence_map` tracking and update
3. **Add Direction-Change exploitation** — new `_anti_velocity_score()` 
4. **Improve circuit running** — better loop selection and directional running
5. **Fix minimax to model speed-2 accurately** — Pacman gets 2 moves per ghost move in game tree
6. **Add opening book** — pre-computed optimal first 3-5 moves for the default map start position
7. **Add Voronoi territory scoring** — accurate speed-2 adjusted territory calculation

## Verification Plan

### Automated Tests
```bash
cd /Users/mac/Documents/AI/Hide-Seek/pacman/src
# Run 10 games and check survival steps
for i in $(seq 1 10); do
  python arena.py --seek 24127561 --hide 24127192 --no-viz --max-steps 200 --pacman-speed 2 --capture-distance 2 2>&1 | grep "Total Steps"
done
```

**Success criteria**: Ghost survives ≥ 30 steps on average (vs current 8 steps), ideally reaching 200 (ghost wins).
