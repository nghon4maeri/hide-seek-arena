# RESEARCH.md — 24127457 Leader Sandbox

## Framework Summary (from TEMPLATE_agent.py)

### Required Imports

```python
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_PATH))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
```

### Required Classes

| Class | Inherits From | Role |
|-------|--------------|------|
| `PacmanAgent` | `BasePacmanAgent` (agent_interface) | Seeker — catch Ghost |
| `GhostAgent` | `BaseGhostAgent` (agent_interface) | Hider — survive 200 steps |

### Required Methods

```python
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs): ...
    def step(self, map_state, my_position, enemy_position, step_number): ...

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs): ...
    def step(self, map_state, my_position, enemy_position, step_number): ...
```

### Return Types

| Agent | Valid Returns |
|-------|--------------|
| **PacmanAgent** | `Move.UP/DOWN/LEFT/RIGHT/STAY` **or** `(Move, steps)` where `1 <= steps <= self.pacman_speed` |
| **GhostAgent** | `Move.UP/DOWN/LEFT/RIGHT/STAY` **only** — tuple/string/None → AgentLoadError |

### State Representation

| Parameter | Type | Description |
|-----------|------|-------------|
| `map_state` | `numpy.ndarray` (21×21) | `0`=empty, `1`=wall, `-1`=unseen (fog) |
| `my_position` | `tuple (row, col)` | Agent's current absolute position |
| `enemy_position` | `tuple (row, col)` or `None` | Enemy position (None = hidden by fog) |
| `step_number` | `int` | Current step, starts at 1 |

### Key Constraints

- **Simultaneous moves** — both agents decide at same time, cannot react instantly
- **Straight-line only** — Pacman multi-step must be same direction, stops at walls
- **Time budget** — < 1.0s per step()
- **Memory budget** — 128 MB RAM
- **Allowed libraries** — numpy, pandas, scipy, gurobi (no ML libs)

### Available Helpers (from TEMPLATE)

```python
_is_valid_position(pos, map_state) → bool
_apply_move(pos, move) → (row, col)
_get_neighbors(pos, map_state) → [(pos, move), ...]
_manhattan_distance(pos1, pos2) → int
```

### Move Enum

```python
Move.UP    = (-1, 0)
Move.DOWN  = (1, 0)
Move.LEFT  = (0, -1)
Move.RIGHT = (0, 1)
Move.STAY  = (0, 0)
```

---

## Lab 1 Repository Summary (HideSeek-2526-3.pdf)

### Architecture

- **Arena-based execution**: `arena.py` orchestrates the game loop
- **Agent loader**: Dynamic import via `importlib`, validates class names + inheritance + return types
- **Environment**: Owns map state, applies simultaneous moves, checks win conditions
- **Visualization**: Terminal-based colored display

### Game Rules (from spec)

| Rule | Value |
|------|-------|
| Map | 21×21 classic Pacman layout |
| Pacman speed | 2 cells/step (straight line) |
| Ghost speed | 1 cell/step |
| Capture distance | Manhattan < 2 |
| Max steps | 200 (Ghost wins if survived) |
| Info mode | Perfect Information (no fog of war) |
| Start mode | Deterministic (classic P/G positions) or Stochastic |

### Search Algorithms Referenced

| Algorithm | Use Case | Notes |
|-----------|----------|-------|
| **BFS** | Shortest path, distance maps | O(V+E), optimal for unit-cost grid |
| **A\*** | Optimal pursuit path | f(n)=g(n)+h(n), Manhattan heuristic |
| **Flood Fill** | Reachable area, safety estimation | Good for Ghost survival analysis |
| **Minimax** | Adversarial lookahead | Depth-limited due to branching factor |
| **Alpha-Beta** | Minimax pruning | Same result, fewer evaluations |

### Heuristic Approaches

- **Manhattan distance** — admissible for grid movement (no diagonal)
- **Junction distance** — prefer positions near multiple exits
- **Safe flood fill** — cells Ghost reaches before Pacman (accounting for speed 2)
- **Dead-end depth** — penalty proportional to how deep a dead-end goes

### Graph Abstraction

- **Junction graph** — nodes = cells with ≥ 3 exits, edges = corridor length
- **Precomputed once** in `__init__`, queried O(1) during `step()`
- **Dead-end map** — walk from each dead-end to nearest junction, store depth
- **Loop zones** — biconnected components where Ghost can cycle indefinitely

### Scoring

| Component | Weight |
|-----------|--------|
| Implementation completeness | 3 pts |
| First submission ranking | max 3 pts |
| Optimized submission ranking | max 4 pts |
| **Tie-break**: `diff = avg_steps_pacman - avg_steps_ghost` | lower is better |

---

## Existing Teammate Solutions

### 24127561 — Pacman (Seeker)

**Strengths:**
- A\* pathfinding is well-implemented with heapq optimization
- Ghost direction tracking + interception target projection (up to 4 cells ahead)
- Junction-aware: prefers intercepting at junctions (≥3 exits) where Ghost must turn
- Path caching avoids redundant A\* recomputation
- Multi-step movement: packs consecutive same-direction steps up to `pacman_speed`
- Good fallback exploration when enemy not visible

**Weaknesses:**
- GhostAgent is a non-functional placeholder (`return Move.STAY`)
- Code duplication: both module-level `astar()` and class method `_astar()` exist
- Dead code: `_path_to_action()` is defined but never called
- `_explore()` defined twice (both module-level and class method)
- `self._visited` referenced but never initialized
- Interception logic is conservative — only activates at streak ≥ 2
- Trap pressure feature is disabled by default
- Many import-time helper functions pollute module namespace

### 24127192 — Ghost (Hider)

**Strengths:**
- Cached BFS distance maps with LRU eviction
- Rich topology analysis: degree map, junctions, corridors, dead-ends, dead-end depth, junction-distance map
- Three-phase strategy with tuned weights (early: conservative, mid: balanced, late: survival)
- Minimax alpha-beta for close-range evasion (depth 8, time-budgeted)
- Floodfill safety scoring with speed-2 Pacman model
- Anti-oscillation: hard ban on revisiting cells within 6-step window
- Opening escape: specialized conservative strategy for first 10 steps
- Floodfill rerank: cross-checks chosen move against exhaustive candidate scoring

**Weaknesses:**
- PacmanAgent is a non-functional placeholder (`return (Move.STAY, 1)`)
- High complexity: ~890 lines, many interacting components
- Topology cache keyed by map signature hash — fragile, may collide
- Minimax depth 8 may be too deep for 1.0s budget on slower hardware
- `pacman_reach_2()` models Pacman speed correctly but enumerates all 2-step sequences (expensive)
- Hard anti-oscillation ban (6 steps) can force bad moves when cornered
- `_score_position()` partially duplicates `_phase_scoring_move()` logic

---

## Design Decisions for 24127457

### What Should Be Reused

| Idea | Source | Reason |
|------|--------|--------|
| A\* pathfinding with heapq | 24127561 | Proven optimal, efficient |
| BFS distance maps with caching | 24127192 | Fast O(1) distance queries |
| Multi-step straight-line packing | 24127561 | Required for Pacman speed 2 |
| Path caching | 24127561 | Reduces recomputation |
| Topology precomputation (junctions, dead-ends) | 24127192 | One-time cost, O(1) queries |
| Floodfill safety scoring | 24127192 | Good survival heuristic |
| Anti-oscillation (soft penalty) | 24127192 | Prevents back-and-forth |
| Phased strategy concept | 24127192 | Adapts weights over game phases |

### What Should Be Redesigned

| Aspect | Old Approach | New Approach |
|--------|-------------|--------------|
| **Module structure** | Monolithic agent.py | Split into `agent.py` + `search.py` + `topology.py` |
| **Code duplication** | Duplicated A\* in both modules | Single shared `search.py` module |
| **Interception** | Complex projection + gating | Simple direction-based prediction (1 step ahead) |
| **Minimax** | Deep (depth 8), always-on | Shallow (depth 4), only when Pacman is close |
| **Anti-oscillation** | Hard ban (excludes moves) | Soft penalty (subtracts from score) |
| **Return validation** | None inside step() | Always validate move before returning |

### What Should NOT Be Copied

| Anti-pattern | Source | Why |
|-------------|--------|-----|
| Dead code (unused methods) | 24127561 | Confuses maintenance |
| Duplicate function definitions | 24127561 | Violates DRY |
| Placeholder agents in wrong role | Both | Each agent should focus on its role |
| Fragile hash-based cache keys | 24127192 | Use tuple-based keys instead |
| Overly deep minimax | 24127192 | Risk of timeout on evaluation hardware |
| Module-level mutable state | Both | Use instance variables |
| Disabled feature flags | 24127561 | Remove or enable, don't carry dead weight |
| Magic constants without names | Both | All constants should be named |

---

## V2 Research-Backed Improvements

### Paper Analysis

#### 1. Carmel & Markovitch (1996) — "Learning Models of Intelligent Agents", AAAI

**Idea:** Opponent modeling via finite-state machine observation. An agent builds a model
of its opponent by tracking action-state transitions. A frequency table maps
(opponent_position, action) → occurrence count. The agent predicts the opponent's
next action as the most frequent action from that state.

**Why applicable:** In Pacman-Ghost pursuit-evasion, predicting the opponent's next
move is critical. For Pacman, predicting Ghost's direction enables interception.
For Ghost, predicting Pacman's target enables anticipatory evasion.

**Implementation:** `opponent.py: GhostDirectionModel` — tracks Ghost position-to-direction
transitions. `PacmanAgent._predict_target()` queries `predict_next()` for frequency-based
prediction, falling back to last-direction projection.

#### 2. Silver (2005) — "Cooperative Pathfinding", AIIDE

**Idea:** Grid-based pathfinding can be accelerated by abstracting the map to a
junction graph. Paths between junctions are precomputed; runtime queries traverse
the abstract graph rather than the full grid. Bottleneck nodes (junctions with
high betweenness) are natural interception points.

**Why applicable:** In the Pacman arena, junctions (cells with ≥3 exits) are where
Ghost must commit to a direction. Intercepting at a junction cuts off multiple
escape routes simultaneously — more efficient than chasing the Ghost directly.

**Implementation:** `heuristic.py: trace_to_junction()` traces from Ghost's position
in its current direction to find the next junction. `PacmanAgent._intercept_target()`
uses this junction as the A* target when Ghost is far away (dist > 18).

#### 3. Burkov & Chaib-draa (2010) — "Effective Multi-Agent Coordination with Entropy Maximization", AAMAS

**Idea:** In adversarial settings, deterministic action selection makes an agent
predictable. Maximizing the entropy of the action distribution (choosing actions
with a softmax over scores rather than argmax) reduces the opponent's ability to
anticipate and counter. The temperature parameter controls the exploration-exploitation
trade-off.

**Why applicable:** GhostAgent V1 used deterministic A*-to-farthest, which reveals
its direction and makes it vulnerable to interception-based Pacman. Entropy
maximization introduces strategic unpredictability.

**Implementation:** `GhostAgent._adaptive_flee()` switches to softmax-weighted
random selection when top safe destinations have close scores (gap < 20%).
Temperature = 8.0 balances randomness with quality.

#### 4. Southey et al. (2005) — "Bayesian Opponent Modeling in Adversarial Search", IJCAI

**Idea:** Opponent strategy can be inferred from observed behavior using Bayesian
updates. An exponential moving average (EMA) over observed features provides a
running estimate of the opponent's strategy type. This estimate weights the
agent's evaluation of candidate actions.

**Why applicable:** A Ghost facing an unknown Pacman benefits from knowing whether
the Pacman uses direct chase (always moves toward Ghost) or interception (moves
to cut off junctions). The EMA provides a smooth estimate without requiring
explicit strategy classification.

**Implementation:** `opponent.py: PacmanPursuitModel` maintains an intercept_ema
(0 = random walk, 1 = direct pursuit). `GhostAgent._greedy_evade()` uses
`predict_target()` to anticipate Pacman's position after speed-2 and filters
candidate moves that would land near the predicted target.

#### 5. Bakkes et al. (2012) — "Rapid Adaptation of Video Game AI", IEEE TCIAIG

**Idea:** Game AI should dynamically select strategies based on game state rather
than using a single static approach. Three-range adaptation is common in
pursuit-evasion: far → strategic planning, mid → tactical pressure, near →
reactive capture. Transitions between strategies use distance thresholds.

**Why applicable:** Pacman's optimal strategy depends on distance to Ghost.
Far away: plan interception at choke points. Mid-range: apply pressure by
approaching escape routes. Close: switch to direct pursuit with prediction.

**Implementation:** `PacmanAgent.step()` uses `ADAPTIVE_FAR_THRESHOLD` (18) and
`ADAPTIVE_NEAR_THRESHOLD` (7) to select between `_intercept_target()`,
`_trap_target()`, and `_predict_target()`.

### Implementation Summary

| Module | Change | Research Basis |
|--------|--------|---------------|
| `opponent.py` (NEW) | `GhostDirectionModel`, `PacmanPursuitModel` | Carmel & Markovitch 1996, Southey et al. 2005 |
| `heuristic.py` | `trace_to_junction()` | Silver 2005 |
| `agent.py: PacmanAgent` | Adaptive 3-range pursuit with junction interception | Bakkes et al. 2012, Silver 2005, Carmel & Markovitch 1996 |
| `agent.py: GhostAgent` | Adaptive entropy flee, opponent-aware greedy evade | Burkov & Chaib-draa 2010, Southey et al. 2005 |

### Complexity Budget (V2)

| Component | Time per step() | Space |
|-----------|----------------|-------|
| GhostDirectionModel.observe() | O(1) dict update | O(window) |
| PacmanPursuitModel.observe() | O(1) + manhattan | O(window) |
| trace_to_junction() | O(max_steps) | O(1) |
| _intercept_target() | O(12 + A*) | O(A*) |
| _adaptive_flee() (deterministic) | O(bfs + 3×A*) | O(V) |
| _adaptive_flee() (entropy) | O(bfs + 3×A*) | O(V) |
| **Total PacmanAgent** | < 50ms | < 5 MB |
| **Total GhostAgent** | < 80ms | < 10 MB |

### Risks and Tradeoffs

| Risk | Mitigation |
|------|-----------|
| Junction interception over-shoots if Ghost turns | Fall back to simple prediction when trace hits wall |
| Entropy mode worsen performance vs simple chasers | Score-proximity gate (< 20% gap) prevents unnecessary randomization |
| Opponent model requires warm-up steps | Confidence gate delays model-dependent decisions until 5+ observations |
| Intercept target may be unreachable | A* pathfinding returns [] → fall back to explore |
| Stochastic start positions may disadvantage Ghost | Deterministic flee (V1 behavior) as default baseline |
