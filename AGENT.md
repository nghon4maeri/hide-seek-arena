# AGENT.md — Hide and Seek Arena

Guide for coding agents working on this repository. Read this **before** editing anything.

---

## 1. Project Overview

Two independent AI labs in one repo. Each lab is a self-contained workspace with its own framework, submissions, tests, and runner.

| Lab | Workspace | Observability | Map | Roles |
|-----|-----------|---------------|-----|-------|
| **Lab 1 — Hide and Seek** | `pacman/` | Full (perfect info) | 21×21 static | **Pacman = Seeker**, **Ghost = Hider** |
| **Lab 2 — Blind Adversary** | `blind/` | Partial (cross-shaped vision, radius 5, walls block LOS) | 21×21 static | Same roles, `enemy_position` may be `None` |

**Role convention inside the labs** (authoritative for all edits): `PacmanAgent` = Seeker (catches), `GhostAgent` = Hider (evades).
The root `agent.py` tournament entry uses the *opposite* naming (`HideAgent→PacmanAgent`, `SeekAgent→GhostAgent`) — do **not** let that confuse you when working inside `pacman/` or `blind/`.

**Win conditions (both labs):**
- Pacman (Seeker) wins when Manhattan distance < `capture_distance` (CLI default **2**).
- Ghost (Hider) wins by surviving `max_steps` (default **200**).
- Turns are **synchronous** — both agents move simultaneously; you cannot react to the opponent's current move.

**Movement:**
- Pacman: returns `Move` **or** `(Move, steps)` with `1 ≤ steps ≤ pacman_speed` (CLI default **2**). Straight-line only; stops at walls.
- Ghost: returns **only** `Move`. Returning a tuple/string is an instant forfeit.

---

## 2. Architecture Overview

```
hide-seek-arena/
├── agent.py                 # Tournament entry (FORBIDDEN). HideAgent/SeekAgent → Pacman/Ghost.
├── src/                     # Reference tournament agent (FORBIDDEN). Different type system (Action).
│   ├── agents/              # HideAgent (minimax+αβ), SeekAgent (A*+minimax)
│   ├── search/              # SearchToolkit (singleton/cache), BFS, A*, flood-fill, minimax, αβ
│   ├── evaluation/          # hide_eval, seek_eval, features (danger/dead-end/cutoff/interception)
│   ├── core/                # official_map, movement, constants, simulator
│   ├── debug/  ui/          # Traces + visualizers (not in tournament path)
│
├── pacman/                  # ← Lab 1 workspace (ONLY editable area for Lab 1)
│   ├── src/                 #   Lab framework — FORBIDDEN to edit
│   │   ├── arena.py         #   Match runner (CLI: --seek --hide --pacman-speed --capture-distance …)
│   │   ├── environment.py   #   Map, Move enum, rules, cross-shaped observation
│   │   ├── agent_interface.py  # PacmanAgent / GhostAgent base classes
│   │   ├── agent_loader.py  #   Dynamic import + validation
│   │   └── visualizer.py
│   ├── submissions/<id>/agent.py  # ← EDIT HERE. team_submission/ is the final merge.
│   ├── tests/               #   FORBIDDEN to edit
│   └── scripts/             #   FORBIDDEN to edit
│
├── blind/                   # ← Lab 2 workspace (ONLY editable area for Lab 2)
│   ├── src/                 #   Same framework shape as pacman/ (with fog-of-war) — FORBIDDEN
│   ├── submissions/<id>/agent.py  # ← EDIT HERE
│   ├── tests/  scripts/     #   FORBIDDEN
│
├── tests/                   # Root pytest (FORBIDDEN)
├── visualizer/              # React/Vite replay UI (FORBIDDEN)
├── ts-backend/              # TypeScript port (FORBIDDEN)
└── scripts/                 # Root tooling (FORBIDDEN)
```

**Key type-system split (important):**
- Root `src/` uses `Action` (custom) and `Grid` (list-of-lists). Cached `SearchToolkit` is keyed by grid hash.
- Lab frameworks use `Move` (enum) and `np.ndarray` maps.
- These two systems are **not interchangeable**. Lab submissions import only from their own `src/` (`agent_interface`, `environment`). Do **not** `import` root `src/` into a lab agent — type mismatch + it lives in a forbidden zone.
- **Reuse ideas/algorithms** from root `src/` (BFS, A*, flood-fill, minimax, αβ, dead-end/trap heuristics) by **reimplementing them inside the submission** using `Move`.

**Lab agent interface** (`pacman/src/agent_interface.py`, mirrored in `blind/`):
```python
class PacmanAgent(AgentInterface):   # Seeker
    def __init__(self, **kwargs): ...           # pacman_speed passed via kwargs
    def step(self, map_state, my_position, enemy_position, step_number): -> Move | (Move, steps)

class GhostAgent(AgentInterface):    # Hider
    def step(self, map_state, my_position, enemy_position, step_number): -> Move
```
- `map_state`: `0`=empty, `1`=wall, **`-1`=unseen** (Lab 2 only).
- `enemy_position`: `None` when hidden by fog (Lab 2 only; Lab 1 always provides it).
- Agents are **stateful** — caches/memory persist across steps. Initialize in `__init__`.

---

## 3. Lab 1 Strategy Guide (`pacman/`)

Full observability, simultaneous turns, Pacman speed 2, capture distance 2.

**Priorities:** Seek win rate ↑, Hide survival rate ↑, avg match performance ↑, search efficiency ↑.

### Seeker (PacmanAgent) — catch the Ghost
- **A\* pursuit** toward ghost as the primary plan; cache the path, replan only when the ghost moved significantly or path is exhausted.
- **Minimax / Alpha-Beta** (depth 3–4) around the capture zone. Move ordering: sort branches by ascending BFS distance to ghost; this makes αβ prune hard.
- **Interception > direct chase:** predict ghost's evasion (model it as a maximizer of distance) and aim for the cell it is fleeing toward.
- **Trap pressure:** push ghost toward dead-ends and degree-≤2 corridors. Score trap potential = `max(0, 4-degree)*12 + max(0, 7-dead_end_dist)*5` (see root `src/evaluation/seek_eval.py`).
- **Cutoff scoring:** prefer moves that shrink ghost's safe reachable area (`safe_reachable_area`) over moves that only reduce raw distance.
- **Use speed:** with `pacman_speed=2`, return `(Move, 2)` on long open corridors to close distance; fall back to `Move` near junctions where a 2-step straight move overshoots.

### Hider (GhostAgent) — survive 200 steps
- **Safe-area heuristic:** maximize flood-fill area that Pacman cannot reach before you (`safe_reachable_area(pacman, ghost, ghost_speed=…)`).
- **Minimax + αβ** (depth 3–4) as evader; ghost is the *minimizing* player on capture probability.
- **Avoid dead-ends and degree-1 cells** — penalize `distance_to_dead_end ≤ 1`.
- **Corridor risk:** degree ≤ 2 near the seeker is dangerous; add a corridor penalty scaled by closeness.
- **Never move toward the seeker** unless the only alternative is a dead-end trap; bias toward maintaining or increasing BFS distance.
- **Tie-breaks matter:** the competition tie-break favors lower `avg_steps_pacman − avg_steps_ghost`, so as Hider, surviving *longer* (not just winning) is valuable.

**Recommended algorithms:** Minimax, Alpha-Beta Pruning, Expectiminimax (for stochastic starts), Monte Carlo rollouts for shallow tactical noise.

---

## 4. Lab 2 Strategy Guide (`blind/`)

Partial observability: cross-shaped vision radius 5, walls block line-of-sight, unseen cells = `-1`, `enemy_position` often `None`.

**Priorities:** enemy tracking ↑, exploration ↑, information gain ↑, belief-state maintenance ↑, capture efficiency ↑.

### Shared mental map
- Maintain a **persistent `known_map`** in agent state: write `0`/`1` for seen cells; keep `-1` for unseen. Walls are always visible (structural knowledge).
- Never read `map_state[r,c] == 0` without first checking it isn't `-1`.

### Enemy tracking (belief state)
- Keep a **belief distribution** over possible enemy cells. Update each step:
  - If `enemy_position is not None`: collapse belief to that cell (then propagate one move of motion model).
  - If `None`: propagate belief forward by the motion model (uniform over each candidate's legal neighbors), then **reweight** by removing cells that *should* be visible but where the enemy was *not* seen (observation update).
- **Particle filter** is the practical implementation: ~200–500 particles, resample when effective count drops.
- Use the belief **mean** or **mode** as the planning target when the enemy is hidden.

### Exploration (when enemy is lost)
- **Frontier-based exploration:** target the nearest unseen boundary cell (`-1` adjacent to a seen open cell). Pick the frontier maximizing information gain (count of unseen cells it would reveal) divided by BFS distance.
- **Information-gain heuristic:** prefer positions whose cross-shaped footprint covers the most unseen open cells.

### Planning under uncertainty
- **A\*** on the mental map toward belief-target or frontier. Treat `-1` as optimistic open (`0`) for pathing, but flag the plan as uncertain.
- **Expectiminimax / αβ on belief states:** branch over the top-k likely enemy cells weighted by belief probability.
- **MCTS** with particle-based rollouts when belief is diffuse; switch to deterministic αβ once belief concentrates.
- **Capture policy:** when belief mode is within capture distance + small margin, commit to the interception move (do not over-explore).

### Hider-specific (GhostAgent) under fog
- Exploit the seeker's blindness: break line-of-sight by rounding wall corners; the seeker loses you the moment you exit its cross.
- Track the seeker's last-known position and a belief over its current location; flee the *closest* plausible seeker cell, not just last-known.
- Prefer loops and high-degree regions where re-observation is unlikely.

**Recommended algorithms:** Belief State Search, Particle Filters, Frontier Exploration, Information Gain Search, Expectiminimax, MCTS over belief states, A* with Memory.

---

## 5. Modification Restrictions

### Hard rules
1. **DO NOT modify** `src/` (root reference agent).
2. **DO NOT modify** `tests/` (root) or `pacman/tests/` or `blind/tests/`.
3. **DO NOT modify** `visualizer/`.
4. **DO NOT modify** `ts-backend/`.
5. **DO NOT modify** `scripts/` (root) or `pacman/scripts/` or `blind/scripts/`.
6. **DO NOT modify** `agent.py` (root tournament entry) unless explicitly requested.
7. **DO NOT modify** `pacman/src/` or `blind/src/` (lab frameworks).
8. **ONLY modify** files inside `pacman/` for Lab 1 — in practice only `pacman/submissions/<id>/agent.py` (and helper modules you add inside that submission folder).
9. **ONLY modify** files inside `blind/` for Lab 2 — same pattern.
10. **Never** refactor the framework, rename framework files, or change public APIs used by the arena.
11. Assume all framework code is production code and must remain compatible.

### Declaration required before every edit
Before editing any file, state exactly:

```
Allowed to modify:
- pacman/submissions/<id>/*        (for Lab 1)
  or
- blind/submissions/<id>/*         (for Lab 2)

Not allowed:
- src/  tests/  visualizer/  ts-backend/  scripts/  agent.py
- pacman/src/  pacman/tests/  pacman/scripts/
- blind/src/   blind/tests/   blind/scripts/
```

If a requested change would require touching a forbidden path, **refuse and explain why** — do not attempt a workaround that edits framework files.

### Submission target
- The final, submitted agent lives at `pacman/submissions/team_submission/agent.py` (Lab 1) and `blind/submissions/team_submission/agent.py` (Lab 2). Individual engineer folders (`24127457/`, `24127192/`, `24127561/`) are sandboxes; `team_submission/` is the merge target.

---

## 6. Performance Constraints

Every `step()` call must satisfy:

| Constraint | Limit |
|------------|-------|
| **Time per step** | **< 1.0 s** (arena enforces ~1s; aim for < 0.85 s budget internally) |
| **Memory (RAM)** | **< 128 MB** |
| **Python** | 3.11 |
| **Allowed libs** | `numpy`, `pandas`, `scipy`, `gurobi`, `pytorch`, `scikit-learn` (Lab 2). Lab 1: stick to `numpy` + stdlib for fast cold-start. |

### Efficiency rules
- **Cache aggressively** across steps in agent state: BFS distance maps, dead-end map, degree map, junction list. The map is static — precompute once in `__init__` (or lazily on first `step`).
- Reuse a single BFS/A* implementation; do not recompute the full map scan each turn.
- **Time-budget your search:** check `time.perf_counter()` inside deep loops and bail out with the best move found so far (iterative deepening). Never block on a full depth-4 minimax if the clock is low.
- **Move ordering first:** in αβ, sort candidate actions by a cheap heuristic (BFS distance / eval) before recursing — pruning is only effective with good ordering.
- Avoid numpy heavy ops in the hot path; plain Python lists are often faster for tiny 21×21 grids.

---

## 7. Development Workflow

For every change, follow this order:

1. **Explain reasoning** — what is the problem and why this approach.
2. **Produce implementation plan** — list the steps.
3. **List affected files** — with full paths.
4. **Verify restrictions** — paste the Allowed / Not-allowed declaration (§5).
5. **Generate code** — edit only the allowed submission file(s).
6. **Validate** — run the tests / smoke test / a sample match (§8).

### Running matches

```bash
# Lab 1 — from repo root
python pacman/src/arena.py --seek team_submission --hide example_student
python pacman/src/arena.py --seek team_submission --hide example_student --no-viz --max-steps 200
python pacman/src/arena.py --seek <id> --hide <id> --pacman-speed 2 --capture-distance 2 --delay 0.3
./pacman/run_game.sh team_submission example_student

# Lab 2 — fog of war ON by default
python blind/src/arena.py --seek team_submission --hide example_student \
    --pacman-obs-radius 5 --ghost-obs-radius 5
./blind/run_game.sh team_submission example_student
```

Key flags: `--max-steps`, `--capture-distance`, `--pacman-speed`, `--start-mode stochastic|deterministic`, `--pacman-obs-radius`, `--ghost-obs-radius`, `--delay`, `--no-viz`, `--step-timeout`.

> Windows: set `PYTHONIOENCODING=utf-8` to avoid Unicode errors from the visualizer.

### Benchmarking

```bash
python pacman/scripts/benchmark_agents.py --seek team_submission --hide example_student --games 10
python blind/scripts/benchmark_agents.py  --seek team_submission --hide example_student --games 10
```

Always benchmark with **both** roles (your agent as Seeker and as Hider) — competition score combines `winrate_seek` and `winrate_hide`.

---

## 8. Testing Workflow

```bash
# Root tests (must stay green — do not edit these)
python -m pytest tests/ -v

# Lab 1 tests (interface + smoke + workspace structure)
python -m pytest pacman/tests/ -v

# Lab 2 tests
python -m pytest blind/tests/ -v

# Quick smoke tests
python pacman/scripts/run_smoke_test.py
python blind/scripts/run_smoke_test.py
```

The lab `test_submission_interface.py` loads each submission in `SUBMISSIONS` list and asserts that `PacmanAgent(pacman_speed=2)` and `GhostAgent()` construct and return a **legal** `Move` / `(Move, steps)` on a tiny 3×3 map. Any new submission folder must be added to that list **only by editing the test file** — which is **forbidden**, so prefer improving `team_submission/agent.py` rather than adding new submission IDs unless the test is already aware of them.

### Pre-submit self-check
1. `python -m pytest pacman/tests/ -v` (or `blind/tests/`) — all green.
2. Run a 10-game benchmark as Seeker **and** as Hider; record win rates.
3. Run a single `--delay 0.3` visual match and watch for: crashes, timeouts, invalid-move forfeits, getting stuck in corners, walking into the enemy.
4. Stress-test edge cases: stochastic starts, `max-steps 50` (short), `max-steps 300` (long), `pacman-speed 1` (slow), `capture-distance 1` (tight).

---

## 9. Safety Checklist Before Every Code Change

Run through this list **every** time, no exceptions:

- [ ] **Target path allowed?** File is inside `pacman/submissions/*/*` or `blind/submissions/*/*`. If not → stop.
- [ ] **Declaration pasted?** Wrote the Allowed / Not-allowed block (§5) before editing.
- [ ] **No framework edits?** Not touching `src/`, `pacman/src/`, `blind/src/`, `agent.py`, `tests/`, `scripts/`, `visualizer/`, `ts-backend/`.
- [ ] **Interface preserved?** Class names `PacmanAgent` / `GhostAgent` unchanged; `__init__(self, **kwargs)` and `step(self, map_state, my_position, enemy_position, step_number)` signatures intact.
- [ ] **Return types correct?** Pacman → `Move` or `(Move, steps)` with `1 ≤ steps ≤ pacman_speed`. Ghost → `Move` only (never a tuple/string).
- [ ] **Fog handled?** Code checks `enemy_position is None` and `map_state[r,c] == -1` before using them (Lab 2; also safe for Lab 1).
- [ ] **Time-safe?** Search has a `perf_counter` deadline check and returns a fallback legal move on timeout.
- [ ] **Memory-safe?** No unbounded growth across steps; caches are static-map-keyed or bounded.
- [ ] **Tests green?** `python -m pytest <lab>/tests/ -v` passes after the change.
- [ ] **Benchmark sane?** Win rate did not regress vs. the previous version (run 10 games each role).
- [ ] **No comments added unless asked.** No emoji unless requested. Follow existing file style.
- [ ] **Did not commit** unless the user explicitly asked.

If any box cannot be checked, stop and report the blocker instead of pushing the change through.
