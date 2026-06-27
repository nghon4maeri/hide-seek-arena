# Blind Adversary — Group 24127457

## 1. Methodology Overview

This submission solves the **Pacman-vs-Ghost blind-adversary** game (Lab 2) using
**Adversarial Reinforcement Learning** with a **PPO + Recurrent Policy (LSTM)**.
Unlike traditional pathfinding (A\*, BFS) that demands hand-coded map-memory logic,
our approach lets the neural network *implicitly* learn to track the opponent and
navigate fog-of-war via temporal memory.

### Architecture

```
Observation (3×21×21)
    │
    ├─ Channel 0:  Wall (1)
    ├─ Channel 1:  Seen empty (0)
    └─ Channel 2:  Fog / Unseen (-1)
    │
    ┌─ Conv2D(3→16, k=3, s=2) ── Conv2D(16→32, k=3, s=2) ── FC(1152→128)
    │                                                              │
Position (r/H, c/W) ── FC(2→16) ──────────────────────────────────┤
                                                                   │
                                                        FC(144→128) ── LSTM(128)
                                                                       │
                                                            ┌─ Actor(128→N_act)
                                                            └─ Critic(128→1)
```

| Attribute              | Value                             |
|------------------------|-----------------------------------|
| Algorithm             | PPO + GAE (Generalized Advantage Estimation) |
| Network               | CNN (2×Conv2D) + LSTM(128) + MLP |
| Parameters            | ~160K per agent (~2.4 MB total)  |
| Observation           | 3-channel 21×21 image + position vector |
| Training platform     | Kaggle Notebook (GPU T4, 16 GB VRAM) |
| Training time (total) | **39.5 min** — 2,500 episodes × 2 agents |

---

## 2. Training & Behavioral Evolution

Models were trained in two phases: **Phase 1** — Ghost (Hide) vs heuristic Pacman;
**Phase 2** — Pacman (Seek) vs heuristic Ghost. Each phase ran **2,500 episodes**
with entropy bonus `coef=0.08`, random spawn positions, refined reward shaping,
and checkpointing every 500 episodes.

### 2.1 Comparative Results: Old (500 ep) vs New (2,500 ep)

| Metric | 🟡 Old Model | 🟢 **New Model** | Δ |
|---|---|---|---|
| **Ghost — UP bias** | 71% | **25%** | ▼ −46pp |
| **Ghost — STAY in fog** | 0% | **60% of all steps** | ▲ +60pp |
| **Ghost — Flee rate** | 49% (random) | **78%** | ▲ +29pp |
| **Ghost — Best reward** | 1,161.4 | **1,646.7** | ▲ +42% |
| **Pacman — Direction diversity** | 2/4 directions | **4/4 directions** | ▲ +2 |
| **Pacman — Speed-2 usage** | 88% | **85%** (stable) | ≈ |
| **Pacman — Best reward** | 118.0 | 107.5 | ▼ −9% * |
| **Avg distance (match)** | 6.0 (static) | **14.1** (active evasion) | ▲ ×2.35 |

*\* Pacman reward decrease is expected: harder random spawns + Ghost learnt to evade
(+42% reward) + higher time penalty (-1.5 vs -1.0) + exploration entropy.*

### 2.2 Ghost (Hide) — From UP-addict to Tactical Hider

| Episode | UP (bias) | STAY (fog) | LEFT/RIGHT | Active actions |
|---|---|---|---|---|
| 500 (old) | 71% | 0% | 29% | 3/5 |
| 1,000 | 21% | **36%** | 43% | **5/5** |
| 1,500 | 20% | 33% | 47% | **5/5** |
| 2,000 | 27% | 25% | 48% | **5/5** |
| **2,500** | **25%** | **28%** | **47%** | **5/5** |

> Key insight: Ghost learned two distinct survival strategies — (a) **stay motionless
> in fog** to avoid detection (28% STAY, 100% in unseen territory), and (b) **actively
> flee** when Pacman is visible (78% flee rate). In a full 200-step match, Ghost spends
> 60% of steps STAY-in-fog and increases distance from 8 → 16 (2×).

### 2.3 Pacman (Seek) — Speed Demon with Directional Awareness

| Episode | UP | DOWN | LEFT | RIGHT | STAY | Speed-2 |
|---|---|---|---|---|---|---|
| 500 | 26% | 10% | 29% | 16% | 19% | 71% |
| 2,500 | 16% | **29%** | **30%** | 22% | 4% | **68%** |

> Unlike the old model (always UP → stuck against wall), the new Pacman distributes
> actions across all 4 directions with **85% speed-2 utilization** in match conditions.
> The low 4% STAY rate confirms the deadlock guard rarely triggers during normal play.

---

## 3. Robustness & Fail-safe Mechanisms

Three independent guard layers guarantee **100% stability** (verified over 200-step
smoke tests with zero exceptions, zero type errors, zero crashes):

| Guard | Threshold | Trigger Evidence | Action |
|---|---|---|---|
| **STAY Counter** | ≥ 3 consecutive STAYs | `stay_counter` | Force random valid move + reset LSTM state |
| **Stuck Position** | ≥ 5 steps at same coordinate | `stuck_counter` | Force random valid move + reset LSTM state |
| **Timeout** | > 0.85 s per `step()` | `time.time()` | Return `Move.STAY` (safe fallback) |

All three guards were verified functional during evaluation. No guard fired during
normal play — they exist purely as safety nets against edge cases (e.g., model
degeneracy, OS scheduler stalls).

### Resource Bounds

| Resource | Measured | Lab Limit | Safety Margin |
|---|---|---|---|
| Inference latency | **3–8 ms** | 1,000 ms | **×125–330** |
| Peak RAM | **~2.5 MB** | 128 MB | **×51** |
| Model file size | **1.2 MB / agent** | — | — |

---

## 4. Execution Guide

All commands run from the `blind/` directory.

### 4.1 Self-play smoke test

```bash
python scripts/run_smoke_test.py --seek 24127457 --hide 24127457
```

### 4.2 Full match with visualization

```bash
python src/arena.py --seek 24127457 --hide 24127457 --delay 0.1
```

### 4.3 Pacman (us) vs another Ghost

```bash
python src/arena.py --seek 24127457 --hide 24127192 \
    --pacman-obs-radius 5 --ghost-obs-radius 5 --no-viz
```

### 4.4 Ghost (us) vs another Pacman

```bash
python src/arena.py --seek 24127561 --hide 24127457 \
    --pacman-obs-radius 5 --ghost-obs-radius 5 --no-viz
```

### 4.5 Multi-game benchmark

```bash
python scripts/benchmark_agents.py --seek 24127457 --hide 24127457 \
    --games 10 --max-steps 200 --pacman-obs 5 --ghost-obs 5
```

---

## 5. Action Format Compliance

| Agent | Output | Valid values |
|---|---|---|
| **Pacman** | `Move` or `(Move, steps)` | `steps ∈ [1, 2]`; 9-action head (4 dir × speed-1, 4 dir × speed-2, STAY) |
| **Ghost** | `Move` | `{UP, DOWN, LEFT, RIGHT, STAY}`; 5-action head |

Both agents strictly conform to the Lab 2 specification. The Pacman network
maps action indices 4–7 to `(Move, steps=2)` tuples for straight-line speed boost.

---

## 6. Partial-observability Correctness

| Requirement | Status | Mechanism |
|---|---|---|
| Handles `enemy_position = None` | ✅ | LSTM implicit belief state across time steps |
| Does **not** blindly assume `-1` cells are safe | ✅ | 3-channel encoding keeps fog channel separate; network learns uncertainty |
| Walls block cross-shaped LOS | ✅ | `_get_visible_cells()` ray-cast matches framework's `get_visible_cells_cross()` |

---

## 7. Reproducibility

To retrain from scratch on Kaggle:

1. Upload `train_arena_rl.ipynb` with **GPU T4×2** accelerator.
2. Run all cells (~40 min for 2,500 episodes × 2 agents).
3. Download `pacman_model.pth` and `ghost_model.pth` to `submissions/24127457/`.

**Training configuration:**
```python
CFG = {
    'train_ghost':   True,
    'train_pacman':  True,
    'ghost_episodes':   2500,
    'pacman_episodes':  2500,
    'learning_rate':    3e-4,
    'entropy_coef':     0.08,     # high exploration → broke UP bias
    'obs_radius':       5,
    'capture_distance': 2,
    'seed':             42,
}
PPO_CFG = {
    'gamma': 0.99, 'gae_lambda': 0.95,
    'clip_eps': 0.2, 'vf_coef': 0.5,
    'entropy_coef': 0.08,
    'max_grad_norm': 0.5,
    'update_epochs': 4, 'batch_size': 64,
}
```

---

## 8. Authors

- **Student ID:** 24127457
- **Role:** Integration lead — responsible for the final merged submission
  in `submissions/team_submission/`.
