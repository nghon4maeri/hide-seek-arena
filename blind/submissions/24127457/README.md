# Blind Adversary

## 1. Introduction

This submission implements a **Reinforcement Learning (RL)** solution for the
Pacman-vs-Ghost blind-adversary game (Lab 2).  Unlike traditional pathfinding
approaches (A\*, BFS) that require explicit map memory and heuristic rules,
our agents use a **PPO + LSTM (recurrent policy)** trained entirely via
self-play against heuristic opponents on Kaggle (GPU T4).

**Key idea:** The agent receives a 3-channel image (wall / visible empty /
fog) each step.  An LSTM hidden state (128-d) carries temporal context across
steps, allowing the network to *implicitly* reconstruct unseen regions and
track the enemy's last known location — no hand-coded memory logic needed.

| Attribute              | Value                              |
|------------------------|------------------------------------|
| Algorithm              | PPO with GAE, clipped surrogate    |
| Architecture           | CNN (2×Conv2D) + LSTM(128) + MLP  |
| Parameters             | ~160K per agent                    |
| Observation            | 3-channel 21×21 image + position   |
| Training platform      | Kaggle Notebook (GPU T4/P100)      |
| Training time          | ~30 min per agent (500 episodes)   |

---

## 2. Directory Structure

Only **5 files** are required for submission.  The notebook and intermediate
checkpoints (`*_ep*.pth`) are excluded from the final zip.

```
submissions/24127457/
├── agent.py               # RL inference wrapper (loads model, preprocesses obs,
│                          #   manages LSTM hidden state, timeout guard)
├── network_architect.py   # PyTorch network definition (must match training
│                          #   notebook exactly for weight loading)
├── pacman_model.pth       # Trained Pacman (Seeker) weights — 1.2 MB
├── ghost_model.pth        # Trained Ghost (Hider) weights — 1.2 MB
└── README.md              # This file
```

---

## 3. Execution Guide

All commands below assume the working directory is `blind/` (i.e., one level
above `submissions/`).

### 3.1  Smoke test — self-play (5 steps)

```bash
python scripts/run_smoke_test.py --seek 24127457 --hide 24127457
```

Runs a 5-step blind match (`--pacman-obs-radius 5 --ghost-obs-radius 5`)
to verify that both agents load, produce legal actions, and do not crash.

### 3.2  Pacman (us) vs Ghost (other member)

```bash
python src/arena.py --seek 24127457 --hide 24127192 \
    --pacman-obs-radius 5 --ghost-obs-radius 5 --no-viz
```

Our trained Pacman hunts another team's Ghost under full blind constraints.

### 3.3  Ghost (us) vs Pacman (other member)

```bash
python src/arena.py --seek 24127561 --hide 24127457 \
    --pacman-obs-radius 5 --ghost-obs-radius 5 --no-viz
```

Our trained Ghost evades another team's Pacman.

### 3.4  Full benchmark (200 steps, 10 games)

```bash
python scripts/benchmark_agents.py --seek 24127457 --hide 24127457 \
    --games 10 --max-steps 200 --pacman-obs 5 --ghost-obs 5
```

### 3.5  Fallback test — full visibility (`--obs-radius 0`)

```bash
python src/arena.py --seek 24127457 --hide 24127457 \
    --pacman-obs-radius 0 --ghost-obs-radius 0 --no-viz
```

Verifies the agents degrade gracefully when fog-of-war is disabled (both
models were trained with radius=5 but generalise to full visibility via
the wall / fog channel structure).

---

## 4. Technical Specifications

### 4.1  Latency

| Component                 | Measured time |
|---------------------------|---------------|
| NumPy preprocessing       | ~1–3 ms       |
| CNN + LSTM inference      | ~0.5–3 ms     |
| **Total `step()` call**   | **~3–8 ms**   |
| *Lab limit*               | *1 000 ms*    |

A **timeout guard** triggers at **0.85 s** and forces `Move.STAY` if the
step function is ever blocked (e.g. OS scheduler pause, swap thrashing).

### 4.2  Memory

| Item                         | Size     |
|------------------------------|----------|
| Model weights (both agents)  | ~2.4 MB  |
| LSTM hidden state            | ~1 KB    |
| Observation tensors          | ~5 KB    |
| **Peak RAM usage**           | **~2.5 MB** |
| *Lab limit*                  | *128 MB* |

### 4.3  Action format

| Agent    | Output format                                          |
|----------|--------------------------------------------------------|
| Pacman   | `Move` (speed 1) or `(Move, steps)` with `steps ∈ [1, 2]` |
| Ghost    | `Move` (UP / DOWN / LEFT / RIGHT / STAY)              |

Both agents return only legal `Move` enum values.  The Pacman network was
trained with a 9-action head (4 directions × speed-1, 4 directions × speed-2,
STAY) and maps indices back to the framework's `(Move, steps)` convention.

### 4.4  Partial-observability correctness

| Requirement                 | Status | Mechanism                                    |
|-----------------------------|--------|----------------------------------------------|
| Handles `enemy_position is None` | ✅ | LSTM maintains belief state across steps; enemy tracking is implicit |
| Does **not** blindly assume `-1` cells are safe | ✅ | 3-channel encoding separates "seen empty" from "fog"; network learns uncertainty from the fog channel |
| Walls block line-of-sight   | ✅ | `_get_visible_cells()` halts rays at walls (matches framework's `get_visible_cells_cross`) |

---

## 5.  Model training (reproducibility)

To retrain from scratch on Kaggle:

1. Upload `train_arena_rl.ipynb` to a Kaggle Notebook with **GPU T4×2**
   accelerator.
2. Run all cells (≈60 min total for both agents × 500 episodes).
3. Download `pacman_model.pth` and `ghost_model.pth` from the output
   directory back to `submissions/24127457/`.

**Training config (default):**
```python
CFG = {
    'train_ghost':   True,
    'train_pacman':  True,
    'ghost_episodes':   500,
    'pacman_episodes':  500,
    'learning_rate':    3e-4,
    'obs_radius':       5,
    'capture_distance': 2,
    'seed':             42,
}
```