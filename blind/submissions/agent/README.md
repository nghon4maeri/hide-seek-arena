# Agent — Lab 2 Blind Adversary

## Thuật toán

### GhostAgent (Hider)
```
Tier 1: Survival Gate  
  ├── Trap detection — BFS lookahead 6 steps
  ├── Escape margin — ratio ghost_reachable / pacman_reachable
  └── Force escape nếu margin < 0.4

Tier 2: Mode Dispatch  
  ├── PANIC (dist < 4) — Minimax 3-ply (Ghost→Pacman→Ghost)
  ├── EVASION (4 ≤ dist < 8) — Strategic flee + future sim  
  └── FORTRESS (dist ≥ 8) — Maximize distance, patrol loops

Tier 3: Exploration  
  └── BFS 6-step + topology scoring + belief-guided
```

**Scoring** dùng `capture_eta` (speed-2-aware turns) thay vì Manhattan, và `safe_area` (floodfill) để đánh giá không gian an toàn.

### PacmanAgent (Seeker)
```
Tier 1: Safety — Dead-end escape + anti-stuck
Tier 2: Pursuit — Interception (chặn đầu) + A* chase + Belief-guided
Tier 3: Exploration — Frontier + Belief-weighted search
```

**Speed packing**: gộp tối đa 2 bước cùng hướng trong 1 lượt.  
**Interception**: dự đoán hướng Ghost (streak ≥ 2) → tìm junction chặn đầu.

## Kết quả Benchmark (stochastic, 30 games)

| Matchup | Win Rate |
|---------|----------|
| Agent vs Agent (self-play) | Pacman 100% / Ghost 100% |
| Agent vs example_student (Pacman) | **93.3%** win |
| Agent vs example_student (Ghost) | 20.8s avg capture |

## Cách chạy

```bash
cd blind/src
python arena.py --seek agent --hide agent --pacman-obs-radius 5 --ghost-obs-radius 5 --delay 0.3

# Benchmark
cd blind
python scripts/benchmark_full.py --seek agent --hide agent --games 50 --json
```

## Cấu trúc thư mục

```
agent/
├── agent.py           # Main entry (PacmanAgent + GhostAgent)
├── pathfinding.py     # A*, BFS, capture_eta, safe_area
├── topology.py        # Junction, loop, core, dead-end analysis
├── belief_state.py    # Bayesian enemy location tracking
├── game_theory.py     # Expectiminimax
├── rollout.py         # Monte Carlo simulation
└── README.md
```
