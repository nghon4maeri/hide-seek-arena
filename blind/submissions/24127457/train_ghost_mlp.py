"""train_ghost_mlp.py — Train lightweight Ghost move predictor via synthetic data.

Uses the V3 Ghost heuristic as oracle on fixed map positions.
No games needed — generates millions of samples in seconds.

Usage:  python train_ghost_mlp.py
Output: ghost_mlp.pth (~15 KB, inference <0.1ms)
"""

import sys, os, random
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------------------------------
# Setup paths
# ------------------------------------------------------------
SUBMISSION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SUBMISSION_DIR.parents[1] / "src"))
sys.path.insert(0, str(SUBMISSION_DIR))

from environment import Move

# ------------------------------------------------------------
# Map
# ------------------------------------------------------------
MAP_LAYOUT = [
    "#####################",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...................#",
    "#.###.#.#####.#.###.#",
    "#.....#...#...#.....#",
    "#####.###.#.###.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#.........#.........#",
    "#####.#.#####.#.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...#.....#.....#...#",
    "###.#.#.#####.#.#.###",
    "#.....#...#...#.....#",
    "#.#######.#.#######.#",
    "#...................#",
    "#####################",
]

MOVE_ORDER = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def _valid(pos, ref_map):
    r, c = pos
    H, W = ref_map.shape
    return 0 <= r < H and 0 <= c < W and ref_map[r, c] == 0

def _apply(pos, move):
    return (pos[0] + move.value[0], pos[1] + move.value[1])

def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _cell_exits(pos, ref_map):
    return sum(1 for m in MOVE_ORDER if _valid(_apply(pos, m), ref_map))

def _legal(pos, ref_map):
    return [m for m in MOVE_ORDER if _valid(_apply(pos, m), ref_map)]

# ------------------------------------------------------------
# BFS
# ------------------------------------------------------------
def bfs_dist(ref_map, start, max_dist=40):
    if not _valid(start, ref_map): return {start: 0}
    d = {start: 0}; q = deque([start])
    while q:
        cur = q.popleft()
        if d[cur] >= max_dist: continue
        for m in MOVE_ORDER:
            nxt = _apply(cur, m)
            if nxt not in d and _valid(nxt, ref_map):
                d[nxt] = d[cur] + 1; q.append(nxt)
    return d

# ------------------------------------------------------------
# Topology (same as V3 Ghost)
# ------------------------------------------------------------
class Topo:
    def __init__(self, ref_map):
        H, W = ref_map.shape
        self.dead_ends = set(); self.junctions = set(); self.core = set()
        self.junction_dist = {}

        deg = {}
        for r in range(H):
            for c in range(W):
                if ref_map[r, c] == 1: continue
                p = (r, c); deg[p] = _cell_exits(p, ref_map)
                if deg[p] >= 3: self.junctions.add(p)
                elif deg[p] <= 1: self.dead_ends.add(p)

        dead_seeds = set(self.dead_ends)
        for seed in dead_seeds:
            cur, prev = seed, None
            while True:
                self.dead_ends.add(cur)
                if cur in self.junctions: break
                nxts = [x for x in (_apply(cur, m) for m in MOVE_ORDER) if _valid(x, ref_map) and x != prev]
                if not nxts: break
                nxt = nxts[0]
                if nxt in self.dead_ends and nxt != seed: break
                prev, cur = cur, nxt

        active = {(r, c) for r in range(H) for c in range(W) if ref_map[r, c] == 0}
        changed = True
        while changed:
            changed = False; to_remove = set()
            for cell in active:
                if sum(1 for m in MOVE_ORDER if _valid(_apply(cell, m), ref_map) and _apply(cell, m) in active) <= 1:
                    to_remove.add(cell)
            if to_remove: active -= to_remove; changed = True
        self.core = active

        self.junction_dist = {p: 99 for p in {(r, c) for r in range(H) for c in range(W) if ref_map[r, c] == 0}}
        q = deque()
        starts = self.junctions or self.core
        for p in starts:
            self.junction_dist[p] = 0; q.append(p)
        while q:
            cur = q.popleft()
            for nxt in (_apply(cur, m) for m in MOVE_ORDER):
                if _valid(nxt, ref_map) and self.junction_dist.get(nxt, 99) > self.junction_dist[cur] + 1:
                    self.junction_dist[nxt] = self.junction_dist[cur] + 1; q.append(nxt)


# ------------------------------------------------------------
# Score cell (same as V3 Ghost._score_cell)
# ------------------------------------------------------------
def score_cell(cell, pacman_dist_map, ghost_dist_map, topo, history_set, ref_map):
    pac_dist = pacman_dist_map.get(cell, 99)
    ghost_dist = ghost_dist_map.get(cell, 99)
    if ghost_dist < 2: return float("-inf")

    pacman_eta = (pac_dist + 1) // 2
    margin = pacman_eta - ghost_dist

    score = pac_dist * 200.0 + max(0, margin) * 150.0
    if cell in topo.core: score += 800.0
    if cell in topo.junctions: score += 1200.0
    if cell in topo.dead_ends: score -= 6000.0
    jd = topo.junction_dist.get(cell, 99)
    score += max(0, 4 - jd) * 600.0
    score += _cell_exits(cell, ref_map) * 300.0
    if cell in history_set: score -= 2000.0
    return score


def v3_ghost_move(ghost_pos, pacman_pos, ref_map, topo, history_set):
    """Replicate V3 Ghost's strategic decision for given positions."""
    dist = _manhattan(ghost_pos, pacman_pos)
    gd = bfs_dist(ref_map, ghost_pos, max_dist=20)
    pd = bfs_dist(ref_map, pacman_pos, max_dist=40)

    # Strategic flee
    best_cell, best_score = ghost_pos, float("-inf")
    for cell in gd:
        s = score_cell(cell, pd, gd, topo, history_set, ref_map)
        if s > best_score:
            best_score = s; best_cell = cell

    if best_cell != ghost_pos:
        # A* to best cell
        path = astar(ref_map, ghost_pos, best_cell)
        if path and path[0] in _legal(ghost_pos, ref_map):
            nxt = _apply(ghost_pos, path[0])
            if _manhattan(nxt, pacman_pos) >= 2:
                return path[0]

    # Minimax flee
    moves = _legal(ghost_pos, ref_map)
    if not moves: return Move.STAY
    best_move, best_d = moves[0], -1
    for m in moves:
        ng = _apply(ghost_pos, m)
        if _manhattan(ng, pacman_pos) < 2: continue
        worst_pac_d = float("inf")
        for pm in _legal(pacman_pos, ref_map):
            np1 = _apply(pacman_pos, pm)
            np2 = _apply(np1, pm) if _valid(np1, ref_map) else np1
            d = pd.get(ng, _manhattan(ng, np2))
            worst_pac_d = min(worst_pac_d, d)
        score = worst_pac_d * 100.0 + (_cell_exits(ng, ref_map) - 2) * 50.0
        if ng in topo.junctions: score += 300.0
        if ng in topo.dead_ends: score -= 2000.0
        if score > best_d: best_d = score; best_move = m

    return best_move


# ------------------------------------------------------------
# A*
# ------------------------------------------------------------
import heapq
def astar(ref_map, start, goal):
    if not _valid(start, ref_map) or not _valid(goal, ref_map): return []
    if start == goal: return []
    open_set = [(0, 0, start)]
    came_from = {}; g_score = {start: 0}; closed = set()
    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current in closed: continue
        closed.add(current)
        if current == goal:
            path = []
            while current != start:
                prev, move = came_from[current]; path.append(move); current = prev
            path.reverse(); return path
        for move in MOVE_ORDER:
            nxt = _apply(current, move)
            if not _valid(nxt, ref_map) or nxt in closed: continue
            ng = g + 1
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng; came_from[nxt] = (current, move)
                heapq.heappush(open_set, (ng + _manhattan(nxt, goal), ng, nxt))
    return []


# ------------------------------------------------------------
# Feature extractor
# ------------------------------------------------------------
_GHOST_ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]

def extract_features(ref_map, ghost_pos, pacman_pos, topo):
    H, W = ref_map.shape
    feats = [ghost_pos[0]/H, ghost_pos[1]/W, pacman_pos[0]/H, pacman_pos[1]/W, 1.0]
    for move in _GHOST_ALL_MOVES:
        nxt = _apply(ghost_pos, move)
        if _valid(nxt, ref_map):
            d = _manhattan(nxt, pacman_pos)
            exits = _cell_exits(nxt, ref_map)
            is_dead = 1.0 if nxt in topo.dead_ends else 0.0
            is_junc = 1.0 if nxt in topo.junctions else 0.0
            jd = topo.junction_dist.get(nxt, 99)
        else:
            d = 99; exits = 0; is_dead = 0.0; is_junc = 0.0; jd = 99
        feats.extend([min(d, 40) / 40.0, exits / 4.0, is_dead, is_junc, min(jd, 10) / 10.0])
    return np.array(feats, dtype=np.float32)


# ------------------------------------------------------------
# MLP Model
# ------------------------------------------------------------
class GhostMoveMLP(nn.Module):
    def __init__(self, input_dim=30, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 5),
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict(self, x_np):
        self.eval()
        x_t = torch.from_numpy(x_np).unsqueeze(0)
        return self.forward(x_t).argmax(dim=-1).item()


# ------------------------------------------------------------
# Generate synthetic data
# ------------------------------------------------------------
def generate_data(ref_map, topo, max_samples=50000):
    H, W = ref_map.shape
    empty_cells = [(r, c) for r in range(H) for c in range(W) if ref_map[r, c] == 0]
    print(f"  Map: {H}x{W}, empty cells: {len(empty_cells)}")

    X, Y = [], []
    move_map = {Move.UP: 0, Move.DOWN: 1, Move.LEFT: 2, Move.RIGHT: 3, Move.STAY: 4}

    # Sample pairs where Pacman is visible to Ghost
    sampled = 0
    while sampled < max_samples:
        gp = random.choice(empty_cells)
        pp = random.choice(empty_cells)
        dist = _manhattan(gp, pp)
        if dist < 2 or dist > 30: continue  # too close or too far

        # Check visibility: is Pacman within Ghost's cross-shaped vision?
        visible = False
        for dr, dc in _DIRS:
            r, c = gp
            for d in range(1, 6):
                nr, nc = r + dr * d, c + dc * d
                if not (0 <= nr < H and 0 <= nc < W): break
                if (nr, nc) == pp: visible = True; break
                if ref_map[nr, nc] == 1: break
            if visible: break

        if not visible: continue  # only train on visible-enemy scenarios

        move = v3_ghost_move(gp, pp, ref_map, topo, set())
        if move is None: continue

        x = extract_features(ref_map, gp, pp, topo)
        y = move_map[move]
        X.append(x); Y.append(y)
        sampled += 1

        if sampled % 10000 == 0:
            print(f"    Generated {sampled}/{max_samples} samples")

    print(f"  Total: {len(X)} samples")
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train_mlp(X, Y, epochs=200):
    model = GhostMoveMLP()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    idx = np.random.permutation(len(X))
    split = int(0.85 * len(X))
    X_tr, Y_tr = torch.from_numpy(X[idx[:split]]), torch.from_numpy(Y[idx[:split]])
    X_va, Y_va = torch.from_numpy(X[idx[split:]]), torch.from_numpy(Y[idx[split:]])

    best_acc = 0.0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(X_tr))
        bs = 256
        for i in range(0, len(X_tr), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(X_tr[b]), Y_tr[b])
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_va).argmax(dim=-1)
            acc = (pred == Y_va).float().mean().item()
        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1:3d}: val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), str(SUBMISSION_DIR / "ghost_mlp.pth"))

    print(f"  Best val acc: {best_acc:.3f}")
    return model


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("=" * 56)
    print("  Ghost Move MLP — Synthetic Training")
    print("=" * 56)

    ref_map = np.array([[(0 if c == '.' else 1) for c in row]
                        for row in MAP_LAYOUT], dtype=np.int64)
    # Fix: replace P and G markers
    ref_map = np.where(ref_map == ord('#') - ord('.'), 1, ref_map)
    ref_map = np.array([[1 if MAP_LAYOUT[r][c] == '#' else 0
                          for c in range(len(MAP_LAYOUT[0]))]
                         for r in range(len(MAP_LAYOUT))], dtype=np.int64)

    print("\nPhase 1: Computing topology...")
    topo = Topo(ref_map)
    print(f"  Junctions: {len(topo.junctions)}, Dead-ends: {len(topo.dead_ends)}")

    print("\nPhase 2: Generating synthetic data...")
    X, Y = generate_data(ref_map, topo, max_samples=60000)

    # Show class distribution
    names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
    print("\n  Class distribution:")
    for i, name in enumerate(names):
        print(f"    {name}: {(Y == i).sum():5d} ({(Y == i).sum()/len(Y)*100:.1f}%)")

    print(f"\nPhase 3: Training MLP...")
    model = train_mlp(X, Y, epochs=200)

    # Final accuracy
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).argmax(dim=-1)
        acc = (pred.numpy() == Y).mean()
    print(f"\n  Overall accuracy: {acc:.3f}")
    print(f"  Model saved: ghost_mlp.pth")
    print(f"  Size: {os.path.getsize(SUBMISSION_DIR / 'ghost_mlp.pth') / 1024:.1f} KB")


if __name__ == '__main__':
    main()
