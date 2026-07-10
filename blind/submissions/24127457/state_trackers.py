"""state_trackers.py — Fog-of-war memory and probabilistic belief tracking.

Provides:
  SpatialHeatmap   — tracks how long each cell has been unseen (fog age)
  BeliefStateTracker — Bayesian probability distribution over enemy location
"""

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from topology_analyzer import (
    DIRS, MOVE_ORDER, TopologyAnalyzer,
    _cell, _shape, _manhattan,
)


# ===================================================================
# Spatial Heatmap — fog-age tracker with topology weighting
# ===================================================================
class SpatialHeatmap:
    """Tracks how many ticks each cell has been unseen.

    Cells that are currently visible get reset to 0.
    Cells that remain unseen accumulate age each step.
    """

    def __init__(self, H: int, W: int):
        self.heat = np.zeros((H, W), dtype=np.int64)
        self.H, self.W = H, W

    def update(self, visible_cells):
        self.heat += 1
        for r, c in visible_cells:
            if 0 <= r < self.H and 0 <= c < self.W:
                self.heat[r, c] = 0

    def best_target(self, ms, my_pos, last_enemy=None, topo=None):
        """Find the best exploration target: high fog-age + topology weight."""
        H, W = self.H, self.W
        best, best_score = None, -1
        for r in range(H):
            for c in range(W):
                if _cell(ms, r, c) != 0:
                    continue
                ticks = self.heat[r, c]
                if ticks < 3:
                    continue
                has_fog = any(
                    0 <= r + dr < H and 0 <= c + dc < W
                    and _cell(ms, r + dr, c + dc) == -1
                    for dr, dc in DIRS
                )
                if not has_fog:
                    continue
                geo_mult = 1.0
                if topo is not None and topo.ready:
                    geo_mult = topo._get_topological_weight((r, c), ms)
                score = ticks * geo_mult * 10
                if last_enemy is not None:
                    d = _manhattan((r, c), last_enemy)
                    if d <= 12:
                        score += (12 - d) * geo_mult * 3
                if score > best_score:
                    best_score = score
                    best = (r, c)
        return best


# ===================================================================
# Belief State Tracker — Bayesian enemy location estimator
# ===================================================================
class BeliefStateTracker:
    """Maintains a probability distribution over enemy location.

    When enemy is seen: belief collapses to a point mass.
    When enemy is lost: belief spreads via BFS expansion weighted by
    the enemy's movement speed, nullifying cells that are currently
    visible (enemy cannot be in a cell we can see).
    """

    def __init__(self, H: int = 21, W: int = 21):
        self.H, self.W = H, W
        self.belief = np.ones((H, W), dtype=np.float64) / (H * W)

    def update(self, ghost_pos, enemy_pos, memory_map):
        if enemy_pos is not None:
            self.belief.fill(0.0)
            er, ec = enemy_pos
            if 0 <= er < self.H and 0 <= ec < self.W:
                self.belief[er, ec] = 1.0
            return
        self._nullify_visible(memory_map)
        self._bfs_expand(ghost_pos, speed=2)
        total = self.belief.sum()
        if total > 0:
            self.belief /= total

    def _nullify_visible(self, memory_map):
        for r in range(self.H):
            for c in range(self.W):
                if _cell(memory_map, r, c) == 0:
                    self.belief[r, c] = 0.0

    def _bfs_expand(self, origin, speed):
        new_belief = np.zeros_like(self.belief)
        for r in range(self.H):
            for c in range(self.W):
                prob = self.belief[r, c]
                if prob <= 0:
                    continue
                queue = deque([(r, c, 0)])
                visited = {(r, c)}
                reachable = [(r, c)]
                while queue:
                    cr, cc, d = queue.popleft()
                    if d >= speed:
                        continue
                    for dr, dc in DIRS:
                        nr, nc = cr + dr, cc + dc
                        if not (0 <= nr < self.H and 0 <= nc < self.W):
                            continue
                        if (nr, nc) in visited:
                            continue
                        visited.add((nr, nc))
                        reachable.append((nr, nc))
                        queue.append((nr, nc, d + 1))
                denom = max(1, len(reachable))
                for cell in reachable:
                    new_belief[cell[0], cell[1]] += prob / denom
        self.belief = new_belief

    def highest_threat_cells(self, top_k: int = 3):
        flat = self.belief.ravel()
        if flat.sum() == 0:
            return []
        indices = np.argsort(flat)[::-1][:top_k]
        cells = []
        for idx in indices:
            r, c = divmod(idx, self.W)
            if self.belief[r, c] > 0:
                cells.append(((r, c), self.belief[r, c]))
        return cells

    def threat_center(self) -> Tuple[int, int]:
        total = self.belief.sum()
        if total == 0:
            return (self.H // 2, self.W // 2)
        r_center = float(np.sum(np.arange(self.H)[:, None] * self.belief) / total)
        c_center = float(np.sum(np.arange(self.W) * self.belief.sum(axis=0)) / total)
        return (int(r_center), int(c_center))
