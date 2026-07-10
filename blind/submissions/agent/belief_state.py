"""belief_state.py — Bayesian belief state tracker for enemy location.

Direction 1: Belief State Estimation
- Maintains 21x21 probability matrix over enemy location
- Collapses to point mass when enemy seen
- Propagates via Markov chain (BFS) when enemy lost
- Computes danger_time[r,c] = estimated steps for enemy to reach (r,c)
"""

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from pathfinding import DIRS, _shape, _cell, _valid, _apply, _manhattan, bfs_dist


class BeliefState:
    """Maintains probability distribution over enemy location.

    When enemy is seen: belief collapses to point mass.
    When enemy is lost: nullify visible cells, propagate via BFS.
    Also maintains danger_time[r,c] = estimated steps for enemy to reach (r,c).
    """

    def __init__(self, H: int = 21, W: int = 21):
        self.H, self.W = H, W
        self.belief = np.ones((H, W), dtype=np.float64) / (H * W)
        self.danger_time = np.full((H, W), 999, dtype=np.int32)
        self.last_seen_pos: Optional[Tuple] = None
        self.steps_since_seen: int = 0

    def update(self, my_pos: Tuple, enemy_pos: Optional[Tuple],
               memory_map: np.ndarray, enemy_speed: int = 2):
        """Update belief state based on observation."""
        if enemy_pos is not None:
            self.belief.fill(0.0)
            er, ec = enemy_pos
            if 0 <= er < self.H and 0 <= ec < self.W:
                self.belief[er, ec] = 1.0
            self.last_seen_pos = enemy_pos
            self.steps_since_seen = 0
            self._update_danger_time(enemy_pos, memory_map, enemy_speed)
            return

        self.steps_since_seen += 1
        self._nullify_visible(memory_map)
        self._propagate(memory_map, enemy_speed)
        total = self.belief.sum()
        if total > 0:
            self.belief /= total
        self._update_danger_time_from_belief(memory_map, enemy_speed)

    def _nullify_visible(self, memory_map):
        """Set belief to 0 for all cells we can currently see (empty = no enemy)."""
        for r in range(self.H):
            for c in range(self.W):
                if _cell(memory_map, r, c) == 0:
                    self.belief[r, c] = 0.0

    def _propagate(self, memory_map, speed):
        """Propagate belief via BFS expansion (Markov chain)."""
        new_belief = np.zeros_like(self.belief)
        for r in range(self.H):
            for c in range(self.W):
                prob = self.belief[r, c]
                if prob <= 0:
                    continue
                reachable = self._bfs_reachable((r, c), memory_map, speed)
                denom = max(1, len(reachable))
                for cell in reachable:
                    new_belief[cell[0], cell[1]] += prob / denom
        self.belief = new_belief

    def _bfs_reachable(self, start, memory_map, max_dist):
        """BFS to find all cells reachable within max_dist steps."""
        reachable = [start]
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            cur, dist = queue.popleft()
            if dist >= max_dist:
                continue
            for dr, dc in DIRS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in visited:
                    continue
                if not _valid(nxt, memory_map):
                    continue
                visited.add(nxt)
                reachable.append(nxt)
                queue.append((nxt, dist + 1))
        return reachable

    def _update_danger_time(self, enemy_pos, memory_map, speed):
        """Update danger_time: BFS from enemy position."""
        self.danger_time.fill(999)
        dist_map = bfs_dist(memory_map, enemy_pos, max_dist=40)
        for cell, dist in dist_map.items():
            r, c = cell
            self.danger_time[r, c] = max(1, dist // speed)

    def _update_danger_time_from_belief(self, memory_map, speed):
        """Update danger_time from belief distribution."""
        self.danger_time.fill(999)
        high_prob = np.argwhere(self.belief > 0.05)
        for idx in high_prob:
            r, c = idx[0], idx[1]
            dist_map = bfs_dist(memory_map, (r, c), max_dist=20)
            for cell, dist in dist_map.items():
                cr, cc = cell
                time_est = max(1, dist // speed)
                self.danger_time[cr, cc] = min(self.danger_time[cr, cc], time_est)

    def threat_center(self) -> Tuple[int, int]:
        """Weighted centroid of belief distribution."""
        total = self.belief.sum()
        if total == 0:
            return (self.H // 2, self.W // 2)
        r_center = float(np.sum(np.arange(self.H)[:, None] * self.belief) / total)
        c_center = float(np.sum(np.arange(self.W) * self.belief.sum(axis=0)) / total)
        return (int(r_center), int(c_center))

    def highest_prob_cells(self, top_k: int = 5) -> List[Tuple[Tuple, float]]:
        """Return top-k cells with highest probability."""
        flat = self.belief.ravel()
        if flat.sum() == 0:
            return []
        indices = np.argsort(flat)[::-1][:top_k]
        cells = []
        for idx in indices:
            r, c = divmod(idx, self.W)
            if self.belief[r, c] > 0:
                cells.append(((r, c), float(self.belief[r, c])))
        return cells

    def prob_at(self, pos: Tuple) -> float:
        """Get probability at a specific position."""
        r, c = pos
        if 0 <= r < self.H and 0 <= c < self.W:
            return float(self.belief[r, c])
        return 0.0

    def danger_at(self, pos: Tuple) -> int:
        """Get estimated time for enemy to reach position."""
        r, c = pos
        if 0 <= r < self.H and 0 <= c < self.W:
            return int(self.danger_time[r, c])
        return 999
