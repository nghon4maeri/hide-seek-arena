"""belief.py — Bayesian belief state over the enemy's position (POMDP-style).

Belief update:
  - enemy visible      -> point mass at its position
  - enemy hidden       -> zero out cells we currently observe empty,
                          then diffuse each remaining mass over the enemy's
                          one-turn reachable set, and renormalize.
The reachable set differs by role: the Ghost moves 1 cell/turn, the
Pacman moves up to 2 straight cells/turn.
"""

import numpy as np

from cch_core import DIRS, INF


class EnemyBelief:
    def __init__(self, grid, reach_fn, prior_cells=None):
        """prior_cells: cells the enemy can plausibly start in (used for
        the initial uniform prior).  If None, uniform over the whole map."""
        self.grid = grid
        self.reach_fn = reach_fn
        self.prob = np.zeros((grid.h, grid.w), dtype=np.float64)
        if prior_cells:
            cells = list(prior_cells)
        else:
            cells = list(grid.free)
        total = 0.0
        for r, c in cells:
            self.prob[r, c] = 1.0
            total += 1.0
        if total > 0:
            self.prob /= total
        self.last_seen = None
        self.steps_since_seen = 0

    def update(self, my_pos, enemy_pos, visible_empty, step, flee_center=None):
        """Bayesian update.

        flee_center: when given, cells within 7 maze-steps of it diffuse
        with flee-weighted transitions (enemy likely running AWAY from
        that position — inherited from team 6's weighted belief)."""
        g = self.grid
        if enemy_pos is not None:
            self.prob.fill(0.0)
            r, c = int(enemy_pos[0]), int(enemy_pos[1])
            if g.walkable((r, c)):
                self.prob[r, c] = 1.0
            self.last_seen = enemy_pos
            self.steps_since_seen = 0
            return

        self.steps_since_seen += 1
        # Cells we currently see as empty cannot hold the enemy.
        for r, c in visible_empty:
            self.prob[r, c] = 0.0

        # Diffuse over one-turn reachable sets.
        new_prob = np.zeros_like(self.prob)
        idx = np.argwhere(self.prob > 0)
        for r, c in idx:
            mass = self.prob[r, c]
            cell = (int(r), int(c))
            reach = self.reach_fn(cell)
            if flee_center is not None and g.dist_between(flee_center, cell) <= 7:
                d0 = g.dist_between(flee_center, cell)
                weights = []
                for rc in reach:
                    d1 = g.dist_between(flee_center, rc)
                    w = 2.0 if d1 > d0 else (1.0 if d1 == d0 else 0.5)
                    weights.append(w)
                total_w = sum(weights)
                for rc, w in zip(reach, weights):
                    new_prob[rc[0], rc[1]] += mass * w / total_w
            else:
                share = mass / max(1, len(reach))
                for (rr, cc) in reach:
                    new_prob[rr, cc] += share
        total = new_prob.sum()
        if total > 0:
            new_prob /= total
        self.prob = new_prob

    def prob_at(self, pos):
        r, c = pos
        if 0 <= r < self.grid.h and 0 <= c < self.grid.w:
            return float(self.prob[r, c])
        return 0.0

    def threat_center(self):
        g = self.grid
        rows = np.arange(g.h)[:, None]
        cols = np.arange(g.w)
        total = self.prob.sum()
        if total <= 0:
            return (g.h // 2, g.w // 2)
        rr = int(np.sum(rows * self.prob) / total)
        cc = int(np.sum(cols * self.prob.sum(axis=0)) / total)
        return (rr, cc)

    def top_k(self, k=6):
        flat = self.prob.ravel()
        order = np.argsort(flat)[::-1][:k]
        out = []
        w = self.grid.w
        for idx in order:
            r, c = divmod(int(idx), w)
            if self.prob[r, c] > 1e-9:
                out.append(((r, c), float(self.prob[r, c])))
        return out

    def mass(self):
        return float(self.prob.sum())

    def threat_samples(self, k=8):
        """Representative enemy positions with weights.

        Concentrated belief -> top-k cells.  Flat belief (early game) ->
        the probability centroid weighted 3x plus evenly-spread support
        cells (weight 1x) so decisions hedge against every direction."""
        peak = float(self.prob.max())
        if peak >= 0.05:
            return [(p, w) for (p, w) in self.top_k(k) if w >= 0.01]
        cells = [(r, c) for r in range(self.grid.h) for c in range(self.grid.w)
                 if self.prob[r, c] > 1e-9]
        out = [(self.threat_center(), 3.0)]
        if cells:
            step = max(1, len(cells) // 6)
            for i in range(0, len(cells), step):
                out.append((cells[i], 1.0))
                if len(out) >= 7:
                    break
        return out


def ghost_reach_fn(grid):
    def reach(pos):
        out = {pos}
        for n in grid.neighbors(pos):
            out.add(n)
        return out
    return reach


def pacman_reach_fn(grid, pac_model):
    def reach(pos):
        return pac_model.reach_set(grid, pos)
    return reach


def danger_time_for(grid, pac_model, belief, cell):
    """Minimum turns any high-probability pacman position needs to reach cell."""
    best = INF
    for (pos, prob) in belief.top_k(6):
        if prob < 0.02:
            continue
        eta = pac_model.eta(grid, pos, cell)
        if eta < best:
            best = eta
    return best
