"""search.py — Adversarial search under partial observability.

Course algorithms implemented here:
  - Minimax with Alpha-Beta pruning (both roles, speed-2 Pacman model)
  - Expectiminimax (chance nodes weighted by the belief state)
  - Monte Carlo rollouts (sample enemy positions, simulate, average)

Every evaluation is an O(1) maze-distance lookup thanks to the
precomputed DistTable, so the trees stay well inside the 1s budget.
"""

import random
import time

from cch_core import INF, CAPTURE_DISTANCE, cross_visible

WIN = 10 ** 7
LOSE = -WIN


class TimeBudget:
    def __init__(self, seconds):
        self.t0 = time.time()
        self.limit = seconds

    def ok(self):
        return (time.time() - self.t0) < self.limit


# ----------------------------------------------------------------------
# Evaluations
# ----------------------------------------------------------------------
def pacman_eval(grid, topo, pac, ghost):
    """Seeker perspective: lower maze distance and fewer ghost exits = better."""
    d = grid.dist_between(pac, ghost)
    if d < CAPTURE_DISTANCE:
        return WIN
    exits = grid.exits(ghost)
    score = -d * 10.0
    score += (4 - min(exits, 4)) * 4.0        # prefer cornering the ghost
    if cross_visible(pac, ghost):
        score += 3.0                          # keep line-of-sight
    w = topo.weight_of(ghost)
    if w <= 0.05:
        score += 30.0                          # ghost trapped in dead end
    elif w <= 0.4:
        score += 10.0                          # ghost in corridor
    return score


def ghost_eval(grid, topo, pac_model, ghost, pac):
    """Hider perspective: safety of standing at `ghost` vs pacman at `pac`."""
    d = grid.dist_between(pac, ghost)
    if d < CAPTURE_DISTANCE:
        return LOSE
    eta = pac_model.eta(grid, pac, ghost)
    score = float(d) * 120.0
    score += min(eta, 15) * 160.0
    score += grid.exits(ghost) * 60.0
    w = topo.weight_of(ghost)
    score += w * 220.0
    if ghost in topo.junctions:
        score += 250.0
    if ghost in topo.loops:
        score += 350.0
    if ghost in topo.dead_ends:
        score -= 9000.0
    if ghost in topo.corridors:
        score -= 1800.0
    jd = topo.junction_dist.get(ghost, 99)
    score += max(0, 4 - jd) * 90.0
    return score


# ----------------------------------------------------------------------
# Alpha-Beta for the Seeker (MAX=Pacman, MIN=Ghost)
# ----------------------------------------------------------------------
class PacmanSearcher:
    def __init__(self, grid, topo, pac_model, time_limit=0.55):
        self.grid = grid
        self.topo = topo
        self.pac_model = pac_model
        self.time_limit = time_limit
        self.nodes = 0
        self.budget = None

    def search(self, pac, ghost, depth=4, time_limit=None):
        """Iterative-deepening-free single alpha-beta pass.

        Returns (action, score)."""
        self.nodes = 0
        self.budget = TimeBudget(time_limit if time_limit is not None
                                 else self.time_limit)
        best_action = None
        best_score = float("-inf")
        alpha, beta = float("-inf"), float("inf")
        actions = self.pac_model.legal_actions(self.grid, pac)
        if not actions:
            return ((0, 0), 0), best_score
        actions.sort(key=lambda a: self.grid.dist_between(
            self.pac_model.apply(self.grid, pac, a[0], a[1]), ghost))
        for action in actions:
            npac = self.pac_model.apply(self.grid, pac, action[0], action[1])
            score = self._min_node(npac, ghost, depth - 1, alpha, beta)
            if score > best_score:
                best_score, best_action = score, action
            alpha = max(alpha, score)
            if not self.budget.ok():
                break
        if best_action is None:
            best_action = actions[0]
        return best_action, best_score

    def _min_node(self, pac, ghost, depth, alpha, beta):
        self.nodes += 1
        if self.grid.dist_between(pac, ghost) < CAPTURE_DISTANCE:
            return WIN + depth
        if depth <= 0 or not self.budget.ok():
            return pacman_eval(self.grid, self.topo, pac, ghost)
        value = float("inf")
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        # Order ghost moves by resulting distance from pacman (far first).
        scored = []
        for m in moves:
            ng = (ghost[0] + m[0], ghost[1] + m[1])
            if not self.grid.walkable(ng):
                continue
            scored.append((self.grid.dist_between(pac, ng), ng))
        scored.sort(reverse=True)
        for _, ng in scored:
            value = min(value, self._max_node(pac, ng, depth - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
            if not self.budget.ok():
                break
        return value

    def _max_node(self, pac, ghost, depth, alpha, beta):
        self.nodes += 1
        if self.grid.dist_between(pac, ghost) < CAPTURE_DISTANCE:
            return WIN + depth
        if depth <= 0 or not self.budget.ok():
            return pacman_eval(self.grid, self.topo, pac, ghost)
        value = float("-inf")
        actions = self.pac_model.legal_actions(self.grid, pac)
        actions.sort(key=lambda a: self.grid.dist_between(
            self.pac_model.apply(self.grid, pac, a[0], a[1]), ghost))
        for action in actions:
            npac = self.pac_model.apply(self.grid, pac, action[0], action[1])
            value = max(value, self._min_node(npac, ghost, depth - 1, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
            if not self.budget.ok():
                break
        return value


# ----------------------------------------------------------------------
# Alpha-Beta for the Hider (MAX=Ghost, MIN=Pacman)
# ----------------------------------------------------------------------
class GhostSearcher:
    def __init__(self, grid, topo, pac_model, time_limit=0.55):
        self.grid = grid
        self.topo = topo
        self.pac_model = pac_model
        self.time_limit = time_limit
        self.nodes = 0
        self.budget = None

    def search(self, ghost, pac, depth=4):
        self.nodes = 0
        self.budget = TimeBudget(self.time_limit)
        best_move = None
        best_score = float("-inf")
        alpha, beta = float("-inf"), float("inf")
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        scored = []
        for m in moves:
            ng = (ghost[0] + m[0], ghost[1] + m[1])
            if self.grid.walkable(ng):
                scored.append((self.grid.dist_between(pac, ng), ng, m))
        scored.sort(reverse=True)  # try safer-looking moves first
        for _, ng, m in scored:
            score = self._min_node(ng, pac, depth - 1, alpha, beta)
            if score > best_score:
                best_score, best_move = score, m
            alpha = max(alpha, score)
            if not self.budget.ok():
                break
        if best_move is None:
            best_move = scored[0][2] if scored else (0, 0)
        return best_move

    def _min_node(self, ghost, pac, depth, alpha, beta):
        self.nodes += 1
        if self.grid.dist_between(pac, ghost) < CAPTURE_DISTANCE:
            return LOSE - depth
        if depth <= 0 or not self.budget.ok():
            return ghost_eval(self.grid, self.topo, self.pac_model, ghost, pac)
        value = float("inf")
        actions = self.pac_model.legal_actions(self.grid, pac)
        # Pacman orders moves that get closer to the ghost first.
        actions.sort(key=lambda a: self.grid.dist_between(
            self.pac_model.apply(self.grid, pac, a[0], a[1]), ghost))
        for action in actions:
            npac = self.pac_model.apply(self.grid, pac, action[0], action[1])
            value = min(value, self._max_node(ghost, npac, depth - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
            if not self.budget.ok():
                break
        return value

    def _max_node(self, ghost, pac, depth, alpha, beta):
        self.nodes += 1
        if self.grid.dist_between(pac, ghost) < CAPTURE_DISTANCE:
            return LOSE - depth
        if depth <= 0 or not self.budget.ok():
            return ghost_eval(self.grid, self.topo, self.pac_model, ghost, pac)
        value = float("-inf")
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        scored = []
        for m in moves:
            ng = (ghost[0] + m[0], ghost[1] + m[1])
            if self.grid.walkable(ng):
                scored.append((self.grid.dist_between(pac, ng), ng))
        scored.sort(reverse=True)
        for _, ng in scored:
            value = max(value, self._min_node(ng, pac, depth - 1, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
            if not self.budget.ok():
                break
        return value


# ----------------------------------------------------------------------
# Expectiminimax — Hider deciding while the Pacman is hidden
# ----------------------------------------------------------------------
def expectiminimax_ghost(grid, topo, pac_model, ghost, belief, depth=1):
    """Chance nodes over sampled pacman positions (belief-weighted).

    For each ghost move:  expected value = sum_samples  prob *
        (value of pacman's best response, evaluated 1 ply later).
    """
    moves = []
    for m in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ng = (ghost[0] + m[0], ghost[1] + m[1])
        if grid.walkable(ng):
            moves.append((ng, m))
    if not moves:
        return (0, 0)

    samples = belief.threat_samples(8)
    total_w = sum(w for _, w in samples) or 1.0

    best_move, best_score = moves[0][1], float("-inf")
    for ng, m in moves:
        expected = 0.0
        for pac_pos, w in samples:
            actions = pac_model.legal_actions(grid, pac_pos)
            worst = float("inf")
            for action in actions:
                npac = pac_model.apply(grid, pac_pos, action[0], action[1])
                val = ghost_eval(grid, topo, pac_model, ng, npac)
                if val < worst:
                    worst = val
            if not actions:
                worst = ghost_eval(grid, topo, pac_model, ng, pac_pos)
            expected += (w / total_w) * worst
        if expected > best_score:
            best_score, best_move = expected, m
    return best_move


# ----------------------------------------------------------------------
# Monte Carlo rollouts — Hider candidate validation
# ----------------------------------------------------------------------
def mc_rollout_ghost(grid, topo, pac_model, ghost, candidate_moves, belief,
                     rollouts=10, horizon=8):
    """Simulate: pacman samples from belief then chases greedily with
    speed 2; ghost flees greedily.  Average the safety of each candidate."""
    rng = random.Random(42)
    samples = belief.threat_samples(10)
    cells = [p for p, _ in samples]
    weights = [w for _, w in samples]

    def greedy_pacman(pac, g):
        best, best_d = pac, grid.dist_between(pac, g)
        for action in pac_model.legal_actions(grid, pac):
            npac = pac_model.apply(grid, pac, action[0], action[1])
            d = grid.dist_between(npac, g)
            if d < best_d:
                best_d, best = d, npac
        return best

    def greedy_ghost(g, pac):
        best, best_s = None, float("-inf")
        for n in grid.neighbors(g):
            s = ghost_eval(grid, topo, pac_model, n, pac)
            if s > best_s:
                best_s, best = s, n
        return best if best is not None else g

    scores = {m: 0.0 for m in candidate_moves}
    for _ in range(rollouts):
        pac = rng.choices(cells, weights=weights, k=1)[0]
        for m in candidate_moves:
            g = (ghost[0] + m[0], ghost[1] + m[1])
            if not grid.walkable(g):
                scores[m] -= 50000.0
                continue
            alive = True
            for _ in range(horizon):
                if grid.dist_between(pac, g) < CAPTURE_DISTANCE:
                    alive = False
                    break
                pac = greedy_pacman(pac, g)
                if grid.dist_between(pac, g) < CAPTURE_DISTANCE:
                    alive = False
                    break
                g = greedy_ghost(g, pac)
            if alive:
                scores[m] += 100.0 + grid.dist_between(pac, g) * 5.0
                if g in topo.loops:
                    scores[m] += 30.0
            else:
                scores[m] -= 4000.0
    return scores
