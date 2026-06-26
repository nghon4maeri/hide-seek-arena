"""Opponent modeling for pursuit-evasion agents.

Based on:
- Carmel & Markovitch (1996), "Learning Models of Intelligent Agents", AAAI
- Southey et al. (2005), "Bayesian Opponent Modeling in Adversarial Search", IJCAI
"""

from typing import Optional, Tuple


class GhostDirectionModel:
    """Tracks Ghost movement patterns from Pacman's perspective.

    Builds a frequency table of direction transitions from observed
    history. Used by PacmanAgent for interception planning.

    Reference — Carmel & Markovitch (1996): opponent modeling via
    finite-state machine observation, learning transition probabilities
    from move history to predict next action.
    """

    def __init__(self, window: int = 20):
        self._history: list = []
        self._dir_history: list = []
        self._freq: dict = {}
        self._window = window

    def observe(self, old_pos, new_pos):
        if old_pos is None or new_pos is None:
            return
        dr = new_pos[0] - old_pos[0]
        dc = new_pos[1] - old_pos[1]
        self._history.append(new_pos)
        self._dir_history.append((dr, dc))
        while len(self._history) > self._window:
            self._history.pop(0)
            self._dir_history.pop(0)
        self._freq.setdefault(old_pos, {})
        self._freq[old_pos][(dr, dc)] = self._freq[old_pos].get((dr, dc), 0) + 1

    def predict_next(self, pos, ms) -> Optional[Tuple]:
        """Predict Ghost's next position.

        Uses frequency table for this position; falls back to last
        direction if no history for this position.
        """
        from heuristic import is_valid

        if pos in self._freq and self._freq[pos]:
            best_dir = max(self._freq[pos], key=self._freq[pos].get)
            nxt = (pos[0] + best_dir[0], pos[1] + best_dir[1])
            if is_valid(nxt, ms):
                return nxt

        if self._dir_history:
            last_dir = self._dir_history[-1]
            nxt = (pos[0] + last_dir[0], pos[1] + last_dir[1])
            if is_valid(nxt, ms):
                return nxt

        return None

    def current_direction(self):
        if self._dir_history:
            return self._dir_history[-1]
        return None

    def is_confident(self) -> bool:
        return len(self._history) >= 3


class PacmanPursuitModel:
    """Tracks Pacman behavior from Ghost's perspective.

    Estimates interception tendency via exponential moving average
    and predicts Pacman's target after speed-2 move.

    Reference — Southey et al. (2005): Bayesian opponent modeling
    with exponential moving average for strategy inference.
    """

    def __init__(self, window: int = 20):
        self._history: list = []
        self._dir_history: list = []
        self._window = window
        self._intercept_ema: float = 0.0

    def observe(self, old_pos, new_pos, ghost_pos):
        if old_pos is None or new_pos is None:
            return
        dr = new_pos[0] - old_pos[0]
        dc = new_pos[1] - old_pos[1]
        self._history.append(new_pos)
        self._dir_history.append((dr, dc))
        while len(self._history) > self._window:
            self._history.pop(0)
            self._dir_history.pop(0)

        from heuristic import manhattan

        old_dist = manhattan(old_pos, ghost_pos)
        new_dist = manhattan(new_pos, ghost_pos)
        toward = 1.0 if new_dist < old_dist else (0.3 if new_dist == old_dist else 0.0)
        self._intercept_ema = 0.85 * self._intercept_ema + 0.15 * toward

    def predict_target(self, pos, ghost_pos, ms) -> Tuple:
        """Predict Pacman's target after speed-2 move.

        Projects 2 steps in Pacman's last direction if it reduces
        distance to Ghost (indicating interception pattern).
        """
        from heuristic import is_valid, manhattan

        if not self._dir_history:
            return pos

        last_dir = self._dir_history[-1]
        p1 = (pos[0] + last_dir[0], pos[1] + last_dir[1])
        p2 = (p1[0] + last_dir[0], p1[1] + last_dir[1])

        if is_valid(p2, ms) and manhattan(p2, ghost_pos) < manhattan(pos, ghost_pos):
            return p2
        if is_valid(p1, ms):
            return p1
        return pos

    def is_aggressive(self) -> bool:
        return self._intercept_ema > 0.4

    def confidence(self) -> float:
        return min(1.0, len(self._history) / 5.0)
