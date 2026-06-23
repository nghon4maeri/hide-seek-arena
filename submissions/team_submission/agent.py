"""Team submission agent for CSC14003 Hide and Seek Arena.

This file follows STUDENT_GUIDE.md:
- defines PacmanAgent and GhostAgent
- keeps the official step(...) signatures
- validates moves before returning them
- handles enemy_position=None for limited observation mode

The implementation is intentionally lightweight and submission-safe. It uses
standard grid search helpers inside this file and does not modify the arena
framework in pacman/src.
"""

from __future__ import annotations

from collections import deque
import sys
from pathlib import Path


SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


MOVE_ORDER = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)


def _shape(map_state):
    if hasattr(map_state, "shape"):
        return int(map_state.shape[0]), int(map_state.shape[1])
    return len(map_state), len(map_state[0]) if map_state else 0


def _cell(map_state, row: int, col: int) -> int:
    return int(map_state[row, col] if hasattr(map_state, "shape") else map_state[row][col])


def _apply_move(pos, move: Move):
    dr, dc = move.value
    return pos[0] + dr, pos[1] + dc


def _is_valid_position(pos, map_state) -> bool:
    row, col = pos
    height, width = _shape(map_state)
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    return _cell(map_state, row, col) == 0


def _valid_moves(pos, map_state):
    return [move for move in MOVE_ORDER if _is_valid_position(_apply_move(pos, move), map_state)]


def _manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _bfs_path(map_state, start, goal):
    """Return a list of moves from start to goal, or [] if unreachable."""
    if goal is None or not _is_valid_position(start, map_state) or not _is_valid_position(goal, map_state):
        return []

    queue = deque([start])
    parent = {start: (None, None)}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for move in MOVE_ORDER:
            nxt = _apply_move(current, move)
            if nxt not in parent and _is_valid_position(nxt, map_state):
                parent[nxt] = (current, move)
                queue.append(nxt)

    if goal not in parent:
        return []

    path = []
    current = goal
    while current != start:
        previous, move = parent[current]
        path.append(move)
        current = previous
    path.reverse()
    return path


def _distance_map(map_state, start):
    if start is None or not _is_valid_position(start, map_state):
        return {}

    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for move in MOVE_ORDER:
            nxt = _apply_move(current, move)
            if nxt not in distances and _is_valid_position(nxt, map_state):
                distances[nxt] = distances[current] + 1
                queue.append(nxt)
    return distances


def _reachable_count(map_state, start, max_depth: int = 8) -> int:
    if not _is_valid_position(start, map_state):
        return 0

    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for move in MOVE_ORDER:
            nxt = _apply_move(current, move)
            if nxt not in seen and _is_valid_position(nxt, map_state):
                seen.add(nxt)
                queue.append((nxt, depth + 1))
    return len(seen)


class PacmanAgent(BasePacmanAgent):
    """Pacman seeker: use BFS toward the visible or last known ghost position."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos = None
        self.visited = set()

    def step(self, map_state, my_position, enemy_position, step_number):
        my_position = tuple(my_position)
        self.visited.add(my_position)

        if enemy_position is not None:
            self.last_known_enemy_pos = tuple(enemy_position)

        target = self.last_known_enemy_pos
        if target is not None:
            path = _bfs_path(map_state, my_position, target)
            if path:
                move = path[0]
                steps = self._straight_steps_from_path(path, my_position, map_state)
                return (move, steps)

        explore_move = self._choose_exploration_move(map_state, my_position)
        return (explore_move, 1)

    def _straight_steps_from_path(self, path, my_position, map_state) -> int:
        first_move = path[0]
        desired = 0
        for move in path:
            if move != first_move:
                break
            desired += 1

        current = my_position
        valid_steps = 0
        for _ in range(min(self.pacman_speed, desired)):
            nxt = _apply_move(current, first_move)
            if not _is_valid_position(nxt, map_state):
                break
            valid_steps += 1
            current = nxt
        return max(1, valid_steps)

    def _choose_exploration_move(self, map_state, my_position):
        candidates = _valid_moves(my_position, map_state)
        if not candidates:
            return Move.STAY

        def score(move):
            nxt = _apply_move(my_position, move)
            unvisited_bonus = 5 if nxt not in self.visited else 0
            return unvisited_bonus + _reachable_count(map_state, nxt, max_depth=5)

        return max(candidates, key=score)


class GhostAgent(BaseGhostAgent):
    """Ghost hider: maximize distance and local mobility."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None

    def step(self, map_state, my_position, enemy_position, step_number):
        my_position = tuple(my_position)

        if enemy_position is not None:
            self.last_known_enemy_pos = tuple(enemy_position)

        candidates = _valid_moves(my_position, map_state)
        if not candidates:
            return Move.STAY

        threat = self.last_known_enemy_pos
        if threat is None:
            return max(candidates, key=lambda move: _reachable_count(map_state, _apply_move(my_position, move), max_depth=6))

        threat_distances = _distance_map(map_state, threat)

        def score(move):
            nxt = _apply_move(my_position, move)
            distance = threat_distances.get(nxt, _manhattan(nxt, threat) + 20)
            mobility = _reachable_count(map_state, nxt, max_depth=6)
            return 10 * distance + mobility

        return max(candidates, key=score)
