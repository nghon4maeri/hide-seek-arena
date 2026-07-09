"""
Initial submission agent for Hide and Seek Arena.

Idea:
- Pacman/Seek: BFS shortest path to Ghost, then move along the first straight segment.
- Ghost/Hide: evaluate all legal moves and choose the safest one using maze distance,
  immediate capture risk, mobility, and dead-end avoidance.

This file defines both required classes: PacmanAgent and GhostAgent.
"""

import sys
from pathlib import Path
from collections import deque
import random
import numpy as np

# Add src to path to import the framework classes.
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


CARDINAL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]


def add_pos(pos, move):
    """Return new position after applying one Move."""
    dr, dc = move.value
    return (pos[0] + dr, pos[1] + dc)


def manhattan(a, b):
    """Manhattan distance on grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(pos, map_state):
    r, c = pos
    h, w = map_state.shape
    return 0 <= r < h and 0 <= c < w


def is_passable(pos, map_state):
    """
    True if position is inside map and not a wall.
    In fog mode, -1 means unseen but walls are still visible, so -1 is considered passable.
    """
    if not in_bounds(pos, map_state):
        return False
    return map_state[pos[0], pos[1]] != 1


def valid_moves(pos, map_state, include_stay=True):
    """List valid moves from a position."""
    moves = []
    base = ALL_MOVES if include_stay else CARDINAL_MOVES
    for move in base:
        if move == Move.STAY:
            moves.append(move)
        elif is_passable(add_pos(pos, move), map_state):
            moves.append(move)
    return moves


def count_mobility(pos, map_state):
    """Number of non-wall neighboring cells. Higher means fewer dead-end risks."""
    return sum(1 for m in CARDINAL_MOVES if is_passable(add_pos(pos, m), map_state))


def bfs_path(map_state, start, goal):
    """
    Breadth-first search from start to goal.
    Returns a list of Move values. Returns [] if start == goal or no path exists.
    """
    if start is None or goal is None:
        return []
    if start == goal:
        return []

    q = deque([start])
    parent = {start: (None, None)}

    while q:
        cur = q.popleft()
        for move in CARDINAL_MOVES:
            nxt = add_pos(cur, move)
            if not is_passable(nxt, map_state) or nxt in parent:
                continue
            parent[nxt] = (cur, move)
            if nxt == goal:
                path = []
                node = goal
                while parent[node][0] is not None:
                    prev, used_move = parent[node]
                    path.append(used_move)
                    node = prev
                path.reverse()
                return path
            q.append(nxt)

    return []


def bfs_distance(map_state, start, goal):
    """Shortest path length through the maze. Large number if unreachable."""
    if start is None or goal is None:
        return 9999
    if start == goal:
        return 0

    q = deque([(start, 0)])
    visited = {start}

    while q:
        cur, dist = q.popleft()
        for move in CARDINAL_MOVES:
            nxt = add_pos(cur, move)
            if not is_passable(nxt, map_state) or nxt in visited:
                continue
            if nxt == goal:
                return dist + 1
            visited.add(nxt)
            q.append((nxt, dist + 1))

    return 9999


def reachable_after_straight_move(start, move, max_steps, map_state):
    """All positions reachable by repeatedly moving in one direction up to max_steps."""
    positions = [start]
    cur = start
    if move == Move.STAY:
        return positions
    for _ in range(max_steps):
        nxt = add_pos(cur, move)
        if not is_passable(nxt, map_state):
            break
        positions.append(nxt)
        cur = nxt
    return positions


def best_pacman_reachable_positions(pacman_pos, pacman_speed, map_state):
    """Positions Pacman could occupy after one turn."""
    out = set()
    for move in ALL_MOVES:
        out.update(reachable_after_straight_move(pacman_pos, move, pacman_speed, map_state))
    return out


def min_distance_after_one_pacman_turn(pacman_pos, ghost_candidate, pacman_speed, map_state):
    """Minimum Manhattan distance from Ghost candidate after any one straight Pacman move."""
    positions = best_pacman_reachable_positions(pacman_pos, pacman_speed, map_state)
    return min(manhattan(p, ghost_candidate) for p in positions)


def first_straight_segment_length(path, max_speed):
    """How many steps can be taken in the path's first direction."""
    if not path:
        return 1
    first = path[0]
    steps = 0
    for move in path:
        if move != first or steps >= max_speed:
            break
        steps += 1
    return max(1, steps)


class PacmanAgent(BasePacmanAgent):
    """Seek agent: use BFS to chase the Ghost."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Initial BFS Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos = None
        self.rng = random.Random(17)
        self.explore_target = None

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        target = enemy_position or self.last_known_enemy_pos

        if target is not None:
            # Normal mode: plan the shortest path to the Ghost.
            path = bfs_path(map_state, my_position, target)
            if path:
                move = path[0]
                steps = first_straight_segment_length(path, self.pacman_speed)
                return (move, steps)

            # If already very close, stay or make any safe move.
            if manhattan(my_position, target) <= 1:
                return (Move.STAY, 1)

        # Fallback exploration for rare cases: fog mode or no path.
        return self._explore(map_state, my_position, step_number)

    def _explore(self, map_state, my_position, step_number):
        """Simple deterministic exploration: go to a far reachable cell."""
        if self.explore_target is None or my_position == self.explore_target or step_number % 15 == 1:
            self.explore_target = self._choose_far_cell(map_state, my_position)

        if self.explore_target is not None:
            path = bfs_path(map_state, my_position, self.explore_target)
            if path:
                return (path[0], first_straight_segment_length(path, self.pacman_speed))

        moves = valid_moves(my_position, map_state, include_stay=False)
        if not moves:
            return (Move.STAY, 1)
        move = self.rng.choice(moves)
        return (move, 1)

    def _choose_far_cell(self, map_state, my_position):
        q = deque([(my_position, 0)])
        visited = {my_position}
        farthest = (my_position, 0)

        while q:
            cur, dist = q.popleft()
            if dist > farthest[1]:
                farthest = (cur, dist)
            for move in CARDINAL_MOVES:
                nxt = add_pos(cur, move)
                if is_passable(nxt, map_state) and nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, dist + 1))
        return farthest[0]


class GhostAgent(BaseGhostAgent):
    """Hide agent: choose the move with the best safety score."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Initial Safety Ghost"
        self.last_known_enemy_pos = None
        self.rng = random.Random(29)
        # Arena default is 2. Ghost does not receive pacman_speed, so keep a safe estimate.
        self.assumed_pacman_speed = 2

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        pacman_pos = enemy_position or self.last_known_enemy_pos
        moves = valid_moves(my_position, map_state, include_stay=True)

        if not moves:
            return Move.STAY

        if pacman_pos is None:
            # Fog fallback: avoid dead ends and keep moving.
            best = max(moves, key=lambda m: count_mobility(add_pos(my_position, m), map_state) - (1 if m == Move.STAY else 0))
            return best

        best_move = Move.STAY
        best_score = -10**18

        for move in moves:
            new_pos = add_pos(my_position, move)

            maze_dist = bfs_distance(map_state, pacman_pos, new_pos)
            direct_dist = manhattan(pacman_pos, new_pos)
            mobility = count_mobility(new_pos, map_state)
            one_turn_dist = min_distance_after_one_pacman_turn(
                pacman_pos, new_pos, self.assumed_pacman_speed, map_state
            )

            score = 0.0

            # Most important: avoid positions Pacman can capture immediately.
            if one_turn_dist <= 1:
                score -= 10000
            elif one_turn_dist == 2:
                score -= 1200
            elif one_turn_dist == 3:
                score -= 250

            # Prefer being far away through real maze distance, not only Manhattan distance.
            score += maze_dist * 18
            score += direct_dist * 4

            # Avoid dead ends. Corridors are acceptable, junctions are better.
            score += mobility * 35
            if mobility <= 1:
                score -= 180

            # Small preference for movement unless staying is clearly safer.
            if move == Move.STAY:
                score -= 25

            # Avoid repeating predictable tie behavior.
            score += self.rng.random() * 0.01

            if score > best_score:
                best_score = score
                best_move = move

        return best_move
