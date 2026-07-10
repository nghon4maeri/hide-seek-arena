"""
Pacman vs Ghost Arena agent.

Strategy:
- Use BFS distance on the maze instead of simple Manhattan distance.
- Pacman tries all legal moves with speed 1 or 2, then chooses the move
  that keeps Ghost closest after Ghost's possible reply.
- Ghost tries all legal moves, predicts Pacman's best chase move, avoids
  capture, avoids dead ends, and maximizes escape distance.
"""

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


CARDINAL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
ALL_GHOST_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
INF = 10**9


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_valid_position(pos, map_state):
    row, col = pos
    height, width = map_state.shape

    if row < 0 or row >= height or col < 0 or col >= width:
        return False

    return map_state[row, col] == 0


def apply_one(pos, move):
    dr, dc = move.value
    return (pos[0] + dr, pos[1] + dc)


def simulate_steps(pos, move, steps, map_state):
    if move == Move.STAY:
        return pos

    cur = pos

    for _ in range(steps):
        nxt = apply_one(cur, move)

        if not is_valid_position(nxt, map_state):
            return None

        cur = nxt

    return cur


def valid_pacman_actions(pos, map_state, pacman_speed):
    actions = []

    for move in CARDINAL_MOVES:
        for steps in range(1, pacman_speed + 1):
            nxt = simulate_steps(pos, move, steps, map_state)

            if nxt is None:
                break

            actions.append((move, steps, nxt))

    actions.append((Move.STAY, 1, pos))
    return actions


def valid_ghost_actions(pos, map_state):
    actions = []

    for move in ALL_GHOST_MOVES:
        nxt = simulate_steps(pos, move, 1, map_state)

        if nxt is not None:
            actions.append((move, nxt))

    return actions


def bfs_distances(start, map_state):
    dist = {start: 0}
    q = deque([start])

    while q:
        cur = q.popleft()

        for move in CARDINAL_MOVES:
            nxt = apply_one(cur, move)

            if nxt in dist:
                continue

            if not is_valid_position(nxt, map_state):
                continue

            dist[nxt] = dist[cur] + 1
            q.append(nxt)

    return dist


def bfs_distance(start, goal, map_state):
    if start == goal:
        return 0

    return bfs_distances(start, map_state).get(goal, INF)


def escape_degree(pos, map_state):
    count = 0

    for move in CARDINAL_MOVES:
        nxt = apply_one(pos, move)

        if is_valid_position(nxt, map_state):
            count += 1

    return count


def choose_random_valid_pacman(pos, map_state, pacman_speed):
    actions = valid_pacman_actions(pos, map_state, pacman_speed)
    move, steps, _ = random.choice(actions)
    return (move, steps)


def choose_random_valid_ghost(pos, map_state):
    actions = valid_ghost_actions(pos, map_state)
    move, _ = random.choice(actions)
    return move


class PacmanAgent(BasePacmanAgent):
    """
    Pacman / Seek agent.
    Goal: catch the Ghost as quickly as possible.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BFS Minimax Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos = None

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            return choose_random_valid_pacman(
                my_position,
                map_state,
                self.pacman_speed,
            )

        best_action = (Move.STAY, 1)
        best_score = -INF

        ghost_replies = valid_ghost_actions(target, map_state)

        if not ghost_replies:
            ghost_replies = [(Move.STAY, target)]

        for move, steps, pac_next in valid_pacman_actions(
            my_position,
            map_state,
            self.pacman_speed,
        ):
            worst_after_reply = -INF
            capture_all_replies = True

            for _, ghost_next in ghost_replies:
                path_dist = bfs_distance(pac_next, ghost_next, map_state)
                close_dist = manhattan(pac_next, ghost_next)

                if close_dist >= 2:
                    capture_all_replies = False

                reply_value = path_dist * 10 + close_dist
                worst_after_reply = max(worst_after_reply, reply_value)

            score = -worst_after_reply

            if capture_all_replies:
                score += 100000

            score += steps * 0.2

            if score > best_score:
                best_score = score
                best_action = (move, steps)

        return best_action


class GhostAgent(BaseGhostAgent):
    """
    Ghost / Hide agent.
    Goal: survive as long as possible.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BFS Evasive Ghost"
        self.last_known_enemy_pos = None

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        threat = enemy_position or self.last_known_enemy_pos

        if threat is None:
            return choose_random_valid_ghost(my_position, map_state)

        assumed_pacman_speed = 2
        pacman_actions = valid_pacman_actions(
            threat,
            map_state,
            assumed_pacman_speed,
        )
        threat_distances = bfs_distances(threat, map_state)

        best_move = Move.STAY
        best_score = -INF

        for move, ghost_next in valid_ghost_actions(my_position, map_state):
            closest_after_pacman = INF
            captured_by_some_action = False

            for _, _, pac_next in pacman_actions:
                d = bfs_distance(pac_next, ghost_next, map_state)
                closest_after_pacman = min(closest_after_pacman, d)

                if manhattan(pac_next, ghost_next) < 2:
                    captured_by_some_action = True

            current_threat_dist = threat_distances.get(ghost_next, INF)
            degree = escape_degree(ghost_next, map_state)

            score = 0
            score += closest_after_pacman * 25
            score += current_threat_dist * 6
            score += degree * 4

            if degree <= 1:
                score -= 30

            if captured_by_some_action:
                score -= 100000

            if move == Move.STAY:
                score -= 1

            if score > best_score:
                best_score = score
                best_move = move

        return best_move