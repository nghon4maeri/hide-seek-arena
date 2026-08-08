"""
A* Hide-and-Seek agents.

Pacman uses Echo-Intercept A*: it predicts the Ghost's next likely tile and
searches for the nearest capture tile around that prediction. Ghost uses
Sanctuary A*: it ranks hideouts by distance, escape space, and corridor safety,
then searches toward the best one.
"""

import heapq
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


MOVES = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
UNKNOWN = -1
EMPTY = 0
WALL = 1


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _apply(pos, move):
    dr, dc = move.value
    return pos[0] + dr, pos[1] + dc


def _is_valid(pos, map_state, allow_unseen=False):
    row, col = pos
    height, width = map_state.shape
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    return map_state[row, col] == 0 or (allow_unseen and map_state[row, col] == -1)


def _neighbors(pos, map_state, include_stay=False, allow_unseen=False):
    result = []
    for move in MOVES:
        nxt = _apply(pos, move)
        if _is_valid(nxt, map_state, allow_unseen):
            result.append((nxt, move))
    if include_stay:
        result.append((pos, Move.STAY))
    return result


def _degree(pos, map_state):
    return len(_neighbors(pos, map_state))


def _line_of_sight(a, b, map_state):
    if a[0] == b[0]:
        row = a[0]
        start, end = sorted((a[1], b[1]))
        for col in range(start + 1, end):
            if map_state[row, col] == 1:
                return False
        return True
    if a[1] == b[1]:
        col = a[1]
        start, end = sorted((a[0], b[0]))
        for row in range(start + 1, end):
            if map_state[row, col] == 1:
                return False
        return True
    return False


def _distance_map(sources, map_state):
    queue = deque()
    dist = {}
    for source in sources:
        if _is_valid(source, map_state) and source not in dist:
            dist[source] = 0
            queue.append(source)

    while queue:
        pos = queue.popleft()
        for nxt, _ in _neighbors(pos, map_state):
            if nxt not in dist:
                dist[nxt] = dist[pos] + 1
                queue.append(nxt)
    return dist


def _maze_distance(start, goal, map_state):
    if start == goal:
        return 0
    distances = _distance_map([goal], map_state)
    return distances.get(start, 10_000)


def _reachable_from(start, map_state, max_depth=None):
    queue = deque([(start, 0)])
    dist = {start: 0}
    while queue:
        pos, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for nxt, _ in _neighbors(pos, map_state):
            if nxt not in dist:
                dist[nxt] = depth + 1
                queue.append((nxt, depth + 1))
    return dist


def _escape_space(pos, map_state, depth=7):
    return len(_reachable_from(pos, map_state, depth))


def _merge_known_map(known_map, map_state, my_position):
    """Remember visible empty cells while keeping unknown cells unexplored."""
    if known_map is None or known_map.shape != map_state.shape:
        known_map = np.full(map_state.shape, UNKNOWN, dtype=int)

    known_map[map_state == WALL] = WALL
    known_map[map_state == EMPTY] = EMPTY
    known_map[my_position] = EMPTY
    return known_map


def _frontier_score(pos, map_state):
    score = 0
    for move in MOVES:
        current = pos
        for distance in range(1, 6):
            current = _apply(current, move)
            row, col = current
            if row < 0 or row >= map_state.shape[0] or col < 0 or col >= map_state.shape[1]:
                break
            if map_state[current] == WALL:
                break
            if map_state[current] == UNKNOWN:
                score += 7 - distance
    for nxt, _ in _neighbors(pos, map_state, allow_unseen=True):
        if map_state[nxt] == UNKNOWN:
            score += 4
    return score


def _frontiers(map_state):
    result = []
    height, width = map_state.shape
    for row in range(height):
        for col in range(width):
            pos = (row, col)
            if map_state[pos] != EMPTY:
                continue
            if any(map_state[nxt] == UNKNOWN for nxt, _ in _neighbors(pos, map_state, allow_unseen=True)):
                result.append(pos)
    return result


class PacmanAgent(BasePacmanAgent):
    """Seeker using A* with a predictive intercept heuristic."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Echo-Intercept A* Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos = None
        self.last_seen_enemy_step = None
        self.known_map = None
        self.visit_count = {}
        self.recent_positions = deque(maxlen=8)
        self._last_target = None

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        self.known_map = _merge_known_map(self.known_map, map_state, my_position)
        self.visit_count[my_position] = self.visit_count.get(my_position, 0) + 1
        self.recent_positions.append(my_position)

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.last_seen_enemy_step = step_number
            self.known_map[enemy_position] = EMPTY

        target = enemy_position or self.last_known_enemy_pos
        if target is None:
            return self._blind_seek(my_position, step_number)

        plan_map = self.known_map
        if enemy_position is None:
            return self._search_lost_enemy(my_position, target, step_number)

        predictions = self._predict_ghost_tiles(target, my_position, plan_map)
        best = None
        for rank, predicted_ghost in enumerate(predictions[:4]):
            capture_tiles = self._capture_tiles(predicted_ghost, plan_map)
            action, turns, landing = self._a_star_to_any(my_position, capture_tiles, plan_map, predicted_ghost)
            if action is None:
                continue

            trap_bonus = max(0, 3 - _degree(predicted_ghost, plan_map)) * 0.22
            escape_penalty = min(20, _escape_space(predicted_ghost, plan_map, depth=5)) * 0.015
            score = turns + rank * 0.16 + escape_penalty - trap_bonus
            if best is None or score < best[0]:
                best = (score, action, landing, predicted_ghost)

        if best is not None:
            self._last_target = best[3]
            return best[1]

        return self._search_lost_enemy(my_position, target, step_number)

    def _search_lost_enemy(self, my_position, last_seen, step_number):
        frontiers = _frontiers(self.known_map)
        reachable = _reachable_from(my_position, self.known_map)
        if not reachable:
            return (Move.STAY, 1)

        age = 0 if self.last_seen_enemy_step is None else max(0, step_number - self.last_seen_enemy_step)
        candidate_scores = []
        for pos in frontiers:
            travel = reachable.get(pos)
            if travel is None:
                continue
            info = _frontier_score(pos, self.known_map)
            last_seen_bias = max(0, 18 - _manhattan(pos, last_seen)) * max(0.25, 1.0 - age * 0.04)
            revisit_penalty = self.visit_count.get(pos, 0) * 4
            score = info * 2.2 + last_seen_bias - travel * 0.45 - revisit_penalty
            candidate_scores.append((score, pos))

        if candidate_scores:
            candidate_scores.sort(reverse=True)
            for _, target in candidate_scores[:8]:
                action, _, _ = self._a_star_to_any(my_position, {target}, self.known_map, target)
                if action is not None and action[0] != Move.STAY:
                    return action

        return self._blind_seek(my_position, step_number)

    def _blind_seek(self, my_position, step_number):
        reachable = _reachable_from(my_position, self.known_map)
        if not reachable:
            return (Move.STAY, 1)

        frontiers = [pos for pos in _frontiers(self.known_map) if pos in reachable]
        if frontiers:
            targets = sorted(
                frontiers,
                key=lambda pos: (
                    -_frontier_score(pos, self.known_map),
                    reachable.get(pos, 10_000),
                    self.visit_count.get(pos, 0),
                ),
            )
        else:
            height, width = self.known_map.shape
            center = (height // 2, width // 2)
            targets = sorted(
                reachable,
                key=lambda pos: (
                    self.visit_count.get(pos, 0),
                    -_frontier_score(pos, self.known_map),
                    _manhattan(pos, center),
                    (pos[0] * 5 + pos[1] * 3 + step_number) % 7,
                ),
            )

        for target in targets[:10]:
            action, _, _ = self._a_star_to_any(my_position, {target}, self.known_map, target)
            if action is not None and action[0] != Move.STAY:
                return action

        actions = self._pacman_actions(my_position, self.known_map)
        scored = []
        for index, (action, pos) in enumerate(actions):
            score = _frontier_score(pos, self.known_map) - self.visit_count.get(pos, 0) * 3
            if pos in self.recent_positions:
                score -= 4
            scored.append((score, -index, action))
        scored.sort(reverse=True)
        return scored[0][2] if scored else (Move.STAY, 1)

    def _predict_ghost_tiles(self, ghost_pos, pacman_pos, map_state):
        options = [ghost_pos] + [pos for pos, _ in _neighbors(ghost_pos, map_state)]
        pacman_dist = _distance_map([pacman_pos], map_state)

        scored = []
        for pos in options:
            dist = pacman_dist.get(pos, _manhattan(pos, pacman_pos))
            openness = _escape_space(pos, map_state, depth=5)
            degree = _degree(pos, map_state)
            sight_penalty = 3.0 if _line_of_sight(pos, pacman_pos, map_state) else 0.0
            dead_end_penalty = 2.5 if degree <= 1 and dist < 8 else 0.0
            score = dist * 1.7 + openness * 0.11 + degree * 0.6 - sight_penalty - dead_end_penalty
            scored.append((score, pos))

        scored.sort(reverse=True)
        return [pos for _, pos in scored]

    def _capture_tiles(self, ghost_pos, map_state):
        tiles = {ghost_pos}
        for pos, _ in _neighbors(ghost_pos, map_state):
            if _manhattan(pos, ghost_pos) < 2:
                tiles.add(pos)
        return tiles

    def _a_star_to_any(self, start, goals, map_state, predicted_ghost):
        valid_goals = {goal for goal in goals if _is_valid(goal, map_state)}
        if not valid_goals:
            return None, math.inf, None
        if start in valid_goals:
            return (Move.STAY, 1), 0, start

        goal_dist = _distance_map(valid_goals, map_state)
        heap = []
        counter = 0
        best_cost = {start: 0}
        first_action = {start: None}
        heapq.heappush(heap, (self._echo_intercept_heuristic(start, goal_dist, predicted_ghost, map_state), 0, counter, start))

        while heap:
            _, cost, _, pos = heapq.heappop(heap)
            if cost != best_cost.get(pos):
                continue
            if pos in valid_goals:
                return first_action[pos], cost, pos

            for action, nxt in self._pacman_actions(pos, map_state):
                new_cost = cost + 1
                if new_cost >= best_cost.get(nxt, 10_000):
                    continue
                best_cost[nxt] = new_cost
                first_action[nxt] = first_action[pos] or action
                counter += 1
                h = self._echo_intercept_heuristic(nxt, goal_dist, predicted_ghost, map_state)
                heapq.heappush(heap, (new_cost + h, new_cost, counter, nxt))

        return None, math.inf, None

    def _echo_intercept_heuristic(self, pos, goal_dist, predicted_ghost, map_state):
        maze_steps = goal_dist.get(pos, 10_000)
        if maze_steps >= 10_000:
            return 10_000

        turn_eta = math.ceil(maze_steps / self.pacman_speed)
        corridor_pressure = _degree(pos, map_state) * 0.045
        sight_discount = 0.12 if _line_of_sight(pos, predicted_ghost, map_state) else 0.0
        closing_discount = max(0, 6 - _manhattan(pos, predicted_ghost)) * 0.018
        return turn_eta + corridor_pressure - sight_discount - closing_discount

    def _pacman_actions(self, pos, map_state):
        actions = []
        for move in MOVES:
            current = pos
            for steps in range(1, self.pacman_speed + 1):
                nxt = _apply(current, move)
                if not _is_valid(nxt, map_state):
                    break
                action = (move, steps)
                actions.append((action, nxt))
                current = nxt
        return actions

    def _explore(self, my_position, map_state, step_number):
        reachable = _reachable_from(my_position, map_state)
        if not reachable:
            return (Move.STAY, 1)

        height, width = map_state.shape
        center = (height // 2, width // 2)
        targets = sorted(
            reachable,
            key=lambda pos: (
                -_escape_space(pos, map_state, depth=5),
                _manhattan(pos, center),
                (pos[0] + pos[1] + step_number) % 5,
            ),
        )
        for target in targets[:8]:
            action, _, _ = self._a_star_to_any(my_position, {target}, map_state, target)
            if action is not None and action[0] != Move.STAY:
                return action

        actions = self._pacman_actions(my_position, map_state)
        return actions[0][0] if actions else (Move.STAY, 1)


class GhostAgent(BaseGhostAgent):
    """Hider using A* to move toward high-survival sanctuary cells."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Sanctuary A* Ghost"
        self.last_known_enemy_pos = None
        self.last_seen_enemy_step = None
        self.known_map = None
        self.visit_count = {}
        self.recent_positions = deque(maxlen=9)
        self.current_sanctuary = None

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        self.known_map = _merge_known_map(self.known_map, map_state, my_position)
        self.visit_count[my_position] = self.visit_count.get(my_position, 0) + 1
        self.recent_positions.append(my_position)
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.last_seen_enemy_step = step_number
            self.known_map[enemy_position] = EMPTY

        threat = enemy_position or self.last_known_enemy_pos
        if threat is None:
            return self._blind_hide(my_position, step_number)

        plan_map = self.known_map
        threat_age = 0 if enemy_position is not None or self.last_seen_enemy_step is None else step_number - self.last_seen_enemy_step
        threat_dist = _distance_map([threat], plan_map)
        sanctuary = self._choose_sanctuary(my_position, threat, plan_map, threat_dist, threat_age)
        self.current_sanctuary = sanctuary
        planned = self._a_star_escape(my_position, sanctuary, threat, plan_map, threat_dist, threat_age)
        planned_pos = _apply(my_position, planned) if planned != Move.STAY else my_position

        if self._is_tactically_dangerous(planned_pos, threat, plan_map, threat_dist, threat_age):
            return self._safest_immediate_move(my_position, threat, plan_map, threat_dist, threat_age)

        return planned

    def _choose_sanctuary(self, my_position, threat, map_state, threat_dist, threat_age=0):
        from_me = _reachable_from(my_position, map_state)

        best_score = -math.inf
        best_pos = my_position
        for pos, my_dist in from_me.items():
            distance_from_threat = threat_dist.get(pos, _manhattan(pos, threat))
            if distance_from_threat < 2:
                continue

            openness = _escape_space(pos, map_state, depth=7)
            degree = _degree(pos, map_state)
            hidden_bonus = 5.0 if not _line_of_sight(pos, threat, map_state) else -2.0
            dead_end_penalty = 7.0 if degree <= 1 and distance_from_threat < 12 else 0.0
            revisit_penalty = 4.0 if pos in self.recent_positions else 0.0
            revisit_penalty += self.visit_count.get(pos, 0) * 1.2
            travel_penalty = my_dist * 0.38
            pacman_turns = math.ceil(distance_from_threat / 2)
            tempo_bonus = max(-4.0, min(6.0, pacman_turns - my_dist)) * 0.9
            threat_reach = max(0, threat_age * 2 + 2 - distance_from_threat)
            belief_penalty = threat_reach * 3.4
            frontier_bonus = _frontier_score(pos, map_state) * 0.18 if threat_age > 1 else 0.0

            score = (
                distance_from_threat * 2.4
                + openness * 0.34
                + degree * 1.3
                + hidden_bonus
                + tempo_bonus
                + frontier_bonus
                - dead_end_penalty
                - revisit_penalty
                - travel_penalty
                - belief_penalty
            )
            if score > best_score:
                best_score = score
                best_pos = pos

        return best_pos

    def _a_star_escape(self, start, goal, threat, map_state, threat_dist, threat_age=0):
        if start == goal:
            return self._safest_immediate_move(start, threat, map_state, threat_dist, threat_age)

        goal_dist = _distance_map([goal], map_state)
        heap = []
        counter = 0
        best_cost = {start: 0}
        first_move = {start: Move.STAY}
        heapq.heappush(heap, (self._sanctuary_heuristic(start, goal_dist, threat, map_state, threat_dist, threat_age), 0, counter, start))

        while heap:
            _, cost, _, pos = heapq.heappop(heap)
            if cost != best_cost.get(pos):
                continue
            if pos == goal:
                return first_move[pos]

            for nxt, move in _neighbors(pos, map_state):
                new_cost = cost + 1
                if new_cost >= best_cost.get(nxt, 10_000):
                    continue
                best_cost[nxt] = new_cost
                first_move[nxt] = first_move[pos] if pos != start else move
                counter += 1
                h = self._sanctuary_heuristic(nxt, goal_dist, threat, map_state, threat_dist, threat_age)
                heapq.heappush(heap, (new_cost + h, new_cost, counter, nxt))

        return self._safest_immediate_move(start, threat, map_state, threat_dist, threat_age)

    def _sanctuary_heuristic(self, pos, goal_dist, threat, map_state, threat_dist, threat_age=0):
        to_goal = goal_dist.get(pos, 10_000)
        if to_goal >= 10_000:
            return 10_000

        distance_from_threat = threat_dist.get(pos, _manhattan(pos, threat))
        cover_bonus = 0.28 if not _line_of_sight(pos, threat, map_state) else -0.22
        room_bonus = min(18, _escape_space(pos, map_state, depth=5)) * 0.018
        revisit_tax = 0.38 if pos in self.recent_positions else 0.0
        threat_reach = max(0, threat_age * 2 + 2 - distance_from_threat)
        return to_goal - distance_from_threat * 0.31 - room_bonus - cover_bonus + revisit_tax + threat_reach * 0.8

    def _is_tactically_dangerous(self, pos, threat, map_state, threat_dist, threat_age=0):
        maze_dist = threat_dist.get(pos, _manhattan(pos, threat))
        if maze_dist <= 2:
            return True
        if threat_age > 0 and maze_dist <= threat_age * 2 + 1:
            return True
        return _line_of_sight(pos, threat, map_state) and _manhattan(pos, threat) <= 4

    def _safest_immediate_move(self, my_position, threat, map_state, threat_dist, threat_age=0):
        candidates = [(my_position, Move.STAY)] + _neighbors(my_position, map_state)
        best_score = -math.inf
        best_move = Move.STAY
        for pos, move in candidates:
            distance_from_threat = threat_dist.get(pos, _manhattan(pos, threat))
            openness = _escape_space(pos, map_state, depth=5)
            degree = _degree(pos, map_state)
            sight_penalty = 3.5 if _line_of_sight(pos, threat, map_state) else 0.0
            revisit_penalty = 2.0 if pos in self.recent_positions else 0.0
            threat_reach = max(0, threat_age * 2 + 2 - distance_from_threat)
            score = distance_from_threat * 2.7 + openness * 0.22 + degree * 0.9 - sight_penalty - revisit_penalty - threat_reach * 4.0
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _blind_hide(self, my_position, step_number):
        reachable = _reachable_from(my_position, self.known_map)
        if not reachable:
            return Move.STAY

        targets = sorted(
            reachable,
            key=lambda pos: (
                -_escape_space(pos, self.known_map, depth=7),
                -_frontier_score(pos, self.known_map),
                self.visit_count.get(pos, 0),
                (pos[0] * 13 + pos[1] * 7 + step_number) % 11,
            ),
        )
        for target in targets[:8]:
            action = self._a_star_patrol(my_position, target, self.known_map)
            if action != Move.STAY:
                return action
        return self._patrol(my_position, self.known_map, step_number)

    def _a_star_patrol(self, start, goal, map_state):
        if start == goal:
            return Move.STAY

        goal_dist = _distance_map([goal], map_state)
        heap = []
        best_cost = {start: 0}
        first_move = {start: Move.STAY}
        counter = 0
        heapq.heappush(heap, (goal_dist.get(start, 10_000), 0, counter, start))

        while heap:
            _, cost, _, pos = heapq.heappop(heap)
            if cost != best_cost.get(pos):
                continue
            if pos == goal:
                return first_move[pos]

            for nxt, move in _neighbors(pos, map_state):
                new_cost = cost + 1 + self.visit_count.get(nxt, 0) * 0.2
                if new_cost >= best_cost.get(nxt, 10_000):
                    continue
                best_cost[nxt] = new_cost
                first_move[nxt] = first_move[pos] if pos != start else move
                counter += 1
                heapq.heappush(heap, (new_cost + goal_dist.get(nxt, 10_000), new_cost, counter, nxt))

        return Move.STAY

    def _patrol(self, my_position, map_state, step_number):
        candidates = _neighbors(my_position, map_state)
        if not candidates:
            return Move.STAY

        scored = []
        for index, (pos, move) in enumerate(candidates):
            score = _escape_space(pos, map_state, depth=6) + _degree(pos, map_state) * 2
            if pos in self.recent_positions:
                score -= 5
            score += ((pos[0] * 7 + pos[1] * 11 + step_number) % 3) * 0.1
            scored.append((score, -index, move))
        scored.sort(reverse=True)
        return scored[0][2]
