import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
from collections import deque
import time

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Team_03_Pacman"
        self.knowledge_map = None
        self.last_enemy_pos = None
        self.last_seen_step = 0
        self.search_depth = 3  
        self.bfs_cache = {}
        self.last_positions = deque(maxlen=6)
        self.stuck_counter = 0

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        self.start_time = time.time()
        if self.knowledge_map is None:
            self.knowledge_map = np.full_like(map_state, -1)
        self._update_knowledge(map_state)
        if enemy_position is not None:
            self.last_enemy_pos = enemy_position
            self.last_seen_step = step_number
        self.last_positions.append(my_position)
        if len(self.last_positions) >= 6 and len(set(self.last_positions)) <= 2:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        if self.stuck_counter > 4:
            return self._escape_loop(my_position)
        if enemy_position is not None:
            return self._pursuit_decision(my_position, enemy_position)
        elif self.last_enemy_pos is not None and (step_number - self.last_seen_step) <= 8:
            path = self._get_bfs_path(my_position, self.last_enemy_pos)
            if path:
                return self._format_action(my_position, path[0])
            return self._pursuit_decision(my_position, self.last_enemy_pos)
        else:
            return self._explore_frontier(my_position)
        
    def _pursuit_decision(self, my_pos: tuple, enemy_pos: tuple):
        best_action = (Move.STAY, 1)
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        candidate_actions = self._get_pacman_candidate_actions(my_pos)
        if not candidate_actions:
            return (Move.STAY, 1)
        for move, steps in candidate_actions:
            next_pacman_pos = self._apply_action(my_pos, move, steps)
            if self._manhattan(next_pacman_pos, enemy_pos) < 2:
                return (move, steps)
            val = self._min_value(next_pacman_pos, enemy_pos, self.search_depth - 1, alpha, beta)
            if val > best_value:
                best_value = val
                best_action = (move, steps)
            alpha = max(alpha, best_value)
        return best_action

    def _max_value(self, pacman_pos: tuple, ghost_pos: tuple, depth: int, alpha: float, beta: float) -> float:
        if depth == 0 or self._manhattan(pacman_pos, ghost_pos) < 2 or (time.time() - self.start_time) > 0.8:
            return self._evaluate_state(pacman_pos, ghost_pos)
        actions = self._get_pacman_candidate_actions(pacman_pos)
        if not actions:
            return self._evaluate_state(pacman_pos, ghost_pos)
        val = -float('inf')
        for move, steps in actions:
            next_pac_pos = self._apply_action(pacman_pos, move, steps)
            val = max(val, self._min_value(next_pac_pos, ghost_pos, depth - 1, alpha, beta))
            if val >= beta:
                return val
            alpha = max(alpha, val)
        return val

    def _min_value(self, pacman_pos: tuple, ghost_pos: tuple, depth: int, alpha: float, beta: float) -> float:
        if depth == 0 or self._manhattan(pacman_pos, ghost_pos) < 2 or (time.time() - self.start_time) > 0.8:
            return self._evaluate_state(pacman_pos, ghost_pos)
        ghost_moves = self._get_valid_ghost_moves(ghost_pos)
        if not ghost_moves:
            return self._evaluate_state(pacman_pos, ghost_pos)
        val = float('inf')
        for g_move in ghost_moves:
            next_ghost_pos = self._apply_move_single(ghost_pos, g_move)
            val = min(val, self._max_value(pacman_pos, next_ghost_pos, depth - 1, alpha, beta))
            if val <= alpha:
                return val
            beta = min(beta, val)
        return val

    def _evaluate_state(self, pacman_pos: tuple, ghost_pos: tuple) -> float:
        if ghost_pos is None:
            return -500.0
        m_dist = self._manhattan(pacman_pos, ghost_pos)
        if m_dist < 2:
            return 10000.0
        path = self._get_bfs_path(pacman_pos, ghost_pos)
        maze_dist = len(path) if path else m_dist * 2.5
        ghost_escape_routes = len(self._get_valid_ghost_moves(ghost_pos))
        trap_bonus = (4 - ghost_escape_routes) * 15.0
        same_line_bonus = 20.0 if (pacman_pos[0] == ghost_pos[0] or pacman_pos[1] == ghost_pos[1]) else 0.0
        score = - (maze_dist * 10.0) + trap_bonus + same_line_bonus - (m_dist * 2.0)
        return score

    def _explore_frontier(self, my_pos: tuple):
        queue = deque([my_pos])
        visited = {my_pos}
        target_unseen = None
        while queue:
            curr = queue.popleft()
            if self.knowledge_map[curr[0], curr[1]] == -1:
                target_unseen = curr
                break
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = self._apply_move_single(curr, move)
                if self._is_traversable(nxt) and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        if target_unseen:
            path = self._get_bfs_path(my_pos, target_unseen)
            if path:
                return self._format_action(my_pos, path[0])
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_traversable(self._apply_move_single(my_pos, move)):
                return (move, 1)
        return (Move.STAY, 1)

    def _get_pacman_candidate_actions(self, pos: tuple):
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            max_s = self._get_max_straight_steps(pos, move, self.pacman_speed)
            for s in range(1, max_s + 1):
                actions.append((move, s))
        return actions if actions else [(Move.STAY, 1)]

    def _get_max_straight_steps(self, pos: tuple, move: Move, max_speed: int) -> int:
        steps = 0
        curr = pos
        for _ in range(max_speed):
            nxt = self._apply_move_single(curr, move)
            if not self._is_traversable(nxt):
                break
            steps += 1
            curr = nxt
        return steps

    def _format_action(self, pos: tuple, move: Move):
        max_steps = self._get_max_straight_steps(pos, move, self.pacman_speed)
        return (move, max_steps) if max_steps > 0 else (Move.STAY, 1)

    def _get_valid_ghost_moves(self, pos: tuple):
        valid = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                 if self._is_traversable(self._apply_move_single(pos, m))]
        return valid if valid else [Move.STAY]

    def _get_bfs_path(self, start: tuple, goal: tuple):
        if start == goal:
            return []
        cache_key = (start, goal)
        if cache_key in self.bfs_cache:
            return self.bfs_cache[cache_key]
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            curr, path = queue.popleft()
            if len(path) > 40:  
                break
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = self._apply_move_single(curr, move)
                if nxt == goal:
                    res = path + [move]
                    self.bfs_cache[cache_key] = res
                    return res
                if self._is_traversable(nxt) and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [move]))
        self.bfs_cache[cache_key] = []
        return []

    def _escape_loop(self, my_pos: tuple):
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = self._apply_move_single(my_pos, move)
            if self._is_traversable(nxt) and nxt not in self.last_positions:
                self.stuck_counter = 0
                return (move, 1)
        return (Move.STAY, 1)

    def _update_knowledge(self, map_state: np.ndarray):
        if self.knowledge_map is not None:
            newly_discovered = (self.knowledge_map == -1) & (map_state != -1)
            if np.any(newly_discovered):
                self.bfs_cache.clear()
            mask = map_state != -1
            self.knowledge_map[mask] = map_state[mask]

    def _is_traversable(self, pos: tuple) -> bool:
        r, c = pos
        if self.knowledge_map is None:
            return True
        h, w = self.knowledge_map.shape
        if 0 <= r < h and 0 <= c < w:
            return self.knowledge_map[r, c] != 1
        return False

    def _apply_move_single(self, pos: tuple, move: Move) -> tuple:
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _apply_action(self, pos: tuple, move: Move, steps: int) -> tuple:
        dr, dc = move.value
        return (pos[0] + dr * steps, pos[1] + dc * steps)

    def _manhattan(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_map = None
        self.last_enemy_pos = None
        self.enemy_visible = False
        self.search_depth = 3
        self.path_cache = {}
        self.last_positions = deque(maxlen=10)
        self.stuck_counter = 0
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.max_bfs_nodes = 300
        self.name = "Team_03_Ghost"

    def step(self, map_state, my_position, enemy_position, step_number):
        self.start_time = time.time()
        if self.knowledge_map is None:
            self.knowledge_map = np.full_like(map_state, -1)
        self._update_knowledge(map_state)
        self.enemy_visible = enemy_position is not None
        if self.enemy_visible:
            self.last_enemy_pos = enemy_position  
        self.last_positions.append(my_position)
        if len(self.last_positions) >= 5:
            if len(set(self.last_positions)) <= 2:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)
        if self.stuck_counter > 5:
            return self._escape_stuck(my_position, map_state)
        return self._decide_move(my_position, map_state)

    def _decide_move(self, my_pos, map_state):
        if self.enemy_visible and self.last_enemy_pos is not None:
            return self._minimax_decision(my_pos, self.last_enemy_pos, map_state)
        elif self.last_enemy_pos is not None:
            target = self._find_best_pos(my_pos, self.last_enemy_pos, map_state)
            if target != my_pos:
                path = self._get_path(my_pos, target, map_state)
                if path:
                    return path[0]
            return self._run_away(my_pos, self.last_enemy_pos, map_state)
        return self._explore_strategy(my_pos, map_state)

    def _minimax_decision(self, my_pos, enemy_pos, map_state):
        best_move = Move.STAY
        best_value = -float('inf')
        alpha, beta = -float('inf'), float('inf')
        valid_moves = self._get_valid_moves(my_pos, map_state)
        if not valid_moves:
            return Move.STAY
        for move in valid_moves:
            new_pos = self._apply_move(my_pos, move)
            value = self._min_value(new_pos, enemy_pos, map_state, self.search_depth - 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def _max_value(self, my_pos, enemy_pos, map_state, depth, alpha, beta):
        if depth == 0 or self._is_caught(my_pos, enemy_pos) or (time.time() - self.start_time) > 0.8:
            return self._evaluate(my_pos, enemy_pos)
        valid_moves = self._get_valid_moves(my_pos, map_state)
        if not valid_moves:
            return self._evaluate(my_pos, enemy_pos)
        value = -float('inf')
        for move in valid_moves:
            new_pos = self._apply_move(my_pos, move)
            value = max(value, self._min_value(new_pos, enemy_pos, map_state, depth - 1, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def _min_value(self, my_pos, enemy_pos, map_state, depth, alpha, beta):
        if depth == 0 or self._is_caught(my_pos, enemy_pos) or (time.time() - self.start_time) > 0.8:
            return self._evaluate(my_pos, enemy_pos)
        pacman_actions = self._get_pacman_actions_for_ghost(enemy_pos, map_state)
        if not pacman_actions:
            return self._evaluate(my_pos, enemy_pos)
        value = float('inf')
        for move, steps in pacman_actions:
            dr, dc = move.value
            new_enemy_pos = (enemy_pos[0] + dr * steps, enemy_pos[1] + dc * steps)
            value = min(value, self._max_value(my_pos, new_enemy_pos, map_state, depth - 1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def _get_pacman_actions_for_ghost(self, enemy_pos, map_state):
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            curr = enemy_pos
            for s in range(1, self.pacman_speed + 1):
                dr, dc = move.value
                nxt = (curr[0] + dr, curr[1] + dc)
                if not self._is_valid(nxt, map_state):
                    break
                actions.append((move, s))
                curr = nxt
        return actions if actions else [(Move.STAY, 1)]

    def _evaluate(self, ghost_pos, pacman_pos):
        if pacman_pos is None:
            return 100.0
        dist = self._manhattan_distance(ghost_pos, pacman_pos)
        return dist * 4.0 + self._wall_score(ghost_pos) * 1.5 + self._escape_routes(ghost_pos) * 2.0

    def _is_caught(self, ghost_pos, pacman_pos):
        return pacman_pos is not None and self._manhattan_distance(ghost_pos, pacman_pos) < 2

    def _get_valid_moves(self, pos, map_state):
        valid = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                 if self._is_valid(self._apply_move(pos, m), map_state)]
        return valid if valid else [Move.STAY]

    def _is_safe(self, pos, map_state):
        return self._is_valid(pos, map_state)

    def _is_valid(self, pos, map_state):
        r, c = pos
        return 0 <= r < map_state.shape[0] and 0 <= c < map_state.shape[1] and map_state[r, c] != 1

    def _apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_path(self, start, goal, map_state):
        if start == goal:
            return []
        cache_key = (start, goal)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        queue = deque([(start, [])])
        visited = {start}
        nodes = 0
        while queue and nodes < self.max_bfs_nodes:
            nodes += 1
            current, path = queue.popleft()
            if len(path) > 50:
                continue
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_pos = self._apply_move(current, move)
                if not self._is_valid(next_pos, map_state):
                    continue
                if next_pos == goal:
                    result = path + [move]
                    self.path_cache[cache_key] = result
                    return result
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
        self.path_cache[cache_key] = []
        return []

    def _find_best_pos(self, my_pos, enemy_pos, map_state):
        queue = deque([(my_pos, 0)])
        visited = {my_pos}
        best_pos, best_score = my_pos, -float('inf')
        while queue:
            pos, dist = queue.popleft()
            if dist > 20:
                continue
            score = self._manhattan_distance(pos, enemy_pos) * 2.0 + self._wall_score(pos) * 1.5 + self._escape_routes(pos)
            if score > best_score:
                best_score, best_pos = score, pos
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                new_pos = self._apply_move(pos, move)
                if self._is_valid(new_pos, map_state) and new_pos not in visited:
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))
        return best_pos

    def _run_away(self, my_pos, enemy_pos, map_state):
        max_dist = self._manhattan_distance(my_pos, enemy_pos)
        best_move = Move.STAY
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            new_pos = self._apply_move(my_pos, move)
            if self._is_valid(new_pos, map_state):
                dist = self._manhattan_distance(new_pos, enemy_pos)
                if dist > max_dist:
                    max_dist = dist
                    best_move = move
        return best_move

    def _explore_strategy(self, my_pos, map_state):
        queue = deque([(my_pos, 0)])
        visited = {my_pos}
        while queue:
            pos, _ = queue.popleft()
            if map_state[pos[0], pos[1]] == -1:
                path = self._get_path(my_pos, pos, map_state)
                if path:
                    return path[0]
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                new_pos = self._apply_move(pos, move)
                if self._is_valid(new_pos, map_state) and new_pos not in visited:
                    visited.add(new_pos)
                    queue.append((new_pos, 0))
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            new_pos = self._apply_move(my_pos, move)
            if self._is_valid(new_pos, map_state):
                return move
        return Move.STAY

    def _escape_stuck(self, my_pos, map_state):
        best_move, max_routes = Move.STAY, 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            new_pos = self._apply_move(my_pos, move)
            if self._is_valid(new_pos, map_state):
                routes = self._escape_routes(new_pos)
                if routes > max_routes:
                    max_routes, best_move = routes, move
        if best_move != Move.STAY:
            self.stuck_counter = 0
        return best_move

    def _wall_score(self, pos):
        r, c = pos
        walls = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.knowledge_map.shape[0] and 0 <= nc < self.knowledge_map.shape[1]):
                walls += 1
            elif self.knowledge_map[nr, nc] == 1:
                walls += 1
            elif self.knowledge_map[nr, nc] == -1:
                walls += 0.5
        if (not (0 <= r - 1 < self.knowledge_map.shape[0]) or self.knowledge_map[r - 1, c] == 1) and \
           (not (0 <= c - 1 < self.knowledge_map.shape[1]) or self.knowledge_map[r, c - 1] == 1):
            walls += 2
        return walls

    def _escape_routes(self, pos):
        return sum(1 for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                   if self._is_valid(self._apply_move(pos, m), self.knowledge_map))

    def _update_knowledge(self, map_state):
        if self.knowledge_map is not None:
            newly_discovered = (self.knowledge_map == -1) & (map_state != -1)
            if np.any(newly_discovered):
                self.path_cache.clear()
            mask = map_state != -1
            self.knowledge_map[mask] = map_state[mask]