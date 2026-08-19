"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
"""

import time
import sys
import random
from pathlib import Path
from collections import deque

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost

    Strategy (Blind Adversary):
    - Tích lũy bản đồ (known_map) qua mỗi bước từ tầm nhìn hạn chế
    - Khi thấy Ghost: dùng A* tìm đường ngắn nhất đuổi theo
    - Khi mất dấu Ghost: A* đến vị trí cuối cùng đã thấy
    - Khi hoàn toàn không biết: exploration — đi về phía vùng chưa khám phá
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "A* Optimized Pacman"
        # Lấy tốc độ từ kwargs
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

        self.current_path = []
        self.last_enemy_pos = None
        self.path_step = 0

        # Bản đồ tích lũy: None = chưa khởi tạo, cập nhật mỗi bước
        self.known_map = None
        # Vị trí cuối cùng thấy Ghost (dùng khi Fog of War)
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):

        #  Cập nhật bản đồ tích lũy từ tầm nhìn hiện tại 
        self._update_known_map(map_state)

        #  Cập nhật vị trí cuối cùng thấy Ghost 
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        # Xác định mục tiêu
        target = enemy_position if enemy_position is not None else self.last_known_enemy_pos

        #  Trường hợp: Hoàn toàn không biết Ghost ở đâu → Exploration 
        if target is None:
            return self._explore(my_position, map_state)

        #  Trường hợp: Có mục tiêu (thấy hoặc đã biết) → A* đuổi theo 
        should_replan = (
            not self.current_path or
            self.last_enemy_pos is None or
            target != self.last_enemy_pos or
            self.path_step >= len(self.current_path)
        )

        if should_replan:
            self.current_path = self._astar(my_position, target)
            self.last_enemy_pos = target
            self.path_step = 0

        # Nếu đã đến vị trí cuối cùng thấy Ghost mà vẫn không thấy → chuyển exploration
        if not self.current_path and enemy_position is None:
            self.last_known_enemy_pos = None
            return self._explore(my_position, map_state)

        if self.current_path and self.path_step < len(self.current_path):
            next_move = self.current_path[self.path_step]

            # Tính số bước liên tiếp cùng hướng để tận dụng speed
            steps_to_take = 0
            for i in range(self.pacman_speed):
                idx = self.path_step + i
                if idx >= len(self.current_path) or self.current_path[idx] != next_move:
                    break
                steps_to_take += 1

            real_steps = self._max_valid_steps(my_position, next_move, steps_to_take)

            if real_steps > 0:
                self.path_step += real_steps
                return (next_move, real_steps)

        # Fallback greedy
        fallback_move = self._greedy_move_direction(my_position, target)
        steps = self._max_valid_steps(my_position, fallback_move, self.pacman_speed)
        return (fallback_move, steps if steps > 0 else 1)

    #  HELPER METHODS 

    def _update_known_map(self, map_state: np.ndarray):
        """Cập nhật bản đồ tích lũy từ observation hiện tại."""
        if self.known_map is None:
            # Khởi tạo: -1 = chưa biết cho tất cả
            self.known_map = np.full(map_state.shape, -1, dtype=int)

        # Cập nhật những ô đã thấy (0 = trống, 1 = tường)
        visible_mask = map_state != -1
        self.known_map[visible_mask] = map_state[visible_mask]

    def _is_valid_on_known_map(self, pos: tuple) -> bool:
        """Kiểm tra vị trí hợp lệ trên bản đồ tích lũy."""
        row, col = pos
        if self.known_map is None:
            return False
        height, width = self.known_map.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return self.known_map[row, col] == 0

    def _is_valid_move_on_known_map(self, pos: tuple, move: Move) -> bool:
        delta_row, delta_col = move.value
        next_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_on_known_map(next_pos)

    def _max_valid_steps(self, pos: tuple, move: Move, desired_steps: int) -> int:
        """Tính số bước tối đa có thể đi theo hướng move trên known_map."""
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_on_known_map(next_pos):
                break
            steps += 1
            current = next_pos
        return steps

    def _explore(self, my_position: tuple, map_state: np.ndarray):
        """
        Exploration: BFS tìm ô trống gần nhất nằm cạnh vùng chưa khám phá (-1).
        Mục đích: mở rộng tầm nhìn để tìm Ghost.
        """
        if self.known_map is None:
            # Fallback: đi bước đầu tiên hợp lệ
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                npos = (my_position[0] + dr, my_position[1] + dc)
                if self._is_valid_position(npos, map_state):
                    return (move, 1)
            return (Move.STAY, 1)

        height, width = self.known_map.shape

        # BFS tìm ô trống gần nhất cạnh vùng unseen
        queue = deque([(my_position, [])])
        visited = {my_position}

        while queue:
            curr, path = queue.popleft()

            # Kiểm tra xem curr có nằm cạnh ô unseen không
            if len(path) > 0:
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    dr, dc = move.value
                    nr, nc = curr[0] + dr, curr[1] + dc
                    if 0 <= nr < height and 0 <= nc < width and self.known_map[nr, nc] == -1:
                        # Tìm thấy ô cạnh vùng chưa biết → đi theo path
                        first_move = path[0]
                        steps = self._max_valid_steps(my_position, first_move, self.pacman_speed)
                        return (first_move, steps if steps > 0 else 1)

            # Mở rộng BFS
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                npos = (curr[0] + dr, curr[1] + dc)
                if npos not in visited and self._is_valid_on_known_map(npos):
                    visited.add(npos)
                    # Lưu move đầu tiên trong path
                    new_path = path + [move] if path else [move]
                    queue.append((npos, new_path))

        # Nếu không tìm được vùng chưa biết → random move
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move_on_known_map(my_position, move):
                valid_moves.append(move)

        if valid_moves:
            chosen = random.choice(valid_moves)
            steps = self._max_valid_steps(my_position, chosen, self.pacman_speed)
            return (chosen, steps if steps > 0 else 1)

        return (Move.STAY, 1)

    def _astar(self, start: tuple, goal: tuple) -> list:
        """A* search trên known_map, trả về danh sách các Move."""
        if start == goal:
            return []

        start_time = time.time()

        # [f_cost, g_cost, h_cost, position, parent_pos]
        frontier = [[0, 0, self._manhattan_distance(start, goal), start, None]]

        g_costs = {start: 0}
        came_from = {}
        closed_set = set()

        move_deltas = {
            Move.UP: (-1, 0), Move.DOWN: (1, 0),
            Move.LEFT: (0, -1), Move.RIGHT: (0, 1)
        }

        nodes_explored = 0
        max_nodes = 2500

        while frontier and nodes_explored < max_nodes:
            # Timeout check mỗi 100 nodes
            if nodes_explored % 100 == 0 and time.time() - start_time > 0.4:
                break

            nodes_explored += 1

            # Tìm node có f_cost nhỏ nhất (thủ công thay vì heapq)
            min_idx = 0
            min_f = frontier[0][0]
            for i in range(1, len(frontier)):
                if frontier[i][0] < min_f:
                    min_f = frontier[i][0]
                    min_idx = i

            current_node = frontier.pop(min_idx)
            _, g_cost, _, current_pos, _ = current_node

            if current_pos in closed_set:
                continue
            closed_set.add(current_pos)

            if current_pos == goal:
                return self._reconstruct_path(came_from, current_pos, start)

            current_row, current_col = current_pos

            for move, (delta_row, delta_col) in move_deltas.items():
                next_pos = (current_row + delta_row, current_col + delta_col)

                if not self._is_valid_on_known_map(next_pos):
                    continue
                if next_pos in closed_set:
                    continue

                tentative_g = g_cost + 1
                if next_pos in g_costs and g_costs[next_pos] <= tentative_g:
                    continue

                g_costs[next_pos] = tentative_g
                came_from[next_pos] = (current_pos, move)

                h_cost = self._manhattan_distance(next_pos, goal)
                f_cost = tentative_g + h_cost
                frontier.append([f_cost, tentative_g, h_cost, next_pos, current_pos])

        return []

    def _reconstruct_path(self, came_from: dict, current: tuple, start: tuple) -> list:
        path = []
        while current != start:
            if current not in came_from:
                break
            parent_pos, move = came_from[current]
            path.append(move)
            current = parent_pos
        path.reverse()
        return path

    def _greedy_move_direction(self, my_position: tuple, enemy_position: tuple) -> Move:
        """Fallback: Chọn hướng đi giúp giảm khoảng cách Manhattan nhiều nhất."""
        row_diff = enemy_position[0] - my_position[0]
        col_diff = enemy_position[1] - my_position[1]
        moves_with_priority = []

        if row_diff != 0:
            move = Move.DOWN if row_diff > 0 else Move.UP
            if self._is_valid_move_on_known_map(my_position, move):
                moves_with_priority.append((abs(row_diff), move))

        if col_diff != 0:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT
            if self._is_valid_move_on_known_map(my_position, move):
                moves_with_priority.append((abs(col_diff), move))

        moves_with_priority.sort(reverse=True, key=lambda x: x[0])

        if moves_with_priority:
            return moves_with_priority[0][1]

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move_on_known_map(my_position, move):
                return move
        return Move.STAY

    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid on raw map_state."""
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught

    Strategy (Blind Adversary):
    - Tích lũy bản đồ (known_map) qua mỗi bước từ tầm nhìn hạn chế
    - Khi thấy Pacman: BFS tính khoảng cách, chọn hướng tối đa hóa distance
    - Khi mất dấu Pacman: dùng vị trí cuối cùng đã biết để né
    - Khi hoàn toàn không biết: di chuyển ngẫu nhiên để tránh đứng yên
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Bản đồ tích lũy
        self.known_map = None
        # Vị trí cuối cùng thấy Pacman (dùng khi Fog of War)
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col)
            enemy_position: Pacman's current (row, col) or None if unseen
            step_number: Current step number (starts at 1)

        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        #  Cập nhật bản đồ tích lũy 
        self._update_known_map(map_state)

        #  Cập nhật vị trí cuối cùng thấy Pacman 
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        # Xác định mối đe dọa
        threat = enemy_position if enemy_position is not None else self.last_known_enemy_pos

        #  Hoàn toàn không biết Pacman ở đâu → random move
        if threat is None:
            return self._random_valid_move(my_position)

        #  Có thông tin về Pacman → BFS max-distance evasion 
        distance_map = self._bfs_full_map(threat)
        best_move = Move.STAY
        max_distance = distance_map[my_position[0], my_position[1]]

        # Đánh giá tất cả hướng đi có thể
        possible_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]

        for move in possible_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)

            if self._is_valid_on_known_map(new_pos):
                new_distance = distance_map[new_pos[0], new_pos[1]]

                # Ô chưa được BFS reach → coi là rất xa (an toàn)
                if new_distance == -1:
                    new_distance = float('inf')

                if new_distance > max_distance:
                    max_distance = new_distance
                    best_move = move

        return best_move

    #  HELPER METHODS 

    def _update_known_map(self, map_state: np.ndarray):
        """Cập nhật bản đồ tích lũy từ observation hiện tại."""
        if self.known_map is None:
            self.known_map = np.full(map_state.shape, -1, dtype=int)

        visible_mask = map_state != -1
        self.known_map[visible_mask] = map_state[visible_mask]

    def _is_valid_on_known_map(self, pos: tuple) -> bool:
        """Kiểm tra vị trí hợp lệ trên bản đồ tích lũy."""
        row, col = pos
        if self.known_map is None:
            return False
        height, width = self.known_map.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return self.known_map[row, col] == 0

    def _random_valid_move(self, my_position: tuple) -> Move:
        """Chọn một hướng đi hợp lệ ngẫu nhiên."""
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            npos = (my_position[0] + dr, my_position[1] + dc)
            if self._is_valid_on_known_map(npos):
                valid_moves.append(move)

        if valid_moves:
            return random.choice(valid_moves)
        return Move.STAY

    def _bfs_full_map(self, start_pos: tuple) -> np.ndarray:
        """
        BFS trên known_map tìm khoảng cách ngắn nhất từ start_pos đến tất cả các ô.
        - grid[r][c] = khoảng cách từ start_pos đến (r, c)
        - grid[r][c] = -1 nếu là tường, unseen, hoặc không thể đến được
        """
        height, width = self.known_map.shape
        distances = np.full(self.known_map.shape, -1, dtype=int)
        queue = deque()

        if self._is_valid_on_known_map(start_pos):
            distances[start_pos[0], start_pos[1]] = 0
            queue.append(start_pos)

        while queue:
            current_pos = queue.popleft()
            current_dist = distances[current_pos[0], current_pos[1]]

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)

                # Chỉ đi qua ô đã biết là trống (== 0)
                if self._is_valid_on_known_map(next_pos) and distances[next_pos[0], next_pos[1]] == -1:
                    distances[next_pos[0], next_pos[1]] = current_dist + 1
                    queue.append(next_pos)
        return distances

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0
