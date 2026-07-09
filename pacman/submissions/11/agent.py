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
- Agents are STATEFUL - you can store memory across steps
- enemy_position may be None when limited observation is enabled
- map_state cells: 1=wall, 0=empty, -1=unseen (fog)
"""

import sys
from pathlib import Path
from collections import deque
import numpy as np
import random

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

# Bảng ánh xạ tọa độ an toàn để tránh lỗi move.value
DIRECTION_MAP = {
    Move.UP: (-1, 0),
    Move.DOWN: (1, 0),
    Move.LEFT: (0, -1),
    Move.RIGHT: (0, 1),
    Move.STAY: (0, 0)
}

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Hide) Agent 
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Evading Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        threat = enemy_position or self.last_known_enemy_pos

        # Nếu không thấy Ghost, di chuyển khám phá ngẫu nhiên
        if threat is None:
            move = self._explore(my_position, map_state)
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            return move, steps

        # Tính toán hướng chạy trốn (Né xa tọa độ của Ghost)
        row_diff = my_position[0] - threat[0]
        col_diff = my_position[1] - threat[1]

        best_moves = []
        if row_diff > 0:
            best_moves.append(Move.DOWN)
        elif row_diff < 0:
            best_moves.append(Move.UP)
        if col_diff > 0:
            best_moves.append(Move.RIGHT)
        elif col_diff < 0:
            best_moves.append(Move.LEFT)

        # Thử các hướng chạy trốn tốt nhất
        for move in best_moves:
            desired_steps = self._desired_steps(move, row_diff, col_diff)
            steps = self._max_valid_steps(my_position, move, map_state, desired_steps)
            if steps > 0:
                return move, steps

        # Nếu hướng tối ưu bị kẹt, thử tất cả các hướng đi được còn lại
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        for move in all_moves:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps > 0:
                return move, steps

        return Move.STAY, 1

    def _explore(self, my_position: tuple, map_state: np.ndarray):
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        for move in all_moves:
            if self._is_valid_move(my_position, move, map_state):
                return move
        return Move.STAY

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        dr, dc = DIRECTION_MAP[move]
        return self._is_valid_position((pos[0] + dr, pos[1] + dc), map_state)

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, desired_steps: int) -> int:
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            dr, dc = DIRECTION_MAP[move]
            next_pos = (current[0] + dr, current[1] + dc)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def _desired_steps(self, move: Move, row_diff: int, col_diff: int) -> int:
        if move in (Move.UP, Move.DOWN):
            return abs(row_diff)
        if move in (Move.LEFT, Move.RIGHT):
            return abs(col_diff)
        return 1


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Seek) Agent 
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
            my_position: tuple, 
            enemy_position: tuple,
            step_number: int) -> Move:
        
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        target = enemy_position or self.last_known_enemy_pos
        
        # Nếu không có thông tin mục tiêu, di chuyển ngẫu nhiên ngắt quãng
        if target is None:
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return move
            return Move.STAY
        
        # Sử dụng BFS tìm đường ngắn nhất để Săn đuổi Pacman
        path = self._bfs(my_position, target, map_state)
        if path:
            return path[0]
        
        # Chiến lược dự phòng (Fallback Greedy) nếu BFS không cho ra đường đi công phá
        row_diff = target[0] - my_position[0]
        col_diff = target[1] - my_position[1]
        
        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT
        
        if self._is_valid_move(my_position, move, map_state):
            return move
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                return move
        
        return Move.STAY

    def _bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            cur, path = queue.popleft()
            if cur == goal:
                return path
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = DIRECTION_MAP[move]
                nb = (cur[0] + dr, cur[1] + dc)
                if nb not in visited and self._is_valid_position(nb, map_state):
                    visited.add(nb)
                    queue.append((nb, path + [move]))
        return []
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        dr, dc = DIRECTION_MAP[move]
        new_pos = (pos[0] + dr, pos[1] + dc)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0