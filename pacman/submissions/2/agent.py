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
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First

    [DFS]
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = 2 # self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        self.name = "BFS Pacman"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.last_position = None

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        # TODO: Implement your search algorithm here
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    self.last_position = my_position
                    return (move, 1)
            return (Move.STAY, 1)

        # find the shortest path by BFS
        full_path = self._bfs_find_path(my_position, target, map_state)

        if full_path:
            step1_pos = full_path[0]

            if self.last_position is not None and step1_pos == self.last_position:
                alternative_move = self._find_non_repeating_move(my_position, map_state)
                if alternative_move:
                    self.last_position = my_position
                    return (alternative_move, 1)
                else:
                    self.last_position = None

            delta_row1 = step1_pos[0] - my_position[0]
            delta_col1 = step1_pos[1] - my_position[1]

            chosen_move = self._get_move_from_delta(delta_row1, delta_col1)

            if len(full_path) >= 2 and self.pacman_speed >= 2:
                step2_pos = full_path[1]
                delta_row2 = step2_pos[0] - step1_pos[0]
                delta_col2 = step2_pos[1] - step1_pos[1]
                next_move = self._get_move_from_delta(delta_row2, delta_col2)

                if next_move == chosen_move:
                    self.last_position = my_position
                    return (chosen_move, 2)

            if chosen_move != Move.STAY:
                self.last_position = my_position
                return (chosen_move, 1)

        fallback_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        action = self._choose_action(my_position, fallback_moves, map_state, self.pacman_speed)
        if action:
            self.last_position = my_position
            return action

        return (Move.STAY, 1)
    
    # Helper methods (you can add more)
    
    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    def _get_move_from_delta(self, delta_row: int, delta_col: int) -> Move:
        for move in Move:
            if move.value == (delta_row, delta_col):
                return move
        return Move.STAY

    def _bfs_find_path(self, start: tuple, target: tuple, map_state: np.ndarray) -> list:
        if start == target:
            return []

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current_pos, path = queue.popleft()

            if current_pos == target:
                return path

            row, col = current_pos
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (row + dr, col + dc)

                if self._is_valid_position(next_pos, map_state) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))

        return []

    def _find_non_repeating_move(self, current_pos: tuple, map_state: np.ndarray) -> Move:
        valid_alternatives = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)
            if self._is_valid_position(next_pos, map_state):
                if self.last_position is None or next_pos != self.last_position:
                    valid_alternatives.append(move)
        if valid_alternatives:
            return random.choice(valid_alternatives)
        return None

class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement search algorithm to evade Pacman as long as possible.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Smart Ghost"
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move for Ghost.
        """
        # Cập nhật vị trí của Pacman nếu nhìn thấy
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        target_pacman = enemy_position or self.last_known_enemy_pos

        # TRƯỜNG HỢP 1: Không biết Pacman ở đâu (Sương mù bao phủ) -> Đi ngẫu nhiên hợp lệ
        if target_pacman is None:
            valid_moves = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    # Ưu tiên không đi vào ngõ cụt ngay cả khi đi ngẫu nhiên
                    next_pos = (my_position[0] + move.value[0], my_position[1] + move.value[1])
                    if not self._is_dead_end(next_pos, map_state):
                        valid_moves.append(move)
            
            # Nếu tất cả đều là ngõ cụt thì mới chấp nhận đi vào
            if not valid_moves:
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    if self._is_valid_move(my_position, move, map_state):
                        valid_moves.append(move)

            return random.choice(valid_moves) if valid_moves else Move.STAY

        # TRƯỜNG HỢP 2: Đã biết vị trí (hoặc vị trí cuối cùng) của Pacman
        current_distance = self._calculate_distance(my_position, target_pacman)
        scored_moves = {}

        # Lan lân cận 4 hướng để đánh giá điểm số
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)

            if self._is_valid_position(next_pos, map_state):
                score = 0

                # Tiêu chí 1: Đi xa Pacman hơn vị trí hiện tại
                next_dist = self._calculate_distance(next_pos, target_pacman)
                if next_dist > current_distance:
                    score += 10
                elif next_dist == current_distance:
                    score += 2  # Đi ngang bằng điểm vẫn tốt hơn là đi lại gần

                # Tiêu chí 2: Tránh chung hàng hoặc chung cột với Pacman
                if next_pos[0] != target_pacman[0] and next_pos[1] != target_pacman[1]:
                    score += 15

                # Tiêu chí 3: Tuyệt đối tránh ngõ cụt (Trừ khi bị dồn vào đường cùng)
                if not self._is_dead_end(next_pos, map_state):
                    score += 30

                # Lưu các nước đi theo thang điểm tương ứng
                if score not in scored_moves:
                    scored_moves[score] = []
                scored_moves[score].append(move)

        # Chọn nước đi có điểm số cao nhất dựa trên các tiêu chí
        if scored_moves:
            max_score = max(scored_moves.keys())
            best_moves = scored_moves[max_score]
            return random.choice(best_moves)

        return Move.STAY
    
    # --- Các hàm bổ trợ (Helper Methods) ---
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Kiểm tra hướng đi từ vị trí hiện tại có hợp lệ không."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Kiểm tra ô chỉ định nằm trong bản đồ và không phải là tường."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        # 0 là đường trống, chấp nhận cả ô -1 (sương mù) đối với Ghost nếu cần di chuyển ẩn nấp
        return map_state[row, col] == 0 or map_state[row, col] == -1

    def _is_dead_end(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Kiểm tra xem một ô có phải ngõ cụt (chỉ có duy nhất 1 đường ra/vào)."""
        row, col = pos
        valid_neighbors = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (row + dr, col + dc)
            if self._is_valid_position(next_pos, map_state):
                valid_neighbors += 1
        return valid_neighbors <= 1

    def _calculate_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Tính khoảng cách Manhattan giữa 2 vị trí."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])