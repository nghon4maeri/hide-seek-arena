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

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

from collections import deque
import random
class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        self.last_path = []
        self.visited = deque(maxlen=8)
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        self.name = "BFS Pacman"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
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
        
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or explore
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            # No information about enemy - explore randomly
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return (move, 1)
            return (Move.STAY, 1)

        self.visited.append(my_position)
        path = self._find_path(my_position, target, map_state)

        if len(path) > 1:
            if self._in_cycle() and len(path) > 3:
                candidates = path[1:len(path)//2]
                if candidates:
                    banned = random.choice(candidates)
                    new_path = self._find_path(my_position, target, map_state, banned)

                    if new_path != path and len(new_path) > 1:
                        path = new_path
                        self.visited.clear()

            self.last_path = path 
            return self._path_to_direction(path)

        # If the primary direction is blocked, try other moves
        fallback_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        action = self._choose_action(my_position, fallback_moves, map_state, self.pacman_speed)
        if action:
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

    def _bfs_parent(self, start, map_state, banned_pos = None):
        queue = deque([start])
        parent = {start: None}
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]
        while queue:
            current = queue.popleft()

            for drow, dcol in directions:
                nrow = current[0] + drow
                ncol = current[1] + dcol
                next_pos = (nrow, ncol)

                if banned_pos is not None and next_pos == banned_pos:
                    continue

                if not self._is_valid_position(next_pos, map_state):
                    continue

                if next_pos in parent:
                    continue

                parent[next_pos] = current
                queue.append(next_pos)
        return parent

    def _find_path(self, start, goal, map_state, banned_pos = None):
        parent = self._bfs_parent(start, map_state, banned_pos)
        if goal not in parent:
            return [start]
        
        path = []
        node = goal 
        while node is not None:
            path.append(node)
            node = parent[node]
        
        path.reverse()
        return path

    def _path_to_direction(self, path):
        if len(path) < 2:
             return (Move.STAY, 1)

        start = path[0]
        next_pos = path[1]

        dr = next_pos[0] - start[0]
        dc = next_pos[1] - start[1]

        if dr == -1:
            move = Move.UP
        elif dr == 1:
            move = Move.DOWN
        elif dc == -1:
            move = Move.LEFT
        else:
            move = Move.RIGHT

        steps = 1

        dir = (dr, dc)

        while steps < self.pacman_speed and steps + 1 < len(path):
            current = path[steps]
            new_dir = ( path[steps + 1][0] - current[0],path[steps + 1][1] - current[1])
            if new_dir != dir:
                break
            steps += 1
        
        return(move, steps)

    def _in_cycle(self):
        v = list(self.visited)
        n = len(v)
        if n < 4:
            return False
        
        for k in range (1, n //2 + 1):
            if 2 * k > n:
                continue
            if v[-k:] == v[-2*k:-k]:
                return True
        return False

class GhostAgent(BaseGhostAgent):
    """
    Minimax + Kiting Ghost: Biết nhìn trước tương lai và tuyệt đối không chui vào ngõ cụt!
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Smart Minimax Ghost"

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0

    def _count_exits(self, pos, map_state):
        exits = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if self._is_valid_position((pos[0] + dr, pos[1] + dc), map_state):
                exits += 1
        return exits

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        if enemy_position is None:
            return Move.STAY
            
        best_move = Move.STAY
        max_eval = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        # Nhìn trước 3 bước (Tăng lên 4 nếu máy bạn chạy nhanh)
        SEARCH_DEPTH = 3 
        
        moves_dict = {
            Move.UP: (-1, 0), Move.DOWN: (1, 0),
            Move.LEFT: (0, -1), Move.RIGHT: (0, 1)
        }
        
        for move, (dr, dc) in moves_dict.items():
            new_ghost_pos = (my_position[0] + dr, my_position[1] + dc)
            
            if self._is_valid_position(new_ghost_pos, map_state):
                eval = self._minimax(map_state, SEARCH_DEPTH, False, alpha, beta, new_ghost_pos, enemy_position)
                
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                    
                alpha = max(alpha, eval)
                
        return best_move

    def _minimax(self, map_state, depth, is_maximizing, alpha, beta, ghost_pos, pacman_pos):
        dist = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        if dist <= 1:
            return -99999 + depth # Chết là trừ điểm cực nặng
            
        if depth == 0:
            return self._evaluate_state(ghost_pos, pacman_pos, map_state)
            
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        if is_maximizing:
            max_eval = -float('inf')
            for dr, dc in moves:
                next_pos = (ghost_pos[0] + dr, ghost_pos[1] + dc)
                if self._is_valid_position(next_pos, map_state):
                    eval = self._minimax(map_state, depth - 1, False, alpha, beta, next_pos, pacman_pos)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            if max_eval == -float('inf'):
                return -99999 + depth
            return max_eval
            
        else:
            min_eval = float('inf')
            for dr, dc in moves:
                next_pos = (pacman_pos[0] + dr, pacman_pos[1] + dc)
                if self._is_valid_position(next_pos, map_state):
                    eval = self._minimax(map_state, depth - 1, True, alpha, beta, ghost_pos, next_pos)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            if min_eval == float('inf'):
                return 99999 - depth
            return min_eval

    def _evaluate_state(self, ghost_pos, pacman_pos, map_state):
        # 1. Ưu tiên hàng đầu: Khoảng cách càng xa càng tốt (Nhân 10 để làm trọng số chính)
        dist = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        score = dist * 10 
        
        # 2. Áp dụng luật sinh tồn (Cảm biến ngõ cụt)
        exits = self._count_exits(ghost_pos, map_state)
        
        if exits <= 1:
            score -= 1000 # Thấy ngõ cụt là bỏ chạy ngay
        elif exits >= 3:
            score += 50   # Thấy ngã tư rẽ được nhiều hướng thì ưu tiên chui vào
            
        return score

