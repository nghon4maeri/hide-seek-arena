"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student
"""

import sys
import time
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
  
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Optimal BFS Pacman"
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return (move, 1)
            return (Move.STAY, 1)

        path = self._bfs_find_path(my_position, target, map_state)
        
        if len(path) > 1:
            next_pos = path[1]
            dr = next_pos[0] - my_position[0]
            dc = next_pos[1] - my_position[1]
            
            if dr == -1: move = Move.UP
            elif dr == 1: move = Move.DOWN
            elif dc == -1: move = Move.LEFT
            else: move = Move.RIGHT
            
            steps = 1
            curr_pos = next_pos
            
            while steps < self.pacman_speed and (steps + 1) < len(path):
                future_pos = path[steps + 1]
                if (future_pos[0] - curr_pos[0] == dr) and (future_pos[1] - curr_pos[1] == dc):
                    steps += 1
                    curr_pos = future_pos
                else:
                    break
                    
            return (move, steps)
            
        return (Move.STAY, 1)

    def _bfs_find_path(self, start, goal, map_state):
        queue = deque([[start]])
        visited = {start}
        
        while queue:
            path = queue.popleft()
            curr = path[-1]
            
            if curr == goal:
                return path
                
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (curr[0] + dr, curr[1] + dc)
                if self._is_valid_position(nxt, map_state) and nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return [start]

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        delta_row, delta_col = move.value
        nxt = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(nxt, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width: return False
        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
  
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Pro Minimax Ghost"
        self.last_known_enemy_pos = None
        # Biến an toàn để tránh bị Timeout (Quá 1 giây)
        self.time_limit = 0.9 

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        threat = enemy_position or self.last_known_enemy_pos
        if threat is None:
            return Move.STAY
            
        self.start_time = time.time()
        best_move = Move.STAY
        max_eval = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        SEARCH_DEPTH = 4 
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            new_ghost_pos = (my_position[0] + dr, my_position[1] + dc)
            
            if self._is_valid_position(new_ghost_pos, map_state):
                eval = self._minimax(map_state, SEARCH_DEPTH - 1, False, alpha, beta, new_ghost_pos, threat)
                
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                    
                alpha = max(alpha, eval)
                
        return best_move

    def _minimax(self, map_state, depth, is_ghost_turn, alpha, beta, ghost_pos, pacman_pos):
        if time.time() - self.start_time > self.time_limit:
            return self._evaluate_state(ghost_pos, pacman_pos, map_state)

        dist = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        if dist <= 1:
            return -99999 
            
        if depth == 0:
            return self._evaluate_state(ghost_pos, pacman_pos, map_state)
            
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        if is_ghost_turn:
            max_eval = -float('inf')
            for dr, dc in moves:
                next_pos = (ghost_pos[0] + dr, ghost_pos[1] + dc)
                if self._is_valid_position(next_pos, map_state):
                    eval = self._minimax(map_state, depth - 1, False, alpha, beta, next_pos, pacman_pos)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break 
            return max_eval if max_eval != -float('inf') else -99999 + depth
            
        else:
            min_eval = float('inf')
            for dr, dc in moves:
                nxt1 = (pacman_pos[0] + dr, pacman_pos[1] + dc)
                if self._is_valid_position(nxt1, map_state):
                    eval1 = self._minimax(map_state, depth - 1, True, alpha, beta, ghost_pos, nxt1)
                    min_eval = min(min_eval, eval1)
                    beta = min(beta, eval1)
                    if beta <= alpha: break
                    
                    nxt2 = (nxt1[0] + dr, nxt1[1] + dc)
                    if self._is_valid_position(nxt2, map_state):
                        eval2 = self._minimax(map_state, depth - 1, True, alpha, beta, ghost_pos, nxt2)
                        min_eval = min(min_eval, eval2)
                        beta = min(beta, eval2)
                        if beta <= alpha: break
                        
            return min_eval if min_eval != float('inf') else 99999 - depth

    def _evaluate_state(self, ghost_pos, pacman_pos, map_state):
        dist = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        score = dist * 10 
        
        exits = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if self._is_valid_position((ghost_pos[0] + dr, ghost_pos[1] + dc), map_state):
                exits += 1
                
        if exits <= 1:
            score -= 1000 
        elif exits >= 3:
            score += 50   
            
        return score

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width: return False
        return map_state[row, col] == 0