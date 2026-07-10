import sys 
from pathlib import Path
from collections import deque
import random
from typing import Optional, Tuple
import heapq
import itertools

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
        self.name = "Team_03_Pacman"
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        target = enemy_position or self.last_known_enemy_pos
        dist_to_target = self._manhattan_distance(my_position, target)
        if enemy_position is not None and dist_to_target > 2:
            intercept_target = self._predict_ghost_target(my_position, target, map_state)
        else:
            intercept_target = target
        path = self._a_star_find_path(my_position, intercept_target, map_state)
        if not path: 
            return (Move.STAY, 1)
        best_move = path[0]
        consecutive_steps = 1
        for i in range(1, min(len(path), self.pacman_speed)):
            if path[i] == best_move:
                consecutive_steps += 1
            else:
                break  
        return (best_move, consecutive_steps)
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _predict_ghost_target(self, my_pos: tuple, ghost_pos: tuple, map_state: np.ndarray) -> tuple:
        best_future_pos = ghost_pos
        max_dist = -1
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            next_pos = (ghost_pos[0] + delta_row, ghost_pos[1] + delta_col)
            if self._is_valid_position(next_pos, map_state):
                dist = self._manhattan_distance(next_pos, my_pos)
                if dist > max_dist:
                    max_dist = dist
                    best_future_pos = next_pos
        return best_future_pos

    def _a_star_find_path(self, start: tuple, goal: tuple, map_state: np.ndarray):
        counter = itertools.count()
        queue = []
        heapq.heappush(queue, (0, next(counter), start, []))
        g_scores = {start: 0}
        while queue:
            f, _, current_pos, path = heapq.heappop(queue)
            if current_pos == goal:
                return path
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)
                if self._is_valid_position(next_pos, map_state):
                    tentative_g = len(path) + 1
                    if tentative_g < g_scores.get(next_pos, float('inf')):
                        g_scores[next_pos] = tentative_g
                        h = self._manhattan_distance(next_pos, goal)
                        new_path = list(path)
                        new_path.append(move)
                        heapq.heappush(queue, (tentative_g + h, next(counter), next_pos, new_path))
        return []

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
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width: return False
        return map_state[row, col] == 0

class GhostAgent(BaseGhostAgent): 
    def __init__(self, **kwargs): 
        super().__init__(**kwargs) 
        self.name = "Team_03_Ghost" 
        self.last_known_pacman = None 
    
    def step(self, map_state: np.ndarray,
             my_position: Tuple[int, int],
             enemy_position: Optional[Tuple[int, int]],
             step_number: int) -> Move:
        if enemy_position is not None: 
            self.last_known_pacman = enemy_position 
        return self._move_to_farthest_cell(map_state, my_position, enemy_position)
    
    def _move_to_farthest_cell(self, map_state, my_pos, enemy_pos):
        reachable = self._get_reachable_cells(map_state, my_pos)
        if not reachable: 
            return Move.STAY
        def manhattan(a, b): 
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        best_cell = my_pos 
        best_score = -float('inf')
        for cell in reachable: 
            dist_to_enemy = manhattan(cell, enemy_pos)  
            corner_bonus = self._corner_bonus(cell) 
            escape_bonus = self._escape_bonus(map_state, cell)
            score = dist_to_enemy * 2.0 + corner_bonus * 0.5 + escape_bonus * 0.3 
            if score > best_score: 
                best_score = score
                best_cell = cell
        path = self._bfs_find_path(map_state, my_pos, best_cell)
        if path and len(path) > 0:
            return path[0]
        return Move.STAY
    
    def _get_reachable_cells(self, map_state, start_pos):
        visited = set()
        queue = deque([start_pos]) 
        visited.add(start_pos)
        while queue: 
            current = queue.popleft() 
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]: 
                new_pos = self._apply_move(current, move) 
                if self._is_valid_position(new_pos, map_state) and new_pos not in visited: 
                    visited.add(new_pos) 
                    queue.append(new_pos) 
        return list(visited) 
    
    def _bfs_find_path(self, map_state, start, goal):
        if start == goal: return []
        queue = deque([(start, [])]) 
        visited = {start} 
        while queue:
            current, path = queue.popleft() 
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]: 
                new_pos = self._apply_move(current, move) 
                if not self._is_valid_position(new_pos, map_state): continue 
                if new_pos == goal: return path + [move] 
                if new_pos not in visited: 
                    visited.add(new_pos) 
                    queue.append((new_pos, path + [move])) 
        return []
    
    def _move_to_safest_corner(self, map_state, my_pos):
        corners = [(0, 0), (0, 20), (20, 0), (20, 20)] 
        best_corner = None 
        best_dist = float('inf') 
        for corner in corners: 
            path = self._bfs_find_path(map_state, my_pos, corner) 
            if path: 
                dist = len(path) 
                if dist < best_dist: 
                    best_dist = dist 
                    best_corner = corner
        if best_corner:
            path = self._bfs_find_path(map_state, my_pos, best_corner)
            if path: return path[0]
        return self._random_move(map_state, my_pos) 
    
    def _corner_bonus(self, pos):
        row, col = pos
        dist_to_center = abs(row - 10) + abs(col - 10)
        return dist_to_center
    
    def _escape_bonus(self, map_state, pos):
        count = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            new_pos = self._apply_move(pos, move)
            if self._is_valid_position(new_pos, map_state): count += 1
        return count

    def _is_valid_position(self, pos, map_state):
        row, col = pos 
        h, w = map_state.shape 
        if row < 0 or row >= h or col < 0 or col >= w: return False 
        return map_state[row][col] == 0 
    
    def _apply_move(self, pos, move):
        row, col = pos
        if move == Move.UP: return (row - 1, col)
        elif move == Move.DOWN: return (row + 1, col)
        elif move == Move.LEFT: return (row, col - 1)
        elif move == Move.RIGHT: return (row, col + 1)
        return pos
    
    def _random_move(self, map_state, pos):
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves) 
        for move in moves: 
            new_pos = self._apply_move(pos, move) 
            if self._is_valid_position(new_pos, map_state): return move 
        return Move.STAY