import sys
from pathlib import Path
from collections import deque
import random
import numpy as np
 
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BFS_Seeker_Pacman" 
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.last_known_enemy_pos = None
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        return 0 <= row < height and 0 <= col < width and map_state[row, col] == 0

    def _bfs_shortest_path(self, start: tuple, goal: tuple, map_state: np.ndarray): 
        if start == goal:
            return None
            
        queue = deque([(start, None)])  
        visited = {start}
        
        while queue:
            curr_pos, first_move = queue.popleft()
            
            if curr_pos == goal:
                return first_move 
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nxt_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
                
                if self._is_valid_position(nxt_pos, map_state) and nxt_pos not in visited:
                    visited.add(nxt_pos)
                     
                    nxt_first_move = first_move if first_move is not None else move
                    queue.append((nxt_pos, nxt_first_move))
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:         
        steps = 0
        current = pos
        for _ in range(max_steps):
            dr, dc = move.value
            next_pos = (current[0] + dr, current[1] + dc)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            random.shuffle(all_moves)
            for m in all_moves:
                steps = self._max_valid_steps(my_position, m, map_state, self.pacman_speed)
                if steps > 0:
                    return (m, steps)
            return (Move.STAY, 1)

        best_move = self._bfs_shortest_path(my_position, target, map_state)
        
        if best_move is not None:
            steps = self._max_valid_steps(my_position, best_move, map_state, self.pacman_speed)
            if steps > 0:
                return (best_move, steps) [cite: 80] 
            
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        for m in all_moves:
            steps = self._max_valid_steps(my_position, m, map_state, 1)
            if steps > 0:
                return (m, steps) [cite: 80]
                
        return (Move.STAY, 1) [cite: 80]


class GhostAgent(BaseGhostAgent): 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BFS_Evasive_Ghost"
        self.last_known_enemy_pos = None

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        return 0 <= row < height and 0 <= col < width and map_state[row, col] == 0

    def _compute_bfs_distances(self, start: tuple, map_state: np.ndarray) -> dict:
        distances = {start: 0}
        queue = deque([start])
        
        while queue:
            curr = queue.popleft()
            curr_dist = distances[curr]
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nxt = (curr[0] + dr, curr[1] + dc)
                if self._is_valid_position(nxt, map_state) and nxt not in distances:
                    distances[nxt] = curr_dist + 1
                    queue.append(nxt)
        return distances

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        threat = enemy_position or self.last_known_enemy_pos
        
        if threat is None:
            all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            random.shuffle(all_moves)
            for m in all_moves:
                dr, dc = m.value
                if self._is_valid_position((my_position[0] + dr, my_position[1] + dc), map_state):
                    return m [cite: 83]
            return Move.STAY [cite: 83]
        
        pacman_distances = self._compute_bfs_distances(threat, map_state)
        
        best_move = Move.STAY [cite: 83] 
        max_distance_from_enemy = pacman_distances.get(my_position, 0) 
        possible_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        random.shuffle(possible_moves)  
        
        for move in possible_moves:
            if move == Move.STAY:
                continue
            
            dr, dc = move.value
            next_pos = (my_position[0] + dr, my_position[1] + dc)
            
            if self._is_valid_position(next_pos, map_state):
                dist_from_enemy = pacman_distances.get(next_pos, 0)
                
                if dist_from_enemy > max_distance_from_enemy:
                    max_distance_from_enemy = dist_from_enemy
                    best_move = move
                    
        return best_move [cite: 83]