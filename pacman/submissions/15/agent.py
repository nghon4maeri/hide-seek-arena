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

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost.
    Uses BFS to find the shortest path and compresses actions based on the maximum allowed speed.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Optimal BFS Pacman"
        # Memory to store the last spotted enemy location in limited observation mode
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move for Pacman.
        """
        # Update memory if the enemy is currently visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        target = enemy_position or self.last_known_enemy_pos
        
        # If there is no information about the Ghost, stay put
        if target is None:
            return (Move.STAY, 1)

        # 1. Run BFS to find the shortest path to the target
        path = self._bfs(my_position, target, map_state)
        
        # 2. If a path is found, follow it using consecutive speed compression
        if path:
            best_move = path[0]
            
            # Count consecutive steps in the same direction along the planned path
            steps = 0
            for move in path:
                if move == best_move and steps < self.pacman_speed:
                    steps += 1
                else:
                    break
            
            return (best_move, steps)
            
        # If no path is found (e.g., trapped or error), stay put
        return (Move.STAY, 1)

    def _bfs(self, start: tuple, goal: tuple, map_state: np.ndarray):
        """
        BFS algorithm to find the shortest path to the goal.
        """
        queue = deque([(start, [])])  # Queue stores tuple: (current_position, path_taken)
        visited = {start}             # Set to track visited coordinates
        
        while queue:
            current_pos, path = queue.popleft()
            
            if current_pos == goal:
                return path
                
            # Explore the 4 adjacent cardinal directions
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)
                
                # Check if the next position is valid and not yet visited
                if self._is_valid_position(next_pos, map_state) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
                    
        return []  # Return an empty list if no path is found
    
    # Helper methods
    
    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        """Choose a valid action from a list of candidate moves."""
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        """Calculate the maximum number of valid steps in a single direction."""
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
        """Check if a move from the current position is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a coordinate is within boundaries and is not a wall."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    Uses Double BFS to find the safest reachable tile on the map.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        # Keep track of Pacman's last seen position
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        threat = enemy_position or self.last_known_enemy_pos
        
        # If no information about Pacman, wander randomly to stay active
        if threat is None:
            valid_moves = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    valid_moves.append(move)
            if valid_moves:
                import random
                return random.choice(valid_moves)
            return Move.STAY
        
        # Step 1: Calculate the shortest path distance from Pacman to all empty tiles
        threat_distances = self._get_distances_from_threat(threat, map_state)
        
        # Step 2: Find all empty tiles reachable by Ghost from its current position
        reachable_tiles = self._get_reachable_tiles(my_position, map_state)
        
        # Step 3: Select the safest tile (the one that Pacman needs the most steps to reach)
        safest_tile = my_position
        max_dist = -1
        
        for pos, path in reachable_tiles.items():
            # If Pacman cannot reach this tile, assign it a very high safety score
            dist = threat_distances.get(pos, 9999)
            
            # Prioritize maximizing distance. On tie, prefer tiles closer to Ghost to avoid erratic moves.
            if dist > max_dist:
                max_dist = dist
                safest_tile = pos
            elif dist == max_dist:
                if len(path) < len(reachable_tiles[safest_tile]):
                    safest_tile = pos
        
        # Step 4: Follow the first step of the shortest path to the safest tile
        path_to_safest = reachable_tiles[safest_tile]
        if path_to_safest:
            return path_to_safest[0]
        
        return Move.STAY
    
    def _get_distances_from_threat(self, threat: tuple, map_state: np.ndarray) -> dict:
        """Run BFS from the threat to calculate graph distances to all tiles."""
        distances = {}
        queue = deque([(threat, 0)])
        visited = {threat}
        
        while queue:
            curr, dist = queue.popleft()
            distances[curr] = dist
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (curr[0] + delta_row, curr[1] + delta_col)
                if next_pos not in visited and self._is_valid_position(next_pos, map_state):
                    visited.add(next_pos)
                    queue.append((next_pos, dist + 1))
        return distances
    
    def _get_reachable_tiles(self, start: tuple, map_state: np.ndarray) -> dict:
        """Run BFS to map all reachable tiles and their corresponding paths from Ghost."""
        reachable = {}  # Format: { tile_position: [list_of_Moves_to_get_there] }
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            curr, path = queue.popleft()
            reachable[curr] = path
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (curr[0] + delta_row, curr[1] + delta_col)
                if next_pos not in visited and self._is_valid_position(next_pos, map_state):
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
        return reachable
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0