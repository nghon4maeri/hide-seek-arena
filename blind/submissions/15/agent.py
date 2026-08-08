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
import random

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
    Features persistent map memory to perform frontier-based systematic exploration
    when the target is out of sight, and optimal BFS pathfinding with speed compression.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Optimal BFS Pacman"
        # Memory to store the last spotted enemy location in limited observation mode
        self.last_known_enemy_pos = None
        # Persistent memory to construct the full layout of the maze over time
        self.internal_map = None
        # Track last move to prevent back-and-forth oscillation
        self.last_move = Move.STAY
        # Memory decay timer to prevent Pacman from chasing old targets indefinitely
        self.threat_memory_timer = 0
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move for Pacman.
        """
        # Initialize or update persistent map memory
        if self.internal_map is None:
            self.internal_map = np.copy(map_state)
        else:
            # Update known areas (anything that is not fog -1 in current observation)
            known_mask = (map_state != -1)
            self.internal_map[known_mask] = map_state[known_mask]
            
        # Update memory dynamically based on maximum map dimension
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            # OPTIMIZED: Increased timer to ensure Pacman has enough time to track down targets on a 21x21 grid
            self.threat_memory_timer = max(15, max(self.internal_map.shape))
        else:
            if self.threat_memory_timer > 0:
                self.threat_memory_timer -= 1
                if self.threat_memory_timer == 0:
                    self.last_known_enemy_pos = None
            
        # Prevent Pacman from standing still when reaching the old ghost position
        if my_position == self.last_known_enemy_pos and enemy_position is None:
            self.last_known_enemy_pos = None
            
        target = enemy_position or self.last_known_enemy_pos
        
        # If there is no information about the Ghost, perform systematic frontier-based exploration
        if target is None:
            path = self._find_nearest_unseen(my_position)
            if path:
                action = self._compress_path(path)
                self.last_move = action[0]
                return action
            
            # Fallback: Move randomly if no exploration frontier is found (avoiding deterministic loops)
            valid_moves = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    valid_moves.append(move)
            if valid_moves:
                opposite_move = self._get_opposite_move(self.last_move)
                preferred_moves = [m for m in valid_moves if m != opposite_move]
                chosen_move = random.choice(preferred_moves) if preferred_moves else random.choice(valid_moves)
                self.last_move = chosen_move
                return (chosen_move, 1)
            return (Move.STAY, 1)

        # 1. Run BFS to find a safe, known path to target using internal_map
        path = self._bfs(my_position, target, self.internal_map)
        
        # 2. If a safe path is found, follow it using speed compression
        if path:
            action = self._compress_path(path)
            self.last_move = action[0]
            return action
            
        # 3. If chasing BFS fails (no safe path known), fall back to frontier exploration instead of staying still
        path = self._find_nearest_unseen(my_position)
        if path:
            action = self._compress_path(path)
            self.last_move = action[0]
            return action

        # 4. Secondary fallback: Random walk
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                valid_moves.append(move)
        if valid_moves:
            opposite_move = self._get_opposite_move(self.last_move)
            preferred_moves = [m for m in valid_moves if m != opposite_move]
            chosen_move = random.choice(preferred_moves) if preferred_moves else random.choice(valid_moves)
            self.last_move = chosen_move
            return (chosen_move, 1)
            
        return (Move.STAY, 1)

    def _bfs(self, start: tuple, goal: tuple, map_state: np.ndarray):
        """
        BFS algorithm to find the shortest path to the goal.
        Optimized: Uses parent-pointers to avoid storing paths in the queue.
        """
        if start == goal:
            return []
            
        queue = deque([start])
        parent = {start: (None, None)}  # child -> (parent, move)
        found = False
        height, width = map_state.shape
        
        while queue:
            current_pos = queue.popleft()
            
            if current_pos == goal:
                found = True
                break
                
            # Explore the 4 adjacent cardinal directions
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)
                
                # Allow the target node as destination even if it is currently unseen (-1)
                if next_pos == goal:
                    if 0 <= next_pos[0] < height and 0 <= next_pos[1] < width:
                        if map_state[next_pos[0], next_pos[1]] != 1:  # Not a wall
                            parent[next_pos] = (current_pos, move)
                            found = True
                            break
                
                # Standard pathfinding is strictly restricted to known empty (0) tiles
                if next_pos not in parent and self._is_valid_position(next_pos, map_state):
                    parent[next_pos] = (current_pos, move)
                    queue.append(next_pos)
            
            if found:
                break
                    
        if not found:
            return []
            
        # Reconstruct path backwards from goal to start
        path = []
        curr = goal
        while curr != start:
            p, move = parent[curr]
            path.append(move)
            curr = p
        path.reverse()
        return path

    def _find_nearest_unseen(self, start: tuple) -> list:
        """
        Frontier-Based Exploration BFS on internal map memory.
        Optimized: Locates the nearest empty tile adjacent to unseen (-1) area using parent-pointers.
        """
        if self.internal_map is None:
            return []
            
        queue = deque([start])
        parent = {start: (None, None)}
        height, width = self.internal_map.shape
        goal = None
        
        while queue:
            curr = queue.popleft()
            
            # Check if current empty tile is adjacent to any unexplored (-1) tile
            frontier_found = False
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (curr[0] + delta_row, curr[1] + delta_col)
                
                if 0 <= next_pos[0] < height and 0 <= next_pos[1] < width:
                    if self.internal_map[next_pos[0], next_pos[1]] == -1:
                        goal = curr  # Stop at curr (empty tile), do NOT add final_move step
                        frontier_found = True
                        break
            
            if frontier_found:
                break
            
            # Otherwise, continue BFS traversal through known traversable (0) paths
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (curr[0] + delta_row, curr[1] + delta_col)
                
                if next_pos not in parent:
                    if 0 <= next_pos[0] < height and 0 <= next_pos[1] < width:
                        if self.internal_map[next_pos[0], next_pos[1]] == 0:
                            parent[next_pos] = (curr, move)
                            queue.append(next_pos)
                            
        if goal is None:
            return []
            
        # OPTIMIZED: If we are already adjacent to the frontier (goal == start), do not step 
        # blindly into the -1 cell (which might be a wall). This avoids wall-crashing.
        if goal == start:
            return []
            
        # Reconstruct path backwards to the frontier-adjacent empty cell
        path = []
        curr = goal
        while curr != start:
            p, move = parent[curr]
            path.append(move)
            curr = p
        path.reverse()
        return path

    def _compress_path(self, path: list) -> tuple:
        """
        Compress BFS single steps into a straight-line multi-step move based on pacman_speed.
        """
        if not path:
            return (Move.STAY, 1)
        first_move = path[0]
        steps = 0
        for move in path:
            if move == first_move and steps < self.pacman_speed:
                steps += 1
            else:
                break
        return (first_move, steps)

    def _get_opposite_move(self, move: Move) -> Move:
        """Helper to get the direct opposite cardinal direction."""
        if move == Move.UP: return Move.DOWN
        if move == Move.DOWN: return Move.UP
        if move == Move.LEFT: return Move.RIGHT
        if move == Move.RIGHT: return Move.LEFT
        return Move.STAY
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from the current position is valid."""
        delta_row, delta_col = move.value
        next_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(next_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a coordinate is within boundaries and is a known empty tile."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught by Pacman.
    Uses Double BFS with strategic penalizations (Interception and Dead-end checks)
    and a Desperate Escape system for extreme threat situations.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Memory to store the last spotted threat location in limited observation mode
        self.last_known_enemy_pos = None
        # Persistent memory to construct the full layout of the maze over time
        self.internal_map = None
        # Memory decay timer to prevent Ghost from evading non-existent threats indefinitely
        self.threat_memory_timer = 0
        # Track last move to prevent back-and-forth oscillation
        self.last_move = Move.STAY
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move for Ghost.
        """
        # Update internal map memory
        if self.internal_map is None:
            self.internal_map = np.copy(map_state)
        else:
            known_mask = (map_state != -1)
            self.internal_map[known_mask] = map_state[known_mask]

        # Update threat memory timer based on maximum map dimension
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            # OPTIMIZED: Increased threat tracking memory to prevent early loss of Pacman's position
            self.threat_memory_timer = max(15, max(self.internal_map.shape))
        else:
            if self.threat_memory_timer > 0:
                self.threat_memory_timer -= 1
                if self.threat_memory_timer == 0:
                    self.last_known_enemy_pos = None
        
        # Determine the target threat position
        threat = enemy_position or self.last_known_enemy_pos
        
        # If no threat is detected, move randomly to stay active
        if threat is None:
            valid_moves = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    valid_moves.append(move)
            if valid_moves:
                opposite_move = self._get_opposite_move(self.last_move)
                preferred_moves = [m for m in valid_moves if m != opposite_move]
                chosen_move = random.choice(preferred_moves) if preferred_moves else random.choice(valid_moves)
                self.last_move = chosen_move
                return chosen_move
            return Move.STAY
        
        # Step 1: Calculate the shortest path distance from Pacman to all empty tiles
        threat_distances = self._get_distances_from_threat(threat, self.internal_map)
        
        # Step 2: Find all empty tiles reachable and their distance in linear V time.
        # Uses path direction inheritance to completely avoid backwards reconstructions.
        reachable_distances = {my_position: 0}
        first_move_to = {my_position: Move.STAY}
        queue = deque([my_position])
        
        while queue:
            curr = queue.popleft()
            dist = reachable_distances[curr]
            first_move = first_move_to[curr]
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (curr[0] + delta_row, curr[1] + delta_col)
                
                if next_pos not in reachable_distances and self._is_valid_position(next_pos, self.internal_map):
                    reachable_distances[next_pos] = dist + 1
                    # Inherit the first move from parent node to achieve O(1) step retrieval
                    first_move_to[next_pos] = move if curr == my_position else first_move
                    queue.append(next_pos)
        
        # Step 3: Select the safest tile based on customized evaluations
        safest_tile = my_position
        max_score = -999999
        opposite_move = self._get_opposite_move(self.last_move)
        
        for pos, path_len in reachable_distances.items():
            if pos == my_position and len(reachable_distances) > 1:
                continue  # Never stay put if there are alternative valid escape tiles
                
            dist = threat_distances.get(pos, 9999)
            score = dist
            
            # OPTIMIZED: Speed-adjusted Interception Penalty. Since Pacman can move 2x faster,
            # he can intercept us if (dist / 2) <= path_len (i.e. dist <= 2 * path_len)
            if dist <= 2 * path_len:
                score -= 50
                
            # Dead-end Penalty (Subtract score if the tile is a dead end/corridor end)
            if self._get_degree(pos, self.internal_map) <= 1:
                score -= 15
                
            # Anti-oscillation Penalty for Ghost while actively fleeing
            first_move = first_move_to[pos]
            if first_move == opposite_move:
                score -= 5
                
            if score > max_score:
                max_score = score
                safest_tile = pos
            elif score == max_score:
                # Tie-breaker: Prefer tiles closer to Ghost to avoid erratic moves
                if path_len < reachable_distances[safest_tile]:
                    safest_tile = pos
        
        # Step 4: Follow the first step towards the safest tile in O(1)
        chosen_move = first_move_to[safest_tile]
        
        # OPTIMIZED: DESPERATE ESCAPE GAMBLE
        # If Pacman is extremely close (<= 3 steps away) and our safest known path is highly dangerous
        # (e.g., we are trapped or Pacman can intercept us), gamble by stepping into an adjacent unseen (-1) tile.
        my_dist_to_threat = threat_distances.get(my_position, 9999)
        if my_dist_to_threat <= 3:
            unseen_escape_moves = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
                if 0 <= next_pos[0] < self.internal_map.shape[0] and 0 <= next_pos[1] < self.internal_map.shape[1]:
                    if self.internal_map[next_pos[0], next_pos[1]] == -1:
                        unseen_escape_moves.append(move)
            
            if unseen_escape_moves and (chosen_move == Move.STAY or max_score < 0):
                chosen_move = random.choice(unseen_escape_moves)
        
        if chosen_move != Move.STAY:
            self.last_move = chosen_move
            return chosen_move
        
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

    def _get_degree(self, pos: tuple, map_state: np.ndarray) -> int:
        """Get the number of valid empty neighbors for a position."""
        degree = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            next_pos = (pos[0] + delta_row, pos[1] + delta_col)
            if self._is_valid_position(next_pos, map_state):
                degree += 1
        return degree

    def _get_opposite_move(self, move: Move) -> Move:
        """Helper to get the direct opposite cardinal direction."""
        if move == Move.UP: return Move.DOWN
        if move == Move.DOWN: return Move.UP
        if move == Move.LEFT: return Move.RIGHT
        if move == Move.RIGHT: return Move.LEFT
        return Move.STAY
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from the current position is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a coordinate is within boundaries and is a known empty tile."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0