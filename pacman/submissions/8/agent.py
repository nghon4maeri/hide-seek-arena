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
import random

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import heapq
from collections import deque

class PacmanAgent(BasePacmanAgent):
    """
        Optimized A* Search Algorithm Seeker (Pacman).
        Strategy:
        - Memory : Remembers the last known enemy position when the target
            disappears in the fog. Falls back to random exploration if no target is known.
        - Predictive Interception: Calculates the ghost's movement vector and uses A* to
            target a future interception point (2 steps ahead) to cut off the ghost,
            rather than just trailing behind it.
        - Path Caching (Performance): Caches the planned A* path to avoid redundant
            calculations at every step, drastically reducing execution time to prevent
            TimeOut errors. Only replans when the path is exhausted or the target escapes.
        - Dynamic Speed (Burst): Scans the cached path for straight lines and groups
            consecutive identical moves to fully leverage the maximum `pacman_speed` advantage.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Pacman"
        self.last_known_enemy_pos = None
        # ADDED: Memory variables for Path Caching and Predictive Movement
        self.current_path = []
        self.previous_enemy_pos = None

    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            self.current_path = []
            return (self._random_valid_move(my_position, map_state), 1)

        # ADDED: Predictive Movement Logic (Intercepting the Ghost)
        predicted_target = target
        if enemy_position is not None and self.previous_enemy_pos is not None:
            # Calculate ghost's movement vector
            dr = enemy_position[0] - self.previous_enemy_pos[0]
            dc = enemy_position[1] - self.previous_enemy_pos[1]

            # Predict 2 steps ahead to intercept
            pred_r = target[0] + (dr * 2)
            pred_c = target[1] + (dc * 2)

            # Fallback to actual target if the predicted cell is a wall/invalid
            if self._is_valid((pred_r, pred_c), map_state):
                predicted_target = (pred_r, pred_c)

        self.previous_enemy_pos = enemy_position

        # ADDED: Path Replanning Logic (Caching)
        # Avoid running A* every single step to save execution time (< 1s)
        should_replan = True
        if self.current_path:
            dist_to_target = abs(my_position[0] - target[0]) + abs(my_position[1] - target[1])
            # Keep using cached path if we are getting closer (distance <= 5 is a safe threshold)
            if dist_to_target <= 5:
                should_replan = False

        if should_replan or not self.current_path:
            # Try to find a path to the predicted intercept point first
            path = self._a_star(my_position, predicted_target, map_state)
            if not path:
                # Fallback to the actual ghost position if intercept path is blocked
                path = self._a_star(my_position, target, map_state)
            self.current_path = path

        # ADDED: Dynamic Speed Execution (Move, steps)
        if not self.current_path or len(self.current_path) < 2:
            self.current_path = []
            return (Move.STAY, 1)

        next_pos = self.current_path[1]
        move = self._get_move_dir(my_position, next_pos)

        # Calculate maximum consecutive steps allowed in the same direction
        steps_to_take = 1
        for i in range(2, min(len(self.current_path), self.pacman_speed + 1)):
            future_pos = self.current_path[i]
            future_move = self._get_move_dir(self.current_path[i - 1], future_pos)

            if future_move == move:
                steps_to_take += 1
            else:
                break

        # Remove the executed steps from the cached path
        self.current_path = self.current_path[steps_to_take:]

        return (move, steps_to_take)

    def _a_star(self, start, goal, map_state):
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal: break

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_node = (current[0] + dr, current[1] + dc)

                if self._is_valid(next_node, map_state):
                    new_cost = cost_so_far[current] + 1
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + abs(next_node[0] - goal[0]) + abs(next_node[1] - goal[1])
                        heapq.heappush(frontier, (priority, next_node))
                        came_from[next_node] = current

        path = []
        curr = goal if goal in came_from else None
        while curr:
            path.append(curr)
            curr = came_from[curr]
        return path[::-1]

    def _get_move_dir(self, p1, p2):
        dr, dc = p2[0] - p1[0], p2[1] - p1[1]
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if m.value == (dr, dc): return m
        return Move.STAY

    def _is_valid(self, pos, map_state):
        r, c = pos
        return 0 <= r < map_state.shape[0] and 0 <= c < map_state.shape[1] and map_state[r, c] == 0

    def _random_valid_move(self, pos, map_state):
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid((pos[0] + m.value[0], pos[1] + m.value[1]), map_state): return m
        return Move.STAY


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ghost"
        self.dead_ends = set() 
        self.valid_cells = set() # Cache lưu sẵn các ô hợp lệ O(1)
        self.is_map_analyzed = False 

    def _analyze_map(self, map_state):
        """Chạy 1 lần duy nhất: Nạp Cache O(1) và Áp dụng Thuật toán Lấp Hành Lang"""
        height, width = map_state.shape
        
        # 1. Cache toàn bộ ô trống
        for r in range(height):
            for c in range(width):
                if map_state[r, c] == 0:
                    self.valid_cells.add((r, c))
                    
        # 2. Đếm số lối thoát của từng ô (Bậc của đồ thị)
        graph_degree = {}
        for cell in self.valid_cells:
            graph_degree[cell] = len(self._get_neighbors(cell))
            
        # 3. Thuật toán Tunnel Filling 
        from collections import deque
        # Bỏ tất cả các ô ngõ cụt thực sự (chỉ có 1 lối thoát) vào hàng đợi
        queue = deque([cell for cell, degree in graph_degree.items() if degree == 1])
        
        while queue:
            curr = queue.popleft()
            self.dead_ends.add(curr)
            
        # Quét các ô lân cận. Nếu đi vào hàng xóm mà chỉ dẫn đến ngõ cụt,thì hàng xóm đó cũng bị coi là ngõ cụt
            for neighbor, _ in self._get_neighbors(curr):
                if neighbor not in self.dead_ends:
                    graph_degree[neighbor] -= 1
                    # Nếu hàng xóm bị giảm xuống chỉ còn 1 lối thoát
                    if graph_degree[neighbor] == 1:
                        queue.append(neighbor)
                        
        self.is_map_analyzed = True
    
    def _is_valid_position_fast(self, pos: tuple) -> bool:
        """Tra sổ Cache O(1) thay vì kiểm tra biên bản đồ"""
        return pos in self.valid_cells

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(self, pos: tuple) -> list:
        """Lấy hướng đi hợp lệ cực nhanh nhờ Cache"""
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position_fast(next_pos):
                neighbors.append((next_pos, move))
        return neighbors

    def _get_danger_map(self, pacman_pos):
        """BFS loang nhanh trên Cache dùng Deque O(1)"""
        
        queue = deque([(pacman_pos, 0)]) 
        danger_map = {pacman_pos: 0}
        
        while queue:
            curr_pos, dist = queue.popleft() 
            for next_pos, _ in self._get_neighbors(curr_pos):
                if next_pos not in danger_map:
                    danger_map[next_pos] = dist + 1
                    queue.append((next_pos, dist + 1))
        return danger_map
    
    def _is_in_line_of_sight(self, ghost_pos, pacman_pos):
        """
        Radar phát hiện đường thẳng: Kiểm tra xem Ghost có đang đứng cùng 
        hàng hoặc cùng cột với Pacman mà KHÔNG CÓ TƯỜNG chắn ở giữa hay không.
        """
        r1, c1 = ghost_pos
        r2, c2 = pacman_pos

        # Nếu không cùng hàng và không cùng cột -> Chắc chắn bị tường/góc cua khuất lấp
        if r1 != r2 and c1 != c2:
            return False

        # Nếu cùng hàng ngang
        if r1 == r2:
            min_c, max_c = min(c1, c2), max(c1, c2)
            # Quét các ô ở giữa
            for c in range(min_c + 1, max_c):
                if (r1, c) not in self.valid_cells:
                    return False  # Có tường chắn ngang -> An toàn
            return True  # Thông suốt -> Nguy hiểm!

        # Nếu cùng cột dọc
        if c1 == c2:
            min_r, max_r = min(r1, r2), max(r1, r2)
            for r in range(min_r + 1, max_r):
                if (r, c1) not in self.valid_cells:
                    return False
            return True
        
    def _evaluate_state(self, ghost_pos, pacman_pos):
        current_dist = self._manhattan_distance(ghost_pos, pacman_pos)
        
        score = current_dist * 10000
        
        # RADAR: Phạt đúng 1.5 bước (15000 điểm).
        # Thà RẼ GÓC mất 1 bước (-10000 đ) còn hơn CHẠY THẲNG (-15000 đ). 
        # Ghost sẽ tự động lách vào ngõ hẻm 
        if current_dist <= 6 and self._is_in_line_of_sight(ghost_pos, pacman_pos):
            score -= 15000  
            
        if ghost_pos in self.dead_ends:
            score -= 50000  # Ngõ cụt là trừ điểm
            
        # Thưởng điểm ngã ba/ngã tư để ưu tiên không gian rộng
        escape_routes = len(self._get_neighbors(ghost_pos))
        score += escape_routes * 100 

        return score
    
    def _get_pacman_moves(self, pacman_pos):
        """Mô phỏng chính xác tốc độ x2 của Pacman nhờ Cache"""
        possible_positions = {pacman_pos} # Trường hợp Pacman ngừng
        
        # Bước 1 của Pacman
        for pos1, _ in self._get_neighbors(pacman_pos):
            possible_positions.add(pos1) # Rẽ L-shape (1 bước)
            
            # Bước 2 của Pacman
            for pos2, _ in self._get_neighbors(pos1):
                possible_positions.add(pos2) # Đường thẳng (2 bước)
                
        # Format trả về để hàm Minimax có thể xử lý
        return [(pos, Move.STAY) for pos in possible_positions]
    
    def _minimax(self, depth, ghost_pos, pacman_pos, is_maximizing, alpha, beta):
        current_dist = self._manhattan_distance(ghost_pos, pacman_pos)
        
        # 1. khi GHOST chết 
        if current_dist == 0:
            # Cộng thêm self._evaluate_state 
            # Ép Ghost phải luôn hướng ra không gian mở ngay cả trong những bước đi cuối 
            return -9999999 + (depth * 10000) + self._evaluate_state(ghost_pos, pacman_pos)
            
        if depth == 0:
            return self._evaluate_state(ghost_pos, pacman_pos)
            
        # 2. LƯỢT CỦA GHOST
        if is_maximizing: 
            max_eval = -float('inf')
            valid_moves = self._get_neighbors(ghost_pos)
            valid_moves.append((ghost_pos, Move.STAY))
            
            random.shuffle(valid_moves) 
            
            for next_pos, _ in valid_moves:
                eval_score = self._minimax(depth - 1, next_pos, pacman_pos, False, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval

        # 3. LƯỢT CỦA PACMAN
        else: 
            min_eval = float('inf')
            valid_moves = self._get_pacman_moves(pacman_pos)
            
            random.shuffle(valid_moves)

            for next_pos, _ in valid_moves:
                eval_score = self._minimax(depth - 1, ghost_pos, next_pos, True, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def _get_minimax_move(self, ghost_pos, pacman_pos, depth=4):
        best_move = Move.STAY
        max_eval = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        valid_moves = self._get_neighbors(ghost_pos)
        valid_moves.append((ghost_pos, Move.STAY))
        
        random.shuffle(valid_moves)

        for next_ghost_pos, move in valid_moves:
            eval_score = self._minimax(depth - 1, next_ghost_pos, pacman_pos, False, alpha, beta)
            if eval_score > max_eval:
                max_eval = eval_score 
                best_move = move
            alpha = max(alpha, eval_score)

        return best_move
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) 

    def step(self, map_state, my_position, enemy_position, step_number):
        if not self.is_map_analyzed:
            self._analyze_map(map_state)
            
        danger_map = self._get_danger_map(enemy_position)
        current_dist_to_pacman = danger_map.get(my_position, 0)

        # Vẫn giữ khoảng cách an toàn 5 ô để tối ưu tốc độ tinh
        if current_dist_to_pacman > 5: 
            valid_moves = self._get_neighbors(my_position)
            valid_moves.append((my_position, Move.STAY))
            
            random.shuffle(valid_moves)
            
            best_move = Move.STAY
            max_safety = -1

            for next_pos, move in valid_moves:
                dist = danger_map.get(next_pos, 0) * 10
                if next_pos in self.dead_ends:
                    dist -= 50
                    
                if len(self._get_neighbors(next_pos)) >= 3:
                    dist += 2

                if dist > max_safety:
                    max_safety = dist
                    best_move = move
            return best_move
        
        else:
            return self._get_minimax_move(my_position, enemy_position, depth=4)

