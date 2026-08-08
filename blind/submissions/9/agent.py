import numpy as np
from collections import deque
import random
from scipy.signal import convolve2d

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


# ==========================================================
# STATE MANAGER
# ==========================================================
class StateManager:
    def __init__(self):
        self.global_map = None
        self.visited = set()

        # Biến caching để tối ưu hiệu suất tính toán
        self.known_cells_count = 0
        self.map_changed = True
        self.last_bfs_target = None
        self.cached_bfs_map = None

        self.enemy_position = None
        self.last_enemy_position = None
        self.predicted_enemy_position = None
        self.enemy_confidence = 0

        self.seek_heatmap = None
        self.hide_heatmap = None
        
        self.step = 0
        self.my_position = None

    def update(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        self.step = step_number
        self.my_position = my_position
        self.visited.add(my_position)

        # Khởi tạo bản đồ toàn cục với giá trị -1
        if self.global_map is None:
            self.global_map = np.full(map_state.shape, -1)

        # Cập nhật vùng nhìn thấy
        visible = (map_state != -1)
        self.global_map[visible] = map_state[visible]
        
        # Kiểm tra xem có ô mới nào được khám phá không để Cache
        current_known = np.count_nonzero(self.global_map != -1)
        if current_known > self.known_cells_count:
            self.known_cells_count = current_known
            self.map_changed = True
        else:
            self.map_changed = False

        # Chỉ cập nhật Heatmap nếu bản đồ thực sự có thêm vùng mới
        if self.map_changed or self.seek_heatmap is None:
            self._update_heat_maps_vectorized()
            
        self._update_enemy(enemy_position)

    def _update_enemy(self, enemy_position):
        if enemy_position is None:
            if self.predicted_enemy_position is not None:
                self.enemy_confidence = max(0, self.enemy_confidence - 1)
            return

        self.last_enemy_position = self.enemy_position
        self.enemy_position = enemy_position
        self.enemy_confidence = 10

        if self.last_enemy_position is not None:
            dx = enemy_position[0] - self.last_enemy_position[0]
            dy = enemy_position[1] - self.last_enemy_position[1]
            self.predicted_enemy_position = (enemy_position[0] + dx, enemy_position[1] + dy)
        else:
            self.predicted_enemy_position = enemy_position

    def get_target(self):
        if self.enemy_position is not None:
            return self.enemy_position
        return self.predicted_enemy_position

    def is_valid_position(self, pos, allow_unknown=False):
        r, c = pos
        h, w = self.global_map.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        value = self.global_map[r, c]
        if allow_unknown:
            return value in (0, -1)
        return value == 0

    def _walkable_neighbors(self, pos):
        r, c = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.is_valid_position((nr, nc)):
                neighbors.append((nr, nc))
        return neighbors

    def _update_heat_maps_vectorized(self):
        """
        Tách biệt 2 Heatmap cho Seek (khám phá) và Hide (ẩn nấp an toàn)
        """
        if self.global_map is None:
            return

        self.seek_heatmap = np.zeros(self.global_map.shape, dtype=float)
        self.hide_heatmap = np.zeros(self.global_map.shape, dtype=float)
        
        empty_mask = (self.global_map == 0).astype(int)
        unknown_mask = (self.global_map == -1).astype(int)

        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])
        
        neighbor_count = convolve2d(empty_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        valid_neighbors = neighbor_count * empty_mask

        dead_ends = (valid_neighbors <= 1) & (empty_mask == 1)
        corridors = (valid_neighbors == 2) & (empty_mask == 1)
        junctions = (valid_neighbors >= 3) & (empty_mask == 1)

        self.seek_heatmap[dead_ends] -= 50
        self.seek_heatmap[corridors] += 5
        self.seek_heatmap[junctions] += 20

        self.hide_heatmap[dead_ends] -= 80  
        self.hide_heatmap[corridors] += 10
        self.hide_heatmap[junctions] += 30  

        frontier_count = convolve2d(unknown_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        
        self.seek_heatmap += (frontier_count * empty_mask * 15)
        self.hide_heatmap -= (frontier_count * empty_mask * 20)

    def get_safe_target(self, target_pos):
        if target_pos is None: 
            return None
            
        if self.is_valid_position(target_pos, allow_unknown=False):
            return target_pos
            
        best_cell = None
        best_dist = float('inf')
        rows, cols = np.where(self.global_map == 0)
        
        for r, c in zip(rows, cols):
            dist = abs(r - target_pos[0]) + abs(c - target_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_cell = (r, c)
                
        return best_cell

    def build_bfs_distance_map(self, target_pos):
        """
        Tạo BFS Map có sử dụng cơ chế Cache để tránh tính lại khi map không đổi.
        """
        if not self.map_changed and self.last_bfs_target == target_pos and self.cached_bfs_map is not None:
            return self.cached_bfs_map

        dist_map = np.full(self.global_map.shape, float('inf'))
        safe_target = self.get_safe_target(target_pos)
        
        if safe_target is not None:
            queue = deque([(safe_target, 0)])
            dist_map[safe_target] = 0

            while queue:
                curr, dist = queue.popleft()
                for nxt in self._walkable_neighbors(curr):
                    if dist_map[nxt] == float('inf'):
                        dist_map[nxt] = dist + 1
                        queue.append((nxt, dist + 1))
        
        # Lưu lại cache
        self.last_bfs_target = target_pos
        self.cached_bfs_map = dist_map
        return dist_map

    def path_clear(self, position, move, steps=1):
        current = position
        dr, dc = move.value
        for _ in range(steps):
            current = (current[0] + dr, current[1] + dc)
            if not self.is_valid_position(current):
                return False
        return True

    def get_legal_actions(self, position, last_direction=None, speed=2):
        actions = []
        directions = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        for move in directions:
            if self.path_clear(position, move, 1):
                actions.append((move, 1))
            if speed >= 2:
                if last_direction is None or last_direction == move:
                    if self.path_clear(position, move, 2):
                        actions.append((move, 2))
        if len(actions) == 0:
            actions.append((Move.STAY, 1))
        return actions


# ==========================================================
# AGENT INTERFACES
# ==========================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_manager = StateManager()
        self.last_move = None
        # Đã nâng Depth lên 3 nhờ có hệ thống Cache tối ưu
        self.max_depth = 3 
        self.bfs_dist_map = None 

    def step(self, map_state, my_position, enemy_position, step_number):
        self.state_manager.update(map_state, my_position, enemy_position, step_number)
        legal_actions = self.state_manager.get_legal_actions(my_position, self.last_move, speed=2)
        
        if not legal_actions:
            return (Move.STAY, 1)

        target = self.state_manager.get_target()
        self.bfs_dist_map = self.state_manager.build_bfs_distance_map(target)

        # Chế độ khám phá khi chưa thấy mục tiêu
        if target is None:
            return self._explore_action(my_position, legal_actions)

        # Chế độ truy đuổi bằng Minimax
        best_score = float('-inf')
        best_action = legal_actions[0]

        for action in legal_actions:
            dr, dc = action[0].value
            steps = action[1]
            next_pos = (my_position[0] + dr * steps, my_position[1] + dc * steps)
            
            score = self.minimax(next_pos, target, depth=self.max_depth, alpha=float('-inf'), beta=float('inf'), is_maximizing=False)
            
            if score > best_score:
                best_score = score
                best_action = action

        self.last_move = best_action[0]
        return best_action 

    def minimax(self, pacman_pos, ghost_pos, depth, alpha, beta, is_maximizing):
        dist_to_ghost = self.bfs_dist_map[pacman_pos]
        
        if depth == 0 or dist_to_ghost < 2:
            return self.evaluate_state(pacman_pos)

        if is_maximizing:
            max_eval = float('-inf')
            actions = self.state_manager.get_legal_actions(pacman_pos, speed=2)
            for action in actions:
                dr, dc = action[0].value
                steps = action[1]
                next_pacman = (pacman_pos[0] + dr * steps, pacman_pos[1] + dc * steps)
                
                eval_score = self.minimax(next_pacman, ghost_pos, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            actions = self.state_manager.get_legal_actions(ghost_pos, speed=1)
            for action in actions:
                dr, dc = action[0].value
                next_ghost = (ghost_pos[0] + dr, ghost_pos[1] + dc)
                
                eval_score = self.minimax(pacman_pos, next_ghost, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_state(self, pacman_pos):
        distance = self.bfs_dist_map[pacman_pos]
        if distance == float('inf'):
            return -9999 
        
        score = -distance * 10
        
        # Thưởng điểm nếu dồn được Ghost vào ngõ cụt (dựa trên điểm kiến trúc)
        if self.state_manager.seek_heatmap is not None:
            score += self.state_manager.seek_heatmap[pacman_pos] * 0.1
            
        # Thêm nhiễu (Tie-breaker) để tránh kẹt đi qua đi lại
        return score + (random.random() * 0.1)
        
    def _explore_action(self, current_pos, legal_actions):
        best_action = legal_actions[0]
        best_score = float('-inf')
        for action in legal_actions:
            dr, dc = action[0].value
            steps = action[1]
            r, c = current_pos[0] + dr * steps, current_pos[1] + dc * steps
            if self.state_manager.seek_heatmap is not None:
                score = self.state_manager.seek_heatmap[r, c]
                # Thêm nhiễu chống kẹt khi explore
                score += (random.random() * 0.1)
                if score > best_score:
                    best_score = score
                    best_action = action
        return best_action


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_manager = StateManager()

    def step(self, map_state, my_position, enemy_position, step_number):
        self.state_manager.update(map_state, my_position, enemy_position, step_number)
        legal_actions = self.state_manager.get_legal_actions(my_position, speed=1) 
        
        if not legal_actions:
            return Move.STAY

        target = self.state_manager.get_target()
        
        bfs_dist_map = None
        if target:
            bfs_dist_map = self.state_manager.build_bfs_distance_map(target)
        elif self.state_manager.last_enemy_position:
            bfs_dist_map = self.state_manager.build_bfs_distance_map(self.state_manager.last_enemy_position)

        best_action = legal_actions[0][0]
        best_score = float('-inf')

        for action_tuple in legal_actions:
            action = action_tuple[0]
            dr, dc = action.value
            nxt_pos = (my_position[0] + dr, my_position[1] + dc)
            
            score = 0
            
            # Đánh giá bằng Hide Heatmap
            if self.state_manager.hide_heatmap is not None:
                score += self.state_manager.hide_heatmap[nxt_pos]
                
            # Đánh giá khoảng cách
            if bfs_dist_map is not None:
                dist_to_pacman = bfs_dist_map[nxt_pos]
                if dist_to_pacman != float('inf'):
                    score += dist_to_pacman * 20
                    
            # Nhiễu tie-breaker
            score += (random.random() * 0.1)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action