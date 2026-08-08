import sys
import time
import math
import random
import heapq
from collections import deque
from pathlib import Path
import numpy as np

# Thêm src vào path để import interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

# ============================================================================
# UTILITIES CHUNG
# ============================================================================
DIRS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# ============================================================================
# PACMAN AGENT (BAYESIAN SENTINEL)
# ============================================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Bayesian_Sentinel"
        self.pacman_speed = int(kwargs.get("pacman_speed", 2))
        self.memory = None
        self.belief = None
        self.last_enemy = None
        self.enemy_history = deque(maxlen=4)
        self.h = 0
        self.w = 0
        
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        if self.memory is None:
            self.h, self.w = map_state.shape
            self.memory = np.full((self.h, self.w), -1, dtype=np.int8)
            self.belief = np.zeros((self.h, self.w), dtype=np.float32)
            walkable = (map_state != 1)
            if walkable.sum() > 0:
                self.belief[walkable] = 1.0 / walkable.sum()
                
        visible_mask = map_state != -1
        self.memory[visible_mask] = map_state[visible_mask]
        
        # 2. Cập nhật Belief (Xác suất vị trí Ghost)
        self._update_belief(visible_mask, enemy_position)
        
        target_pos = None
        
        # 3. Chiến thuật mục tiêu
        if enemy_position:
            self.last_enemy = enemy_position
            self.enemy_history.append(enemy_position)
            target_pos = self._get_interception_target(my_position, enemy_position)
        else:
            # Tối ưu việc lấy vị trí max prob
            max_prob = np.max(self.belief)
            if max_prob > 0.01:
                candidates = np.argwhere(self.belief == max_prob)
                # Dùng generator để tiết kiệm RAM & thời gian
                target_pos = tuple(min((tuple(c) for c in candidates), key=lambda c: manhattan(my_position, c)))
            
            # Nếu không có dấu vết, đi thám hiểm vùng mù gần nhất
            if not target_pos:
                target_pos = self._find_nearest_frontier(my_position)

        # 4. Tìm đường với Weighted A* (Phạt ô -1)
        if target_pos:
            path = self._weighted_astar(my_position, target_pos)
            if path:
                return self._pack_speed(my_position, path)
                
        # Fallback: Chỉ đi vào ô không phải tường
        valid_moves = [m for m in DIRS if self._is_not_wall((my_position[0] + m.value[0], my_position[1] + m.value[1]))]
        return (random.choice(valid_moves), 1) if valid_moves else (Move.STAY, 1)

    def _update_belief(self, visible_mask, enemy_pos):
        new_belief = np.zeros_like(self.belief)
        
        if enemy_pos:
            new_belief[enemy_pos] = 1.0
        else:
            # TỐI ƯU: Chỉ xét những ô có xác suất > 0 để tránh lặp O(h*w)
            active_r, active_c = np.nonzero(self.belief)
            for r, c in zip(active_r, active_c):
                p = self.belief[r, c]
                neighbors = [(r, c)]
                for m in DIRS:
                    nr, nc = r + m.value[0], c + m.value[1]
                    if 0 <= nr < self.h and 0 <= nc < self.w and self.memory[nr, nc] != 1:
                        neighbors.append((nr, nc))
                
                share = p / len(neighbors)
                for nr, nc in neighbors:
                    new_belief[nr, nc] += share
            
            # QUAN TRỌNG: Loại bỏ xác suất ở những ô ta đang nhìn thấy mà không có Ghost
            new_belief[visible_mask] = 0.0
            
            total = new_belief.sum()
            if total > 1e-9:
                new_belief /= total
            else:
                walkable = (self.memory != 1)
                if walkable.sum() > 0:
                    new_belief[walkable] = 1.0 / walkable.sum()
        self.belief = new_belief

    def _weighted_astar(self, start, goal):
        if start == goal: return []
        counter = 0  # Ngăn chặn lỗi khi tuple (cost, pos) trùng nhau thì so sánh path (Enum không so sánh được)
        pq = [(manhattan(start, goal), counter, start, [], 0)]
        visited = {start: 0}
        
        while pq:
            f, _, curr, path, g = heapq.heappop(pq)
            if curr == goal: return path
            
            # Prune nhánh rác
            if visited.get(curr, float('inf')) < g: continue
            
            for m in DIRS:
                nxt = (curr[0] + m.value[0], curr[1] + m.value[1])
                if self._is_not_wall(nxt):
                    step_cost = 1 if self.memory[nxt] == 0 else 5
                    new_g = g + step_cost
                    
                    if nxt not in visited or new_g < visited[nxt]:
                        visited[nxt] = new_g
                        h = manhattan(nxt, goal)
                        counter += 1
                        heapq.heappush(pq, (new_g + h, counter, nxt, path + [m], new_g))
        return []

    def _pack_speed(self, pos, path):
        first = path[0]
        if self.pacman_speed >= 2 and len(path) >= 2 and path[0] == path[1]:
            p1 = (pos[0] + first.value[0], pos[1] + first.value[1])
            p2 = (p1[0] + first.value[0], p1[1] + first.value[1])
            if self.memory[p1] == 0 and self.memory[p2] == 0:
                return (first, 2)
        return (first, 1)

    def _get_interception_target(self, my_pos, ghost_pos):
        if len(self.enemy_history) >= 2:
            prev = self.enemy_history[-2]
            dr, dc = ghost_pos[0] - prev[0], ghost_pos[1] - prev[1]
            predict = (ghost_pos[0] + dr * 2, ghost_pos[1] + dc * 2)
            if 0 <= predict[0] < self.h and 0 <= predict[1] < self.w:
                if self.memory[predict] != 1: return predict
        return ghost_pos

    def _find_nearest_frontier(self, start):
        queue = deque([start])
        visited = {start}
        while queue:
            curr = queue.popleft()
            for m in DIRS:
                nxt = (curr[0] + m.value[0], curr[1] + m.value[1])
                if 0 <= nxt[0] < self.h and 0 <= nxt[1] < self.w:
                    if self.memory[nxt] == -1: return curr
                    if self.memory[nxt] == 0 and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
        return start

    def _is_not_wall(self, pos):
        return 0 <= pos[0] < self.h and 0 <= pos[1] < self.w and self.memory[pos] != 1


# ============================================================================
# GHOST AGENT (MINIMAX CAMPER)
# ============================================================================
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory = None
        self.last_pacman = None
        self.phase = "OPENING"
        self.camp_target = (5, 12)
        self.camp_path = []
        self.TIME_LIMIT = 0.85
        self.h = 0
        self.w = 0

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        start_time = time.time()
        if self.memory is None:
            self.h, self.w = map_state.shape
            self.memory = np.full((self.h, self.w), -1, dtype=np.int8)
            # Ngăn lỗi văng IndexError nếu map bé hơn kích thước hardcode (5,12)
            if not (0 <= self.camp_target[0] < self.h and 0 <= self.camp_target[1] < self.w) or map_state[self.camp_target] == 1: 
                self.camp_target = my_position
                
        self.memory[map_state != -1] = map_state[map_state != -1]
        if enemy_position:
            self.last_pacman = enemy_position
            self.phase = "EVADING"
            
        if self.phase == "OPENING":
            if my_position == self.camp_target: self.phase = "CAMPING"
            else:
                if not self.camp_path: self.camp_path = self._weighted_bfs_path(my_position, self.camp_target)
                if self.camp_path: return self.camp_path.pop(0)
                else: self.phase = "CAMPING"
                    
        if self.phase == "CAMPING": return Move.STAY
            
        # EVADING: Minimax
        best_move = Move.STAY
        time_check_counter = [0] # List để truyền tham chiếu Counter
        
        for depth in range(1, 4):
            try:
                if time.time() - start_time > self.TIME_LIMIT: break
                _, move = self._minimax(my_position, self.last_pacman, depth, True, start_time, time_check_counter)
                if move is not None:
                    best_move = move
            except TimeoutError: 
                break
                
        return best_move

    def _weighted_bfs_path(self, start, target):
        pq = [(0, 0, start, [])]
        visited = {start: 0}
        counter = 0
        while pq:
            cost, _, curr, path = heapq.heappop(pq)
            if curr == target: return path
            if visited.get(curr, float('inf')) < cost: continue
            
            for m in DIRS:
                nxt = (curr[0] + m.value[0], curr[1] + m.value[1])
                if 0 <= nxt[0] < self.h and 0 <= nxt[1] < self.w and self.memory[nxt] != 1:
                    new_cost = cost + (1 if self.memory[nxt] == 0 else 10)
                    if nxt not in visited or new_cost < visited[nxt]:
                        visited[nxt] = new_cost
                        counter += 1
                        heapq.heappush(pq, (new_cost, counter, nxt, path + [m]))
        return []

    def _minimax(self, ghost_pos, pac_pos, depth, is_ghost_turn, start_time, counter):
        counter[0] += 1
        # TỐI ƯU: Chỉ check thời gian mỗi 50 nodes chạy qua. Tránh overhead quá lớn của time.time()
        if counter[0] % 50 == 0:
            if time.time() - start_time > self.TIME_LIMIT: 
                raise TimeoutError()
                
        if depth == 0 or manhattan(ghost_pos, pac_pos) <= 1:
            return self._evaluate(ghost_pos, pac_pos), Move.STAY

        if is_ghost_turn:
            max_eval = -float('inf')
            best_move = Move.STAY
            valid = [m for m in DIRS if self._is_not_wall((ghost_pos[0]+m.value[0], ghost_pos[1]+m.value[1]))] + [Move.STAY]
            for m in valid:
                nxt = ghost_pos if m == Move.STAY else (ghost_pos[0]+m.value[0], ghost_pos[1]+m.value[1])
                ev, _ = self._minimax(nxt, pac_pos, depth - 1, False, start_time, counter)
                if ev > max_eval: 
                    max_eval, best_move = ev, m
            return max_eval, best_move
        else:
            min_eval = float('inf')
            # Giả định Pacman cực kỳ thông minh (tốc độ 2)
            for nxt_pac in self._get_reachables(pac_pos, 2):
                ev, _ = self._minimax(ghost_pos, nxt_pac, depth - 1, True, start_time, counter)
                if ev < min_eval: min_eval = ev
            return min_eval, None

    def _evaluate(self, ghost_pos, pac_pos):
        d = manhattan(ghost_pos, pac_pos)
        if d <= 1: return -10000
        score = d * 10
        if not self._has_los(ghost_pos, pac_pos): score += 500
        if self.memory[ghost_pos] == -1: score -= 50
        return score

    def _get_reachables(self, start, dist):
        res = {start}
        q = deque([(start, 0)])
        while q:
            curr, d = q.popleft()
            if d >= dist: continue
            for m in DIRS:
                nxt = (curr[0] + m.value[0], curr[1] + m.value[1])
                if self._is_not_wall(nxt) and nxt not in res:
                    res.add(nxt)
                    q.append((nxt, d + 1))
        return res

    def _is_not_wall(self, pos):
        return 0 <= pos[0] < self.h and 0 <= pos[1] < self.w and self.memory[pos] != 1

    def _has_los(self, p1, p2):
        if p1[0] != p2[0] and p1[1] != p2[1]: return False
        if p1[0] == p2[0]:
            s = 1 if p2[1] > p1[1] else -1
            for c in range(p1[1] + s, p2[1], s):
                if self.memory[p1[0], c] == 1: return False
        else:
            s = 1 if p2[0] > p1[0] else -1
            for r in range(p1[0] + s, p2[0], s):
                if self.memory[r, p1[1]] == 1: return False
        return True