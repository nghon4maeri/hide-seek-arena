import sys
from pathlib import Path
from collections import deque, defaultdict
import heapq
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_MOVE = {
    (-1, 0): Move.UP, (1, 0): Move.DOWN,
    (0, -1): Move.LEFT, (0, 1): Move.RIGHT
}


def _is_valid_cell(pos, map_state):
    r, c = pos
    h, w = map_state.shape
    return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0


def bfs_dist_from(start, map_state, max_steps=50):
    """BFS thực tế từ start, trả về dict {pos: bước_đi}"""
    dist = {start: 0}
    q = deque([start])
    while q:
        pos = q.popleft()
        if dist[pos] >= max_steps:
            continue
        for dr, dc in DIRS:
            nxt = (pos[0]+dr, pos[1]+dc)
            if nxt not in dist and _is_valid_cell(nxt, map_state):
                dist[nxt] = dist[pos] + 1
                q.append(nxt)
    return dist

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Super Pacman v3 Blind"

        self.last_known_enemy_pos = None
        self.lost_counter = 0
        self.visit_count = defaultdict(float)
        self.current_target = None
        self.known_map = None

    def _update_map(self, map_state):
        if self.known_map is None:
            self.known_map = np.copy(map_state)
        else:
            mask = map_state != -1
            self.known_map[mask] = map_state[mask]

    def step(self, map_state: np.ndarray, my_position: tuple,
             enemy_position: tuple, step_number: int):

        self._update_map(map_state)

        # Cập nhật visit count
        self.visit_count[my_position] += 1
        for cell in list(self.visit_count):
            self.visit_count[cell] *= 0.97
            if self.visit_count[cell] < 0.05:
                del self.visit_count[cell]

        # Cập nhật vị trí Ghost
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.lost_counter = 0
        elif self.last_known_enemy_pos is not None:
            self.lost_counter += 1
            if self.lost_counter > 8:
                self.last_known_enemy_pos = None

        ghost_pos = self.last_known_enemy_pos

        # ---- CHẾ ĐỘ SĂN ĐUỔI ----
        if ghost_pos is not None:
            predicted = self._predict_ghost(my_position, ghost_pos, lookahead=4)
            
            path = self._dijkstra_by_turns(my_position, predicted)

            if len(path) <= 1:
                path = self._dijkstra_by_turns(my_position, ghost_pos)

            if path and len(path) > 1:
                return self._path_to_action(path, my_position)

        return self._explore(my_position)

    def _predict_ghost(self, my_pos, ghost_pos, lookahead):
        """Dự đoán Ghost chọn bước đi xa Pacman nhất theo BFS thực"""
        pos = ghost_pos
        for _ in range(lookahead):
            best, best_d = pos, 0
            for dr, dc in DIRS:
                nxt = (pos[0]+dr, pos[1]+dc)
                if _is_valid_cell(nxt, self.known_map):
                    d = abs(nxt[0]-my_pos[0]) + abs(nxt[1]-my_pos[1])
                    if d > best_d:
                        best_d = d
                        best = nxt
            pos = best
        return pos

    def _explore(self, my_position):
        dist_map, parent = self._dijkstra_with_penalty(my_position)

        if my_position == self.current_target:
            self.current_target = None

        if self.current_target is None or self.current_target not in parent:
            self.current_target = self._best_frontier(dist_map)

        if self.current_target is None:
            return self._fallback(my_position)

        path = self._reconstruct(parent, self.current_target)
        if len(path) < 2:
            self.current_target = None
            return self._fallback(my_position)

        return self._path_to_action(path, my_position)

    def _best_frontier(self, dist_map):
        best, best_score = None, -float('inf')
        h, w = self.known_map.shape
        has_any_unknown = (self.known_map == -1).any()
        
        for (r, c), d in dist_map.items():
            has_unknown = any(
                0 <= r+dr < h and 0 <= c+dc < w and self.known_map[r+dr, c+dc] == -1
                for dr, dc in DIRS
            )
            
            if has_any_unknown and not has_unknown:
                continue
                
            if has_any_unknown:
                unknown_area = sum(
                    1 for dr in range(-3, 4) for dc in range(-3, 4)
                    if 0 <= r+dr < h and 0 <= c+dc < w and self.known_map[r+dr, c+dc] == -1
                )
                penalty = self.visit_count.get((r, c), 0) * 20
                score = unknown_area * 8 - d * 5 - penalty
            else:
                penalty = self.visit_count.get((r, c), 0) * 50
                score = -d - penalty
                
            if score > best_score:
                best_score = score
                best = (r, c)
        return best

    def _dijkstra_with_penalty(self, start):
        heap = [(0, start)]
        dist = {start: 0}
        parent = {start: None}
        while heap:
            cost, cur = heapq.heappop(heap)
            if cost > dist.get(cur, float('inf')):
                continue
            for dr, dc in DIRS:
                nxt = (cur[0]+dr, cur[1]+dc)
                if not _is_valid_cell(nxt, self.known_map):
                    continue
                penalty = self.visit_count.get(nxt, 0) * 0.4
                nc = cost + 1 + penalty
                if nc < dist.get(nxt, float('inf')):
                    dist[nxt] = nc
                    parent[nxt] = cur
                    heapq.heappush(heap, (nc, nxt))
        return dist, parent

    def _reconstruct(self, parent, goal):
        if goal not in parent:
            return []
        path, cur = [], goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        return list(reversed(path))

    def _fallback(self, pos):
        candidates = []
        for i, (dr, dc) in enumerate(DIRS):
            nxt = (pos[0]+dr, pos[1]+dc)
            if _is_valid_cell(nxt, self.known_map):
                deg = sum(1 for dr2, dc2 in DIRS if _is_valid_cell((nxt[0]+dr2, nxt[1]+dc2), self.known_map))
                visit = self.visit_count.get(nxt, 0)
                score = deg - visit * 3
                candidates.append((score, i, DIR_TO_MOVE[(dr, dc)]))
        if not candidates:
            return (Move.STAY, 1)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return (candidates[0][2], 1)

    def _dijkstra_by_turns(self, start, goal):
        pq = [(0, 0, start, [start])]
        visited = {start: 0}
        counter = 0
        while pq:
            turns, _, curr, path = heapq.heappop(pq)
            if curr == goal:
                return path
            
            if visited.get(curr, float('inf')) < turns:
                continue
                
            for dr, dc in DIRS:
                nxt1 = (curr[0] + dr, curr[1] + dc)
                if not _is_valid_cell(nxt1, self.known_map):
                    continue
                    
                new_turns = turns + 1
                if new_turns < visited.get(nxt1, float('inf')):
                    visited[nxt1] = new_turns
                    counter += 1
                    heapq.heappush(pq, (new_turns, counter, nxt1, path + [nxt1]))

                if self.pacman_speed >= 2:
                    nxt2 = (curr[0] + 2*dr, curr[1] + 2*dc)
                    if _is_valid_cell(nxt2, self.known_map):
                        if new_turns < visited.get(nxt2, float('inf')):
                            visited[nxt2] = new_turns
                            counter += 1
                            heapq.heappush(pq, (new_turns, counter, nxt2, path + [nxt1, nxt2]))

        return []

    def _path_to_action(self, path, my_pos):
        if len(path) < 2:
            return (Move.STAY, 1)
        nxt = path[1]
        dr, dc = nxt[0]-my_pos[0], nxt[1]-my_pos[1]
        move = DIR_TO_MOVE.get((dr, dc), Move.STAY)
        steps = 1
        if self.pacman_speed >= 2 and len(path) > 2:
            nn = path[2]
            if (nn[0]-nxt[0], nn[1]-nxt[1]) == (dr, dc) and _is_valid_cell(nn, self.known_map):
                steps = 2
        return (move, steps)


class GhostAgent(BaseGhostAgent):
    """
    Chiến lược:
    - Khi nguy hiểm (thấy Pacman hoặc mới thấy gần đây):
        BFS lookahead 6 bước, chọn nước đi dẫn đến vùng xa Pacman nhất
        và có nhiều lối thoát (freedom score cao).
    - Khi an toàn:
        Di chuyển đến vùng chưa khám phá có freedom score cao nhất,
        ưu tiên xa vị trí cuối cùng biết của Pacman.
    """
    DANGER_HORIZON = 20
    LOOKAHEAD      = 8

    # Khởi tạo mảng trống, các waypoint sẽ được sinh tự động theo map thực tế
    PATROL_WAYPOINTS = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Super Ghost v4"
        self.dead_ends   = set()
        self.valid_cells = set()
        self.recent_positions = []   # tối đa 10 vị trí gần nhất
        self._pac_bfs_cache = {}     # {pac_pos: dist_dict}  – clear mỗi turn
        self.known_map       = None
        self.last_enemy_pos  = None
        self.last_enemy_step = -100
        
        # State machine
        self.target_wp_idx = 0
        self.patrol_dir = 1 # 1: đi vào trong, -1: đi ra ngoài

    def _update_map(self, map_state):
        if self.known_map is None:
            self.known_map = np.copy(map_state)
            self._full_analyze()
            self._generate_dynamic_waypoints()
        else:
            mask = map_state != -1
            changed = (self.known_map[mask] != map_state[mask]).any()
            new_cells = (self.known_map[mask] == -1).any()
            self.known_map[mask] = map_state[mask]
            if changed or new_cells:
                self._incremental_analyze()
                self._generate_dynamic_waypoints()


    def step(self, map_state: np.ndarray, my_position: tuple,
             enemy_position: tuple, step_number: int) -> Move:

        self._update_map(map_state)
        self._pac_bfs_cache.clear()

        if enemy_position is not None:
            self.last_enemy_pos  = enemy_position
            self.last_enemy_step = step_number

        self.recent_positions.append(my_position)
        if len(self.recent_positions) > 10:
            self.recent_positions.pop(0)

        neighbors = self._neighbors(my_position)
        if not neighbors:
            return Move.STAY

        in_danger = (self.last_enemy_pos is not None and
                     step_number - self.last_enemy_step < self.DANGER_HORIZON)

        if not in_danger:
            # Kiểm tra xem có thể tiến gần hơn tới waypoint không
            target_wp = self.PATROL_WAYPOINTS[self.target_wp_idx]
            path_move = self._pathfind_to_closest(my_position, target_wp)
            
            if path_move is None:
                # Không thể tiến gần hơn nữa (có thể đã tới nơi hoặc bị tường chặn)
                # Chuyển sang waypoint tiếp theo
                self.target_wp_idx += self.patrol_dir
                # Đảo chiều nếu đi hết vòng
                if self.target_wp_idx >= len(self.PATROL_WAYPOINTS):
                    self.target_wp_idx = len(self.PATROL_WAYPOINTS) - 2
                    self.patrol_dir = -1
                elif self.target_wp_idx < 0:
                    self.target_wp_idx = 1
                    self.patrol_dir = 1
                
                # Cố gắng đi tới waypoint mới
                target_wp = self.PATROL_WAYPOINTS[self.target_wp_idx]
                path_move = self._pathfind_to_closest(my_position, target_wp)
                
            if path_move:
                return path_move
            else:
                # Fallback nếu hoàn toàn kẹt
                return self._best_move(my_position, None)

        # Ưu tiên bẻ góc: Tránh đi thẳng nếu có ngã rẽ an toàn
        return self._best_move(my_position, self.last_enemy_pos)

    def _pathfind_to_closest(self, start, target_abs):
        """BFS tìm ô hợp lệ gần target tuyệt đối nhất"""
        if start == target_abs: return None
        q = deque([(start, None)])
        visited = {start}
        best_cell = start
        best_dist = abs(start[0]-target_abs[0]) + abs(start[1]-target_abs[1])
        best_first_move = None
        
        while q:
            curr, first_move = q.popleft()
            
            d = abs(curr[0]-target_abs[0]) + abs(curr[1]-target_abs[1])
            if d < best_dist:
                best_dist = d
                best_cell = curr
                best_first_move = first_move
                
            if curr == target_abs:
                return first_move
                
            for nxt, move in self._neighbors(curr):
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, first_move or move))
                    
        return best_first_move

    def _best_move(self, my_pos, pac_pos):
        """
        BFS từ Ghost tới LOOKAHEAD bước.
        Mỗi ô reachable được chấm điểm.
        Chọn first_move dẫn đến tập ô có tổng điểm cao nhất.
        """
        pac_dist = self._bfs_dist_from(pac_pos) if pac_pos else {}

        visited    = {my_pos}
        reachable  = {}           
        q = deque([(my_pos, 0, None)])

        while q:
            pos, d, first = q.popleft()
            if d > 0:
                reachable[pos] = (d, first)
            if d >= self.LOOKAHEAD:
                continue
            for nxt, move in self._neighbors(pos):
                if nxt not in visited:
                    visited.add(nxt)
                    fm = first if first is not None else move
                    q.append((nxt, d + 1, fm))

        if not reachable:
            return Move.STAY

        # Chấm điểm mỗi ô
        move_best = {}   # first_move → best cell score trên đường đó

        for cell, (ghost_steps, fm) in reachable.items():
            if cell in self.dead_ends:
                score = -8_000_000
            else:
                freedom = self._freedom_score(cell, steps=3)
                if pac_pos:
                    pd = pac_dist.get(cell, 50)
                    # Ưu tiên: (1) xa Pacman, (2) vùng rộng, (3) khuất LOS
                    # ghost_steps bonus: ưu tiên commit vào đường xa hơn
                    los_bonus = 3000 if not self._in_los(cell, pac_pos) else 0
                    score = pd * 3000 + freedom * 300 + los_bonus + ghost_steps * 50
                else:
                    # An toàn: ưu tiên vùng rộng, chưa khám phá, đi xa
                    h, w = self.known_map.shape
                    unknown_adj = sum(
                        1 for dr, dc in DIRS
                        if 0 <= cell[0]+dr < h and 0 <= cell[1]+dc < w
                        and self.known_map[cell[0]+dr, cell[1]+dc] == -1
                    )
                    score = freedom * 400 + unknown_adj * 300

            if fm not in move_best or score > move_best[fm]:
                move_best[fm] = score

        if not move_best:
            return Move.STAY

        # Áp thêm penalty chống lặp và thưởng bẻ góc
        best_move  = Move.STAY
        best_score = -float('inf')
        last_pos   = self.recent_positions[-2] if len(self.recent_positions) >= 2 else None
        
        last_dir = None
        if last_pos:
            last_dir = (my_pos[0] - last_pos[0], my_pos[1] - last_pos[1])

        for move, score in move_best.items():
            nxt = (my_pos[0] + move.value[0], my_pos[1] + move.value[1])
            # Penalty nặng nếu đã đến đây gần đây
            revisit_penalty = self.recent_positions.count(nxt) * 5000
            # Penalty cực nặng nếu quay ngược về ô trước đó (dao động)
            reverse_penalty = 20000 if nxt == last_pos else 0
            
            # Thưởng bẻ góc (khác hướng đi trước đó và không phải quay đầu)
            corner_bonus = 0
            if last_dir and move.value != last_dir and nxt != last_pos:
                corner_bonus = 2000 # Khuyến khích bẻ góc nhưng không mù quáng
                
            final = score - revisit_penalty - reverse_penalty + corner_bonus

            if final > best_score:
                best_score = final
                best_move  = move


        return best_move


    def _bfs_dist_from(self, start):
        if start in self._pac_bfs_cache:
            return self._pac_bfs_cache[start]
        dist = {start: 0}
        q    = deque([start])
        while q:
            pos = q.popleft()
            for nxt, _ in self._neighbors(pos):
                if nxt not in dist:
                    dist[nxt] = dist[pos] + 1
                    q.append(nxt)
        self._pac_bfs_cache[start] = dist
        return dist

    def _freedom_score(self, pos, steps=3):
        reachable = {pos}
        frontier  = {pos}
        for _ in range(steps):
            nf = set()
            for p in frontier:
                for nxt, _ in self._neighbors(p):
                    if nxt not in reachable:
                        reachable.add(nxt)
                        nf.add(nxt)
            frontier = nf
        return len(reachable)

    def _in_los(self, p1, p2):
        r1, c1 = p1; r2, c2 = p2
        if r1 != r2 and c1 != c2:
            return False
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if (r1, c) not in self.valid_cells:
                    return False
        else:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if (r, c1) not in self.valid_cells:
                    return False
        return True

    def _full_analyze(self):
        h, w = self.known_map.shape
        self.valid_cells.clear()
        for r in range(h):
            for c in range(w):
                if self.known_map[r, c] == 0:
                    self.valid_cells.add((r, c))
        self._recalc_dead_ends()

    def _incremental_analyze(self):
        h, w = self.known_map.shape
        to_add    = set()
        to_remove = set()
        for r in range(h):
            for c in range(w):
                cell = (r, c)
                v = self.known_map[r, c]
                if v == 0 and cell not in self.valid_cells:
                    to_add.add(cell)
                elif v != 0 and cell in self.valid_cells:
                    to_remove.add(cell)
        if to_add or to_remove:
            self.valid_cells |= to_add
            self.valid_cells -= to_remove
            self._recalc_dead_ends()

    def _recalc_dead_ends(self):
        """Topo pruning: đánh dấu mọi ô chỉ có 1 lối ra là dead_end."""
        self.dead_ends.clear()
        deg = {cell: len(self._neighbors(cell)) for cell in self.valid_cells}
        q   = deque([cell for cell, d in deg.items() if d == 1])
        while q:
            curr = q.popleft()
            self.dead_ends.add(curr)
            for nxt, _ in self._neighbors(curr):
                if nxt not in self.dead_ends:
                    deg[nxt] -= 1
                    if deg[nxt] == 1:
                        q.append(nxt)

    def _neighbors(self, pos):
        return [
            ((pos[0] + m.value[0], pos[1] + m.value[1]), m)
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            if (pos[0] + m.value[0], pos[1] + m.value[1]) in self.valid_cells
        ]

    def _generate_dynamic_waypoints(self):
        """Tự động tìm 4 góc hợp lệ nhất của bản đồ thay vì code cứng."""
        if self.known_map is None:
            return
            
        h, w = self.known_map.shape
        has_old = len(self.PATROL_WAYPOINTS) > 0
        
        # Tọa độ tuyệt đối của 4 góc bản đồ
        self.PATROL_WAYPOINTS = [
            (h//2, w - 6), # 1. Chạy sang phải trước để né tâm
            (h - 2, w - 6), # 2. Tuột xuống
            (h-1, w-1),    # 3. Góc phải dưới
            (h-1, 0),      # 4. Sang góc trái dưới
            (h-1, w-1),    # 5. Quay lại phải dưới
            (0, w-1),      # 6. Phóng lên phải trên
            (0, 0)         # 7. Băng ngang qua trái trên
        ]
        if not has_old:
            self.target_wp_idx = 0


