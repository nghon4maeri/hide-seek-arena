import math
"""
Hide and Seek Arena - Agent Implementation (Nhóm 3)
====================================================
File này chứa toàn bộ code agent cho đồ án Hide and Seek Arena.

Cấu trúc:
    1. Import và setup path
    2. Helper functions dùng chung (BFS, validation, distance)
    3. PacmanAgent (Seek) - Capture Zone + BFS Distance + Belief Tracking   [Người T2]
    4. GhostAgent (Hide) - Danger Map + Monte Carlo Rollout + Minimax       [Người T3]

Phân công:
    Người T1 - Core, format, tích hợp:
        - Tạo khung agent.py chuẩn theo TEMPLATE_agent.py
        - Viết bộ hàm helper dùng chung
        - Đảm bảo đúng format framework (numpy array, Move enum)

    Người T2 - PacmanAgent (Seek):
        - Capture Zone Pursuit: tính vùng bắt được sau 1 lượt, ép Ghost vào vùng hẹp
        - BFS Distance: dùng khoảng cách thực tế qua tường thay vì Manhattan
        - A* Search: tìm đường ngắn nhất có heuristic
        - Belief Tracking: xác suất vị trí Ghost khi fog-of-war
        - Interception: chặn đầu Ghost dựa trên BFS + velocity prediction
        - Mode Switching: Pursuit → Persistent → Patrol
        - Speed Optimization: đi thẳng max speed khi có lợi
        - Anti-loop: phát hiện và phá vòng lặp

    Người T3 - GhostAgent (Hide):
        - Danger Map: tính vùng Pacman có thể bắt trong lượt kế, tránh vào
        - Monte Carlo Rollout: simulate 8 games (depth 12) cho mỗi nước đi, chọn nước sống lâu nhất
        - Minimax (depth=2): phòng thủ cục bộ khi bị dồn ép
        - BFS Distance Scoring + Escape routes
        - Dead-end penalty + Escape area analysis

LƯU Ý KHI NỘP:
    - Copy file này vào: pacman/submissions/<group_id>/agent.py
    - KHÔNG sửa class name, method signature
    - PacmanAgent.step trả Move hoặc (Move, steps)
    - GhostAgent.step chỉ trả Move
"""

import sys
import random
import heapq
import time
from pathlib import Path
from collections import deque
from typing import List, Optional, Tuple, Dict, Set

# ============================================================
# Setup path để import được các module từ framework
# Hỗ trợ 2 vị trí:
#   1. Khi dev:  AI/agent.py           → tìm AI/pacman/src/
#   2. Khi nộp:  pacman/submissions/3/agent.py → tìm pacman/src/
# ============================================================
_current_dir = Path(__file__).parent
_src_candidates = [
    _current_dir / "pacman" / "src",             # Khi file ở AI/agent.py (dev)
    _current_dir.parent.parent / "src",          # Khi file ở submissions/3/agent.py (nộp)
]
for _src_path in _src_candidates:
    if _src_path.exists():
        sys.path.insert(0, str(_src_path))
        break
else:
    # Fallback: thử cả 2 dù chưa chắc tồn tại
    for _src_path in _src_candidates:
        sys.path.insert(0, str(_src_path))

from agent_interface import PacmanAgent as BasePacmanAgent  # type: ignore
from agent_interface import GhostAgent as BaseGhostAgent  # type: ignore
from environment import Move  # type: ignore
import numpy as np


# ============================================================
# ============ HELPER FUNCTIONS DÙNG CHUNG ===================
# ============================================================
# (Người T1 - Core helpers dùng cho cả PacmanAgent và GhostAgent)

# Danh sách 4 hướng di chuyển (không tính STAY)
FOUR_DIRECTIONS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]


def can_plan(map_state: np.ndarray, pos: tuple) -> bool:
    return is_valid_position(map_state, pos)

def can_step_safely(map_state: np.ndarray, pos: tuple) -> bool:
    return is_walkable(map_state, pos)

def is_valid_position(map_state: np.ndarray, pos: tuple) -> bool:
    """
    Kiểm tra vị trí pos = (row, col) có hợp lệ không.
    Cho phép đi qua cả ô empty (0) và unseen (-1), chỉ chặn tường (1).

    Args:
        map_state: numpy 2D array, 0=empty, 1=wall, -1=fog
        pos: Tuple (row, col)

    Returns:
        bool: True nếu vị trí hợp lệ và đi được (không phải tường)
    """
    row, col = pos
    height, width = map_state.shape
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    return map_state[row, col] != 1  # 0=empty hoặc -1=unseen đều cho đi


def is_walkable(map_state: np.ndarray, pos: tuple) -> bool:
    """
    Kiểm tra vị trí chắc chắn đi được (chỉ ô empty=0).
    Khác is_valid_position ở chỗ: fog (-1) KHÔNG được coi là đi được.

    Args:
        map_state: numpy 2D array, 0=empty, 1=wall, -1=fog
        pos: Tuple (row, col)

    Returns:
        bool: True nếu vị trí là ô trống (empty)
    """
    row, col = pos
    height, width = map_state.shape
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    return map_state[row, col] == 0



def apply_move(pos: tuple, move: Move) -> tuple:
    """
    Tính vị trí mới sau khi áp dụng một move từ vị trí hiện tại.

    Args:
        pos: Tuple (row, col) - vị trí hiện tại
        move: Move enum (UP, DOWN, LEFT, RIGHT, STAY)

    Returns:
        Tuple (new_row, new_col) - vị trí mới
    """
    delta_row, delta_col = move.value
    return (pos[0] + delta_row, pos[1] + delta_col)


def get_neighbors(map_state: np.ndarray, pos: tuple, allow_unseen: bool = True) -> list:
    """
    Lấy tất cả các vị trí kề hợp lệ (đi được) từ vị trí pos.

    Chỉ xét 4 hướng (UP, DOWN, LEFT, RIGHT), không xét STAY.

    Args:
        map_state: numpy 2D array
        pos: Tuple (row, col)
        allow_unseen: Nếu True, cho phép đi vào ô unseen (-1).
                      Nếu False, chỉ cho đi vào ô empty (0).

    Returns:
        List[Tuple[Tuple[int,int], Move]]:
            Danh sách các (new_pos, move) hợp lệ
    """
    neighbors = []
    for move in FOUR_DIRECTIONS:
        new_pos = apply_move(pos, move)
        if allow_unseen:
            if is_valid_position(map_state, new_pos):
                neighbors.append((new_pos, move))
        else:
            if is_walkable(map_state, new_pos):
                neighbors.append((new_pos, move))
    return neighbors


def manhattan_distance(a: tuple, b: tuple) -> int:
    """
    Tính khoảng cách Manhattan giữa 2 điểm.

    Manhattan distance = |row_a - row_b| + |col_a - col_b|

    Đây là ước lượng nhanh nhưng KHÔNG xét tường.
    Dùng để so sánh nhanh, không thay thế BFS distance.

    Args:
        a: Tuple (row, col) - điểm thứ nhất
        b: Tuple (row, col) - điểm thứ hai

    Returns:
        int: Khoảng cách Manhattan
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_distance_between(map_state: np.ndarray, a: tuple, b: tuple) -> int:
    """
    BFS tính khoảng cách thực tế ngắn nhất giữa 2 điểm trên map.

    Khác Manhattan: có xét tường, trả về khoảng cách đi được thực tế.
    Nếu không tìm được đường → trả về 999.

    Args:
        map_state: numpy 2D array
        a: Tuple (row, col) - điểm xuất phát
        b: Tuple (row, col) - điểm đích

    Returns:
        int: Khoảng cách BFS thực tế, 999 nếu không có đường
    """
    if a == b:
        return 0
    queue = deque([(a, 0)])
    visited = {a}
    while queue:
        cur, d = queue.popleft()
        for move in FOUR_DIRECTIONS:
            nxt = apply_move(cur, move)
            if nxt == b:
                return d + 1
            if nxt not in visited and is_valid_position(map_state, nxt):
                visited.add(nxt)
                queue.append((nxt, d + 1))
    return 999


def bfs_distance_map(map_state: np.ndarray, start: tuple) -> Dict[Tuple[int, int], int]:
    """
    BFS tính khoảng cách ngắn nhất từ start đến TẤT CẢ ô reachable.

    Trả về dict {(row,col): distance}.

    Args:
        map_state: numpy 2D array
        start: Tuple (row, col) - điểm xuất phát

    Returns:
        Dict mapping (row,col) -> BFS distance
    """
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for mv in FOUR_DIRECTIONS:
            nxt = apply_move(cur, mv)
            if nxt not in dist and is_valid_position(map_state, nxt):
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


def move_from_to(cur: tuple, nxt: tuple) -> Optional[Move]:
    """
    Tìm Move enum từ vị trí cur đến vị trí nxt liền kề.

    Args:
        cur: Tuple (row, col) - vị trí hiện tại
        nxt: Tuple (row, col) - vị trí tiếp theo

    Returns:
        Move hoặc None nếu không phải ô liền kề
    """
    dr = nxt[0] - cur[0]
    dc = nxt[1] - cur[1]
    for m in FOUR_DIRECTIONS:
        if m.value == (dr, dc):
            return m
    return None


def get_cells_reachable(map_state: np.ndarray, start: tuple, max_steps: int) -> Set[tuple]:
    """
    BFS lấy tất cả ô có thể đến được trong max_steps bước.
    Dùng cho capture zone (Pacman) và danger map (Ghost).

    Args:
        map_state: numpy 2D array
        start: Tuple (row, col)
        max_steps: Số bước tối đa

    Returns:
        Set of (row, col) reachable positions
    """
    cells = {start}
    queue = deque([(start, 0)])
    while queue:
        cur, d = queue.popleft()
        if d >= max_steps:
            continue
        for mv in FOUR_DIRECTIONS:
            nxt = apply_move(cur, mv)
            if nxt not in cells and is_valid_position(map_state, nxt):
                cells.add(nxt)
                queue.append((nxt, d + 1))
    return cells


# ============================================================
# ================ PACMAN AGENT (SEEK) =======================
# ============================================================
# (Người T2 - Capture Zone + BFS Distance + A* + Belief Tracking)

class PacmanAgent(BasePacmanAgent):
    """
    Agent Pacman (Seek) - Tìm và bắt Ghost càng nhanh càng tốt.

    Chiến thuật nâng cao (Người T2):
        1. Capture Zone Pursuit: tính vùng bắt được sau 1 lượt,
           chọn nước đi ép Ghost vào vùng hẹp thay vì đuổi thẳng
        2. BFS Distance: dùng khoảng cách thực tế (có xét tường)
           trong mọi quyết định quan trọng, thay vì Manhattan
        3. A* Search: tìm đường ngắn nhất có heuristic, nhanh hơn BFS
        4. Belief Tracking: xác suất vị trí Ghost khi fog-of-war
        5. Interception: chặn đầu Ghost dựa trên BFS + velocity prediction
        6. Mode Switching: Pursuit → Persistent → Patrol
        7. Speed Optimization: đi thẳng max speed khi có lợi
        8. Anti-loop: phát hiện và phá vòng lặp

    Return format:
        - Tuple (Move, steps) với steps từ 1 đến pacman_speed
    """

    def __init__(self, **kwargs):
        """Khởi tạo PacmanAgent."""
        super().__init__(**kwargs)
        self.name = "Group3 CaptureZone Seek"
        # pacman_speed lấy từ kwargs (framework truyền vào)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))

        # ----- Map Memory (Fog-of-War) -----
        # global_map lưu tất cả ô đã từng nhìn thấy
        # Ban đầu = -1 (unseen), dần dần fill khi khám phá
        self.global_map = None  # Khởi tạo khi biết kích thước map

        # ----- Belief Tracking -----
        # belief[r][c] = xác suất Ghost đang ở ô (r,c)
        # Khi thấy Ghost → belief = 1.0 tại đó, 0.0 ở chỗ khác
        # Khi mất dấu → belief phân tán sang ô liền kề mỗi step
        self.belief = None

        # ----- Ghost Memory -----
        self.last_known_enemy_pos = None  # Vị trí Ghost cuối cùng nhìn thấy
        self.enemy_history = deque(maxlen=5)  # Lịch sử vị trí Ghost (tính velocity)
        self.steps_since_seen = 0  # Số step đã mất dấu Ghost

        # ----- Mode Control -----
        # pursuit: đang thấy Ghost → đuổi/chặn
        # persistent: vừa mất dấu (< PERSISTENCE turns) → dùng belief
        # patrol: mất dấu lâu → tuần tra các điểm chiến lược
        self.pursuit_mode = False
        self.PURSUIT_PERSISTENCE = 8  # Số turn tiếp tục đuổi sau khi mất dấu

        # ----- Patrol System -----
        # Khi không biết Ghost ở đâu, Pacman đi tuần tra qua các vị trí này
        # Các vị trí được chọn để bao phủ toàn map 21x21
        self.patrol_index = 0
        self.PATROL_TARGETS = [
            (5, 5), (5, 15), (10, 10), (15, 5), (15, 15),
            (3, 10), (10, 3), (10, 17), (17, 10), (9, 10),
        ]

        # ----- Anti-loop -----
        self.my_history = deque(maxlen=12)  # Lưu vị trí gần đây

        # ----- BFS Cache -----
        self._bfs_cache: Dict[tuple, Dict[Tuple[int, int], int]] = {}
        self._bfs_cache_step = -1

    def _init_patrol_targets(self, h, w):
        raw_targets = [
            (h // 4, w // 4),
            (h // 4, 3 * w // 4),
            (h // 2, w // 2),
            (3 * h // 4, w // 4),
            (3 * h // 4, 3 * w // 4),
            (h // 2, w // 4),
            (h // 2, 3 * w // 4),
        ]

        self.PATROL_TARGETS = []
        for r, c in raw_targets:
            r = max(0, min(h - 1, r))
            c = max(0, min(w - 1, c))
            self.PATROL_TARGETS.append((r, c))

    # =================================================================
    # MAIN STEP - được gọi mỗi lượt
    # =================================================================

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):
        """
        Logic chính mỗi lượt:
        1. Cập nhật map memory + belief
        2. Xác định mode (pursuit / persistent / patrol)
        3. Nếu thấy Ghost + gần → dùng Capture Zone Pursuit
        4. Chọn target phù hợp theo mode
        5. Tìm đường đến target bằng A*
        6. Tối ưu speed (đi nhiều bước nếu thẳng)
        7. Anti-loop nếu bị kẹt

        Args:
            map_state: numpy 2D array (0=empty, 1=wall, -1=fog)
            my_position: Tuple (row, col) vị trí Pacman
            enemy_position: Tuple (row, col) vị trí Ghost, có thể None
            step_number: Số bước hiện tại

        Returns:
            Tuple (Move, steps) hoặc Move
        """
        # --- 0. Khởi tạo lần đầu ---
        h, w = map_state.shape
        if not hasattr(self, "_patrol_initialized"):
            self._init_patrol_targets(h, w)
            self._patrol_initialized = True

        if self.global_map is None:
            self.global_map = np.full((h, w), -1, dtype=int)
            self.belief = np.ones((h, w), dtype=float)
            # Xác suất ban đầu đều nhau trên tất cả ô
            self.belief /= self.belief.sum()

        # --- Cache management ---
        if step_number != self._bfs_cache_step:
            self._bfs_cache.clear()
            self._bfs_cache_step = step_number

        # --- 1. Cập nhật map memory ---
        self._update_map_memory(map_state)

        # --- 2. Cập nhật belief ---
        self._update_belief(map_state, my_position, enemy_position)

        # --- 2.5. Immediate capture check ---
        if enemy_position is not None:
            kill = self._immediate_capture_move(my_position, enemy_position)
            if kill is not None:
                return kill
            
            bfs_dist = self._bfs_dist(my_position, enemy_position)
            if bfs_dist <= 3: # Ghost is near
                aggressive = self._aggressive_chase_move(my_position, enemy_position)
                if aggressive is not None:
                    return aggressive

        # --- 3. Anti-loop check ---
        self.my_history.append(my_position)
        if self._is_looping(my_position):
            return self._escape_loop(my_position)

        # --- 4. Capture Zone Pursuit khi thấy Ghost ---
        if enemy_position is not None:
            bfs_dist = self._bfs_dist(my_position, enemy_position)
            if bfs_dist <= 8:
                # Gần → dùng Capture Zone Pursuit (tính toán thông minh hơn)
                cz_move = self._capture_zone_pursuit(my_position, enemy_position)
                if cz_move is not None:
                    return cz_move

        # --- 5. Xác định mode & target ---
        target, mode = self._determine_target(
            my_position, enemy_position, step_number
        )

        # --- 6. Nếu không có target → fallback explore ---
        if target is None:
            return self._frontier_explore(my_position)

        # --- 7. Straight-line speed advantage ---
        # Nếu cùng hàng/cột với target, đi thẳng max speed ngay
        if self._on_same_axis(my_position, target):
            straight = self._straight_line_move(my_position, target)
            if straight is not None:
                return straight

        # --- 8. A* pathfinding ---
        path = self._astar_path(my_position, target)

        if path and len(path) >= 2:
            next_pos = path[1]  # path[0] = start
            move = move_from_to(my_position, next_pos)

            if move is not None:
                # Tính số bước thẳng theo path
                steps = self._calc_straight_steps(path, move)
                steps = self._validate_steps(my_position, move, steps)
                if steps > 0:
                    return (move, steps)

        # --- 9. Fallback: greedy toward target ---
        move = self._greedy_toward(my_position, target)
        if move != Move.STAY:
            steps = self._validate_steps(my_position, move, 1)
            return (move, max(1, steps))

        # --- 10. Absolute fallback ---
        return self._any_valid_move(my_position)

    # =================================================================
    # BFS DISTANCE CACHE (Ưu tiên 2)
    # =================================================================

    def _get_bfs_map(self, start: tuple) -> Dict[Tuple[int, int], int]:
        """
        BFS tính khoảng cách từ start đến tất cả ô, CÓ CACHE.
        Dùng global_map để có đường đi qua cả vùng đã khám phá.
        """
        if start in self._bfs_cache:
            return self._bfs_cache[start]

        dist = bfs_distance_map(self.global_map, start)
        self._bfs_cache[start] = dist
        return dist

    def _bfs_dist(self, a: tuple, b: tuple) -> int:
        """
        Trả BFS distance giữa 2 điểm cụ thể, dùng cache.
        Nhanh hơn gọi bfs_distance_between mỗi lần.
        """
        dist_map = self._get_bfs_map(a)
        return dist_map.get(b, 999)

    # =================================================================
    # CAPTURE ZONE PURSUIT (Ưu tiên 1)
    # =================================================================

    def _compute_capture_zone(self, pac_pos: tuple) -> Set[tuple]:
        """
        Capture zone sau khi Pacman đã di chuyển:
        Pacman bắt Ghost nếu Manhattan distance < 2.
        Gồm: ô hiện tại + 4 ô kề (UP/DOWN/LEFT/RIGHT).
        """
        zone = {pac_pos}
        for mv in FOUR_DIRECTIONS:
            nxt = apply_move(pac_pos, mv)
            if is_valid_position(self.global_map, nxt):
                zone.add(nxt)
        return zone


    def _aggressive_chase_move(self, my_position, enemy_position):
        best = None
        best_score = float("-inf")

        for mv in FOUR_DIRECTIONS:
            pos = my_position
            for steps in range(1, self.pacman_speed + 1):
                nxt = apply_move(pos, mv)
                if not is_valid_position(self.global_map, nxt):
                    break
                pos = nxt

                dist = self._bfs_dist(pos, enemy_position)
                ghost_exits = sum(
                    1 for gm in FOUR_DIRECTIONS
                    if is_valid_position(self.global_map, apply_move(enemy_position, gm))
                )

                score = -dist * 10 - ghost_exits * 3 + steps
                if score > best_score:
                    best_score = score
                    best = (mv, steps)

        return best

    def _immediate_capture_move(self, my_position: tuple, enemy_position: tuple):
        """
        Kiểm tra nước bắt Ghost ngay trong lượt hiện tại.
        Trả về (Move, steps) nếu có thể bắt chắc chắn bất kể Ghost đi đâu.
        """
        if enemy_position is None:
            return None

        ghost_next_moves = [enemy_position]
        for gm in FOUR_DIRECTIONS:
            gnxt = apply_move(enemy_position, gm)
            if is_valid_position(self.global_map, gnxt):
                ghost_next_moves.append(gnxt)

        for mv in FOUR_DIRECTIONS:
            pos = my_position
            for steps in range(1, self.pacman_speed + 1):
                nxt = apply_move(pos, mv)
                if not is_valid_position(self.global_map, nxt):
                    break
                pos = nxt
                cap_zone = self._compute_capture_zone(pos)
                if all(g in cap_zone for g in ghost_next_moves):
                    return (mv, steps)

        cap_zone_here = self._compute_capture_zone(my_position)
        if all(g in cap_zone_here for g in ghost_next_moves):
            return (Move.STAY, 1)

        return None

    def _capture_zone_pursuit(self, my_position: tuple, ghost_position: tuple):
        """
        Capture Zone Pursuit: chọn nước đi tối ưu dựa trên tư duy
        "ép Ghost vào vùng hẹp" thay vì "đuổi đến vị trí hiện tại".

        Thuật toán:
        1. Với mỗi candidate move của Pacman → tính vị trí mới → capture zone mới
        2. Với mỗi candidate move của Ghost (4 hướng + STAY) → vị trí Ghost mới
        3. Chọn move Pacman sao cho: maximize số vị trí Ghost mới nằm trong
           capture zone, đồng thời minimize BFS distance đến Ghost

        Đây là minimax 1-ply: Pacman MAX, Ghost MIN.
        Ghost sẽ chọn nước tránh capture zone tốt nhất →
        Pacman phải chọn nước mà DÙ Ghost tránh tốt nhất,
        vẫn ép được Ghost nhiều nhất.

        Returns:
            (Move, steps) hoặc None nếu không tìm được nước tốt
        """
        best_move = None
        best_score = float('-inf')

        # Các nước đi Pacman có thể chọn
        pac_candidates = []
        for move in FOUR_DIRECTIONS:
            # Pacman đi 1 bước hoặc nhiều bước (tùy speed)
            for steps in range(1, self.pacman_speed + 1):
                new_pac = my_position
                valid = True
                for _ in range(steps):
                    nxt = apply_move(new_pac, move)
                    if not is_valid_position(self.global_map, nxt):
                        valid = False
                        break
                    new_pac = nxt
                if valid and new_pac != my_position:
                    pac_candidates.append((move, steps, new_pac))

        if not pac_candidates:
            return None

        # Các nước đi Ghost có thể chọn
        ghost_moves = []
        for gm in FOUR_DIRECTIONS:
            gnxt = apply_move(ghost_position, gm)
            if is_valid_position(self.global_map, gnxt):
                ghost_moves.append(gnxt)
        ghost_moves.append(ghost_position)  # STAY

        for move, steps, new_pac in pac_candidates:
            capture_zone = self._compute_capture_zone(new_pac)

            # 2-ply: xét Ghost phản ứng tốt nhất (worst case cho Pacman)
            worst_score = float('inf')

            for ghost_new in ghost_moves:
                if ghost_new in capture_zone:
                    ghost_outcome = 10000.0
                else:
                    next_pac_dist = self._bfs_dist(new_pac, ghost_new)
                    turns_to_catch = max(0, next_pac_dist - 1) / max(1, self.pacman_speed)
                    ghost_outcome = 500.0 / (turns_to_catch + 1)

                    # Bonus: Ghost bị ít đường thoát
                    ghost_exits = sum(
                        1 for m2 in FOUR_DIRECTIONS
                        if is_valid_position(self.global_map,
                                             apply_move(ghost_new, m2))
                    )
                    if ghost_exits <= 1:
                        ghost_outcome += 200.0  # dead-end
                    elif ghost_exits == 2:
                        ghost_outcome += 50.0   # corridor

                # Ghost chọn nước tối thiểu hóa score Pacman
                if ghost_outcome < worst_score:
                    worst_score = ghost_outcome

            score = worst_score if worst_score != float('inf') else 0
            score += steps * 0.01  # Tie-breaker ưu tiên đi bước dài nhất

            if score > best_score:
                best_score = score
                best_move = (move, steps)

        return best_move

    # =================================================================
    # MAP MEMORY (Fog-of-War)
    # =================================================================

    def _update_map_memory(self, map_state):
        """
        Gộp thông tin từ map_state hiện tại vào global_map.

        Tại sao cần? Trong fog-of-war, mỗi lượt chỉ thấy 1 phần map.
        Global map lưu lại TẤT CẢ ô đã từng thấy, giúp A* tìm đường
        qua cả vùng đã khám phá trước đó.
        """
        # Chỉ cập nhật ô KHÔNG PHẢI unseen (-1)
        visible_mask = map_state != -1
        self.global_map[visible_mask] = map_state[visible_mask]

    # =================================================================
    # BELIEF TRACKING
    # =================================================================

    def _update_belief(self, map_state, my_position, enemy_position):
        """
        Cập nhật xác suất vị trí Ghost theo chu trình:
        PREDICT → UPDATE → NORMALIZE

        - PREDICT: Ghost có thể di chuyển 1 bước → phân tán xác suất
        - UPDATE: Loại bỏ xác suất ở ô visible mà không thấy Ghost
        - NORMALIZE: Đảm bảo tổng xác suất = 1.0

        Tại sao? Trong fog-of-war, khi mất dấu Ghost, ta cần ước lượng
        Ghost đang ở đâu để chọn hướng đi hợp lý thay vì đi mò.
        """
        h, w = map_state.shape

        if enemy_position is not None:
            # === THẤY GHOST: belief = 100% tại vị trí Ghost ===
            self.belief[:] = 0.0
            self.belief[enemy_position[0], enemy_position[1]] = 1.0
            self.last_known_enemy_pos = enemy_position
            self.enemy_history.append(enemy_position)
            self.steps_since_seen = 0
            self.pursuit_mode = True
            return

        # === MẤT DẤU GHOST ===
        self.steps_since_seen += 1

        # --- PREDICT: Ghost di chuyển 1 bước ---
        # Mỗi ô có xác suất p → phân phối p/(số_neighbors+1) cho mỗi neighbor + chính nó
        new_belief = np.zeros_like(self.belief)
        for r in range(h):
            for c in range(w):
                if self.belief[r, c] < 1e-9:
                    continue  # Bỏ qua ô xác suất ≈ 0
                p = self.belief[r, c]
                neighbors = []
                for move in FOUR_DIRECTIONS:
                    nr, nc = r + move.value[0], c + move.value[1]
                    if 0 <= nr < h and 0 <= nc < w and self.global_map[nr, nc] != 1:
                        neighbors.append((nr, nc))
                # Ghost có thể STAY hoặc đi sang neighbor
                total = len(neighbors) + 1
                new_belief[r, c] += p / total  # STAY
                for nr, nc in neighbors:
                    new_belief[nr, nc] += p / total

        # --- UPDATE: Xóa xác suất ở ô visible mà không thấy Ghost ---
        # Nếu ta nhìn thấy 1 ô trống (map_state == 0) mà không có Ghost,
        # thì Ghost chắc chắn KHÔNG ở đó
        visible_empty = (map_state == 0)
        new_belief[visible_empty] = 0.0

        # Ghost không thể ở trên tường
        walls = (self.global_map == 1)
        new_belief[walls] = 0.0

        # --- NORMALIZE ---
        total = new_belief.sum()
        if total > 1e-9:
            self.belief = new_belief / total
        else:
            # Reset nếu mọi xác suất về 0 (rare case)
            self.belief = np.ones((h, w), dtype=float)
            self.belief[self.global_map == 1] = 0.0
            visible_empty = (map_state == 0)
            self.belief[visible_empty] = 0.0
            total = self.belief.sum()
            if total > 0:
                self.belief /= total

    def _get_belief_target(self):
        """
        Trả về ô có xác suất Ghost cao nhất.
        Đây là "best guess" khi không thấy Ghost.
        """
        if self.belief is None:
            return self.last_known_enemy_pos
        idx = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        return (int(idx[0]), int(idx[1]))

    # =================================================================
    # MODE SWITCHING & TARGET SELECTION
    # =================================================================

    def _determine_target(self, my_position, enemy_position, step_number):
        """
        Xác định target dựa trên mode hiện tại (dùng BFS distance thay Manhattan):

        1. PURSUIT (thấy Ghost): đuổi/chặn đầu Ghost
        2. PERSISTENT (vừa mất dấu < 8 turns): dùng belief target
        3. PATROL (mất dấu lâu): tuần tra điểm chiến lược

        Returns: (target_position, mode_name)
        """
        if enemy_position is not None:
            # === PURSUIT MODE: Thấy Ghost ===
            bfs_dist = self._bfs_dist(my_position, enemy_position)

            if bfs_dist <= 4:
                # Gần → đuổi thẳng, không cần chặn (capture zone đã xử lý)
                return enemy_position, "pursuit_direct"
            else:
                # Xa → thử chặn đầu bằng BFS intercept
                intercept = self._find_intercept_point(
                    my_position, enemy_position
                )
                if intercept is not None:
                    return intercept, "pursuit_intercept"
                return enemy_position, "pursuit_direct"

        elif (self.pursuit_mode and
              self.steps_since_seen <= self.PURSUIT_PERSISTENCE and
              self.last_known_enemy_pos is not None):
            # === PERSISTENT MODE: Vừa mất dấu ===
            belief_target = self._get_belief_target()
            if belief_target is not None:
                # Dùng BFS distance thay Manhattan để check đã đến gần
                bfs_d = self._bfs_dist(my_position, belief_target)
                if bfs_d <= 1:
                    self.pursuit_mode = False
                    return self._get_patrol_target(my_position), "patrol"
                return belief_target, "persistent"
            return self.last_known_enemy_pos, "persistent"

        else:
            # === PATROL MODE: Mất dấu lâu ===
            self.pursuit_mode = False
            return self._get_patrol_target(my_position), "patrol"

    def _get_patrol_target(self, my_position):
        """
        Trả về điểm tuần tra tiếp theo.
        Khi đến gần target hiện tại → chuyển sang target kế.

        Tại sao tuần tra? Thay vì đứng yên hoặc đi random,
        tuần tra giúp bao phủ map nhanh hơn và có cơ hội
        phát hiện Ghost sớm.
        """
        target = self.PATROL_TARGETS[self.patrol_index]

        # Dùng BFS distance để check đã đến gần (thay Manhattan)
        bfs_d = self._bfs_dist(my_position, target)
        if bfs_d <= 2:
            self.patrol_index = (self.patrol_index + 1) % len(self.PATROL_TARGETS)
            target = self.PATROL_TARGETS[self.patrol_index]

        # Nếu patrol target là tường → skip
        if self.global_map is not None and self.global_map[target[0], target[1]] == 1:
            self.patrol_index = (self.patrol_index + 1) % len(self.PATROL_TARGETS)
            target = self.PATROL_TARGETS[self.patrol_index]

        return target

    # =================================================================
    # INTERCEPTION - Chặn đầu Ghost (BFS-based)
    # =================================================================

    def _find_intercept_point(self, my_position, ghost_position):
        """
        Dự đoán Ghost sẽ đi đâu dựa trên velocity, rồi tìm điểm chặn.

        CẢI TIẾN so với bản cũ:
        - Dùng BFS distance thay Manhattan để so sánh thời gian đến
        - Chính xác hơn trên map có nhiều tường

        Ý tưởng:
        - Tính velocity Ghost = pos hiện tại - pos trước đó
        - Predict Ghost sẽ tiếp tục đi hướng đó 2-3 bước
        - Nếu Pacman có thể đến điểm predict TRƯỚC Ghost → chặn đầu
        - So sánh dùng BFS distance / speed thay vì Manhattan

        Tại sao? Ghost thường chạy thẳng ra xa Pacman. Nếu ta đuổi thẳng,
        Ghost luôn giữ khoảng cách. Chặn đầu giúp "cắt đường" Ghost.
        """
        if len(self.enemy_history) < 2:
            return None  # Chưa đủ data để tính velocity

        # Tính velocity (hướng + tốc độ di chuyển của Ghost)
        prev = self.enemy_history[-2]
        curr = self.enemy_history[-1]
        dr = curr[0] - prev[0]
        dc = curr[1] - prev[1]

        if dr == 0 and dc == 0:
            return None  # Ghost đứng yên, không predict được

        # Predict vị trí Ghost sau 2-3 bước
        h, w = self.global_map.shape
        best_intercept = None
        best_score = float('inf')

        for look_ahead in range(1, 4):  # Predict 1-3 bước
            pred_r = curr[0] + dr * look_ahead
            pred_c = curr[1] + dc * look_ahead

            # Clamp vào biên map
            pred_r = max(0, min(h - 1, pred_r))
            pred_c = max(0, min(w - 1, pred_c))
            pred_pos = (pred_r, pred_c)

            # Bỏ qua nếu predict vào tường
            if self.global_map[pred_r, pred_c] == 1:
                # Thử tìm ô walkable gần nhất
                pred_pos = self._nearest_walkable(pred_pos)
                if pred_pos is None:
                    continue

            # ★ THAY ĐỔI: Dùng BFS distance thay Manhattan ★
            pac_bfs = self._bfs_dist(my_position, pred_pos)
            ghost_bfs = self._bfs_dist(ghost_position, pred_pos)

            # Pacman đi nhanh hơn nhờ speed
            pac_time = pac_bfs / max(1, self.pacman_speed)

            if pac_time <= ghost_bfs + 1:
                # Pacman đến trước hoặc cùng lúc → intercept tốt!
                choke = self._choke_score(pred_pos)
                score = pac_bfs - choke * 2  # Ưu tiên choke point
                if score < best_score:
                    best_score = score
                    best_intercept = pred_pos

        return best_intercept

    def _nearest_walkable(self, pos):
        """BFS tìm ô walkable gần nhất từ pos (dùng khi predict trúng tường)."""
        if self.global_map is None:
            return None
        h, w = self.global_map.shape
        if (0 <= pos[0] < h and 0 <= pos[1] < w
                and self.global_map[pos[0], pos[1]] != 1):
            return pos
        queue = deque([pos])
        visited = {pos}
        while queue:
            cur = queue.popleft()
            if (0 <= cur[0] < h and 0 <= cur[1] < w
                    and self.global_map[cur[0], cur[1]] != 1):
                return cur
            for m in FOUR_DIRECTIONS:
                nxt = apply_move(cur, m)
                if (0 <= nxt[0] < h and 0 <= nxt[1] < w
                        and nxt not in visited):
                    visited.add(nxt)
                    queue.append(nxt)
        return None

    def _choke_score(self, pos: tuple) -> int:
        """Đánh giá mức độ 'choke point': exits càng ít → càng dễ chặn."""
        exits = sum(
            1 for mv in FOUR_DIRECTIONS
            if is_valid_position(self.global_map, apply_move(pos, mv))
        )
        if exits == 1:
            return 8    # dead-end
        if exits == 2:
            return 4    # corridor
        if exits == 3:
            return 2    # junction
        return 1

    # =================================================================
    # A* SEARCH
    # =================================================================

    def _astar_path(self, start, goal):
        """
        A* search tìm đường ngắn nhất từ start đến goal.

        Khác BFS ở chỗ: dùng heuristic (Manhattan distance) để ưu tiên
        khám phá các ô GẦN goal trước → nhanh hơn đáng kể.

        f(n) = g(n) + h(n)
        - g(n): chi phí thực tế từ start đến n (số bước đã đi)
        - h(n): ước lượng chi phí từ n đến goal (Manhattan distance)
        - f(n): tổng chi phí ước lượng, ô có f nhỏ nhất được xét trước

        Trả về: list vị trí [start, pos1, pos2, ..., goal]
        Nếu không tìm được đường → trả về []

        Safety: giới hạn 3000 nodes để tránh timeout.

        LƯU Ý: heuristic vẫn dùng Manhattan (admissible, không overestimate)
        để A* đảm bảo tìm đường ngắn nhất. BFS distance chỉ dùng trong
        quyết định chiến thuật (intercept, mode switching), không phải A*.
        """
        if start == goal:
            return [start]

        # map dùng cho pathfinding: ưu tiên global_map (đầy đủ hơn)
        search_map = self.global_map

        # Priority queue: (f_cost, counter, position)
        # counter để break tie khi f bằng nhau
        counter = 0
        pq = [(manhattan_distance(start, goal), counter, start)]
        counter += 1

        came_from = {start: None}  # Truy vết đường đi
        g_cost = {start: 0}  # Chi phí thực tế

        max_nodes = 3000  # Safety limit
        nodes_explored = 0

        while pq and nodes_explored < max_nodes:
            f, _, current = heapq.heappop(pq)
            nodes_explored += 1

            if current == goal:
                # Reconstruct path
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path

            # Bỏ qua nếu đã tìm được đường tốt hơn đến current
            if g_cost.get(current, float('inf')) < f - manhattan_distance(current, goal):
                continue

            for nxt, _ in get_neighbors(search_map, current, allow_unseen=True):
                new_g = g_cost[current] + 1

                if nxt not in g_cost or new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    h = manhattan_distance(nxt, goal)
                    f_new = new_g + h
                    came_from[nxt] = current
                    heapq.heappush(pq, (f_new, counter, nxt))
                    counter += 1

        return []  # Không tìm được đường

    # =================================================================
    # SPEED OPTIMIZATION
    # =================================================================

    def _on_same_axis(self, pos1, pos2):
        """Kiểm tra 2 vị trí cùng hàng hoặc cùng cột."""
        return pos1[0] == pos2[0] or pos1[1] == pos2[1]

    def _straight_line_move(self, my_pos, target):
        """
        Nếu target cùng hàng/cột VÀ đường thẳng không bị tường chắn,
        đi thẳng với max speed.

        Tại sao? Pacman có speed >= 1, nếu đi thẳng không rẽ,
        có thể đi nhiều ô trong 1 lượt → bắt Ghost nhanh hơn.
        """
        dr = target[0] - my_pos[0]
        dc = target[1] - my_pos[1]

        if dr == 0 and dc == 0:
            return None

        # Xác định hướng
        if dr == 0:
            move = Move.RIGHT if dc > 0 else Move.LEFT
            dist = abs(dc)
        elif dc == 0:
            move = Move.DOWN if dr > 0 else Move.UP
            dist = abs(dr)
        else:
            return None  # Không cùng trục

        # Kiểm tra đường thẳng có bị chắn không
        steps = self._validate_steps(my_pos, move, min(dist, self.pacman_speed))
        if steps > 0:
            return (move, steps)
        return None

    def _calc_straight_steps(self, path, first_move):
        """
        Tính số bước THẲNG liên tiếp theo path.
        Nếu path rẽ hướng ở bước 2 → chỉ đi 1 bước.
        Tránh đi lố qua ngã rẽ.
        """
        steps = 0
        for i in range(1, min(len(path), self.pacman_speed + 1)):
            m = move_from_to(path[i - 1], path[i])
            if m == first_move:
                steps += 1
            else:
                break
        return max(1, steps)

    def _validate_steps(self, pos, move, desired_steps):
        """
        Kiểm tra Pacman có thể đi bao nhiêu bước thẳng mà không đụng tường.
        Trả về số bước thực tế hợp lệ (0 nếu không đi được).
        """
        steps = 0
        current = pos
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        for _ in range(max_steps):
            nxt = apply_move(current, move)
            if not is_valid_position(self.global_map, nxt):
                break
            steps += 1
            current = nxt
        return steps

    # =================================================================
    # ANTI-LOOP
    # =================================================================

    def _is_looping(self, my_position):
        """
        Phát hiện Pacman bị kẹt vòng lặp.
        Nếu vị trí hiện tại xuất hiện >= 4 lần trong 12 bước gần nhất
        → đang bị loop.
        """
        return self.my_history.count(my_position) >= 4

    def _escape_loop(self, my_position):
        """
        Phá vòng lặp bằng cách đi random theo hướng CHƯA đi gần đây.
        """
        self.my_history.clear()  # Reset history
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)

        # Ưu tiên ô chưa đến gần đây
        for m in moves:
            nxt = apply_move(my_position, m)
            if is_valid_position(self.global_map, nxt):
                return (m, 1)
        return (Move.STAY, 1)

    # =================================================================
    # FRONTIER EXPLORATION (Fog-of-War)
    # =================================================================

    def _frontier_explore(self, my_position):
        """
        Khi không có target, tìm biên giữa vùng đã khám phá và chưa khám phá.
        BFS từ vị trí hiện tại, tìm ô empty có ô -1 (unseen) liền kề.

        Tại sao? Trong fog-of-war, phần map chưa khám phá có thể chứa Ghost.
        Đi đến frontier giúp mở rộng tầm nhìn → tăng cơ hội phát hiện Ghost.
        """
        if self.global_map is None:
            return self._any_valid_move(my_position)

        queue = deque([(my_position, None)])  # (pos, first_move)
        visited = {my_position}

        while queue:
            cur, first_move = queue.popleft()

            # Kiểm tra cur có phải frontier không
            if self._is_frontier(cur):
                if first_move is not None:
                    steps = self._validate_steps(my_position, first_move, self.pacman_speed)
                    if steps > 0:
                        return (first_move, steps)

            for nxt, m in get_neighbors(self.global_map, cur, allow_unseen=False):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, first_move or m))

        # Không tìm thấy frontier → đi random
        return self._any_valid_move(my_position)

    def _is_frontier(self, pos):
        """
        Ô frontier = ô đã biết (empty) có ít nhất 1 ô liền kề chưa biết (-1).
        """
        if self.global_map[pos[0], pos[1]] != 0:
            return False
        h, w = self.global_map.shape
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < h and 0 <= nc < w and self.global_map[nr, nc] == -1:
                return True
        return False

    # =================================================================
    # FALLBACK MOVES
    # =================================================================

    def _greedy_toward(self, my_position, target):
        """
        Fallback: chọn nước đi giảm BFS distance đến target nhiều nhất.
        Dùng khi A* không tìm được đường (do map chưa khám phá đủ).
        """
        best_move = Move.STAY
        best_dist = self._bfs_dist(my_position, target)

        for m in FOUR_DIRECTIONS:
            nxt = apply_move(my_position, m)
            if is_valid_position(self.global_map, nxt):
                d = self._bfs_dist(nxt, target)
                if d < best_dist:
                    best_dist = d
                    best_move = m
        return best_move

    def _any_valid_move(self, my_position):
        """Absolute fallback: đi bất kỳ hướng hợp lệ nào."""
        for m in FOUR_DIRECTIONS:
            nxt = apply_move(my_position, m)
            if is_valid_position(self.global_map, nxt):
                steps = self._validate_steps(my_position, m, self.pacman_speed)
                if steps > 0:
                    return (m, steps)
        return (Move.STAY, 1)


# ============================================================
# ================ GHOST AGENT (HIDE) ========================
# ============================================================
# (Người T3 - Advanced Ghost: Multi-Layer Danger + Alpha-Beta Minimax
#  + Monte Carlo Nâng Cao + Voronoi + Predictive Pacman Modeling)

class GhostAgent(BaseGhostAgent):
    """
    Agent Ghost (Hide) - Trốn tránh Pacman, sống càng lâu càng tốt.

    Chiến lược Nâng cao (8 cải tiến):
        1. Multi-Layer Danger Map: 3 tầng nguy hiểm (1/2/3 turns)
           → Ghost nhìn xa hơn, tránh bị dồn ép trung hạn
        2. Monte Carlo Rollout: 8 rollouts, depth 12 để tránh timeout.
           → Pacman simulation chính xác hơn (không bị stuck ở tường)
        3. Alpha-Beta Minimax: depth 3 khi Pacman gần, depth 2 khi khoảng cách trung bình.
           → Tìm sâu hơn mà vẫn nhanh, Pacman chỉ đi thẳng (đúng luật)
        4. Evaluate cải tiến: corridor + voronoi territory + center attraction
           → Tránh hành lang hẹp, ưu tiên vùng Ghost kiểm soát
        5. Fog-of-War aware: ưu tiên ô ít thăm khi không thấy Pacman
        6. Smart Anti-Loop: game-wide frequency + direction momentum
           → Không chạy lòng vòng cùng khu vực, không oscillate
        7. Soft Filter: penalty thay vì skip → không bao giờ bị kẹt STAY
           → Khi tất cả ô đều nguy hiểm, vẫn chọn ô ít nguy hiểm nhất
        8. Predictive Pacman Modeling: dự đoán hướng đi Pacman kế tiếp
           → Tránh chạy vào hướng Pacman SẼ đi
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Nhom3 Advanced Ghost"
        # --- Core params ---
        self.last_known_enemy_pos: Optional[Tuple[int, int]] = None
        self.step_count = 0
        self.enemy_speed = max(1, int(kwargs.get('pacman_speed', 2)))

        # --- Global Map Memory ---
        self.global_map: Optional[np.ndarray] = None

        # --- Caching System ---
        self._dist_cache: Dict[tuple, Dict[Tuple[int, int], int]] = {}
        self._dist_cache_step = -1

        # --- Anti-loop: Game-wide frequency + recent history (Cải tiến 6) ---
        self._pos_history: deque = deque(maxlen=20)   # Lịch sử gần đây (mở rộng)
        self._pos_frequency: Dict[tuple, int] = {}    # Đếm số lần thăm cả game
        self._last_move: Optional[Move] = None
        self._direction_history: deque = deque(maxlen=8)  # Track đổi hướng

        # --- Pacman Tracking: Predictive Modeling (Cải tiến 8) ---
        self._enemy_history: deque = deque(maxlen=10)
        self._enemy_velocity: Tuple[int, int] = (0, 0)

        # --- Topology State ---
        self._last_unseen_count = -1
        self.valid = set()
        self.edges = {}
        self.deg = {}
        self.dead_ends = set()
        self.core = set()
        self.entry = {}
        self.depth_de = {}
        self.junctions = set()
        self.apsp = {}


    # ══════════════════════════════════════════════════════════════════
    #  ENTRY POINT
    # ══════════════════════════════════════════════════════════════════

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:
        self.step_count = step_number
        step_start_time = time.perf_counter()

        # ── Cập nhật Global Map ──
        if self.global_map is None:
            self.global_map = np.copy(map_state)
        visible = (map_state != -1)
        self.global_map[visible] = map_state[visible]

        # Update topology if new cells are discovered
        unseen_count = np.sum(map_state == -1)
        if self._last_unseen_count == -1:
            self._build_topology()
            self._last_unseen_count = unseen_count

        # ── Cache management ──
        if step_number != self._dist_cache_step:
            self._dist_cache.clear()
            self._dist_cache_step = step_number

        # ── History & frequency tracking (Cải tiến 6) ──
        self._pos_history.append(my_position)
        self._pos_frequency[my_position] = \
            self._pos_frequency.get(my_position, 0) + 1

        # ── Pacman tracking & velocity prediction (Cải tiến 8) ──
        if enemy_position is not None:
            self._enemy_history.append(enemy_position)
            self.last_known_enemy_pos = enemy_position
            if len(self._enemy_history) >= 2:
                prev = self._enemy_history[-2]
                curr = self._enemy_history[-1]
                self._enemy_velocity = (curr[0] - prev[0],
                                        curr[1] - prev[1])

        threat = enemy_position or self.last_known_enemy_pos
        if threat is None:
            return self._safe_random_move(my_position)

        # ── BFS distance map từ Pacman ──
        pacman_dist_map = self._get_bfs_distance(threat)
        current_dist = pacman_dist_map.get(my_position, 999)

        # ══════════════════════════════════════════════════════════════
        #  MULTI-LAYER DANGER ZONES (Cải tiến 1)
        #  3 tầng nguy hiểm thay vì 1: Ghost nhìn xa hơn
        # ══════════════════════════════════════════════════════════════
        immediate_danger, near_danger, far_danger = \
            self._compute_danger_layers(pacman_dist_map)

        # ══════════════════════════════════════════════════════════════
        #  PREDICTIVE PACMAN POSITION (Cải tiến 8)
        #  Dự đoán Pacman sẽ ở đâu turn kế → tránh chạy về đó
        # ══════════════════════════════════════════════════════════════
        predicted_pac = self._predict_pacman_position(threat, my_position)
        pred_pac_dist = None
        if predicted_pac != threat:
            pred_pac_dist = self._get_bfs_distance(predicted_pac)

        # ══════════════════════════════════════════════════════════════
        #  SMART SAFE TARGET
        # ══════════════════════════════════════════════════════════════
        safe_target = self._find_best_safe_target(
            my_position, threat, pacman_dist_map, immediate_danger
        )
        safe_dist_map = self._get_bfs_distance(safe_target)

        # ── Duyệt tất cả candidates ──
        safe_cands = []
        unknown_cands = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            nxt = apply_move(my_position, move) \
                if move != Move.STAY else my_position
            if is_walkable(self.global_map, nxt):
                safe_cands.append((move, nxt))
            elif is_valid_position(self.global_map, nxt):
                unknown_cands.append((move, nxt))
        candidates = safe_cands if safe_cands else unknown_cands

        if not candidates:
            return Move.STAY

        safe_candidates = []
        risky_candidates = []
        for move, new_pos in candidates:
            dist_to_pacman = pacman_dist_map.get(new_pos, 999)
            if dist_to_pacman > self.enemy_speed + 1 and new_pos not in immediate_danger:
                safe_candidates.append((move, new_pos))
            else:
                risky_candidates.append((move, new_pos))

        if safe_candidates:
            candidates = safe_candidates
        else:
            candidates = risky_candidates

        best_move = Move.STAY
        best_score = float('-inf')

        for move, new_pos in candidates:
            dist_to_pacman = pacman_dist_map.get(new_pos, 999)

            # ══════════════════════════════════════════════════════════
            #  SOFT FILTER (Cải tiến 7)
            #  Penalty nặng thay vì skip → Ghost không bao giờ bị kẹt
            # ══════════════════════════════════════════════════════════
            proximity_penalty = 0.0
            if dist_to_pacman <= 1:
                proximity_penalty = 200.0   # Gần như chắc chết
            elif dist_to_pacman == 2:
                proximity_penalty = 40.0    # Rất nguy hiểm

            # ══════════════════════════════════════════════════════════
            #  MULTI-LAYER DANGER PENALTY (Cải tiến 1)
            # ══════════════════════════════════════════════════════════
            danger_penalty = 0.0
            if new_pos in immediate_danger:
                danger_penalty = 300.0      # 1 turn → cực kỳ nguy hiểm
            elif new_pos in near_danger:
                danger_penalty = 40.0       # 2 turns → cảnh giác cao
            elif new_pos in far_danger:
                danger_penalty = 8.0        # 3 turns → cảnh giác

            # ══════════════════════════════════════════════════════════
            #  PREDICTIVE DANGER (Cải tiến 8)
            #  Phạt thêm nếu ô gần vị trí Pacman DỰ ĐOÁN
            # ══════════════════════════════════════════════════════════
            pred_penalty = 0.0
            if pred_pac_dist is not None:
                pred_d = pred_pac_dist.get(new_pos, 999)
                if pred_d <= self.enemy_speed:
                    pred_penalty = 15.0
                elif pred_d <= self.enemy_speed * 2:
                    pred_penalty = 5.0

            # ── Escape analysis ──
            escape_routes = self._escape_routes(new_pos)
            dead_pen = self._dead_end_penalty(new_pos)

            # ── Corridor penalty (Cải tiến 4) ──
            corridor_pen = 5.0 if self._is_corridor(new_pos) else 0.0

            # ══════════════════════════════════════════════════════════
            #  ALPHA-BETA MINIMAX (Cải tiến 3)
            #  Depth 3 khi gần, 2 khi trung bình, evaluate khi xa
            # ══════════════════════════════════════════════════════════
            if dist_to_pacman <= 4:
                mm_score = self._minimax_ab(
                    new_pos, threat, 3, False,
                    float('-inf'), float('inf'))
            elif dist_to_pacman <= 7:
                mm_score = self._minimax_ab(
                    new_pos, threat, 2, False,
                    float('-inf'), float('inf'))
            else:
                mm_score = self._evaluate_state(new_pos, threat)

            # ── Hướng về safe target ──
            dist_to_safe = safe_dist_map.get(new_pos, 999)

            # ══════════════════════════════════════════════════════════
            #  SCORING TỔNG HỢP
            # ══════════════════════════════════════════════════════════
            score = mm_score
            score -= proximity_penalty          # Soft close penalty
            score -= danger_penalty             # Multi-layer danger
            score -= pred_penalty               # Predictive danger
            score += escape_routes * 4.0        # ƯU TIÊN nhiều đường thoát
            score -= dead_pen * 3.0             # Phạt ngõ cụt nặng
            score -= corridor_pen               # Phạt hành lang hẹp
            score -= 0.3 * dist_to_safe         # Hướng về safe target

            # ── Trap detection ──
            safe_space = self._safe_space_count(new_pos, pacman_dist_map)
            score += safe_space * 1.5
            if safe_space <= 3:
                score -= 60.0

            # ── Bonus/penalty khoảng cách ──
            if dist_to_pacman > current_dist:
                score += 5.0                    # Tăng khoảng cách = tốt
            elif dist_to_pacman < current_dist:
                score -= 3.0                    # Giảm khoảng cách = xấu

            # ══════════════════════════════════════════════════════════
            #  SMART ANTI-LOOP (Cải tiến 6)
            # ══════════════════════════════════════════════════════════
            # Phạt theo game-wide frequency
            freq = self._pos_frequency.get(new_pos, 0)
            if freq > 3:
                score -= min(freq * 1.5, 15.0)

            # Phạt theo recent history (mạnh hơn)
            recent_visits = sum(1 for p in self._pos_history
                                if p == new_pos)
            if recent_visits >= 2:
                score -= 6.0 * recent_visits

            # Direction momentum: phạt quay đầu (oscillation)
            if self._last_move is not None and move != Move.STAY:
                if self._is_opposite_move(move, self._last_move):
                    score -= 4.0

            # ── Phạt STAY ──
            if move == Move.STAY:
                score -= 8.0

            # ── Phạt biên map ──
            h, w = self.global_map.shape
            if new_pos[0] <= 1 or new_pos[0] >= h - 2:
                score -= 3.0
            if new_pos[1] <= 1 or new_pos[1] >= w - 2:
                score -= 3.0

            # ── Center attraction (Cải tiến 4) ──
            center_r, center_c = h // 2, w // 2
            dist_center = manhattan_distance(new_pos,
                                             (center_r, center_c))
            score += max(0, ((h + w) // 2 - dist_center)) * 0.15

            if score > best_score:
                best_score = score
                best_move = move

        # ══════════════════════════════════════════════════════════════
        #  MONTE CARLO REFINEMENT (Cải tiến 2)
        #  Chạy rộng hơn (threshold 12) và sâu hơn (30 depth)
        # ══════════════════════════════════════════════════════════════
        elapsed = time.perf_counter() - step_start_time
        time_budget = 0.8

        if elapsed < time_budget * 0.25:
            if current_dist > 8 and current_dist <= 12:
                mc_best = self._monte_carlo_select(
                    my_position, threat, candidates,
                    immediate_danger, pacman_dist_map,
                    step_start_time, time_budget
                )
                if mc_best is not None:
                    best_move = mc_best

        # ── Timeout guard ──
        if time.perf_counter() - step_start_time > 0.95:
            self._last_move = self._flee_fallback(
                my_position, threat, pacman_dist_map)
            return self._last_move

        # ── Absolute fallback ──
        if best_score == float('-inf'):
            best_move = self._flee_fallback(
                my_position, threat, pacman_dist_map)

        self._last_move = best_move
        self._direction_history.append(best_move)
        return best_move

    # ══════════════════════════════════════════════════════════════════
    #  MULTI-LAYER DANGER MAP (Cải tiến 1)
    # ══════════════════════════════════════════════════════════════════

    def _compute_danger_layers(self, pacman_dist_map: dict) \
            -> Tuple[Set[tuple], Set[tuple], Set[tuple]]:
        """
        Tính 3 tầng danger zone từ BFS distance map của Pacman.

        Dùng BFS distance đã tính sẵn → O(1) thêm (chỉ phân loại).

        Tầng 1 (immediate): BFS dist ≤ speed → Pacman bắt trong 1 turn
        Tầng 2 (near):      BFS dist ≤ 2×speed → 2 turns
        Tầng 3 (far):       BFS dist ≤ 3×speed → 3 turns

        Tại sao 3 tầng? Ghost cũ chỉ tránh immediate danger, nên dễ bị
        Pacman "dồn ép" bằng cách ép Ghost vào vùng hẹp qua 2-3 turns.
        Với 3 tầng, Ghost thấy trước và tránh sớm hơn.
        """
        sp = self.enemy_speed
        immediate = set()
        near = set()
        far = set()
        for pos, d in pacman_dist_map.items():
            if d <= sp + 1:
                immediate.add(pos)
            elif d <= sp * 2 + 1:
                near.add(pos)
            elif d <= sp * 3 + 1:
                far.add(pos)
        return immediate, near, far

    # ══════════════════════════════════════════════════════════════════
    #  PREDICTIVE PACMAN MODELING (Cải tiến 8)
    # ══════════════════════════════════════════════════════════════════

    def _predict_pacman_position(self, current_threat: tuple,
                                  ghost_pos: tuple) -> tuple:
        """
        Dự đoán vị trí Pacman ở turn tiếp theo.

        Giả sử Pacman chơi BFS-greedy: chọn hướng giảm BFS distance
        đến Ghost nhiều nhất, rồi đi thẳng tối đa enemy_speed bước.

        Tại sao? Ghost cần tránh không chỉ vị trí hiện tại của Pacman,
        mà cả vị trí Pacman SẼ ở. Nếu Ghost chạy về hướng Pacman đang
        tiến tới, Ghost tự đâm vào tay Pacman.
        """
        if not is_valid_position(self.global_map, current_threat):
            return current_threat

        # BFS từ ghost_pos: dist[pos] = khoảng cách từ ghost đến pos
        # BFS đối xứng trên unweighted graph → = khoảng cách từ pos đến ghost
        ghost_dist_map = self._get_bfs_distance(ghost_pos)

        # Tìm hướng Pacman nên đi (minimize distance to Ghost)
        best_dir = None
        best_d = ghost_dist_map.get(current_threat, 999)
        for mv in FOUR_DIRECTIONS:
            nxt = apply_move(current_threat, mv)
            if is_valid_position(self.global_map, nxt):
                d = ghost_dist_map.get(nxt, 999)
                if d < best_d:
                    best_d = d
                    best_dir = mv

        if best_dir is None:
            return current_threat

        # Pacman đi thẳng hướng đó tối đa enemy_speed bước
        pos = current_threat
        for _ in range(self.enemy_speed):
            nxt = apply_move(pos, best_dir)
            if is_valid_position(self.global_map, nxt):
                pos = nxt
            else:
                break

        return pos

    # ══════════════════════════════════════════════════════════════════
    #  SAFE TARGET SELECTION
    # ══════════════════════════════════════════════════════════════════

    def _find_best_safe_target(self, my_position: tuple, threat: tuple,
                                pacman_dist_map: dict,
                                immediate_danger: Set[tuple]) -> tuple:
        """
        Tìm safe target: vị trí xa Pacman, thoáng, Ghost đến trước Pacman.

        Cải tiến: thêm corridor penalty, giới hạn search radius để nhanh hơn.
        """
        ghost_dist_map = self._get_bfs_distance(my_position)
        h, w = self.global_map.shape

        best_target = my_position
        best_score = float('-inf')

        for pos, g_dist in ghost_dist_map.items():
            if g_dist > 15:
                continue  # Không xét ô quá xa Ghost

            p_dist = pacman_dist_map.get(pos, 999)
            pac_turns_to_catch = max(0, p_dist - 1) / max(1, self.enemy_speed)

            # Ghost phải đến TRƯỚC Pacman
            if pac_turns_to_catch <= g_dist:
                continue

            # Không chọn ô trong immediate danger
            if pos in immediate_danger:
                continue

            # Vùng phải đủ thoáng
            area = self._escape_area(pos, radius=3)
            if area < 6:
                continue

            # Tránh biên map
            border_pen = 0
            if pos[0] <= 1 or pos[0] >= h - 2:
                border_pen += 3
            if pos[1] <= 1 or pos[1] >= w - 2:
                border_pen += 3

            # Corridor penalty
            corr_pen = 4 if self._is_corridor(pos) else 0

            score = (p_dist * 2.0
                     + area * 0.5
                     - g_dist * 0.3
                     - border_pen
                     - corr_pen)

            if score > best_score:
                best_score = score
                best_target = pos

        return best_target

    # ══════════════════════════════════════════════════════════════════
    #  MONTE CARLO ROLLOUT NÂNG CAO (Cải tiến 2)
    # ══════════════════════════════════════════════════════════════════

    def _monte_carlo_select(self, ghost_pos: tuple, pacman_pos: tuple,
                            candidates: list, immediate_danger: Set[tuple],
                            pacman_dist_map: dict,
                            start_time: float,
                            time_budget: float) -> Optional[Move]:
        """
        Monte Carlo nâng cao:
        - 8 rollouts (trước là 25) và depth 12 → thống kê ổn định, tránh timeout.
        - Depth 12 → đủ dự đoán ngắn hạn, tránh timeout
        - Ghost simulate: corridor-aware biased flee
        - Pacman simulate: wall-aware greedy (xử lý stuck)
        - Soft candidate filter (không loại bỏ hoàn toàn)
        """
        NUM_ROLLOUTS = 8
        MAX_DEPTH = 12

        # Lọc candidates (soft: ưu tiên safe, nhưng giữ risky nếu cần)
        safe_candidates = []
        risky_candidates = []
        for move, new_pos in candidates:
            dist = pacman_dist_map.get(new_pos, 999)
            if dist <= 1:
                continue  # Skip chỉ khi chắc chắn chết
            elif dist <= 2 or new_pos in immediate_danger:
                risky_candidates.append((move, new_pos))
            else:
                safe_candidates.append((move, new_pos))

        eval_candidates = safe_candidates if safe_candidates \
            else risky_candidates
        if not eval_candidates:
            return None

        best_move = None
        best_avg_survival = -1

        for move, new_pos in eval_candidates:
            if time.perf_counter() - start_time > time_budget * 0.85:
                break

            total_survival = 0
            rollouts_done = 0

            for _ in range(NUM_ROLLOUTS):
                if time.perf_counter() - start_time > time_budget * 0.9:
                    break

                survival = self._simulate_rollout(
                    new_pos, pacman_pos, MAX_DEPTH)
                total_survival += survival
                rollouts_done += 1

            if rollouts_done > 0:
                avg_survival = total_survival / rollouts_done
                if avg_survival > best_avg_survival:
                    best_avg_survival = avg_survival
                    best_move = move

        return best_move

    def _choose_simulated_pacman_pos(self, pacman_pos: tuple,
                                      ghost_pos: tuple) -> tuple:
        """
        Mô phỏng Pacman dùng BFS: chọn hướng giảm BFS distance nhiều nhất,
        đi thẳng tối đa enemy_speed bước.
        """
        ghost_dist_map = self._get_bfs_distance(ghost_pos)
        best_pos = pacman_pos
        best_d = ghost_dist_map.get(pacman_pos, 999)
        for mv in FOUR_DIRECTIONS:
            pos = pacman_pos
            for _ in range(self.enemy_speed):
                nxt = apply_move(pos, mv)
                if not is_valid_position(self.global_map, nxt):
                    break
                pos = nxt
                d = ghost_dist_map.get(pos, 999)
                if d < best_d:
                    best_d = d
                    best_pos = pos
        return best_pos

    def _simulate_rollout(self, ghost_pos: tuple, pacman_pos: tuple,
                          max_depth: int) -> int:
        """
        Simulate 1 game nâng cao:
        - Ghost: biased flee (distance + exits + corridor penalty)
        - Pacman: wall-aware greedy (xử lý stuck khi gặp tường)

        Cải tiến so với bản cũ:
        - Pacman không bị stuck: khi greedy bị chặn bởi tường,
          Pacman di chuyển lateral (ngang) thay vì đứng yên
        - Ghost thêm corridor penalty vào weight → tránh bị dồn
        - Weight cân bằng hơn (d*4 + exits*3)
        """
        g_pos = ghost_pos
        p_pos = pacman_pos

        for step in range(max_depth):
            # --- Ghost move: corridor-aware biased flee (+ STAY) ---
            g_moves = []
            for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                nxt = apply_move(g_pos, mv) if mv != Move.STAY else g_pos
                if is_valid_position(self.global_map, nxt):
                    g_moves.append(nxt)

            if g_moves:
                scored = []
                for nxt in g_moves:
                    d = manhattan_distance(nxt, p_pos)
                    exits = sum(
                        1 for m2 in FOUR_DIRECTIONS
                        if is_valid_position(
                            self.global_map, apply_move(nxt, m2)))
                    # Corridor penalty: 2 aligned exits = dễ bị dồn
                    corr_pen = 2 if exits == 2 else 0
                    weight = max(1, d * 4 + exits * 3 - corr_pen)
                    scored.append((weight, nxt))

                total_w = sum(w for w, _ in scored)
                r = random.random() * total_w
                cumulative = 0
                g_pos = scored[0][1]
                for w, nxt in scored:
                    cumulative += w
                    if cumulative >= r:
                        g_pos = nxt
                        break

            # --- Pacman move: BFS-greedy ---
            if manhattan_distance(p_pos, g_pos) < 2:
                return step
            p_pos = self._choose_simulated_pacman_pos(p_pos, g_pos)

            if manhattan_distance(p_pos, g_pos) < 2:
                return step

        return max_depth

    # ══════════════════════════════════════════════════════════════════
    #  ALPHA-BETA MINIMAX (Cải tiến 3)
    # ══════════════════════════════════════════════════════════════════

    def _minimax_ab(self, ghost_pos: tuple, pacman_pos: tuple,
                    depth: int, is_ghost_turn: bool,
                    alpha: float, beta: float) -> float:
        """
        Minimax với Alpha-Beta pruning.

        Cải tiến so với bản cũ:
        1. Alpha-Beta pruning: cắt nhánh không cần xét
           → Tăng depth từ 2-3 lên 4 mà vẫn nhanh
        2. Pacman moves realistic: 4 hướng × speed (thay vì BFS-reachable)
           → Đúng luật game (Pacman chỉ đi thẳng 1 hướng)
           → Ít nước đi hơn → pruning hiệu quả hơn
        3. Fast evaluate cho leaf nodes (không tính voronoi/escape_area)
           → Minimax chạy nhẹ nhàng để tránh timeout

        Ghost = MAX (muốn sống lâu), Pacman = MIN (muốn bắt sớm).
        """
        # Check capture
        if manhattan_distance(ghost_pos, pacman_pos) < 2:
            return -1000.0 + (5 - depth) * 100  # Chết sớm = tệ hơn

        if depth == 0:
            return self._evaluate_state_fast(ghost_pos, pacman_pos)

        if is_ghost_turn:
            best = float('-inf')
            ghost_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT,
                           Move.STAY]
            for mv in ghost_moves:
                nxt = apply_move(ghost_pos, mv) if mv != Move.STAY \
                    else ghost_pos
                if not is_valid_position(self.global_map, nxt):
                    continue
                val = self._minimax_ab(nxt, pacman_pos, depth - 1,
                                       False, alpha, beta)
                best = max(best, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break  # Beta cutoff
            if best == float('-inf'):
                return self._evaluate_state_fast(ghost_pos, pacman_pos)
            return best
        else:
            best = float('inf')
            pac_moves = self._generate_pacman_moves(pacman_pos)
            for pac_new in pac_moves:
                val = self._minimax_ab(ghost_pos, pac_new, depth - 1,
                                       True, alpha, beta)
                best = min(best, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break  # Alpha cutoff
            if best == float('inf'):
                return self._evaluate_state_fast(ghost_pos, pacman_pos)
            return best

    def _generate_pacman_moves(self, pacman_pos: tuple) -> List[tuple]:
        """
        Tạo các nước đi realistic của Pacman.

        Pacman đi THẲNG 1 hướng tối đa enemy_speed bước.
        Tối đa: 4 hướng × speed + STAY = 4×2+1 = 9 nước.

        Cải tiến so với _get_cells_within_dist (BFS-reachable):
        - Chính xác hơn: Pacman chỉ đi thẳng, không rẽ giữa chừng
        - Ít nước đi hơn: ~9 thay vì ~13 → alpha-beta hiệu quả hơn
        - BFS-reachable tạo nước đi không hợp lệ (rẽ giữa turn)
        """
        moves = set()
        moves.add(pacman_pos)  # STAY
        for mv in FOUR_DIRECTIONS:
            pos = pacman_pos
            for _ in range(self.enemy_speed):
                nxt = apply_move(pos, mv)
                if is_valid_position(self.global_map, nxt):
                    pos = nxt
                    moves.add(pos)
                else:
                    break
        return list(moves)

    # ══════════════════════════════════════════════════════════════════
    #  EVALUATION FUNCTIONS (Cải tiến 4)
    # ══════════════════════════════════════════════════════════════════

    def _evaluate_state(self, ghost_pos: tuple,
                        pacman_pos: tuple) -> float:
        """
        Hàm đánh giá ĐẦY ĐỦ (dùng cho top-level candidate scoring).

        Cải tiến: thêm voronoi territory, corridor penalty, center bonus,
        effective_turns (xét Pacman speed).

        Factors:
        1. BFS distance / enemy_speed = effective turns to catch
        2. Escape routes (nhiều = khó bắt)
        3. Escape area (vùng thoáng)
        4. Dead-end penalty
        5. Corridor penalty (hành lang hẹp = dễ bị dồn)
        6. Border penalty
        7. Center attraction (gần tâm = nhiều đường thoát)
        8. Voronoi territory (vùng Ghost kiểm soát)
        """
        pac_dist_map = self._get_bfs_distance(pacman_pos)
        dist = float(pac_dist_map.get(ghost_pos, 999))

        # Effective turns: xét Pacman speed khi đánh giá khoảng cách
        effective_turns = dist / max(1, self.enemy_speed)

        escape = float(self._escape_area(ghost_pos, radius=4))
        penalty = self._dead_end_penalty(ghost_pos)
        routes = self._escape_routes(ghost_pos)

        corridor_pen = 5.0 if self._is_corridor(ghost_pos) else 0.0

        h, w = self.global_map.shape
        border_pen = 0.0
        if ghost_pos[0] <= 1 or ghost_pos[0] >= h - 2:
            border_pen += 2.5
        if ghost_pos[1] <= 1 or ghost_pos[1] >= w - 2:
            border_pen += 2.5

        # Center attraction
        center = (h // 2, w // 2)
        dist_center = manhattan_distance(ghost_pos, center)
        center_bonus = max(0, ((h + w) // 2 - dist_center)) * 0.1

        # Voronoi territory: ô Ghost đến trước Pacman
        voronoi = self._voronoi_territory_size(ghost_pos, pac_dist_map)

        return (effective_turns * 3.0
                + routes * 4.0
                + escape * 0.3
                + voronoi * 0.2
                + center_bonus
                - penalty * 2.5
                - corridor_pen
                - border_pen)

    def _evaluate_state_fast(self, ghost_pos: tuple,
                             pacman_pos: tuple) -> float:
        """
        Hàm đánh giá NHẸ (dùng cho minimax leaf nodes).

        Chỉ dùng các phép tính O(1) (BFS cached, neighbor count).
        Không gọi escape_area, escape_routes, voronoi (tốn thời gian).

        Tại sao cần 2 hàm? Minimax gọi evaluate nhiều lần ở nút lá.
        Nếu dùng evaluate đầy đủ → ~200 × 7000ops = 1.4M ops → chậm.
        Evaluate nhẹ: ~200 × 30ops = 6K ops → rất nhanh.
        """
        pac_dist = self._get_bfs_distance(pacman_pos)
        dist = float(pac_dist.get(ghost_pos, 999))
        effective_turns = dist / max(1, self.enemy_speed)

        # Đếm exits trực tiếp (O(4))
        exits = sum(1 for mv in FOUR_DIRECTIONS
                    if is_valid_position(
                        self.global_map, apply_move(ghost_pos, mv)))

        # Dead-end nhanh
        dead_pen = 0.0
        if exits == 0:
            dead_pen = 10.0
        elif exits == 1:
            dead_pen = 6.0

        # Border
        h, w = self.global_map.shape
        border_pen = 0.0
        if ghost_pos[0] <= 1 or ghost_pos[0] >= h - 2:
            border_pen += 2.0
        if ghost_pos[1] <= 1 or ghost_pos[1] >= w - 2:
            border_pen += 2.0

        return (effective_turns * 3.0
                + exits * 4.0
                - dead_pen * 2.5
                - border_pen)

    # ══════════════════════════════════════════════════════════════════
    #  TOPOLOGY ANALYSIS
    # ══════════════════════════════════════════════════════════════════

    def _is_corridor(self, pos: tuple) -> bool:
        """
        Kiểm tra vị trí có phải hành lang hẹp không.

        Hành lang = chỉ có 2 hướng đi và 2 hướng đó đối nhau.
        VD: chỉ đi được UP↔DOWN hoặc LEFT↔RIGHT.

        Ghost bị kẹt trong corridor rất dễ bị Pacman đuổi kịp
        vì chỉ có 1 hướng thoát (hướng xa Pacman). Pacman speed=2
        sẽ bắt Ghost speed=1 trong corridor dài.
        """
        exits = []
        for mv in FOUR_DIRECTIONS:
            nxt = apply_move(pos, mv)
            if is_valid_position(self.global_map, nxt):
                exits.append(mv)
        if len(exits) != 2:
            return False
        return self._is_opposite_move(exits[0], exits[1])

    def _is_opposite_move(self, m1: Move, m2: Move) -> bool:
        """Kiểm tra 2 move đối nhau (UP↔DOWN, LEFT↔RIGHT)."""
        return (m1.value[0] + m2.value[0] == 0 and
                m1.value[1] + m2.value[1] == 0)

    def _voronoi_territory_size(self, ghost_pos: tuple,
                                 pacman_dist_map: dict) -> int:
        """
        Đếm số ô Ghost kiểm soát (đến trước Pacman, xét speed).

        Voronoi territory lớn = Ghost có nhiều "không gian sống",
        nhiều đường thoát. Ô mà Ghost đến trước Pacman
        là ô Ghost có thể trốn vào an toàn.

        Giới hạn bán kính 8 từ Ghost để tối ưu tốc độ tính toán.
        """
        ghost_dist = self._get_bfs_distance(ghost_pos)
        territory = 0
        for pos, g_dist in ghost_dist.items():
            if g_dist > 8:
                continue
            p_dist = pacman_dist_map.get(pos, 999)
            pac_time = p_dist / max(1, self.enemy_speed)
            if g_dist < pac_time:
                territory += 1
        return territory

    def _escape_area(self, pos: tuple, radius: int) -> int:
        """BFS đếm số ô có thể đi được trong vòng `radius` bước."""
        visited = {pos}
        queue = deque([(pos, 0)])
        while queue:
            cur, d = queue.popleft()
            if d >= radius:
                continue
            for mv in FOUR_DIRECTIONS:
                nxt = apply_move(cur, mv)
                if nxt not in visited and \
                        is_valid_position(self.global_map, nxt):
                    visited.add(nxt)
                    queue.append((nxt, d + 1))
        return len(visited)

    def _escape_routes(self, pos: tuple, min_area: int = 5) -> int:
        """Đếm số hướng thoát thực sự (không phải ngõ cụt hẹp)."""
        routes = 0
        for mv in FOUR_DIRECTIONS:
            nxt = apply_move(pos, mv)
            if is_valid_position(self.global_map, nxt):
                area = self._escape_area(nxt, radius=3)
                if area >= min_area:
                    routes += 1
        return routes

    def _dead_end_penalty(self, pos: tuple) -> float:
        """Phạt động khi rơi vào bẫy hoặc hành lang hẹp."""
        routes = self._escape_routes(pos)
        if routes == 0:
            return 10.0
        elif routes == 1:
            return 6.0
        elif routes == 2:
            return 2.0
        return 0.0

    def _safe_space_count(self, start: tuple,
                           pacman_dist_map: dict,
                           radius: int = 3) -> int:
        """Đếm số ô an toàn (Pacman không bắt kịp) trong radius bước."""
        q = deque([(start, 0)])
        visited = {start}
        count = 0
        while q:
            cur, d = q.popleft()
            if d > radius:
                continue
            pac_d = pacman_dist_map.get(cur, 999)
            if pac_d > self.enemy_speed + 1:
                count += 1
            for mv in FOUR_DIRECTIONS:
                nxt = apply_move(cur, mv)
                if nxt not in visited and \
                        is_valid_position(self.global_map, nxt):
                    visited.add(nxt)
                    q.append((nxt, d + 1))
        return count

    # ══════════════════════════════════════════════════════════════════
    #  UTILITY & FALLBACK
    # ══════════════════════════════════════════════════════════════════

    def _safe_random_move(self, pos: tuple) -> Move:
        """
        Khi không biết Pacman ở đâu, đi về hướng thoáng nhất.
        Cải tiến: xét game-wide frequency để tránh quanh quẩn.
        """
        # Explore logic from testpacman
        if getattr(self, 'junctions', None):
            g_map = self.apsp.get(pos, {})
            if self.junctions:
                target = max(self.junctions, key=lambda p: g_map.get(p, 0))
                t_map = self.apsp.get(target, {})
                best_move = Move.STAY
                best_score = float('inf')
                for mv in FOUR_DIRECTIONS:
                    nxt = apply_move(pos, mv)
                    if is_valid_position(self.global_map, nxt):
                        d = t_map.get(nxt, float('inf'))
                        if d < best_score:
                            best_score = d
                            best_move = mv
                if best_move != Move.STAY:
                    return best_move

        best_move = Move.STAY
        best_score = float('-inf')
        for mv in FOUR_DIRECTIONS:
            nxt = apply_move(pos, mv)
            if is_valid_position(self.global_map, nxt):
                area = self._escape_area(nxt, radius=4)
                freq = self._pos_frequency.get(nxt, 0)
                score = area - freq * 2  # Ưu tiên ô ít thăm
                if score > best_score:
                    best_score = score
                    best_move = mv
        return best_move

    def _flee_fallback(self, my_position: tuple, threat: tuple,
                       pacman_dist_map: dict) -> Move:
        """
        Fallback khi mọi đường đều nguy hiểm:
        chọn hướng xa Pacman nhất + nhiều đường thoát nhất.
        """
        best_move = Move.STAY
        best_score = float('-inf')
        for mv in FOUR_DIRECTIONS:
            nxt = apply_move(my_position, mv)
            if is_valid_position(self.global_map, nxt):
                d = pacman_dist_map.get(nxt, 999)
                routes = self._escape_routes(nxt)
                score = d * 2.0 + routes * 3.0
                if score > best_score:
                    best_score = score
                    best_move = mv
        return best_move

    # ══════════════════════════════════════════════════════════════════
    #  BFS DISTANCE CACHING (Tối ưu tốc độ)
    # ══════════════════════════════════════════════════════════════════

    def _get_bfs_distance(self, start: tuple) \
            -> Dict[Tuple[int, int], int]:
        """BFS khoảng cách ngắn nhất, CÓ CACHE per step."""
        if start in self._dist_cache:
            return self._dist_cache[start]

        dist = {start: 0}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for mv in FOUR_DIRECTIONS:
                nxt = apply_move(cur, mv)
                if nxt not in dist and \
                        is_valid_position(self.global_map, nxt):
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)

        self._dist_cache[start] = dist
        return dist


    # ══════════════════════════════════════════════════════════════════
    #  TOPOLOGY METHODS (From testpacman.py)
    # ══════════════════════════════════════════════════════════════════
    def _build_topology(self):
        map_state = self.global_map
        h, w = map_state.shape
        self.valid = {(r, c) for r in range(h) for c in range(w) if map_state[r, c] != 1}

        self.edges = {}
        for r, c in self.valid:
            nb = [(r+dr, c+dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (r+dr, c+dc) in self.valid]
            self.edges[(r, c)] = nb
        self.deg = {p: len(self.edges[p]) for p in self.valid}

        self.apsp = {}
        for s in self.valid:
            d = {s: 0}
            q = deque([s])
            while q:
                u = q.popleft()
                for v in self.edges.get(u, []):
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
            self.apsp[s] = d

        deg2 = dict(self.deg)
        self.dead_ends = set()
        q = deque([p for p in self.valid if deg2[p] <= 1])
        seen = set()
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            self.dead_ends.add(u)
            for v in self.edges.get(u, []):
                if v not in seen:
                    deg2[v] -= 1
                    if deg2[v] <= 1:
                        q.append(v)

        self.core = self.valid - self.dead_ends
        if not self.core:
            self.core = {next(iter(self.valid))} if self.valid else set()

        self.depth_de = {p: 0 for p in self.core}
        self.entry = {p: p for p in self.core}
        q = deque(self.core)
        vis = set(self.core)
        while q:
            u = q.popleft()
            for v in self.edges.get(u, []):
                if v not in vis:
                    vis.add(v)
                    self.depth_de[v] = self.depth_de.get(u, 0) + 1
                    self.entry[v] = self.entry.get(u)
                    q.append(v)

        self.junctions = {p for p in self.core if self.deg.get(p, 0) >= 3}

    def _topo_dist(self, a, b):
        return self.apsp.get(a, {}).get(b, 9999)

    def _topo_pac_turns(self, bfs_dist):
        if bfs_dist <= 0: return 0
        return math.ceil(bfs_dist / self.enemy_speed)

    def _topo_dead_end_trap(self, cell, ppos):
        if cell not in self.dead_ends:
            return False
        ent = self.entry.get(cell)
        if ent is None:
            return True
        depth = self.depth_de.get(cell, 0)
        pac_to_entry = self._topo_pac_turns(self._topo_dist(ppos, ent))
        return pac_to_entry <= depth + 1

    def _simulate_pacman_bfs_speed(self, ppos, gpos):
        if ppos == gpos:
            return ppos

        parent = {ppos: None}
        q = deque([ppos])
        found = False
        while q:
            cur = q.popleft()
            if cur == gpos:
                found = True
                break
            for nb in self.edges.get(cur, []):
                if nb not in parent:
                    parent[nb] = cur
                    q.append(nb)

        if not found:
            return ppos

        path = []
        cur = gpos
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()

        if len(path) < 2:
            return ppos

        step1 = path[1]
        dir1 = (step1[0] - ppos[0], step1[1] - ppos[1])

        # Simulate multiple steps up to self.enemy_speed
        current_pos = step1
        current_dir = dir1
        for i in range(2, self.enemy_speed + 1):
            if len(path) > i:
                next_pos = path[i]
                next_dir = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
                if next_dir == current_dir:
                    current_pos = next_pos
                else:
                    break
            else:
                break
        return current_pos

    def _tree_survival_score(self, gpos, ppos, prev_g, max_turns, depth_budget, t0):
        g = gpos
        p = ppos
        prev = prev_g

        for t in range(max_turns):
            if t0 and time.perf_counter() - t0 > 0.8:
                return max_turns

            if self._topo_dist(g, p) < 2:
                return t

            g_nbs = self.edges.get(g, [])
            if not g_nbs:
                return t

            good = [nb for nb in g_nbs if nb != prev and not (nb in self.dead_ends and self._topo_dead_end_trap(nb, p))]
            if not good:
                good = [nb for nb in g_nbs if nb != prev]
            if not good:
                good = g_nbs

            non_de = [nb for nb in good if nb not in self.dead_ends]
            if non_de:
                good = non_de

            if len(good) == 1:
                prev = g
                g = good[0]
            elif depth_budget > 0:
                best_survived = 0
                for nb in good:
                    new_p = self._simulate_pacman_bfs_speed(p, nb)
                    if self._topo_dist(nb, new_p) < 2:
                        survived = t + 1
                    else:
                        survived = self._tree_survival_score(nb, new_p, g, max_turns - t - 1, depth_budget - 1, t0) + t + 1
                    best_survived = max(best_survived, survived)
                    if best_survived >= max_turns:
                        return max_turns
                return min(best_survived, max_turns)
            else:
                p_map = self.apsp.get(p, {})
                best_nb = max(good, key=lambda nb: (
                    p_map.get(nb, 0) * 10
                    + self.deg.get(nb, 0) * 3
                    + (5 if nb in self.junctions else 0)
                ))
                prev = g
                g = best_nb

            p = self._simulate_pacman_bfs_speed(p, g)

            if self._topo_dist(g, p) < 2:
                return t + 1

        return max_turns
