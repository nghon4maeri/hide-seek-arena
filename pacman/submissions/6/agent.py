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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Tốc độ của Pacman (1 hoặc 2)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Smart Minimax Pacman"
        self.last_known_enemy_pos = None
        
        # Biến lưu "Từ điển khoảng cách" để tra cứu siêu tốc O(1)
        self.all_pairs_distances = None

    # =========================================================================
    # MODULE 1: BẢN ĐỒ TOÀN NĂNG (PRECOMPUTE DISTANCES)
    # Tính toán trước mọi khoảng cách để tiết kiệm thời gian khi chạy AI
    # =========================================================================
    def _precompute_all_distances(self, map_state):
        """
        Chạy ĐÚNG 1 LẦN đầu trận.
        Tính và lưu khoảng cách BFS từ MỌI ô trống đến MỌI ô trống khác.
        """
        self.all_pairs_distances = {}
        height, width = map_state.shape

        # 1. Tìm tất cả các ô hợp lệ (đường đi)
        valid_cells = []
        for r in range(height):
            for c in range(width):
                if map_state[r, c] == 0:
                    valid_cells.append((r, c))

        # 2. Chạy BFS từ mỗi ô để lập bản đồ khoảng cách
        for start_cell in valid_cells:
            self.all_pairs_distances[start_cell] = self._bfs_single_source(start_cell, map_state)

    def _bfs_single_source(self, start_pos, map_state):
        """Hàm BFS lan tỏa để đo khoảng cách từ 1 điểm đến các điểm xung quanh."""
        distances = {start_pos: 0}
        queue = deque([start_pos])

        while queue:
            curr = queue.popleft()
            curr_dist = distances[curr]

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nxt = (curr[0] + dr, curr[1] + dc)

                if self._is_valid(nxt, map_state) and nxt not in distances:
                    distances[nxt] = curr_dist + 1
                    queue.append(nxt)

        return distances

    def _is_valid(self, pos, map_state):
        """Kiểm tra xem tọa độ có nằm trong map và không phải là tường không."""
        r, c = pos
        return (0 <= r < map_state.shape[0]) and (0 <= c < map_state.shape[1]) and map_state[r, c] == 0

    def get_maze_distance(self, pos1, pos2):
        """Tra cứu khoảng cách siêu tốc O(1). Nếu là tường thì trả về vô cực."""
        if pos1 not in self.all_pairs_distances or pos2 not in self.all_pairs_distances.get(pos1, {}):
            return float('inf')
        return self.all_pairs_distances[pos1][pos2]

    def _count_valid_moves(self, pos, map_state):
        """Đếm số đường thoát (ngã rẽ) xung quanh một vị trí."""
        count = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if self._is_valid(nxt, map_state):
                count += 1
        return count

    # =========================================================================
    # MODULE 2: HÀM ĐÁNH GIÁ (EVALUATION FUNCTION)
    # Chấm điểm trạng thái hiện tại (Điểm càng cao thì tình thế càng có lợi)
    # =========================================================================
    def _evaluate_state(self, pacman_pos, ghost_pos, map_state):
        """
        Bản vá Heuristic tuyến tính:
        Ưu tiên TUYỆT ĐỐI việc thu hẹp khoảng cách. Ép góc chỉ là tiêu chí phụ.
        """
        dist = self.get_maze_distance(pacman_pos, ghost_pos)
        
        # Nếu đã ăn được ma -> Cho điểm tuyệt đối để AI phải chọn ngay
        if dist == 0:
            return 999999
            
        ghost_escape_routes = self._count_valid_moves(ghost_pos, map_state)
        
        # Công thức: Khoảng cách càng xa càng bị trừ nặng (trọng số 10.0). 
        # Số đường thoát của ma chỉ dùng để tie-break (trọng số 1.0).
        score = -(10.0 * dist) - ghost_escape_routes
        return score

    # =========================================================================
    # MODULE 3: THUẬT TOÁN MINIMAX (TÌM KIẾM ĐỐI KHÁNG)
    # =========================================================================
    def _get_pacman_next_states(self, pos, map_state):
        """Sinh ra các trạng thái Pacman có thể đi tới, hỗ trợ kịch bản đi 2 bước."""
        next_states = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            pos_1 = (pos[0] + dr, pos[1] + dc)
            
            if self._is_valid(pos_1, map_state):
                next_states.append(pos_1) # Kịch bản đi 1 bước
                
                # Nếu được đi tốc độ x2, tưởng tượng thêm kịch bản phi thẳng 2 bước
                if self.pacman_speed >= 2:
                    pos_2 = (pos_1[0] + dr, pos_1[1] + dc)
                    if self._is_valid(pos_2, map_state):
                        next_states.append(pos_2) # Kịch bản đi 2 bước
        return next_states

    def _minimax(self, pacman_pos, ghost_pos, map_state, depth, is_pacman_turn):
        """Thuật toán Minimax dự đoán tương lai, đã vá lỗi 'Lười biếng'."""
        
        # Thưởng điểm dựa trên độ sâu (Discount factor). 
        # Ăn ma ở depth lớn (tức là ăn sớm hơn) thì điểm càng cao, tránh việc "từ từ rồi ăn".
        if pacman_pos == ghost_pos:
            return 999999 + (depth * 1000)

        # Đạt giới hạn tìm kiếm thì trả về điểm đánh giá
        if depth == 0:
            return self._evaluate_state(pacman_pos, ghost_pos, map_state)

        if is_pacman_turn:
            # LƯỢT PACMAN (Maximize - Cố gắng tối đa hóa điểm số)
            best_score = -float('inf')
            valid_next_positions = self._get_pacman_next_states(pacman_pos, map_state)
            
            if not valid_next_positions: # Nếu kẹt cứng
                return self._evaluate_state(pacman_pos, ghost_pos, map_state)
                
            for nxt_pos in valid_next_positions:
                score = self._minimax(nxt_pos, ghost_pos, map_state, depth - 1, False)
                if score > best_score:
                    best_score = score
            return best_score
            
        else:
            # LƯỢT GHOST (Minimize - Ma sẽ phản kháng để làm giảm điểm của Pacman)
            min_score = float('inf')
            has_valid_move = False
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nxt_pos = (ghost_pos[0] + dr, ghost_pos[1] + dc)
                
                if self._is_valid(nxt_pos, map_state):
                    has_valid_move = True
                    score = self._minimax(pacman_pos, nxt_pos, map_state, depth - 1, True)
                    if score < min_score:
                        min_score = score
                        
            return min_score if has_valid_move else self._evaluate_state(pacman_pos, ghost_pos, map_state)

    # =========================================================================
    # MODULE 4: HÀM RA QUYẾT ĐỊNH CHÍNH (ĐƯỢC GỌI MỖI LƯỢT)
    # =========================================================================
    def step(self, map_state, my_position, enemy_position, step_number):
        # 1. Cập nhật trí nhớ về kẻ thù
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        # 2. Khởi tạo "Bản đồ toàn năng" vào đầu trận (Chỉ chạy đúng 1 lần)
        if self.all_pairs_distances is None:
            self._precompute_all_distances(map_state)
            print("🚀 [Pacman] Đã nạp xong bản đồ toàn năng vào bộ nhớ!")
            
        target = enemy_position or self.last_known_enemy_pos
        if target is None:
            return (Move.STAY, 1) # Đứng im nếu không biết ma ở đâu từ đầu game

        best_move = Move.STAY
        best_steps = 1
        
        # ---------------------------------------------------------------------
        # KỊCH BẢN 1: MA TÀNG HÌNH (SHADOW BOXING)
        # Nếu chưa nhìn thấy ma, không dùng Minimax (vì không lường trước được)
        # Áp dụng thuật toán Greedy dồn dập tiến về vị trí cuối cùng nhìn thấy.
        # ---------------------------------------------------------------------
        if enemy_position is None:
            best_dist = float('inf')
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                pos_1 = (my_position[0] + dr, my_position[1] + dc)
                
                if self._is_valid(pos_1, map_state):
                    # Đánh giá 1 bước
                    dist_1 = self.get_maze_distance(pos_1, target)
                    if dist_1 < best_dist:
                        best_dist = dist_1
                        best_move = move
                        best_steps = 1
                    
                    # Đánh giá 2 bước (nếu có kỹ năng tốc độ x2)
                    if self.pacman_speed >= 2:
                        pos_2 = (pos_1[0] + dr, pos_1[1] + dc)
                        if self._is_valid(pos_2, map_state):
                            dist_2 = self.get_maze_distance(pos_2, target)
                            if dist_2 < best_dist:
                                best_dist = dist_2
                                best_move = move
                                best_steps = 2
            return (best_move, best_steps)

        # ---------------------------------------------------------------------
        # KỊCH BẢN 2: MA HIỆN HÌNH - KÍCH HOẠT MINIMAX ĐỂ ÉP GÓC
        # ---------------------------------------------------------------------
        best_score = -float('inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            pos_1 = (my_position[0] + dr, my_position[1] + dc)
            
            if self._is_valid(pos_1, map_state):
                # Nhìn trước 3 bước (Pacman -> Ghost -> Pacman)
                score_1 = self._minimax(pos_1, target, map_state, depth=3, is_pacman_turn=False)
                
                # Tie-breaker: Trừ một chút điểm theo khoảng cách hiện tại 
                # để phá vỡ hiện tượng "mù phương hướng" (AI loay hoay tại chỗ)
                score_1 -= 0.1 * self.get_maze_distance(pos_1, target)
                
                if score_1 > best_score:
                    best_score = score_1
                    best_move = move
                    best_steps = 1
                
                # Tính toán xem có nên chạy 2 bước không (Nếu có kỹ năng)
                if self.pacman_speed >= 2:
                    pos_2 = (pos_1[0] + dr, pos_1[1] + dc)
                    if self._is_valid(pos_2, map_state):
                        score_2 = self._minimax(pos_2, target, map_state, depth=3, is_pacman_turn=False)
                        score_2 -= 0.1 * self.get_maze_distance(pos_2, target)
                        
                        if score_2 > best_score:
                            best_score = score_2
                            best_move = move
                            best_steps = 2
                            
        return (best_move, best_steps)
    
    
class GhostAgent(BaseGhostAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_positions_cache = set()
        self.degree_cache = {}

        self.last_known_enemy_pos = None
        self.my_history = deque(maxlen=16)
        self.enemy_history = deque(maxlen=6)

        self.last_move = Move.STAY

        self.moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        self.all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]

        # ===== Cấu hình chính =====
        self.win_step = int(kwargs.get("win_step", 200))

        # Nếu Pacman ở cùng ô hoặc cách <= capture_threshold thì coi như bị bắt.
        self.capture_threshold = int(kwargs.get("capture_threshold", 1))

        # Map 21x21 nên tối đa 441 ô.
        self.max_dp_positions = int(kwargs.get("max_dp_positions", 441))

        # Cấu hình Pacman
        self.pacman_can_stay = kwargs.get("pacman_can_stay", True)
        self.pacman_speed = int(kwargs.get("pacman_speed", 2))
        self.fog_is_walkable = kwargs.get("fog_is_walkable", True)

        self.INF = 10**9

        # Cache
        self._dist_map_cache = {}
        self._junction_cache = {}

        self._survival_cache_key = None
        self._survival_data = None

    # =========================================================
    # Main step
    # =========================================================

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:

        self._update_valid_positions(map_state)

        # Đảm bảo vị trí hiện tại của Ghost luôn hợp lệ
        self.valid_positions_cache.add(my_position)

        self._rebuild_degree_cache()

        # Reset cache mỗi turn
        self._dist_map_cache = {}
        self._junction_cache = {}

        if not self.my_history or self.my_history[-1] != my_position:
            self.my_history.append(my_position)

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

            if not self.enemy_history or self.enemy_history[-1] != enemy_position:
                self.enemy_history.append(enemy_position)

        pacman_pos = enemy_position if enemy_position is not None else self.last_known_enemy_pos

        # Nếu chưa biết Pacman ở đâu, đi ra vùng mở
        if pacman_pos is None:
            move = self._move_to_open_area(my_position)
            self.last_move = move
            return move

        # Nếu đã tới step 200 thì chỉ cần sống
        if step_number >= self.win_step:
            if self._distance(pacman_pos, my_position) > self.capture_threshold:
                return Move.STAY

        remaining_steps = self.win_step - step_number + 1

        # Ưu tiên tuyệt đối: chọn move đảm bảo sống đến step 200 nếu có
        survival_move = self._choose_survival_dp_move(
            ghost_pos=my_position,
            pacman_pos=pacman_pos,
            remaining_steps=remaining_steps
        )

        if survival_move is not None:
            self.last_move = survival_move
            return survival_move

        # Nếu DP không tìm được move đảm bảo, fallback sang heuristic sống lâu nhất
        move = self._emergency_long_survival_move(my_position, pacman_pos)
        self.last_move = move
        return move

    # =========================================================
    # Survival DP
    # =========================================================

    def _choose_survival_dp_move(self, ghost_pos, pacman_pos, remaining_steps):
        if remaining_steps <= 0:
            return Move.STAY

        positions = list(self.valid_positions_cache)

        if len(positions) > self.max_dp_positions:
            return None

        target_h = min(remaining_steps, self.win_step)
        data = self._ensure_survival_data(target_h)

        if data is None:
            return None

        pos_to_idx = data["pos_to_idx"]
        idx_to_pos = data["idx_to_pos"]
        ghost_move_idx = data["ghost_move_idx"]
        pacman_legal_idx = data["pacman_legal_idx"]
        layers = data["layers"]

        if ghost_pos not in pos_to_idx or pacman_pos not in pos_to_idx:
            return None

        g0 = pos_to_idx[ghost_pos]
        p0 = pos_to_idx[pacman_pos]

        max_h = min(target_h, len(layers) - 1)
        best_h = -1

        for h in range(max_h, 0, -1):
            if layers[h][g0, p0]:
                best_h = h
                break

        if best_h <= 0:
            
            return None

        child_layer = layers[best_h - 1]

        return self._select_first_move_from_dp_layer(
            ghost_pos=ghost_pos,
            pacman_pos=pacman_pos,
            pos_to_idx=pos_to_idx,
            idx_to_pos=idx_to_pos,
            ghost_move_idx=ghost_move_idx,
            pacman_legal_idx=pacman_legal_idx,
            child_layer=child_layer
        )

    def _ensure_survival_data(self, target_h):
        positions = sorted(self.valid_positions_cache)
        cache_key = (
            tuple(positions),
            self.capture_threshold,
            self.pacman_can_stay,
            self.pacman_speed
        )

        if self._survival_cache_key != cache_key:
            n = len(positions)

            if n == 0 or n > self.max_dp_positions:
                return None

            pos_to_idx = {pos: i for i, pos in enumerate(positions)}
            idx_to_pos = positions

            ghost_legal_idx, ghost_move_idx = self._build_legal_idx_with_moves(
                positions=positions,
                pos_to_idx=pos_to_idx,
                allow_stay=True
            )

            pacman_legal_idx = self._build_pacman_legal_idx(
                positions=positions,
                pos_to_idx=pos_to_idx,
                allow_stay=self.pacman_can_stay
            )

            safe = self._build_safe_matrix(positions, n)
            layers = [safe.copy()]

            self._survival_cache_key = cache_key
            self._survival_data = {
                "positions": positions,
                "pos_to_idx": pos_to_idx,
                "idx_to_pos": idx_to_pos,
                "ghost_legal_idx": ghost_legal_idx,
                "ghost_move_idx": ghost_move_idx,
                "pacman_legal_idx": pacman_legal_idx,
                "safe": safe,
                "layers": layers
            }

        data = self._survival_data
        n = len(data["positions"])
        safe = data["safe"]
        layers = data["layers"]
        ghost_legal_idx = data["ghost_legal_idx"]
        pacman_legal_idx = data["pacman_legal_idx"]

        while len(layers) <= target_h:
            prev = layers[-1]
            good_after_pacman = np.zeros((n, n), dtype=bool)

            for p in range(n):
                p_nexts = pacman_legal_idx[p]
                good_after_pacman[:, p] = (safe[:, p] & prev[:, p_nexts].all(axis=1))

            current = np.zeros((n, n), dtype=bool)
            for g in range(n):
                g_nexts = ghost_legal_idx[g]
                current[g, :] = good_after_pacman[g_nexts, :].any(axis=0)

            current &= safe
            layers.append(current)

            if not current.any():
                break

        return data

    def _build_legal_idx_with_moves(self, positions, pos_to_idx, allow_stay=True):
        legal_idx = []
        move_idx = []

        for pos in positions:
            next_indices = []
            move_pairs = []

            if allow_stay:
                idx = pos_to_idx[pos]
                next_indices.append(idx)
                move_pairs.append((Move.STAY, idx))

            for move in self.moves:
                dr, dc = move.value
                nxt = (pos[0] + dr, pos[1] + dc)

                if nxt in pos_to_idx and self._is_valid_position_fast(nxt):
                    idx = pos_to_idx[nxt]
                    next_indices.append(idx)
                    move_pairs.append((move, idx))

            legal_idx.append(np.array(next_indices, dtype=np.int32))
            move_idx.append(move_pairs)

        return legal_idx, move_idx
    
    def _build_pacman_legal_idx(self, positions, pos_to_idx, allow_stay=True):
        legal_idx = []

        for pos in positions:
            next_indices = []

            if allow_stay:
                next_indices.append(pos_to_idx[pos])

            for move in self.moves:
                dr, dc = move.value
                cur = pos

                for _ in range(self.pacman_speed):
                    nxt = (cur[0] + dr, cur[1] + dc)

                    if nxt not in pos_to_idx or not self._is_valid_position_fast(nxt):
                        break

                    next_indices.append(pos_to_idx[nxt])
                    cur = nxt

            legal_idx.append(np.array(sorted(set(next_indices)), dtype=np.int32))

        return legal_idx

    def _pacman_legal_moves_from(self, pos, allow_stay=True):
        seen = set()

        if allow_stay and self._is_valid_position_fast(pos):
            seen.add(pos)
            yield Move.STAY, pos

        for move in self.moves:
            dr, dc = move.value
            cur = pos

            for _ in range(self.pacman_speed):
                nxt = (cur[0] + dr, cur[1] + dc)

                if not self._is_valid_position_fast(nxt):
                    break

                if nxt not in seen:
                    seen.add(nxt)
                    yield move, nxt

                cur = nxt

    def _build_safe_matrix(self, positions, n):
        safe = np.zeros((n, n), dtype=bool)

        for p_idx, p_pos in enumerate(positions):
            dist_map = self._get_dist_map(p_pos)

            for g_idx, g_pos in enumerate(positions):
                dist = dist_map.get(g_pos, self.INF)

                if dist > self.capture_threshold:
                    safe[g_idx, p_idx] = True

        return safe

    def _select_first_move_from_dp_layer(self, ghost_pos, pacman_pos, pos_to_idx, idx_to_pos,
                                         ghost_move_idx, pacman_legal_idx, child_layer):
        g0 = pos_to_idx[ghost_pos]
        p0 = pos_to_idx[pacman_pos]

        best_move = None
        best_score = -float("inf")

        for move, g2 in ghost_move_idx[g0]:
            ok = True

            for p2 in pacman_legal_idx[p0]:
                if not child_layer[g2, p2]:
                    ok = False
                    break

            if not ok:
                continue

            g2_pos = idx_to_pos[g2]

            score = self._survival_tiebreak_score(
                move=move,
                ghost_pos=g2_pos,
                old_ghost_pos=ghost_pos,
                pacman_pos=pacman_pos
            )

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # =========================================================
    # Heuristic khi co nhieu move cung song duoc
    # =========================================================

    def _survival_tiebreak_score(self, move, ghost_pos, old_ghost_pos, pacman_pos):
        score = 0.0

        dist = self._distance(pacman_pos, ghost_pos)
        old_dist = self._distance(pacman_pos, old_ghost_pos)

        escape_routes = self._count_escape_routes(ghost_pos)

        safe_area = self._safe_area_score(
            start_pos=ghost_pos,
            pacman_pos=pacman_pos,
            max_depth=18
        )

        junction_dist = self._distance_to_junction(
            start_pos=ghost_pos,
            max_depth=25
        )

        score += min(dist, 100) * 120
        score += (min(dist, 100) - min(old_dist, 100)) * 220
        score += safe_area * 15

        if escape_routes <= 1:
            score -= 6000
        elif escape_routes == 2:
            score -= junction_dist * 15
        elif escape_routes == 3:
            score += 1000
        elif escape_routes >= 4:
            score += 1500

        if move == Move.STAY:
            if dist <= 8:
                score -= 5000
            else:
                score -= 300

        if move != Move.STAY and self._is_reverse(move, self.last_move):
            score -= 100

        score -= self._recent_position_penalty(ghost_pos)

        return score

    # =========================================================
    # Emergency fallback
    # =========================================================

    def _emergency_long_survival_move(self, ghost_pos, pacman_pos):
        """
        Khi DP thất bại, dùng Alpha-Beta dự đoán trước 4 bước (Depth=4).
        """
        best_move = Move.STAY
        best_score = -float("inf")

        # Độ sâu tìm kiếm: Mỗi "depth" là 1 lượt của 1 phe. 
        # depth=4 nghĩa là: Ghost đi -> Pacman đi -> Ghost đi -> Pacman đi
        search_depth = 4 

        for move, g2 in self._legal_moves_from(ghost_pos, allow_stay=True):
            # Cắt bỏ ngay những nước đi vào miệng Pacman
            dist_now = self._distance(pacman_pos, g2)
            if dist_now <= self.capture_threshold:
                continue

            # Gọi cây tìm kiếm tương lai. Sau khi Ghost đi, đến lượt Pacman (False)
            score = self._alpha_beta(
                ghost_pos=g2, 
                pacman_pos=pacman_pos, 
                depth=search_depth - 1, 
                alpha=-float('inf'), 
                beta=float('inf'), 
                is_ghost_turn=False
            )

            # Phạt nhẹ nếu lặp lại vị trí cũ để chống kẹt (chỉ phạt rất nhẹ lúc khẩn cấp)
            score -= self._recent_position_penalty(g2) * 0.2

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # =========================================================
    # Map helpers
    # =========================================================

    def _update_valid_positions(self, map_state: np.ndarray):
        self.valid_positions_cache.clear()

        height, width = map_state.shape

        for row in range(height):
            for col in range(width):
                cell = map_state[row, col]

                if self.fog_is_walkable:
                    if cell != 1:
                        self.valid_positions_cache.add((row, col))
                else:
                    if cell == 0:
                        self.valid_positions_cache.add((row, col))

    def _is_valid_position_fast(self, pos):
        return pos in self.valid_positions_cache

    def _legal_moves_from(self, pos, allow_stay=True):
        moves = self.all_moves if allow_stay else self.moves

        for move in moves:
            dr, dc = move.value
            nxt = (pos[0] + dr, pos[1] + dc)

            if self._is_valid_position_fast(nxt):
                yield move, nxt

    def _neighbors(self, pos):
        for move in self.moves:
            dr, dc = move.value
            nxt = (pos[0] + dr, pos[1] + dc)

            if self._is_valid_position_fast(nxt):
                yield nxt

    def _rebuild_degree_cache(self):
        self.degree_cache.clear()

        for pos in self.valid_positions_cache:
            count = sum(1 for _ in self._neighbors(pos))
            self.degree_cache[pos] = count

    def _count_escape_routes(self, pos):
        return self.degree_cache.get(pos, 0)

    # =========================================================
    # Distance BFS
    # =========================================================

    def _get_dist_map(self, start_pos):
        if start_pos in self._dist_map_cache:
            return self._dist_map_cache[start_pos]

        dists = {start_pos: 0}
        queue = deque([start_pos])

        while queue:
            curr = queue.popleft()

            for nxt in self._neighbors(curr):
                if nxt not in dists:
                    dists[nxt] = dists[curr] + 1
                    queue.append(nxt)

        self._dist_map_cache[start_pos] = dists
        return dists

    def _distance(self, start_pos, end_pos):
        if start_pos == end_pos:
            return 0

        dist_map = self._get_dist_map(start_pos)
        return dist_map.get(end_pos, self.INF)

    # =========================================================
    # Area / junction
    # =========================================================

    def _safe_area_score(self, start_pos, pacman_pos, max_depth=18):
        pacman_dist_map = self._get_dist_map(pacman_pos)

        queue = deque([(start_pos, 0)])
        visited = {start_pos}

        score = 0

        while queue:
            curr, depth = queue.popleft()

            ghost_arrival_time = depth
            pacman_cell_dist = pacman_dist_map.get(curr, self.INF)

            if pacman_cell_dist >= self.INF:
                pacman_arrival_time = self.INF
            else:
                need_to_move = max(0, pacman_cell_dist - self.capture_threshold)
                pacman_arrival_time = (need_to_move + self.pacman_speed - 1) // self.pacman_speed

            if pacman_arrival_time <= ghost_arrival_time:
                continue

            score += max_depth - depth + 1

            if depth >= max_depth:
                continue

            for nxt in self._neighbors(curr):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))

        return score

    def _distance_to_junction(self, start_pos, max_depth=25):
        if start_pos in self._junction_cache:
            return self._junction_cache[start_pos]

        if self._count_escape_routes(start_pos) >= 3:
            self._junction_cache[start_pos] = 0
            return 0

        queue = deque([(start_pos, 0)])
        visited = {start_pos}

        while queue:
            curr, depth = queue.popleft()

            if depth > 0 and self._count_escape_routes(curr) >= 3:
                self._junction_cache[start_pos] = depth
                return depth

            if depth >= max_depth:
                continue

            for nxt in self._neighbors(curr):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))

        result = max_depth + 1
        self._junction_cache[start_pos] = result
        return result

    # =========================================================
    # Anti-loop
    # =========================================================

    def _is_reverse(self, move, last_move):
        if move == Move.STAY or last_move == Move.STAY:
            return False

        dr1, dc1 = move.value
        dr2, dc2 = last_move.value

        return dr1 + dr2 == 0 and dc1 + dc2 == 0

    def _recent_position_penalty(self, pos):
        penalty = 0.0

        for age, old_pos in enumerate(reversed(self.my_history), start=1):
            if pos == old_pos:
                penalty += 500 / age

        return penalty

    # =========================================================
    # Khi khong thay Pacman
    # =========================================================

    def _move_to_open_area(self, pos):
        best_move = Move.STAY
        best_score = -float("inf")

        for move, nxt in self._legal_moves_from(pos, allow_stay=True):
            escape_routes = self._count_escape_routes(nxt)
            open_area = self._open_area_score(start_pos=nxt, max_depth=12)
            junction_dist = self._distance_to_junction(start_pos=nxt, max_depth=25)

            score = 0.0
            score += open_area * 10

            if escape_routes <= 1:
                score -= 5000
            elif escape_routes == 2:
                score -= junction_dist * 180
            elif escape_routes == 3:
                score += 1000
            elif escape_routes >= 4:
                score += 1600

            if move == Move.STAY:
                score -= 500

            if move != Move.STAY and self._is_reverse(move, self.last_move):
                score -= 300

            score -= self._recent_position_penalty(nxt)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _open_area_score(self, start_pos, max_depth=12):
        queue = deque([(start_pos, 0)])
        visited = {start_pos}
        score = 0

        while queue:
            curr, depth = queue.popleft()
            score += max_depth - depth + 1

            if depth >= max_depth:
                continue

            for nxt in self._neighbors(curr):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))

        return score
    # =========================================================
    # Alpha-Beta Pruning (Dự đoán tương lai)
    # =========================================================

    def _evaluate_state(self, ghost_pos, pacman_pos):
        """Đánh giá trạng thái bàn cờ tại điểm dừng của cây tìm kiếm"""
        dist = self._distance(pacman_pos, ghost_pos)
        if dist <= self.capture_threshold:
            return -999999  # Trạng thái chết

        safe_area = self._safe_area_score(ghost_pos, pacman_pos, max_depth=12)
        escape_routes = self._count_escape_routes(ghost_pos)
        
        # Công thức: Khoảng cách càng xa càng tốt + Vùng an toàn rộng + Nhiều lối thoát
        return (dist * 200) + (safe_area * 50) + (escape_routes * 300)

    def _alpha_beta(self, ghost_pos, pacman_pos, depth, alpha, beta, is_ghost_turn):
        """Duyệt cây Minimax cắt tỉa Alpha-Beta"""
        dist = self._distance(pacman_pos, ghost_pos)
        
        # Điều kiện dừng: Chết hoặc hết độ sâu
        if dist <= self.capture_threshold:
            return -999999
        if depth == 0:
            return self._evaluate_state(ghost_pos, pacman_pos)

        if is_ghost_turn:
            max_eval = -float('inf')
            for _, nxt_g in self._legal_moves_from(ghost_pos, allow_stay=True):
                ev = self._alpha_beta(nxt_g, pacman_pos, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, ev)
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break 
            return max_eval
        else:
            min_eval = float('inf')
            for _, nxt_p in self._pacman_legal_moves_from(pacman_pos, allow_stay=self.pacman_can_stay):
                ev = self._alpha_beta(ghost_pos, nxt_p, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, ev)
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break  
            return min_eval