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
"""

import sys
import random
from pathlib import Path
from collections import deque

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


DIRECTIONS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Minimax + Alpha-Beta pruning.

    Ý tưởng: coi đây là trò chơi đối kháng 2 người.
      - Pacman (mình)  : muốn khoảng cách tới Ghost càng NHỎ càng tốt.
      - Ghost (đối thủ): giả định nó cũng "thông minh" và sẽ né để khoảng
        cách càng LỚN càng tốt (trường hợp xấu nhất cho mình).
    Minimax duyệt trước vài lượt đi luân phiên của 2 bên để chọn nước đi
    mà DÙ Ghost né kiểu gì thì khoảng cách cuối cùng vẫn nhỏ nhất có thể.
    Alpha-Beta pruning chỉ là kỹ thuật cắt bớt nhánh chắc chắn không tốt hơn,
    kết quả giống hệt Minimax thường nhưng tính nhanh hơn nhiều.

    Khi không thấy Ghost:
      - Còn nhớ vị trí cuối -> BFS đuổi tới đó (không đoán mò được nữa nên
        không dùng Minimax được, vì Minimax cần biết vị trí thật của đối thủ).
      - Chưa từng thấy -> BFS đi tới ô sương mù (-1) gần nhất để khám phá bản đồ.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Minimax Pacman"
        self.last_known_enemy_pos = None
        # Số nửa-nước nhìn trước. depth=6 nghĩa là 3 lượt Pacman + 3 lượt Ghost.
        # Tăng lên nếu muốn nhìn xa hơn (vẫn rất nhanh nhờ Alpha-Beta),
        # giảm xuống nếu bản đồ khiến agent phản hồi chậm.
        self.search_depth = 6
        # Chống dao động vô hạn: vì minimax tính lại từ đầu MỖI bước (không
        # nhớ kế hoạch cũ), khi có nhiều nước đi hòa điểm nhau, nó có thể
        # chọn luân phiên khiến Pacman đi tới rồi quay lại y hệt vị trí cũ
        # mãi mãi (đặc biệt khi Ghost đứng yên hoặc bị chặn tường phải vòng).
        # Lưu vài vị trí gần nhất để phạt nhẹ các nước đi dẫn quay lại đó,
        # phá vỡ chu trình lặp mà không ảnh hưởng khi minimax có 1 lựa chọn
        # rõ ràng tốt nhất (chỉ tác động khi đang hòa điểm).
        self.recent_positions = deque(maxlen=6)
        # Bản đồ TÍCH LŨY những gì đã từng quan sát được, khác với map_state
        # (chỉ là ảnh chụp tức thời của lượt hiện tại). Bắt buộc phải có vì
        # đề "Blind Adversary" dùng tầm nhìn hình CHỮ THẬP (4 tia thẳng,
        # KHÔNG phải hình tròn/vùng) - nghĩa là rất nhiều ô ngay sát cạnh
        # (lệch theo đường chéo) sẽ hiện -1 dù đang đứng rất gần. Nếu không
        # tự nhớ lại, mỗi lượt agent coi như "mù" gần hết map xung quanh,
        # kể cả ô mình vừa đi qua -> gây dò đường sai và dao động vô hạn.
        # Tường là TĨNH (không đổi) nên ô nào đã từng xác định (0 hoặc 1)
        # thì lưu vĩnh viễn, không tin lại -1 khi ô đó tạm ra khỏi tầm nhìn.
        self.known_map = None
        # Lưu 2 lần THẤY Ghost gần nhất (không phải mọi lượt) để ước lượng
        # vector di chuyển của nó, dùng cho việc dự đoán hướng khi khám phá.
        self.enemy_sightings = deque(maxlen=2)

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):

        # Cập nhật bản đồ tích lũy: ghi đè MỌI ô mà lượt này thấy rõ (!= -1),
        # còn ô nào lượt này không thấy (-1) thì GIỮ NGUYÊN giá trị đã biết
        # trước đó (nếu có) thay vì bị "quên" thành sương mù trở lại.
        if self.known_map is None:
            self.known_map = map_state.copy()
        else:
            seen_mask = map_state != -1
            self.known_map[seen_mask] = map_state[seen_mask]

        # Từ đây trở đi, toàn bộ pathfinding (BFS, Minimax...) dùng bản đồ
        # TÍCH LŨY thay vì ảnh chụp tức thời của riêng lượt này.
        map_state = self.known_map

        self.recent_positions.append(my_position)

        # --- Trường hợp 1: thấy Ghost -> Minimax ---
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            if not self.enemy_sightings or self.enemy_sightings[-1] != enemy_position:
                self.enemy_sightings.append(enemy_position)
            move = self._best_move_minimax(my_position, enemy_position, map_state)

            if move == Move.STAY:
                return (Move.STAY, 1)

            desired_steps = self._steps_towards(my_position, enemy_position, move)
            action = self._choose_action(my_position, [move], map_state, desired_steps)
            if action:
                return action

        # --- Trường hợp 2: mất dấu nhưng còn nhớ vị trí cuối -> BFS đuổi tới
        # điểm DỰ ĐOÁN (ngoại suy hướng di chuyển) thay vì điểm đã "nguội" ---
        if self.last_known_enemy_pos is not None:
            target = self._predict_enemy_pos(map_state) or self.last_known_enemy_pos
            path = self._bfs(my_position, target, map_state)
            if not path:
                # Điểm dự đoán có thể rơi vào vùng chưa biết/không tới được
                # -> fallback về đúng vị trí cuối từng thấy thật.
                path = self._bfs(my_position, self.last_known_enemy_pos, map_state)
            if path:
                move = path[0]
                action = self._choose_action(my_position, [move], map_state, len(path))
                if action:
                    return action

        # --- Trường hợp 3: chưa từng thấy Ghost -> đi khám phá sương mù ---
        explore_path = self._explore_path(my_position, map_state)
        if explore_path:
            move = explore_path[0]
            action = self._choose_action(my_position, [move], map_state, len(explore_path))
            if action:
                return action

        # --- Fallback cuối cùng ---
        for move in DIRECTIONS:
            if self._is_valid_move(my_position, move, map_state):
                return (move, 1)
        return (Move.STAY, 1)

    # ------------------------------------------------------------------
    # MINIMAX + ALPHA-BETA
    # ------------------------------------------------------------------

    def _best_move_minimax(self, pac_pos, ghost_pos, map_state):
        """Chọn nước đi của Pacman ở gốc cây Minimax (lượt MIN)."""
        # Cache các bảng khoảng cách mê cung THẬT (đi vòng qua tường),
        # keyed theo vị trí Ghost. Ghost di chuyển qua từng lượt trong cây
        # tìm kiếm nên KHÔNG thể dùng chung 1 bảng tính từ vị trí ban đầu -
        # phải tính (hoặc tra cache) riêng cho vị trí Ghost ở từng nút lá.
        # Cache giúp tránh tính lại BFS khi nhiều nhánh cùng dẫn tới 1
        # ghost_pos giống nhau ở độ sâu lá.
        dist_cache = {}

        best_move, best_value = Move.STAY, float("inf")
        best_revisit_penalty = float("inf")
        alpha, beta = -float("inf"), float("inf")

        for move in self._valid_moves(pac_pos, map_state):
            nxt_pac = self._apply_move(pac_pos, move, map_state)
            value = self._minimax(
                nxt_pac, ghost_pos, self.search_depth - 1,
                maximizing=True, alpha=alpha, beta=beta,
                map_state=map_state, dist_cache=dist_cache,
            )
            # Chống dao động: nếu nước đi này dẫn tới 1 ô vừa ghé qua gần
            # đây, coi như "kém hấp dẫn hơn 1 chút" CHỈ để phá hòa - không
            # đủ lớn để lấn át 1 nước đi thực sự tốt hơn về khoảng cách.
            revisit_penalty = 1 if nxt_pac in self.recent_positions else 0
            if (value < best_value or
                    (value == best_value and revisit_penalty < best_revisit_penalty)):
                best_value, best_move = value, move
                best_revisit_penalty = revisit_penalty
            beta = min(beta, best_value)

        return best_move

    def _minimax(self, pac_pos, ghost_pos, depth, maximizing, alpha, beta, map_state, dist_cache):
        # Bắt được Ghost -> giá trị cực tốt, dừng ngay.
        # Trừ thêm `depth` (số nửa-nước còn lại) để ưu tiên phương án bắt được
        # CÀNG SỚM CÀNG TỐT, thay vì coi mọi cách "chắc thắng" là như nhau.
        if pac_pos == ghost_pos:
            return -1000 - depth

        # Hết độ sâu -> ước lượng bằng khoảng cách mê cung thật (đi vòng qua
        # tường), chính xác hơn nhiều so với Manhattan. Bảng khoảng cách
        # được tính (và cache lại) từ vị trí Ghost THỰC TẾ tại nút lá này -
        # tức là vị trí Ghost SAU khi đã "né" qua các lượt giả lập, chứ
        # không phải vị trí Ghost lúc bắt đầu suy nghĩ. Nếu tính từ vị trí
        # ban đầu, kết quả BFS sẽ không phản ánh đúng nước né của Ghost và
        # Minimax coi như bỏ qua tác dụng của lượt MAX.
        if depth == 0:
            dist_map = dist_cache.get(ghost_pos)
            if dist_map is None:
                dist_map = self._bfs_distance_map(ghost_pos, map_state)
                dist_cache[ghost_pos] = dist_map
            # Nếu pac_pos ngoài vùng BFS phủ tới (hiếm, ví dụ 2 khu vực bị
            # chia cắt) thì fallback về Manhattan cho an toàn, không bị lỗi.
            return dist_map.get(pac_pos, self._manhattan(pac_pos, ghost_pos))

        if maximizing:
            # Lượt của Ghost (giả định): muốn khoảng cách CÀNG LỚN
            value = -float("inf")
            for move in self._valid_moves(ghost_pos, map_state):
                nxt_ghost = self._apply_move(ghost_pos, move, map_state)
                value = max(value, self._minimax(
                    pac_pos, nxt_ghost, depth - 1, False, alpha, beta, map_state, dist_cache
                ))
                if value >= beta:
                    break  # nhánh Alpha cắt: Pacman sẽ không bao giờ chọn đường này
                alpha = max(alpha, value)
            return value
        else:
            # Lượt của Pacman: muốn khoảng cách CÀNG NHỎ
            value = float("inf")
            for move in self._valid_moves(pac_pos, map_state):
                nxt_pac = self._apply_move(pac_pos, move, map_state)
                value = min(value, self._minimax(
                    nxt_pac, ghost_pos, depth - 1, True, alpha, beta, map_state, dist_cache
                ))
                if value <= alpha:
                    break  # nhánh Beta cắt: Ghost sẽ không bao giờ chọn đường này
                beta = min(beta, value)
            return value

    def _bfs_distance_map(self, source, map_state):
        """BFS từ 1 điểm gốc ra toàn bộ ô đã biết là trống (0).
        Trả về dict {vị_trí: khoảng_cách_thật_qua_mê_cung}.
        Đây chính là "bảng khoảng cách" - khác precomputed table ở chỗ
        tính on-the-fly (nhanh vì bản đồ nhỏ, chỉ O(số_ô) mỗi lượt)."""
        dist = {source: 0}
        queue = deque([source])
        while queue:
            cur = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (cur[0] + dr, cur[1] + dc)
                if nb not in dist and self._is_valid_position(nb, map_state):
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        return dist

    def _valid_moves(self, pos, map_state):
        moves = [m for m in DIRECTIONS if self._is_valid_move(pos, m, map_state)]
        return moves if moves else [Move.STAY]

    def _apply_move(self, pos, move, map_state):
        if move == Move.STAY:
            return pos
        delta_row, delta_col = move.value
        nxt = (pos[0] + delta_row, pos[1] + delta_col)
        return nxt if self._is_valid_position(nxt, map_state) else pos

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _steps_towards(self, pos, target, move):
        """Số bước muốn đi theo hướng move (dựa trên khoảng cách thật tới target),
        để tận dụng pacman_speed > 1 mà không đi vọt qua Ghost.

        QUAN TRỌNG: chỉ "đi nhiều bước" khi move thực sự đang tiến GẦN target
        trên trục đó. Nếu minimax chọn move này để VÒNG qua tường (tức đang đi
        ra xa target trên trục row/col dù kết quả cuối vẫn tốt), tuyệt đối
        không được nhân số bước theo khoảng cách thô - làm vậy sẽ lao thẳng
        ra xa target nhiều ô một lúc, gây dao động qua lại vô hạn giữa 2 vị
        trí. Trường hợp move không tiến gần target trên trục tương ứng, chỉ
        đi đúng 1 bước để minimax được đánh giá lại ngay ở lượt kế tiếp."""
        dr, dc = move.value
        if dr != 0:
            raw = target[0] - pos[0]
            if (raw > 0 and dr > 0) or (raw < 0 and dr < 0):
                return max(1, abs(raw))
            return 1
        if dc != 0:
            raw = target[1] - pos[1]
            if (raw > 0 and dc > 0) or (raw < 0 and dc < 0):
                return max(1, abs(raw))
            return 1
        return 1

    # ------------------------------------------------------------------
    # BFS (đuổi theo vị trí cụ thể / khám phá sương mù)
    # ------------------------------------------------------------------

    def _bfs(self, start, goal, map_state):
        """BFS tìm đường ngắn nhất từ start đến goal, trả về list Move."""
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            cur, path = queue.popleft()
            if cur == goal:
                return path
            for move, (dr, dc) in [(Move.UP, (-1, 0)), (Move.DOWN, (1, 0)),
                                    (Move.LEFT, (0, -1)), (Move.RIGHT, (0, 1))]:
                nb = (cur[0] + dr, cur[1] + dc)
                if nb not in visited and self._is_valid_position(nb, map_state):
                    visited.add(nb)
                    queue.append((nb, path + [move]))
        return []

    def _predict_enemy_pos(self, map_state, steps_ahead=4):
        """Ước lượng Ghost hiện đang ở đâu dựa trên vector di chuyển giữa
        2 lần thấy gần nhất (giống ý tưởng "predictive interception"):
        v = vị_trí_thấy_sau - vị_trí_thấy_trước, rồi ngoại suy thêm
        `steps_ahead` bước theo hướng đó. Chỉ để ĐỊNH HƯỚNG khám phá,
        không dùng cho Minimax (Minimax cần vị trí THẬT, không đoán mò).
        Trả về None nếu chưa đủ dữ liệu (mới thấy Ghost < 2 lần)."""
        if len(self.enemy_sightings) < 2:
            return self.last_known_enemy_pos

        prev, last = self.enemy_sightings[0], self.enemy_sightings[1]
        dr, dc = last[0] - prev[0], last[1] - prev[1]
        if dr == 0 and dc == 0:
            return last

        height, width = map_state.shape
        predicted = (last[0] + dr * steps_ahead, last[1] + dc * steps_ahead)
        # Kẹp về trong biên bản đồ để không đề xuất 1 điểm đích ngoài map -
        # BFS sẽ tự nhắm gần điểm này nhất có thể trong phần bản đồ đã biết.
        row = min(max(predicted[0], 0), height - 1)
        col = min(max(predicted[1], 0), width - 1)
        return (row, col)

    def _explore_path(self, start, map_state):
        """BFS liệt kê TẤT CẢ ô "frontier" (ô đã biết, giáp ít nhất 1 ô
        sương mù -1), rồi CHỌN frontier tốt nhất thay vì cứ nhắm gần nhất:

          - Ưu tiên frontier giáp NHIỀU ô sương mù cùng lúc (đứng đó 1 lần
            là "mở khóa" được nhiều hướng luôn, do tầm nhìn hình chữ thập
            tỏa ra 4 tia - frontier ở góc/ngã ba lộ nhiều tia hơn frontier
            ở giữa hành lang thẳng).
          - Nếu từng thấy Ghost, CỘNG điểm cho frontier nằm gần hướng dự
            đoán Ghost đang di chuyển tới (_predict_enemy_pos), để việc
            "khám phá" đồng thời cũng là "truy tìm có định hướng" thay vì
            lan tỏa đều ra mọi phía một cách mù quáng.
          - Trừ điểm theo khoảng cách phải đi (ưu tiên gần vẫn quan trọng).
        """
        height, width = map_state.shape

        def fog_neighbor_count(pos):
            row, col = pos
            count = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                r, c = row + dr, col + dc
                if 0 <= r < height and 0 <= c < width and map_state[r, c] == -1:
                    count += 1
            return count

        queue = deque([(start, [])])
        visited = {start}
        frontiers = []  # list[(path, fog_count)]
        while queue:
            cur, path = queue.popleft()
            if path:
                fc = fog_neighbor_count(cur)
                if fc > 0:
                    frontiers.append((cur, path, fc))
            for move, (dr, dc) in [(Move.UP, (-1, 0)), (Move.DOWN, (1, 0)),
                                    (Move.LEFT, (0, -1)), (Move.RIGHT, (0, 1))]:
                nb = (cur[0] + dr, cur[1] + dc)
                if nb not in visited and self._is_valid_position(nb, map_state):
                    visited.add(nb)
                    queue.append((nb, path + [move]))

        if not frontiers:
            return []

        predicted = self._predict_enemy_pos(map_state)

        def score(item):
            pos, path, fc = item
            s = fc * 3 - len(path)
            if predicted is not None:
                s -= self._manhattan(pos, predicted) * 0.5
            return s

        _, best_path, _ = max(frontiers, key=score)
        return best_path

    # ------------------------------------------------------------------
    # Di chuyển nhiều ô / kiểm tra hợp lệ
    # ------------------------------------------------------------------

    def _choose_action(self, pos, moves, map_state, desired_steps):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos, move, map_state, max_steps):
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

    def _is_valid_move(self, pos, move, map_state):
        return self._max_valid_steps(pos, move, map_state, 1) == 1

    def _is_valid_position(self, pos, map_state):
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
    """
    Ghost Agent cho bản Initial:
    - Nhớ bản đồ và ghi nhận tần suất thăm các ô (Heatmap).
    - Cảnh báo Pacman: Nhớ vị trí Pacman và tự quên sau 5 bước.
    - BFS Né tránh: Tìm đường thoát xa nhất trên bản đồ nội bộ.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.internal_map = np.full((21, 21), -1, dtype=int)
        self.visited_heatmap = np.zeros((21, 21), dtype=int)
        self.last_known_enemy_pos = None
        self.forget_timer = 0
        self.FORGET_LIMIT = 5 # Giảm thời gian nhớ xuống để dũng cảm khám phá hơn

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        # 1. Cập nhật trí nhớ
        self.visited_heatmap[my_position] += 1
        visible_mask = map_state != -1
        self.internal_map[visible_mask] = map_state[visible_mask]

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.forget_timer = 0
        elif self.last_known_enemy_pos is not None:
            self.forget_timer += 1
            if self.forget_timer >= self.FORGET_LIMIT:
                self.last_known_enemy_pos = None

        # 2. Tìm nước đi hợp lệ ngay lập tức
        valid_moves = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = my_position[0] + m.value[0], my_position[1] + m.value[1]
            if 0 <= nr < 21 and 0 <= nc < 21 and map_state[nr, nc] == 0:
                valid_moves.append(m)

        if not valid_moves:
            return Move.STAY

        # 3. Chế độ An Toàn (Đi khám phá sương mù)
        if self.last_known_enemy_pos is None:
            return self._explore(my_position, valid_moves)

        # 4. Chế độ Báo Động (Chạy trốn bằng BFS)
        return self._evade(my_position, valid_moves)

    def _explore(self, my_position, valid_moves):
        # Ưu tiên đi vào những ô ít bị đi qua nhất (tránh lẩn quẩn)
        best_move = Move.STAY
        min_score = float('inf')
        random.shuffle(valid_moves) # Tránh việc luôn ưu tiên hướng UP, DOWN
        
        for move in valid_moves:
            nr, nc = my_position[0] + move.value[0], my_position[1] + move.value[1]
            # Điểm = Số lần đã đi qua - Bonus nếu hướng đó có nhiều sương mù
            score = self.visited_heatmap[nr, nc] - self._count_fog((nr, nc), move) * 0.2
            if score < min_score:
                min_score = score
                best_move = move
        return best_move

    def _evade(self, my_position, valid_moves):
        # Dùng BFS đo khoảng cách từ chỗ Ghost dự định bước tới Pacman
        best_move = valid_moves[0]
        max_safety_score = -1
        threat = self.last_known_enemy_pos

        for move in valid_moves:
            nr, nc = my_position[0] + move.value[0], my_position[1] + move.value[1]
            next_pos = (nr, nc)
            
            # Đo khoảng cách bằng BFS trên bản đồ ảo (coi sương mù là đường trống)
            dist_to_threat = self._bfs_distance(next_pos, threat)
            
            # Tính số đường thoát từ ô tiếp theo (đo độ rộng rãi)
            freedom = self._count_freedom(next_pos)
            
            # Phạt nặng ngõ cụt
            penalty = -1000 if freedom <= 1 else 0
            
            # Điểm: Khoảng cách là quan trọng nhất, sau đó là sự rộng rãi
            score = dist_to_threat * 10 + freedom + penalty
            
            if score > max_safety_score:
                max_safety_score = score
                best_move = move
                
        return best_move

    def _bfs_distance(self, start, target):
        if start == target: return 0
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            curr, dist = queue.popleft()
            if curr == target: return dist
            if dist > 10: break # Tối ưu tốc độ: không cần quét quá xa
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = curr[0] + m.value[0], curr[1] + m.value[1]
                if 0 <= nr < 21 and 0 <= nc < 21 and self.internal_map[nr, nc] in (0, -1):
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), dist + 1))
        return 99 # Xa vô cực

    def _count_freedom(self, start, depth=3):
        # Đo số ô có thể đi tới trong 3 bước
        queue = deque([(start, 0)])
        visited = {start}
        count = 0
        while queue:
            curr, d = queue.popleft()
            count += 1
            if d < depth:
                for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    nr, nc = curr[0] + m.value[0], curr[1] + m.value[1]
                    if 0 <= nr < 21 and 0 <= nc < 21 and self.internal_map[nr, nc] in (0, -1):
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append(((nr, nc), d + 1))
        return count

    def _count_fog(self, pos, move):
        # Đếm sương mù theo một hướng cố định (nhìn xa 3 ô)
        count = 0
        curr = pos
        for _ in range(3):
            nr, nc = curr[0] + move.value[0], curr[1] + move.value[1]
            if 0 <= nr < 21 and 0 <= nc < 21 and self.internal_map[nr, nc] != 1:
                if self.internal_map[nr, nc] == -1: count += 1
                curr = (nr, nc)
            else: break
        return count
