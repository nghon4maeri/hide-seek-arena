import sys
from pathlib import Path
from collections import deque
import numpy as np

# Thêm đường dẫn thư mục src để import module bài tập
src_path = Path(__file__).parent.parent.parent / "src"
if not (str(src_path) in sys.path):
    sys.path.insert(0, str(str(src_path)))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    """
    Pacman Agent: Xử lý dò đường và tìm bắt Ghost trong sương mù.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "24127053 Pacman"

        # Tốc độ di chuyển tối đa là 2 ô / lượt
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))

        # Ma trận lưu xác suất vị trí của Ghost
        self.belief = None

        # Tập hợp các ô đã từng nhìn thấy
        self.global_visited = set()

        # Cache lưu tầm nhìn và các ô kề để đỡ phải tính lại nhiều lần
        self.los_cache = {}
        self.neighbors_cache = {}

    def _reset_episode(self, map_state: np.ndarray):
        """Khởi tạo lại dữ liệu khi bắt đầu trận mới."""
        h, w = map_state.shape
        self.belief = np.zeros((h, w), dtype=float)

        # Chưa biết Ghost ở đâu nên chia đều xác suất cho các ô đi được
        valid_mask = map_state < 1
        num_valid = np.sum(valid_mask)
        if num_valid > 0:
            self.belief[valid_mask] = 1.0 / num_valid
        self.global_visited = set()

        # Pre-compute tầm nhìn và ô kề cho toàn bộ map
        self.los_cache.clear()
        self.neighbors_cache.clear()
        for r in range(h):
            for c in range(w):
                if map_state[r, c] < 1:
                    pos = (r, c)
                    # Cache tầm nhìn chữ thập bán kính 5 ô
                    self.los_cache[pos] = self._compute_visible_cells(pos, map_state, radius=5)

                    # Cache các ô đi được xung quanh
                    neighbors = [pos]
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] < 1:
                            neighbors.append((nr, nc))
                    self.neighbors_cache[pos] = neighbors

    def _compute_visible_cells(self, my_pos: tuple, map_state: np.ndarray, radius: int = 5) -> set:
        """Tầm nhìn chữ thập 4 hướng (tối đa 5 ô), bị cản bởi tường."""
        h, w = map_state.shape
        visible = {my_pos}
        r_start, c_start = my_pos

        # Duyệt 4 hướng: Lên, Xuống, Trái, Phải
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, radius + 1):
                nr, nc = r_start + dr * dist, c_start + dc * dist
                if 0 <= nr < h and 0 <= nc < w:
                    # Đụng tường (giá trị 1) thì dừng tầm nhìn
                    if map_state[nr, nc] == 1:
                        break
                    visible.add((nr, nc))
                else:
                    break
        return visible

    def _get_visible_cells(self, my_pos: tuple, map_state: np.ndarray) -> set:
        """Lấy tầm nhìn từ cache ra dùng cho nhanh."""
        if my_pos in self.los_cache:
            return self.los_cache[my_pos]
        return self._compute_visible_cells(my_pos, map_state, radius=5)

    def _update_belief_state(self, map_state: np.ndarray, my_pos: tuple, enemy_pos: tuple):
        """Cập nhật ma trận xác suất vị trí Ghost theo thuật toán Bayes-Markov."""
        h, w = map_state.shape
        visible_cells = self._get_visible_cells(my_pos, map_state)
        self.global_visited.update(visible_cells)

        # Nếu nhìn thấy Ghost thì chốt luôn vị trí 100%
        if enemy_pos is not None:
            self.belief.fill(0.0)
            self.belief[enemy_pos] = 1.0
            return

        # Bước dự đoán (Markov): Lan truyền xác suất sang các ô kề
        new_belief = np.zeros_like(self.belief)
        nonzero_coords = np.argwhere(self.belief > 1e-6)

        for r, c in nonzero_coords:
            p_val = self.belief[r, c]
            neighbors = self.neighbors_cache.get((r, c), [(r, c)])
            p_split = p_val / len(neighbors)
            for nr, nc in neighbors:
                new_belief[nr, nc] += p_split

        # Bước cập nhật (Bayes): Nhìn không thấy thì ô đó bằng 0
        for r, c in visible_cells:
            new_belief[r, c] = 0.0

        # Chuẩn hóa lại tổng xác suất = 1
        total_p = np.sum(new_belief)
        if total_p > 1e-8:
            self.belief = new_belief / total_p
        else:
            # Nếu mất dấu hoàn toàn thì phân bố đều vào các ô chưa đi qua
            self.belief.fill(0.0)
            unvisited = [(r, c) for r in range(h) for c in range(w)
                         if map_state[r, c] < 1 and (r, c) not in self.global_visited]
            if unvisited:
                for r, c in unvisited:
                    self.belief[r, c] = 1.0 / len(unvisited)
            else:
                valid = [(r, c) for r in range(h) for c in range(w) if map_state[r, c] < 1]
                if valid:
                    for r, c in valid:
                        self.belief[r, c] = 1.0 / len(valid)

    def _bfs_turn_distances(self, start: tuple, map_state: np.ndarray) -> dict:
        """Đếm số lượt ít nhất để đi tới các ô (tính cả nước di chuyển 2 ô)."""
        dist = {start: 0}
        queue = deque([start])
        h, w = map_state.shape

        while queue:
            curr = queue.popleft()
            d = dist[curr]

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                r, c = curr
                # Thử đi 2 ô trước, nếu vướng tường thì thử đi 1 ô
                for steps in range(self.pacman_speed, 0, -1):
                    if steps == 2:
                        mid_r, mid_c = r + dr, c + dc
                        if not (0 <= mid_r < h and 0 <= mid_c < w and map_state[mid_r, mid_c] < 1):
                            continue

                    nr, nc = r + dr * steps, c + dc * steps
                    if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] < 1:
                        nxt = (nr, nc)
                        if not (nxt in dist):
                            dist[nxt] = d + 1
                            queue.append(nxt)
        return dist

    def _bfs_path_to_target(self, start: tuple, target: tuple, map_state: np.ndarray):
        """Tìm đường ngắn nhất tới target bằng BFS (ưu tiên đi 2 ô)."""
        if start == target:
            return None

        h, w = map_state.shape
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            curr, path = queue.popleft()

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                r, c = curr

                # Ưu tiên nhảy 2 ô
                for steps in range(self.pacman_speed, 0, -1):
                    if steps == 2:
                        mid_r, mid_c = r + dr, c + dc
                        if not (0 <= mid_r < h and 0 <= mid_c < w and map_state[mid_r, mid_c] < 1):
                            continue

                    nr, nc = r + dr * steps, c + dc * steps
                    if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] < 1:
                        nxt = (nr, nc)
                        new_path = path + [(move, steps)]

                        if nxt == target:
                            return new_path

                        if not (nxt in visited):
                            visited.add(nxt)
                            queue.append((nxt, new_path))
        return None

    def _get_max_valid_move(self, my_pos: tuple, map_state: np.ndarray):
        """Nước đi mặc định: Chọn hướng đi được xa nhất."""
        h, w = map_state.shape
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            steps = 0
            r, c = my_pos
            for _ in range(self.pacman_speed):
                r += dr
                c += dc
                if 0 <= r < h and 0 <= c < w and map_state[r, c] < 1:
                    steps += 1
                else:
                    break
            if steps > 0:
                return (move, steps)
        return (Move.STAY, 1)

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        # Lượt đầu thì reset dữ liệu
        if step_number == 1 or self.belief is None or self.belief.shape != map_state.shape:
            self._reset_episode(map_state)

        self._update_belief_state(map_state, my_position, enemy_position)

        # 1. BẮT TRỰC TIẾP: Nếu Ghost ở cùng hàng/cột và không bị cản tường
        if enemy_position is not None:
            r_diff = enemy_position[0] - my_position[0]
            c_diff = enemy_position[1] - my_position[1]

            # Cùng hàng
            if r_diff == 0 and not (c_diff == 0):
                step_dir = 1 if c_diff > 0 else -1
                blocked = False
                for c_step in range(1, abs(c_diff)):
                    if map_state[my_position[0], my_position[1] + c_step * step_dir] == 1:
                        blocked = True
                        break
                if not blocked:
                    move = Move.RIGHT if c_diff > 0 else Move.LEFT
                    actual_steps = min(self.pacman_speed, abs(c_diff))
                    return (move, actual_steps)

            # Cùng cột
            if c_diff == 0 and not (r_diff == 0):
                step_dir = 1 if r_diff > 0 else -1
                blocked = False
                for r_step in range(1, abs(r_diff)):
                    if map_state[my_position[0] + r_step * step_dir, my_position[1]] == 1:
                        blocked = True
                        break
                if not blocked:
                    move = Move.DOWN if r_diff > 0 else Move.UP
                    actual_steps = min(self.pacman_speed, abs(r_diff))
                    return (move, actual_steps)

            # Nếu không thẳng hàng thì đi BFS tìm đường ngắn nhất
            path = self._bfs_path_to_target(my_position, enemy_position, map_state)
            if path:
                return path[0]

        # 2. MỞ SƯƠNG MÙ: Chọn vị trí giúp soi được nhiều ô có xác suất cao nhất
        turn_dists = self._bfs_turn_distances(my_position, map_state)
        best_score = -1.0
        best_target = None

        candidate_coords = np.argwhere(self.belief > 1e-4)

        for r, c in candidate_coords:
            target_pos = (int(r), int(c))
            if target_pos == my_position or not (target_pos in turn_dists):
                continue

            turns = turn_dists[target_pos]
            target_los = self._get_visible_cells(target_pos, map_state)
            visible_p = sum(self.belief[tr, tc] for tr, tc in target_los)

            score = visible_p / (turns ** 1.1 + 0.05)
            if score > best_score:
                best_score = score
                best_target = target_pos

        if best_target is not None:
            path = self._bfs_path_to_target(my_position, best_target, map_state)
            if path:
                return path[0]

        return self._get_max_valid_move(my_position, map_state)


class GhostAgent(BaseGhostAgent):
    """
    Ghost Agent: Tìm cách né Pacman và sống sót càng lâu càng tốt.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "24127053 Ghost"
        self.last_seen_pacman = None
        self.threat_turns = 0

    def _reset_episode(self):
        """Reset vị trí Pacman khi bắt đầu trận mới."""
        self.last_seen_pacman = None
        self.threat_turns = 0

    def _compute_pacman_speed2_dist_map(self, start_pos: tuple, map_state: np.ndarray) -> dict:
        """Loang BFS tính số lượt Pacman (đi 2 ô) cần để tới các ô trên bản đồ."""
        if start_pos is None:
            return {}

        h, w = map_state.shape
        dist_map = {start_pos: 0}
        queue = deque([start_pos])

        while queue:
            curr = queue.popleft()
            d = dist_map[curr]

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                r, c = curr
                for s in range(1, 3):
                    nr, nc = r + dr * s, c + dc * s
                    if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] < 1:
                        nxt = (nr, nc)
                        if not (nxt in dist_map):
                            dist_map[nxt] = d + 1
                            queue.append(nxt)
                    else:
                        break
        return dist_map

    def _is_in_line_of_sight(self, pos1: tuple, pos2: tuple, map_state: np.ndarray, radius: int = 5) -> bool:
        """Kiểm tra xem vị trí có nằm trong tầm nhìn thẳng của Pacman hay không."""
        r1, c1 = pos1
        r2, c2 = pos2
        if not (r1 == r2) and not (c1 == c2):
            return False

        dist = abs(r1 - r2) + abs(c1 - c2)
        if dist > radius:
            return False

        dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
        dc = 0 if c1 == c2 else (1 if c2 > c1 else -1)

        curr_r, curr_c = r1 + dr, c1 + dc
        while not ((curr_r, curr_c) == (r2, c2)):
            if map_state[curr_r, curr_c] == 1:
                return False
            curr_r += dr
            curr_c += dc
        return True

    def _count_exits(self, pos: tuple, map_state: np.ndarray) -> int:
        """Đếm số đường đi xung quanh ô này."""
        exits = 0
        h, w = map_state.shape
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] < 1:
                exits += 1
        return exits

    def _evaluate_position(self, pos: tuple, threat_pos: tuple, dist_map: dict, map_state: np.ndarray) -> float:
        """Tính điểm an toàn cho nước đi (điểm càng cao càng an toàn)."""
        exits = self._count_exits(pos, map_state)

        # Chưa thấy Pacman thì chọn ô rộng có >= 2 lối thoát
        if threat_pos is None:
            return 20.0 if exits >= 2 else 0.0

        p_turns = dist_map.get(pos, 999)

        # Pacman tới được trong 1 lượt thì bỏ qua ngay
        if p_turns <= 1:
            return -99999.0

        in_los = self._is_in_line_of_sight(threat_pos, pos, map_state, radius=5)
        decay = max(0.2, 1.0 - (self.threat_turns * 0.05))

        # Trừ điểm nặng nếu lộ diện, cộng điểm nếu nấp kín
        los_penalty = (-200.0 * decay) if in_los else (50.0 * decay)

        # Né ngõ cụt khi Pacman ở gần
        dead_end_penalty = (-500.0 * decay) if (exits <= 1 and p_turns < 6) else 0.0

        # Ô càng nhiều lối ra càng tốt
        exit_bonus = exits * 15.0

        return (p_turns * 30.0 * decay) + los_penalty + exit_bonus + dead_end_penalty

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        if step_number == 1:
            self._reset_episode()

        if enemy_position is not None:
            self.last_seen_pacman = enemy_position
            self.threat_turns = 0
        else:
            self.threat_turns += 1

        threat = enemy_position or self.last_seen_pacman
        dist_map = self._compute_pacman_speed2_dist_map(threat, map_state) if threat else {}

        h, w = map_state.shape
        valid_moves = []

        # Nước đi hợp lệ (tính cả đứng yên STAY)
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            dr, dc = move.value
            nr, nc = my_position[0] + dr, my_position[1] + dc

            if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] < 1:
                valid_moves.append((move, (nr, nc)))

        if not valid_moves:
            return Move.STAY

        best_move = Move.STAY
        best_score = float('-inf')

        # Thử từng nước đi và chọn ô an toàn nhất
        for move, new_pos in valid_moves:
            score = self._evaluate_position(new_pos, threat, dist_map, map_state)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move