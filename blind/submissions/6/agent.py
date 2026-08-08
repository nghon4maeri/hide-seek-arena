

import sys
import time
from collections import Counter, deque
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

_DIRS = ((Move.UP, -1, 0), (Move.DOWN, 1, 0), (Move.LEFT, 0, -1), (Move.RIGHT, 0, 1))
_WIN = 1_000_000      # điểm thắng tuyệt đối trong alpha-beta
_UNREACH = 30_000     # khoảng cách "không tới được"


class _TimeUp(Exception):
    """Hết ngân sách thời gian tìm kiếm trong một lượt"""


class PacmanAgent(BasePacmanAgent):
    """
    Chiến thuật Pacman: Lọc Bayes tìm dấu vết kết hợp Alpha-Beta để chặn đường

    1 Ghi nhớ bản đồ (_init_map, _refresh_map)
      - Lưu lại vị trí tường qua từng lượt đi và mặc định các vùng sương mù (-1) là đường có thể đi
      - Hệ thống sẽ lập tức xóa bộ nhớ đệm khoảng cách (cache BFS) để tính toán lại mỗi khi phát hiện thêm một bức tường mới

    2 Bộ lọc Bayes dự đoán vị trí Ghost (_update_belief)
      - Dự báoo: Phân tán xác suất vị trí của Ghost ra các ô xung quanh, đặc biệt tăng trọng số cho hướng bỏ chạy nếu Ghost đang ở gần Pacman
      - Cập nhật: Điều chỉnh xác suất dựa trên tầmm nhìn thực tế — tập trung 100% vào ô thấy Ghost, xóa bỏ xác suất ở các ô trống trong tầm nhìn và reset những vị trí mâu thuẫn dữ liệu

    3 Chế độ truy đuổi (_chase)
      - Sử dụng cây Minimax luân phiên kết hợp Alpha-Beta và Iterative Deepening để nhìn trước từ 2 đến 12 bước với ngân sách tính toán chỉ 0,4 giây
      - Áp dụng "Move ordering" để ưu tiên quét trước các nước đi hướng về phía đối thủ giúp cắt tỉa nhánh nhanh hơn
      - Hàm lượng giá ưu tiên các trạng thái áp sátt và ép gót Ghost triệt để
      - Có thể kích hoạt ngay cả khi vừa mất dấu Ghost nếu xác suất dự đoán vị trí hiện tại đủ cao (trên 30%)

    4 Định tuyến BFS trên lưới (_dist_from, _turn_bfs)
      - Sử dụng BFS để đo khoảng cách thực tế (có né tường) thay vì dùng đường chim bay, giúp hàm lượng giá đánh giá chính xác hơn
      - Tích hợp khả năng đi 2 bước/lượt của Pacman vào thuật toán tìm đường để tính toán chính xác chi phí di chuyển

    5 Chế độ săn lùng khi mất dấu (_hunt)
      - Chấm điểm các khu vực trên bản đồ dựa trên tỷ lệ giữa xác suất có Ghost và thời gian di chuyển đến đó
      - Sử dụngg cơ chế quán tính để giữ vững mục tiêu (chỉ đổi mục tiêu nếu điểm chênh lệch đủ lớn), tránh tình trạng Pacman ngập ngừng đổi hướng liên tục
      - Tự động bẻ lái sang vùng sương mù gần nhất để khám phá nếu không còn bất cứ manh mối nào


    6 Đảm bảo an toàn thời gian chạy
      - Thiết lập cơ chế khẩn cấp (_panic) để bắt các lỗi ngoại lệ hoặc quá giờ, đảm bảo Pacman luôn tung ra một nước đi hợp lệ trước giới hạn 1 giây của hệ thống
    """

    def __init__(self, **kwargs):
        
        self.speed = max(1, int(kwargs.get("pacman_speed", 2)))
        # Luật blind: bắt được khi khoảng cách Manhattan sau khi đi < 2.
        self.capture = max(1, int(kwargs.get("capture_distance", 2)))
        self.belief = None
        self.prev_pos = None
        self.hunt_target = None
        self.time_budget = 0.4  # số giây dành cho search (giới hạn chấm: 1s)
        self._nodes = 0
        self._deadline = 0.0
        
        
        # Bản đồ tĩnh + cache (gộp từ class _Board cũ vào trong agent)
        self.h = 0
        self.w = 0
        self.wall = None   # mảng bool cùng kích thước bản đồ: True = tường
        self._dist = {}    # cache trường khoảng cách BFS theo từng ô nguồn
        self._nbrs = {}    # cache danh sách ô kề đi được của từng ô

    # tiện ích bản đồ
    def _init_map(self, map_state):
        arr = np.asarray(map_state)
        self.h, self.w = arr.shape
        self.wall = (arr == 1)
        self._dist = {}
        self._nbrs = {}


    def _refresh_map(self, map_state):
        # Tường chỉ có thể được phát hiện thêm; nếu có tường mới thì cache
        # khoảng cách và ô kề không còn đúng nên phải xoá
        new_wall = self.wall | (np.asarray(map_state) == 1)
        
        if (new_wall != self.wall).any():
            self.wall = new_wall
            self._dist = {}
            self._nbrs = {}


    def _free(self, r, c):
        
        # Ô nằm trong bản đồ và không phải tường thì đi được
        return 0 <= r < self.h and 0 <= c < self.w and not self.wall[r, c]


    def _neighbors(self, cell):
        cached = self._nbrs.get(cell)
        
        
        if cached is None:
            r, c = cell
            cached = [(r + dr, c + dc) for _, dr, dc in _DIRS if self._free(r + dr, c + dc)]
            self._nbrs[cell] = cached
            
        return cached



    def _dist_from(self, src):
        
        """Trường khoảng cách đường đi ngắn nhất thật (1 ô / bước) từ src"""
        src = (int(src[0]), int(src[1]))
        cached = self._dist.get(src)
        
        if cached is not None:
            
            return cached
        
        dist = np.full((self.h, self.w), _UNREACH, dtype=np.int32)
        
        
        if self._free(*src):
            dist[src] = 0
            dq = deque([src])
            
            while dq:
                
                cell = dq.popleft()
                nd = dist[cell] + 1
                
                for nb in self._neighbors(cell):
                    
                    if dist[nb] > nd:
                        dist[nb] = nd
                        dq.append(nb)
                        
        self._dist[src] = dist
        
        
        return dist

    # step
    def step(self, map_state, my_position, enemy_position, step_number):
        t0 = time.perf_counter()
        my = (int(my_position[0]), int(my_position[1]))
        enemy = None
        
        if enemy_position is not None:
            enemy = (int(enemy_position[0]), int(enemy_position[1]))
            
        try:
            return self._step(np.asarray(map_state), my, enemy, step_number, t0)
        
        except Exception:
            
            return self._panic(map_state, my)



    def _step(self, map_state, my, enemy, step_number, t0):
        
        if self.wall is None:
            
            self._init_map(map_state)
            
        else:
            self._refresh_map(map_state)
            
            
        # Tự hiệu chỉnh ngưỡng bắt: vị trí nhận được là vị trí sau khi cả hai
        # đã đi, nên nếu ta đang trong tầm bắt giả định mà ván vẫn tiếp diễn
        # thì ngưỡng thật phải nhỏ hơn giá trị đang giả định
        if enemy is not None and step_number > 1:
            
            d = abs(my[0] - enemy[0]) + abs(my[1] - enemy[1])
            
            
            if d < self.capture:
                self.capture = max(1, d)
                
        self._update_belief(map_state, my, enemy)
        
        self.prev_pos = my
        
        if enemy is not None:
            
            self.hunt_target = None
            
            return self._chase(my, enemy, t0)
        
        
        # Ghost vừa khuất tầm nhìn: nếu belief còn tập trung thì chặn ngay
        # ô khả dĩ nhất bằng search đối kháng đầy đủ
        peak = np.unravel_index(int(np.argmax(self.belief)), self.belief.shape)
        peak = (int(peak[0]), int(peak[1]))
        
        
        if self.belief[peak] >= 0.3 and peak != my:
            
            self.hunt_target = None
            return self._chase(my, peak, t0)
        
        
        return self._hunt(my, map_state)


    # belief
    def _update_belief(self, map_state, my, enemy):
        
        if self.belief is None:
            bel = np.zeros((self.h, self.w))
            
            if enemy is not None:
                bel[enemy] = 1.0
                
            else:
                bel[map_state == -1] = 1.0
                
                if bel.sum() <= 0:  # trường hợp biên: chế độ nhìn toàn bản đồ
                    bel[map_state == 0] = 1.0
                    bel[my] = 0.0
                    
            self.belief = bel / max(bel.sum(), 1e-12)
            return

        # Khuếch tán: Ghost đã đi đúng một bước kể từ lần quan sát trước.
        old = self.belief
        new = np.zeros_like(old)
        flee = self._dist_from(self.prev_pos) if self.prev_pos is not None else None
        
        for r, c in np.argwhere(old > 1e-12):
            cell = (int(r), int(c))
            p = old[cell]
            opts = [cell] + self._neighbors(cell)
            
            
            if flee is not None and flee[cell] <= 7:
                # Ghost ở gần nhiều khả năng đã biết ta ở đâu và đang bỏ chạy.
                d0 = flee[cell]
                ws = [2.0 if flee[o] > d0 else (1.0 if flee[o] == d0 else 0.5) for o in opts]
            else:
                ws = [1.0] * len(opts)
                
            tot = sum(ws)
            
            for o, wgt in zip(opts, ws):
                new[o] += p * wgt / tot

        # Điều kiện hoá theo quan sát của lượt này.
        if enemy is not None:
            new[:] = 0.0
            new[enemy] = 1.0
            
        else:
            new[map_state == 0] = 0.0  # ô trống đang thấy: Ghost không ở đó
            new[self.wall] = 0.0
            
            if new.sum() <= 1e-12:  # belief mâu thuẫn quan sát -> đặt lại
                new[map_state == -1] = 1.0
                new[self.wall] = 0.0
                
                
                if new.sum() <= 1e-12:
                    new[map_state == 0] = 1.0
                    new[my] = 0.0
                    
        self.belief = new / max(new.sum(), 1e-12)

    # chase
    def _chase(self, my, ghost, t0):
        self._deadline = t0 + self.time_budget
        self._nodes = 0
        best = self._greedy_towards(my, ghost)
        
        try:
            for rounds in range(2, 13):
                val, act = self._root(my, ghost, rounds)
                
                
                if act is not None:
                    best = act
                    
                if val >= _WIN:  # đã tìm thấy nước bắt chắc chắn
                    break
                
        except _TimeUp:
            pass
        
        return best

    def _pac_opts(self, p):
        # Mọi điểm đến hợp lệ của Pacman trong một lượt (kể cả đứng yên).
        opts = [(p, Move.STAY)]
        
        for mv, dr, dc in _DIRS:
            r, c = p
            
            for s in range(1, self.speed + 1):
                r += dr
                c += dc
                
                if not self._free(r, c):
                    break
                opts.append(((r, c), (mv, s)))
                
        return opts

    def _root(self, p, g, rounds):
        dist_g = self._dist_from(g)
        opts = sorted(self._pac_opts(p), key=lambda o: dist_g[o[0]])
        alpha, best_act = -float("inf"), None
        
        
        for new_p, act in opts:
            val = self._ghost_node(new_p, g, rounds, alpha, float("inf"))
            if val > alpha:
                alpha, best_act = val, act
                
                
        return alpha, best_act

    def _pac_node(self, p, g, rounds, alpha, beta):
        dist_g = self._dist_from(g)
        opts = sorted(self._pac_opts(p), key=lambda o: dist_g[o[0]])
        
        for new_p, _ in opts:
            val = self._ghost_node(new_p, g, rounds, alpha, beta)
            
            
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
            
            
        return alpha

    def _ghost_node(self, p, g, rounds, alpha, beta):
        self._nodes += 1
        
        
        if (self._nodes & 127) == 0 and time.perf_counter() > self._deadline:
            raise _TimeUp()
        
        dist_p = self._dist_from(p)
        gopts = sorted([g] + self._neighbors(g), key=lambda o: dist_p[o], reverse=True)
        best = float("inf")
        
        for ng in gopts:
            
            if abs(p[0] - ng[0]) + abs(p[1] - ng[1]) < self.capture:
                val = _WIN + rounds  # bắt được; rounds còn lớn = bắt càng sớm
                
            elif rounds <= 1:
                val = self._eval(p, ng)
                
            else:
                val = self._pac_node(p, ng, rounds - 1, alpha, beta)
                
            if val < best:
                best = val
                
            if best < beta:
                beta = best
                
            if beta <= alpha:
                break
            
        return best
    

    def _eval(self, p, g):
        # Lá: cànng gần Ghost càng tốt, Ghost càng ít lối thoát càng tốt
        return -(4 * int(self._dist_from(g)[p]) + len(self._neighbors(g)))


    def _greedy_towards(self, my, target):
        dist, acts = self._turn_bfs(my)
        
        
        if dist[target] < _UNREACH and target in acts:
            return acts[target]
        
        
        return self._panic(None, my)

    # hunt
    def _hunt(self, my, map_state):
        dist, acts = self._turn_bfs(my)
        bel = self.belief
        
        # Làm mượt nhẹ để khối xác suất nằm sát một ô vẫn hút ta về phía đó
        sm = bel.copy()
        sm[1:, :] += 0.25 * bel[:-1, :]
        sm[:-1, :] += 0.25 * bel[1:, :]
        sm[:, 1:] += 0.25 * bel[:, :-1]
        sm[:, :-1] += 0.25 * bel[:, 1:]
        score = sm / (1.0 + dist)
        score[dist >= _UNREACH] = 0.0
        score[my] = 0.0
        target = np.unravel_index(int(np.argmax(score)), score.shape)
        target = (int(target[0]), int(target[1]))
        
        # Giữ quán tính: bám mục tiêu cũ khi nó vẫn đáng giá, tránh dao động
        # qua lại giữa hai cụm xác suất ở xa nhau
        prev = self.hunt_target
        
        
        if (prev is not None and prev != my and prev in acts
                and score[prev] >= 0.6 * score[target]):
            target = prev
            
            
        if score[target] > 0 and target in acts:
            self.hunt_target = target
            return acts[target]
        
        self.hunt_target = None
        
        # Không còn khối xác suất nào tới được: quét ô chưa thấy gần nhất
        unseen = np.argwhere(np.asarray(map_state) == -1)
        best, best_d = None, _UNREACH
        
        
        for r, c in unseen:
            cell = (int(r), int(c))
            
            if cell in acts and dist[cell] < best_d:
                best, best_d = cell, dist[cell]
                
        if best is not None:
            return acts[best]
        
        
        return self._panic(map_state, my)

    def _turn_bfs(self, start):
        """BFS theo lượt có xét tốc độ chạy thẳng. Trả về lưới khoảng cách và
        hành động đầu tiên cần đi cho từng ô tới được"""
        dist = np.full((self.h, self.w), _UNREACH, dtype=np.int32)
        dist[start] = 0
        acts = {}
        dq = deque([start])
        
        
        while dq:
            cell = dq.popleft()
            base = acts.get(cell)
            nd = dist[cell] + 1
            
            
            for mv, dr, dc in _DIRS:
                r, c = cell
                
                
                for s in range(1, self.speed + 1):
                    r += dr
                    c += dc
                    
                    if not self._free(r, c):
                        break
                    nb = (r, c)
                    
                    if dist[nb] > nd:
                        dist[nb] = nd
                        acts[nb] = base if base is not None else (mv, s)
                        dq.append(nb)
                        
                        
        return dist, acts



    def _panic(self, _map_state, my):
        # Nước đi an toàn cuối cùng khi mọi bước xử lý khác thất bại.
        if self.wall is not None:
            
            
            for mv, dr, dc in _DIRS:
                
                if self._free(my[0] + dr, my[1] + dc):
                    return (mv, 1)
                
                
        return Move.STAY

class _SearchTimeout(Exception):
    """Internal exception used to return a safe move before the time limit."""

class GhostAgent(BaseGhostAgent):
    """
    Ghost kết hợp ba lớp chiến thuật:
    1. Hiding:
       Tự chấm điểm toàn bộ ô có thể tới được và chọn nơi ít lộ, không phải
       ngõ cụt
    2. Belief state:
       Khi Pacman khuất tầm nhìn, lưu tập vị trí khả dĩ rồi mở rộng tập đó theo
       đúng luật Pacman đi thẳng tối đa hai ô
    3. Minimax + alpha-beta + iterative deepening:
       Ghost tối đa hóa khả năng sống; Pacman tối thiểu hóa điểm của Ghost
       Chỉ cập nhật kết quả sau khi hoàn tất trọn vẹn một độ sâu
    """

    # Luật mặc định trong arena.py
    PACMAN_SPEED = 2
    CAPTURE_DISTANCE = 2       # Bị bắt khi Manhattan < 2
    VISION_RANGE = 5

    # 1 giây/lượt trong Arena; chỉ dùng một phần nhỏ để có biên an toàn
    TIME_BUDGET = 0.18
    MAX_DEPTH = 3
    MAX_BELIEF_SAMPLES = 10     # vị trí ghost cho rằng pacman có thẻ đang đứng khi pacman ngoài tầm nhìn
    MAX_DISTANCE_CACHE = 128

    # Chỉ chạy minimax theo belief khi thông tin còn tương đối mới. Sáu lượt
    # tương ứng Pacman có thể đi xa tối đa 12 ô kể từ lần cuối được nhìn thấy
    ACTIVE_BELIEF_TURNS = 6

    # Nếu đã tới target nhưng vị trí đó không thật sự phù hợp để camp, Ghost
    # chỉ chờ một vài lượt rồi đánh giá và chọn target khác. Ở một camp tốt,
    # Ghost vẫn được phép đứng yên lâu dài để tránh tự làm lộ vị trí
    MAX_UNSAFE_CAMP_TURNS = 6

    CARDINAL_MOVES = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
    GHOST_MOVES = (
        Move.UP,
        Move.DOWN,
        Move.LEFT,
        Move.RIGHT,
        Move.STAY,
    )

    CAPTURE_SCORE = 1_000_000.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Belief Minimax Ghost StableCamp"

        self.pacman_speed = max(
            1, int(kwargs.get("pacman_speed", self.PACMAN_SPEED))
        )
        self.capture_distance = max(
            1,
            int(
                kwargs.get(
                    "capture_distance",
                    self.CAPTURE_DISTANCE,
                )
            ),
        )
        # Nhóm lịch sử di chuyển
        self.history = deque(maxlen=12)
        self.visit_count = Counter()

        # Nhóm belief về pacman
        self.belief = set() # tập các vị trí có thể pacman đang đứng
        self.ever_seen_pacman = False
        self.last_seen_pacman = None        # -> giúp chọn belief quanh vị trí cuối
        self.turns_since_seen = 0       # -> xác định belief còn đủ mới không

        self.hide_target = None
        self.camp_turns = 0
        self.target_generation = 0

        self._shape = None
        self._wall_mask = None
        self._walkable = None
        self._positions = []
        self._neighbors = {}
        self._degree = {}
        self._exposure = {}

        self._distance_cache = {}   # lưu dist map đã chạy 
        self._deadline = 0.0
        self._node_count = 0
        self.debug = bool(kwargs.get("debug", False))

    
    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:
        self._deadline = time.perf_counter() + self.TIME_BUDGET
        self._node_count = 0
        ghost = self._to_pos(my_position)
        pacman = self._to_pos(enemy_position)

        self._ensure_topology(map_state)

        # Fallback được tính trước search. Nếu minimax hết giờ, agent vẫn trả một Move hợp lệ và có ý nghĩa
        fallback = self._greedy_fallback(ghost, pacman)

        # Fallback cũng không được đứng yên trước Pacman ở gần nếu vẫn còn
        # một nước di chuyển chắc chắn an toàn. Nhờ vậy nhánh timeout/exception
        # không làm mất tác dụng của visible escape override phía dưới
        if (
            pacman is not None
            and fallback == Move.STAY
            and self._manhattan(ghost, pacman) <= self.VISION_RANGE
        ):
            fallback_escape = self._best_safe_non_stay_move(
                ghost,
                pacman,
            )
            if fallback_escape is not None:
                fallback = fallback_escape

        self.history.append(ghost)
        self.visit_count[ghost] += 1

        try:
            self._update_belief(
                map_state=map_state,
                ghost=ghost,
                visible_pacman=pacman,
            )

            if pacman is not None:
                self.hide_target = None
                self.camp_turns = 0

            # Nếu đã mất dấu Pacman đủ lâu và đang ở một vị trí ẩn tốt,
            # tiếp tục camp tại đây. Không tự rời chỗ chỉ vì hết timer
            if (
                pacman is None
                and self.turns_since_seen > self.ACTIVE_BELIEF_TURNS
                and self._is_good_camp_position(ghost)
            ):
                self.hide_target = ghost
                self.camp_turns += 1
                return Move.STAY

            use_belief_minimax = (
                self.ever_seen_pacman
                and self.turns_since_seen <= self.ACTIVE_BELIEF_TURNS
                and len(self.belief)
                <= max(1, int(0.80 * len(self._positions)))
            )

            if pacman is not None:
                root_hypotheses = [pacman]
            elif use_belief_minimax:
                root_hypotheses = self._select_belief_samples(ghost)
            else:
                root_hypotheses = []

            if root_hypotheses:
                # Greedy một-lượt trên đúng các giả thuyết hiện tại là fallback
                # tốt hơn fallback ban đầu khi Pacman đang bị khuất
                fallback = self._robust_one_round_move(
                    ghost,
                    root_hypotheses,
                )

                avoid_visible_stay = (
                    pacman is not None
                    and self._manhattan(ghost, pacman)
                    <= self.VISION_RANGE
                )

                # Không để fallback bị đổi ngược về STAY trước Pacman đang
                # nhìn thấy nếu vẫn tồn tại một nước di chuyển chắc chắn an
                # toàn trong vòng kế tiếp
                if avoid_visible_stay and fallback == Move.STAY:
                    fallback_escape = self._best_safe_non_stay_move(
                        ghost,
                        pacman,
                    )
                    if fallback_escape is not None:
                        fallback = fallback_escape

                chosen = self._iterative_minimax(
                    ghost=ghost,
                    pacman_hypotheses=root_hypotheses,
                    fallback=fallback,
                    avoid_root_stay=avoid_visible_stay,
                )
            else:
                chosen = self._choose_hiding_move(
                    ghost=ghost,
                    step_number=step_number,
                )
            if (
                pacman is not None
                and chosen == Move.STAY
                and self._manhattan(ghost, pacman) <= self.VISION_RANGE
            ):
                moving_escape = self._best_safe_non_stay_move(
                    ghost,
                    pacman,
                )

                if moving_escape is not None:
                    chosen = moving_escape

            if chosen is None:
                return fallback

            next_ghost = self._next_position(ghost, chosen)
            if not self._is_walkable(next_ghost):
                return fallback

            return chosen

        except _SearchTimeout:
            return fallback
        except Exception:
            if self.debug:
                raise
            return fallback

    
    # Topology: -1 và 0 đều đi được; chỉ 1 là tường
    def _ensure_topology(self, map_state):
        wall_mask = np.asarray(map_state == 1, dtype=bool)

        if (
            self._shape == tuple(map_state.shape)
            and self._wall_mask is not None
            and np.array_equal(self._wall_mask, wall_mask)
        ):
            return

        self._shape = tuple(map_state.shape)
        self._wall_mask = wall_mask.copy()
        self._walkable = ~self._wall_mask

        height, width = self._shape
        self._positions = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if self._walkable[r, c]
        ]

        self._neighbors = {}
        for pos in self._positions:
            neighbors = []
            for move in self.CARDINAL_MOVES:
                nxt = self._next_position(pos, move)
                if self._is_walkable(nxt):
                    neighbors.append(nxt)
            self._neighbors[pos] = tuple(neighbors)

        self._degree = {
            pos: len(neighbors)
            for pos, neighbors in self._neighbors.items()
        }
        self._exposure = {
            pos: len(self._visible_cells_from(pos))
            for pos in self._positions
        }

        self._distance_cache.clear()
        # Belief và target cũ không còn đáng tin nếu topology đổi
        self.belief.clear()
        self.hide_target = None
        self.camp_turns = 0

    def _is_walkable(self, pos):
        if pos is None or self._walkable is None:
            return False
        r, c = pos
        height, width = self._shape
        return (
            0 <= r < height
            and 0 <= c < width
            and bool(self._walkable[r, c])
        )

    def _legal_ghost_actions(self, ghost):
        actions = []
        for move in self.GHOST_MOVES:
            nxt = self._next_position(ghost, move)
            if self._is_walkable(nxt):
                actions.append((move, nxt))
        return actions

    def _pacman_end_positions(self, pacman):
        """
        Các endpoint hợp lệ của Pacman trong một lượt

        Environment chỉ kiểm tra bắt sau khi hoàn tất lượt, nên minimax cũng
        chỉ dùng endpoint. STAY được thêm để mô hình bảo thủ
        """
        endpoints = {pacman}

        for move in self.CARDINAL_MOVES:
            dr, dc = move.value
            current = pacman

            for _ in range(self.pacman_speed):
                nxt = (current[0] + dr, current[1] + dc)
                if not self._is_walkable(nxt):
                    break
                endpoints.add(nxt)
                current = nxt

        return tuple(endpoints)

    # Belief state cho Pacman bị khuất
    def _update_belief(self, map_state, ghost, visible_pacman):
        if visible_pacman is not None:
            self.belief = {visible_pacman}
            self.ever_seen_pacman = True
            self.last_seen_pacman = visible_pacman
            self.turns_since_seen = 0
            return

        self.turns_since_seen += 1

        if self.belief:
            expanded = set()
            for position in self.belief:
                expanded.update(self._pacman_end_positions(position))
            possible = expanded
        else:
            # Trước lần nhìn thấy đầu tiên, Pacman có thể ở bất kỳ ô đường nào
            # không nằm trong vùng quan sát hiện tại
            possible = set(self._positions)

        # Ô 0 đang nhìn thấy nhưng enemy_position lại là None => Pacman chắc
        # chắn không ở đó. Ô -1 không được loại vì nó chỉ là "chưa nhìn thấy"
        visible_empty = {
            (int(r), int(c))
            for r, c in np.argwhere(map_state == 0)
        }
        possible.difference_update(visible_empty)
        possible.discard(ghost)

        if not possible:
            # Khôi phục bảo thủ nếu belief cũ mâu thuẫn với quan sát mới
            possible = set(self._positions)
            possible.difference_update(visible_empty)
            possible.discard(ghost)

        self.belief = possible

    def _select_belief_samples(self, ghost):
        """
        Chọn các giả thuyết nguy hiểm nhất thay vì đưa hàng trăm vị trí vào
        minimax. Ưu tiên khoảng cách mê cung gần Ghost, sau đó bổ sung một vài
        vị trí gần last-seen để giữ tính đa dạng
        """
        if not self.belief:
            return []

        from_ghost = self._distance_map(ghost)

        ranked = sorted(
            self.belief,
            key=lambda pos: (
                self._grid_distance(from_ghost, pos, default=10_000),
                self._manhattan(ghost, pos),
                pos,
            ),
        )

        # Dành khoảng 70% mẫu cho những vị trí đang đe dọa Ghost nhất và phần
        # còn lại cho vùng gần vị trí Pacman được nhìn thấy lần cuối. Cách cũ
        # lấy đủ 10 vị trí gần Ghost trước, khiến nhánh last-seen không bao giờ
        # được sử dụng khi belief có từ 10 phần tử trở lên
        nearest_quota = max(
            1,
            int(round(0.70 * self.MAX_BELIEF_SAMPLES)),
        )
        selected = ranked[:nearest_quota]

        if self.last_seen_pacman is not None:
            around_last_seen = sorted(
                self.belief,
                key=lambda pos: (
                    self._manhattan(self.last_seen_pacman, pos),
                    pos,
                ),
            )
            for pos in around_last_seen:
                if pos not in selected:
                    selected.append(pos)
                if len(selected) >= self.MAX_BELIEF_SAMPLES:
                    break

        # Nếu last-seen trùng nhiều với nhóm gần Ghost, lấp đầy số mẫu còn
        # thiếu bằng các giả thuyết nguy hiểm kế tiếp
        if len(selected) < self.MAX_BELIEF_SAMPLES:
            for pos in ranked:
                if pos not in selected:
                    selected.append(pos)
                if len(selected) >= self.MAX_BELIEF_SAMPLES:
                    break

        return selected

    # Minimax theo vòng: MAX Ghost -> MIN Pacman
    def _iterative_minimax(
        self,
        ghost,
        pacman_hypotheses,
        fallback,
        avoid_root_stay=False,
    ):
        best_completed_move = fallback

        nearest_distance = min(
            self._maze_distance(ghost, pacman)
            for pacman in pacman_hypotheses
        )

        # Khi Pacman thấy rõ và ở gần, depth 3 đáng giá nhất. Khi belief có
        # nhiều giả thuyết, depth 2 cho tỉ lệ chi phí/lợi ích tốt hơn
        if len(pacman_hypotheses) == 1 and nearest_distance <= 10:
            target_depth = self.MAX_DEPTH
        else:
            target_depth = min(2, self.MAX_DEPTH)

        for depth in range(1, target_depth + 1):
            self._check_time(force=True)
            move = self._minimax_root(
                ghost=ghost,
                pacman_hypotheses=pacman_hypotheses,
                depth=depth,
                avoid_root_stay=avoid_root_stay,
            )
            # Chỉ ghi nhận sau khi toàn bộ search ở depth này hoàn tất
            best_completed_move = move

        return best_completed_move

    def _minimax_root(
        self,
        ghost,
        pacman_hypotheses,
        depth,
        avoid_root_stay=False,
    ):
        alpha = -float("inf")
        beta = float("inf")
        best_value = -float("inf")
        best_move = Move.STAY

        actions = self._ordered_ghost_actions(
            ghost,
            pacman_hypotheses[0],
        )

        if avoid_root_stay:
            pacman_replies = {
                pacman: self._pacman_end_positions(pacman)
                for pacman in pacman_hypotheses
            }
            safe_moving_actions = [
                (move, next_ghost)
                for move, next_ghost in actions
                if move != Move.STAY
                and all(
                    not self._is_caught(
                        next_ghost,
                        next_pacman,
                    )
                    for pacman in pacman_hypotheses
                    for next_pacman in pacman_replies[pacman]
                )
            ]

            # Chỉ loại STAY khi thực sự có ít nhất một nước di chuyển an toàn
            # Các nước còn lại vẫn được so sánh bằng toàn bộ minimax, thay vì
            # override bằng lượng giá greedy một vòng sau khi search kết thúc
            if safe_moving_actions:
                actions = safe_moving_actions

        for move, next_ghost in actions:
            self._check_time()
            worst_hypothesis_value = float("inf")

            # Belief là một nút MIN bổ sung: giả định vị trí thật của Pacman là
            # giả thuyết bất lợi nhất cho Ghost
            ordered_hypotheses = sorted(
                pacman_hypotheses,
                key=lambda p: self._manhattan(next_ghost, p),
            )

            for pacman in ordered_hypotheses:
                value = self._pacman_min_value(
                    next_ghost=next_ghost,
                    pacman=pacman,
                    depth=depth,
                    alpha=alpha,
                    beta=min(beta, worst_hypothesis_value),
                )
                worst_hypothesis_value = min(
                    worst_hypothesis_value,
                    value,
                )

                if worst_hypothesis_value <= alpha:
                    break

            if worst_hypothesis_value > best_value:
                best_value = worst_hypothesis_value
                best_move = move

            alpha = max(alpha, best_value)

        return best_move

    def _ghost_max_value(self, ghost, pacman, depth, alpha, beta):
        self._check_time()

        best = -float("inf")

        for _, next_ghost in self._ordered_ghost_actions(ghost, pacman):
            value = self._pacman_min_value(
                next_ghost=next_ghost,
                pacman=pacman,
                depth=depth,
                alpha=alpha,
                beta=beta,
            )
            best = max(best, value)
            alpha = max(alpha, best)

            if alpha >= beta:
                break

        return best

    def _pacman_min_value(
        self,
        next_ghost,
        pacman,
        depth,
        alpha,
        beta,
    ):
        self._check_time()

        worst = float("inf")

        endpoints = sorted(
            self._pacman_end_positions(pacman),
            key=lambda p: self._manhattan(next_ghost, p),
        )

        for next_pacman in endpoints:
            if self._is_caught(next_ghost, next_pacman):
                # Bị bắt càng sớm càng tệ: depth còn lớn tạo phạt lớn hơn.
                value = -self.CAPTURE_SCORE - 10_000.0 * depth
            elif depth <= 1:
                value = self._evaluate_state(
                    ghost=next_ghost,
                    pacman=next_pacman,
                )
            else:
                value = self._ghost_max_value(
                    ghost=next_ghost,
                    pacman=next_pacman,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                )

            worst = min(worst, value)
            beta = min(beta, worst)

            if beta <= alpha:
                break

        return worst

    def _ordered_ghost_actions(self, ghost, pacman):
        actions = self._legal_ghost_actions(ghost)

        def quick_score(item):
            move, nxt = item
            score = 0.0
            score += 25.0 * self._manhattan(nxt, pacman)
            score += 35.0 * self._degree.get(nxt, 0)
            score -= 4.0 * self._exposure.get(nxt, 0)
            if not self._has_line_of_sight(pacman, nxt):
                score += 80.0
            if move == Move.STAY:
                score -= 15.0
            return score

        return sorted(actions, key=quick_score, reverse=True)

    # Hàm lượng giá ở lá minimax
    def _evaluate_state(self, ghost, pacman):
        if self._is_caught(ghost, pacman):
            return -self.CAPTURE_SCORE

        maze_distance = self._maze_distance(ghost, pacman)
        manhattan_distance = self._manhattan(ghost, pacman)

        degree = self._degree.get(ghost, 0)
        exposure = self._exposure.get(ghost, 0)

        safe_next_moves = 0
        pacman_next = self._pacman_end_positions(pacman)
        for _, next_ghost in self._legal_ghost_actions(ghost):
            if all(
                not self._is_caught(next_ghost, next_pacman)
                for next_pacman in pacman_next
            ):
                safe_next_moves += 1

        score = 0.0
        score += 170.0 * min(maze_distance, 30)
        score += 20.0 * min(manhattan_distance, 20)
        score += 260.0 * safe_next_moves
        score += 65.0 * degree
        score -= 7.0 * exposure

        if safe_next_moves == 0:
            score -= 20_000.0

        if degree <= 1:
            score -= 900.0
        elif degree == 2 and not self._is_corner(ghost):
            score -= 100.0
        elif degree >= 3:
            score += 160.0

        if not self._has_line_of_sight(pacman, ghost):
            score += 220.0

        score -= 35.0 * self.visit_count.get(ghost, 0)
        if ghost in self.history:
            score -= 100.0
        if len(self.history) >= 2 and ghost == self.history[-2]:
            score -= 180.0

        return score

    # Hiding động khi chưa có belief đủ tốt
    def _choose_hiding_move(self, ghost, step_number):
        if self.hide_target is not None:
            target_map = self._distance_map(ghost)
            if self._grid_distance(
                target_map,
                self.hide_target,
                default=-1,
            ) < 0:
                self.hide_target = None
                self.camp_turns = 0
                self.target_generation += 1

        if self.hide_target == ghost:
            self.camp_turns += 1

            if (
                self._is_good_camp_position(ghost)
                or self.camp_turns <= self.MAX_UNSAFE_CAMP_TURNS
            ):
                return Move.STAY

            # Target hiện tại không phải camp tốt. Sau một khoảng chờ ngắn,
            # cho phép chọn lại thay vì đứng tại một giao lộ lộ thiên mãi mãi
            self.hide_target = None
            self.camp_turns = 0
            self.target_generation += 1

        if self.hide_target is None:
            self.hide_target = self._select_hide_target(
                ghost=ghost,
                step_number=step_number,
            )
            self.camp_turns = 0

        if self.hide_target is None or self.hide_target == ghost:
            return Move.STAY

        to_target = self._distance_map(self.hide_target)
        best_move = Move.STAY
        best_score = -float("inf")

        for move, next_ghost in self._legal_ghost_actions(ghost):
            distance = self._grid_distance(
                to_target,
                next_ghost,
                default=10_000,
            )
            score = -120.0 * distance
            score += 20.0 * self._degree.get(next_ghost, 0)
            score -= 4.0 * self._exposure.get(next_ghost, 0)
            score -= 25.0 * self.visit_count.get(next_ghost, 0)

            if next_ghost in self.history:
                score -= 80.0
            if move == Move.STAY:
                score -= 200.0

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _select_hide_target(self, ghost, step_number):
        from_ghost = self._distance_map(ghost)
        from_last_seen = (
            self._distance_map(self.last_seen_pacman)
            if self._is_walkable(self.last_seen_pacman)
            else None
        )

        height, width = self._shape
        center = ((height - 1) / 2.0, (width - 1) / 2.0)
        candidates = []

        for pos in self._positions:
            self._check_time()
            distance_from_ghost = self._grid_distance(
                from_ghost,
                pos,
                default=-1,
            )
            if distance_from_ghost < 0:
                # Loại các túi đường bị tách rời dù ô đó không phải tường
                continue

            degree = self._degree.get(pos, 0)
            exposure = self._exposure.get(pos, 0)
            center_distance = (
                abs(pos[0] - center[0])
                + abs(pos[1] - center[1])
            )

            score = 0.0
            if degree <= 1:
                score -= 900.0
            elif degree == 2:
                score += 210.0 if self._is_corner(pos) else 50.0
            elif degree == 3:
                score += 260.0
            else:
                score += 290.0

            score -= 11.0 * exposure
            score += 6.0 * center_distance
            score += 20.0 * min(self._adjacent_wall_count(pos), 2)
            score += 1.0 * min(distance_from_ghost, 25)
            score -= 32.0 * self.visit_count.get(pos, 0)

            if pos in self.history:
                score -= 120.0

            if from_last_seen is not None:
                distance_from_last_seen = self._grid_distance(
                    from_last_seen,
                    pos,
                    default=-1,
                )
                if distance_from_last_seen >= 0:
                    stale_weight = max(
                        0.20,
                        1.0 - self.turns_since_seen / 20.0,
                    )
                    score += (
                        20.0
                        * stale_weight
                        * distance_from_last_seen
                    )

            candidates.append((score, pos))

        if not candidates:
            return ghost

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

        # Chọn trong top 5 để tránh lần nào cũng camp đúng một tọa độ.
        top_count = min(5, len(candidates))
        index = (
            step_number
            + self.target_generation * 3
            + ghost[0] * 31
            + ghost[1] * 17
        ) % top_count
        return candidates[index][1]

    # Fallback nhanh
    def _greedy_fallback(self, ghost, visible_pacman):
        legal = self._legal_ghost_actions(ghost)
        if not legal:
            return Move.STAY

        if visible_pacman is None:
            return Move.STAY

        return self._robust_one_round_move(ghost, [visible_pacman])

    def _robust_one_round_move(self, ghost, hypotheses):
        best_move = Move.STAY
        best_score = -float("inf")

        for move, next_ghost in self._legal_ghost_actions(ghost):
            worst_score = float("inf")

            for pacman in hypotheses:
                pacman_endpoints = self._pacman_end_positions(pacman)
                capture_exists = any(
                    self._is_caught(next_ghost, next_pacman)
                    for next_pacman in pacman_endpoints
                )

                minimum_maze_distance = min(
                    self._maze_distance(next_ghost, next_pacman)
                    for next_pacman in pacman_endpoints
                )

                score = 180.0 * minimum_maze_distance
                score += 55.0 * self._degree.get(next_ghost, 0)
                score -= 6.0 * self._exposure.get(next_ghost, 0)

                if capture_exists:
                    score -= self.CAPTURE_SCORE
                if not self._has_line_of_sight(pacman, next_ghost):
                    score += 160.0
                if move == Move.STAY:
                    score -= 20.0

                worst_score = min(worst_score, score)

            if worst_score > best_score:
                best_score = worst_score
                best_move = move

        return best_move

    def _best_safe_non_stay_move(self, ghost, pacman):
        """
        Tìm một nước di chuyển không phải STAY và an toàn trước mọi
        endpoint Pacman có thể tới trong lượt kế tiếp

        Nếu không có nước di chuyển an toàn, trả None để giữ lại kết quả
        minimax, kể cả khi kết quả đó là STAY
        """
        pacman_endpoints = self._pacman_end_positions(pacman)

        best_move = None
        best_score = -float("inf")

        for move, next_ghost in self._legal_ghost_actions(ghost):
            if move == Move.STAY:
                continue

            # Không chọn nước mà Pacman có thể bắt ngay.
            capture_exists = any(
                self._is_caught(next_ghost, next_pacman)
                for next_pacman in pacman_endpoints
            )

            if capture_exists:
                continue

            minimum_maze_distance = min(
                self._maze_distance(next_ghost, next_pacman)
                for next_pacman in pacman_endpoints
            )

            score = 0.0
            score += 200.0 * minimum_maze_distance
            score += 60.0 * self._degree.get(next_ghost, 0)
            score -= 7.0 * self._exposure.get(next_ghost, 0)

            if not self._has_line_of_sight(pacman, next_ghost):
                score += 220.0

            if next_ghost in self.history:
                score -= 120.0

            score -= 30.0 * self.visit_count.get(next_ghost, 0)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # Khoảng cách, tầm nhìn và hình học
    def _distance_map(self, source):
        source = self._to_pos(source)
        cached = self._distance_cache.get(source)
        if cached is not None:
            return cached

        height, width = self._shape
        distances = np.full((height, width), -1, dtype=np.int16)

        if not self._is_walkable(source):
            return distances

        if len(self._distance_cache) >= self.MAX_DISTANCE_CACHE:
            oldest = next(iter(self._distance_cache))
            self._distance_cache.pop(oldest, None)

        distances[source[0], source[1]] = 0
        queue = deque([source])

        while queue:
            current = queue.popleft()
            base = int(distances[current[0], current[1]])

            for nxt in self._neighbors.get(current, ()):
                if distances[nxt[0], nxt[1]] < 0:
                    distances[nxt[0], nxt[1]] = base + 1
                    queue.append(nxt)

        self._distance_cache[source] = distances
        return distances

    def _maze_distance(self, first, second):
        distances = self._distance_map(first)
        distance = self._grid_distance(distances, second, default=-1)
        if distance < 0:
            # Hai component tách rời: Pacman không thể tới Ghost. Dùng số lớn
            # hữu hạn để lượng giá ổn định
            return len(self._positions) + self._manhattan(first, second)
        return distance

    def _visible_cells_from(self, observer):
        visible = {observer}

        for move in self.CARDINAL_MOVES:
            dr, dc = move.value
            for distance in range(1, self.VISION_RANGE + 1):
                pos = (
                    observer[0] + dr * distance,
                    observer[1] + dc * distance,
                )
                if not self._is_walkable(pos):
                    break
                visible.add(pos)

        return visible

    def _has_line_of_sight(self, observer, target):
        if observer is None or target is None:
            return False

        if self._manhattan(observer, target) > self.VISION_RANGE:
            return False

        r1, c1 = observer
        r2, c2 = target

        if r1 == r2:
            step = 1 if c2 > c1 else -1
            for c in range(c1 + step, c2, step):
                if self._wall_mask[r1, c]:
                    return False
            return True

        if c1 == c2:
            step = 1 if r2 > r1 else -1
            for r in range(r1 + step, r2, step):
                if self._wall_mask[r, c1]:
                    return False
            return True

        return False

    def _is_good_camp_position(self, position):
        """
        Ô camp tốt phải:
        - Có ít nhất hai lối thoát, không phải ngõ cụt
        - Có exposure thấp
        - Là một góc để phá line-of-sight
        """
        degree = self._degree.get(position, 0)
        exposure = self._exposure.get(position, 999)

        return (
            degree >= 2
            and exposure <= 5
            and self._is_corner(position)
        )

    def _is_corner(self, pos):
        neighbors = self._neighbors.get(pos, ())
        if len(neighbors) != 2:
            return False

        first, second = neighbors
        vector_1 = (first[0] - pos[0], first[1] - pos[1])
        vector_2 = (second[0] - pos[0], second[1] - pos[1])
        dot_product = (
            vector_1[0] * vector_2[0]
            + vector_1[1] * vector_2[1]
        )
        return dot_product == 0

    def _adjacent_wall_count(self, pos):
        height, width = self._shape
        count = 0

        for move in self.CARDINAL_MOVES:
            nr = pos[0] + move.value[0]
            nc = pos[1] + move.value[1]
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                count += 1
            elif self._wall_mask[nr, nc]:
                count += 1

        return count

    # mini helper
    def _check_time(self, force=False):
        self._node_count += 1
        if force or self._node_count % 32 == 0:
            if time.perf_counter() >= self._deadline:
                raise _SearchTimeout()

    def _is_caught(self, ghost, pacman):
        return (
            self._manhattan(ghost, pacman)
            < self.capture_distance
        )

    @staticmethod
    def _grid_distance(grid, pos, default=-1):
        if pos is None:
            return default
        value = int(grid[pos[0], pos[1]])
        return value if value >= 0 else default

    @staticmethod
    def _to_pos(pos):
        if pos is None:
            return None
        return (int(pos[0]), int(pos[1]))

    @staticmethod
    def _next_position(pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    @staticmethod
    def _manhattan(first, second):
        return (
            abs(first[0] - second[0])
            + abs(first[1] - second[1])
        )