import sys
from pathlib import Path
import heapq
import random

SRC_PATH = Path(__file__).resolve().parents[2] / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

# ===================================================================
# Constants
# ===================================================================
MOVE_ORDER = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
CHOKE_SCOUT_DIST = 7     # How far ahead to scan for junctions
LOCK_DURATION = 3        # Turns to hold a choke-point lock
A_STAR_PHASE_END = 10    # Switch from pure A* to interception after this step


# ===================================================================
# Grid utilities (self-contained)
# ===================================================================

def _shape(ms):
    if hasattr(ms, "shape"):
        return int(ms.shape[0]), int(ms.shape[1])
    return len(ms), len(ms[0]) if ms else 0


def _cell(ms, r, c):
    return int(ms[r, c]) if hasattr(ms, "shape") else int(ms[r][c])


def _apply(pos, move):
    return (pos[0] + move.value[0], pos[1] + move.value[1])


def _valid(pos, ms):
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1


def _legal(pos, ms):
    return [m for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _cell_exits(pos, ms):
    """Count how many valid exits a cell has (degree)."""
    return sum(1 for m in MOVE_ORDER if _valid(_apply(pos, m), ms))


# ===================================================================
# A* Search — heapq-optimised O(N log N)
# ===================================================================

def astar(ms, start, goal):
    """A* shortest path returning list of Move from start to goal."""
    if goal is None or not _valid(start, ms) or not _valid(goal, ms):
        return []
    if start == goal:
        return []

    open_set = [(0, 0, start)]  # (f, g, pos)
    came_from = {}
    g_score = {start: 0}
    closed = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = []
            while current != start:
                prev, move = came_from[current]
                path.append(move)
                current = prev
            path.reverse()
            return path

        for move in MOVE_ORDER:
            nxt = _apply(current, move)
            if not _valid(nxt, ms) or nxt in closed:
                continue
            ng = g + 1
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng
                came_from[nxt] = (current, move)
                heapq.heappush(open_set, (ng + _manhattan(nxt, goal), ng, nxt))
    return []


# ===================================================================
# BFS distance map (for junction distance checks)
# ===================================================================

def bfs_dist(ms, start, max_dist=30):
    """BFS distance map from start, up to max_dist steps."""
    if not _valid(start, ms):
        return {}
    dist = {start: 0}
    q = [start]
    for cur in q:
        if dist[cur] >= max_dist:
            continue
        for m in MOVE_ORDER:
            nxt = _apply(cur, m)
            if nxt not in dist and _valid(nxt, ms):
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


# ===================================================================
# PacmanAgent — SOTA Seeker
# ===================================================================

class PacmanAgent(BasePacmanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pacman_speed = max(
            1,
            int(kwargs.get("pacman_speed", 1))
        )

        self.last_seen_enemy = None

        # Ghost direction tracking
        self._enemy_direction = None      # (dr, dc) of last Ghost move
        self._direction_streak = 0        # consecutive steps in same direction

        # Path cache
        self._cached_target = None
        self._cached_path = []            # list of positions
        self._cached_my_pos = None

        # Counters
        self.interception_used = 0
        self.path_cache_hit = 0
        self.path_cache_miss = 0

        # Feature gates
        self.enable_interception = True   # projection-based interception
        self.enable_trap_pressure = False  # step extension (disabled by default)

    def step(
        self,
        map_state,
        my_position,
        enemy_position,
        step_number
    ):

        target = None

        if enemy_position is not None:

            # ---- Ghost direction tracking ----
            self._update_ghost_tracking(enemy_position)

            # ---- Interception planning ----
            # Only when enabled AND Ghost has stable direction >= 2 steps
            if self.enable_interception and self._direction_streak >= 2:
                inter_target = self._compute_interception_target(
                    map_state, enemy_position, my_position
                )
                if inter_target is not None:
                    # Compare A* to interception vs A* to Ghost current
                    path_to_inter = self._astar(
                        map_state, my_position, inter_target
                    )
                    path_to_direct = self._astar(
                        map_state, my_position, enemy_position
                    )

                    # Conservative gate: use interception only if it is
                    # strictly not worse, OR interception cell is very close
                    # to Ghost's current position (≤ 2 cells away)
                    dist_inter_to_ghost = (
                        abs(inter_target[0] - enemy_position[0])
                        + abs(inter_target[1] - enemy_position[1])
                    )
                    if path_to_inter and (
                        not path_to_direct
                        or len(path_to_inter) <= len(path_to_direct)
                        or dist_inter_to_ghost <= 2
                    ):
                        target = inter_target
                        self.interception_used += 1

            if target is None:
                target = enemy_position

            self.last_seen_enemy = enemy_position

        else:

            if self.last_seen_enemy is None:
                return self._explore(
                    my_position,
                    map_state
                )

            if my_position == self.last_seen_enemy:

                self.last_seen_enemy = None

                return self._explore(
                    my_position,
                    map_state
                )

            target = self.last_seen_enemy

        if my_position == target:
            return (Move.STAY, 1)

        # ---- Path caching ----
        path = None
        cache_valid = (
            self._cached_target == target
            and self._cached_my_pos == my_position
            and self._cached_path
        )
        if cache_valid:
            path = self._cached_path
            self.path_cache_hit += 1
        else:
            path = self._astar(map_state, my_position, target)
            self._cached_target = target
            self._cached_path = path
            self._cached_my_pos = my_position
            self.path_cache_miss += 1

        if not path:
            return self._explore(
                my_position,
                map_state
            )

        # Extract move + steps from path
        result = self._path_to_move(path, my_position)

        # ---- Trap pressure tie-break (disabled by default) ----
        if self.enable_trap_pressure:
            result = self._apply_trap_pressure(
                map_state, my_position, target, path, result
            )

        # Update path cache: remove consumed steps and track expected position
        if isinstance(result, tuple):
            consumed = result[1]
            mv = result[0]
        else:
            consumed = 1
            mv = result
        self._cached_path = path[consumed:]
        # Compute expected position after move for next cache validation
        exp_pos = my_position
        for _ in range(consumed):
            nxt = (exp_pos[0] + mv.value[0], exp_pos[1] + mv.value[1])
            if 0 <= nxt[0] < len(map_state) and 0 <= nxt[1] < len(map_state[0]) \
                    and map_state[nxt[0]][nxt[1]] == 0:
                exp_pos = nxt
            else:
                break
        self._cached_my_pos = exp_pos

        return result

    def _astar(
        self,
        map_state,
        start,
        goal
    ):

        rows = len(map_state)
        cols = len(map_state[0])

        def heuristic(a, b):
            return (
                abs(a[0] - b[0])
                + abs(a[1] - b[1])
            )

        open_set = []

        heapq.heappush(
            open_set,
            (
                heuristic(start, goal),
                start
            )
        )

        closed_set = set()

        came_from = {}

        g_score = {
            start: 0
        }

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        while open_set:

            _, current = heapq.heappop(
                open_set
            )

            if current in closed_set:
                continue

            closed_set.add(current)

            if current == goal:

                path = []

                while current != start:
                    path.append(current)
                    current = came_from[current]

                path.reverse()
                return path

            for dr, dc in directions:

                nr = current[0] + dr
                nc = current[1] + dc

                if not (
                    0 <= nr < rows
                    and 0 <= nc < cols
                ):
                    continue

                if map_state[nr][nc] != 0:
                    continue

                neighbor = (nr, nc)

                tentative_g = (
                    g_score[current] + 1
                )

                if (
                    neighbor not in g_score
                    or tentative_g < g_score[neighbor]
                ):

                    came_from[neighbor] = current

                    g_score[neighbor] = tentative_g

                    f_score = (
                        tentative_g
                        + heuristic(
                            neighbor,
                            goal
                        )
                    )

                    heapq.heappush(
                        open_set,
                        (
                            f_score,
                            neighbor
                        )
                    )

        return []

    def _path_to_move(
        self,
        path,
        my_position
    ):

        first = path[0]

        dr = first[0] - my_position[0]
        dc = first[1] - my_position[1]

        if dr == -1:
            move = Move.UP
        elif dr == 1:
            move = Move.DOWN
        elif dc == -1:
            move = Move.LEFT
        elif dc == 1:
            move = Move.RIGHT
        else:
            return (Move.STAY, 1)

        steps = 1
        current = first

        for next_cell in path[1:]:

            next_dr = next_cell[0] - current[0]
            next_dc = next_cell[1] - current[1]

            if (
                next_dr == dr
                and next_dc == dc
                and steps < self.pacman_speed
            ):
                steps += 1
                current = next_cell
            else:
                break

        return (
            move,
            steps
        )

    def _explore(
        self,
        my_position,
        map_state
    ):

        moves = [
            Move.UP,
            Move.DOWN,
            Move.LEFT,
            Move.RIGHT
        ]

        random.shuffle(moves)

        rows = len(map_state)
        cols = len(map_state[0])

        for move in moves:

            dr, dc = move.value

            steps = 0
            r, c = my_position

            for _ in range(self.pacman_speed):

                nr = r + dr
                nc = c + dc

                if not (
                    0 <= nr < rows
                    and 0 <= nc < cols
                ):
                    break

                if map_state[nr][nc] != 0:
                    break

                steps += 1
                r, c = nr, nc

            if steps > 0:
                return (
                    move,
                    steps
                )

        return (
            Move.STAY,
            1
        )

    # ------------------------------------------------------------------
    # Ghost direction tracking
    # ------------------------------------------------------------------
    def _update_ghost_tracking(self, enemy_pos):
        """Update direction streak from last known enemy position."""
        if self.last_seen_enemy is None:
            return
        dr = enemy_pos[0] - self.last_seen_enemy[0]
        dc = enemy_pos[1] - self.last_seen_enemy[1]
        new_dir = (dr, dc)
        if new_dir == self._enemy_direction and (dr != 0 or dc != 0):
            self._direction_streak += 1
        else:
            self._enemy_direction = new_dir
            self._direction_streak = 1 if (dr != 0 or dc != 0) else 0

    # ------------------------------------------------------------------
    # Interception target: project Ghost forward along stable direction
    # ------------------------------------------------------------------
    def _compute_interception_target(self, ms, enemy_pos, my_pos):
        """Project Ghost up to 4 cells forward; return first junction or corridor cell."""
        dr, dc = self._enemy_direction
        cur_row, cur_col = enemy_pos
        best = None

        for i in range(1, 5):
            nr = cur_row + dr * i
            nc = cur_col + dc * i
            if not (0 <= nr < len(ms) and 0 <= nc < len(ms[0])):
                break
            if ms[nr][nc] != 0:
                break
            nxt = (nr, nc)

            exits = self._count_exits(ms, nxt)
            if exits >= 3:
                return nxt          # junction — best interception point
            if exits == 2 and best is None:
                best = nxt           # corridor cell — fallback

        return best  # may be None

    def _count_exits(self, ms, pos):
        r, c = pos
        exits = 0
        if r > 0 and ms[r - 1][c] == 0:
            exits += 1
        if r < len(ms) - 1 and ms[r + 1][c] == 0:
            exits += 1
        if c > 0 and ms[r][c - 1] == 0:
            exits += 1
        if c < len(ms) - 1 and ms[r][c + 1] == 0:
            exits += 1
        return exits

    # ------------------------------------------------------------------
    # Trap pressure: prefer moves closer to Ghost’s nearest junction
    # ------------------------------------------------------------------
    def _apply_trap_pressure(self, ms, my_pos, ghost_pos, path, move_result):
        """If Pacman can move farther but path turns, check if continuing
        straight pressures Ghost escape routes better.

        Returns potentially adjusted (Move, steps).
        """
        if not isinstance(move_result, tuple):
            return move_result
        if move_result[1] >= self.pacman_speed:
            return move_result  # already moving at max speed

        if len(path) < 2:
            return move_result

        mv = move_result[0]
        steps = move_result[1]

        # Check if extending current direction would get closer to Ghost junctions
        cur = my_pos
        extra_step = steps  # already taking this many
        nxt = (cur[0] + mv.value[0] * (extra_step + 1),
               cur[1] + mv.value[1] * (extra_step + 1))

        if not (0 <= nxt[0] < len(ms) and 0 <= nxt[1] < len(ms[0])):
            return move_result
        if ms[nxt[0]][nxt[1]] != 0:
            return move_result

        # Check if this extra straight cell is closer to Ghost
        # or a junction near Ghost, compared to where path turns
        dist_straight = abs(nxt[0] - ghost_pos[0]) + abs(nxt[1] - ghost_pos[1])
        if len(path) > steps:
            turn_cell = path[steps]
            dist_turn = abs(turn_cell[0] - ghost_pos[0]) + abs(turn_cell[1] - ghost_pos[1])

            # Prefer straight if it reduces distance to Ghost more
            if dist_straight <= dist_turn:
                return (mv, steps + 1)

        return move_result

    # ------------------------------------------------------------------
    # Convert A* path → (Move, steps) action
    # ------------------------------------------------------------------
    def _path_to_action(self, me, path, ms):
        """Extract first move from A* path, pack consecutive same-direction
        steps up to pacman_speed (straight-line constraint)."""
        first_move = path[0]

        # Count desired consecutive same-direction steps from path
        desired = 1
        for i in range(1, min(len(path), self.pacman_speed)):
            if path[i] == first_move:
                desired += 1
            else:
                break

        # Walk up to `desired` steps, stopping if wall encountered
        steps = 0
        cur = me
        for _ in range(min(self.pacman_speed, desired)):
            nxt = _apply(cur, first_move)
            if not _valid(nxt, ms):
                break
            steps += 1
            cur = nxt

        return (first_move, max(1, steps))

    # ------------------------------------------------------------------
    # Exploration (enemy not visible — fog of war fallback)
    # ------------------------------------------------------------------
    def _explore(self, me, ms):
        candidates = _legal(me, ms)
        if not candidates:
            return (Move.STAY, 1)

        def score(m):
            nxt = _apply(me, m)
            unvisited_bonus = 5 if nxt not in self._visited else 0
            return unvisited_bonus + _cell_exits(nxt, ms)

        best = max(candidates, key=score)
        steps = 0
        cur = me
        for _ in range(self.pacman_speed):
            nxt = _apply(cur, best)
            if not _valid(nxt, ms):
                break
            steps += 1
            cur = nxt
        return (best, max(1, steps))


# ===================================================================
# GhostAgent — minimal placeholder (not primary deliverable)
# ===================================================================

class GhostAgent(BaseGhostAgent):

    def step(
        self,
        map_state,
        my_position,
        enemy_position,
        step_number
    ):
        return Move.STAY