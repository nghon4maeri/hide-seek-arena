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

    def step(
        self,
        map_state,
        my_position,
        enemy_position,
        step_number
    ):

        target = None

        if enemy_position is not None:

            if self.last_seen_enemy is not None:

                predicted_row = (
                    enemy_position[0]
                    + (enemy_position[0] - self.last_seen_enemy[0])
                )

                predicted_col = (
                    enemy_position[1]
                    + (enemy_position[1] - self.last_seen_enemy[1])
                )

                rows = len(map_state)
                cols = len(map_state[0])

                if (
                    0 <= predicted_row < rows
                    and 0 <= predicted_col < cols
                    and map_state[predicted_row][predicted_col] == 0
                ):
                    target = (
                        predicted_row,
                        predicted_col
                    )

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

        path = self._astar(
            map_state,
            my_position,
            target
        )

        if not path:
            return self._explore(
                my_position,
                map_state
            )

        return self._path_to_move(
            path,
            my_position
        )

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