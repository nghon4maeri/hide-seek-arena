"""
Search-based Lab 1 submission for the official Pacman vs Ghost arena.

This file is intentionally self-contained.  It imports only the official arena
interfaces plus Python standard-library modules, so it can be zipped directly
for Moodle with no dependency on the repository's visualizer or root dev code.
"""

from __future__ import annotations

import heapq
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


Position = Tuple[int, int]
Grid = Tuple[Tuple[int, ...], ...]
INF = 10**8
MOVE_ORDER = (Move.UP, Move.LEFT, Move.RIGHT, Move.DOWN)
GHOST_MOVE_ORDER = (Move.UP, Move.LEFT, Move.RIGHT, Move.DOWN, Move.STAY)


def _normalize_map(map_state) -> Grid:
    """Convert numpy/list map data to an immutable grid; -1 fog is treated as open."""
    if hasattr(map_state, "tolist"):
        rows = map_state.tolist()
    else:
        rows = map_state
    return tuple(tuple(1 if int(cell) == 1 else 0 for cell in row) for row in rows)


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _in_bounds(grid: Grid, p: Position) -> bool:
    return 0 <= p[0] < len(grid) and 0 <= p[1] < len(grid[0])


def _is_open(grid: Grid, p: Position) -> bool:
    return _in_bounds(grid, p) and grid[p[0]][p[1]] == 0


def _step(grid: Grid, pos: Position, move: Move) -> Position:
    dr, dc = move.value
    nxt = (pos[0] + dr, pos[1] + dc)
    return nxt if _is_open(grid, nxt) else pos


def _neighbors(grid: Grid, pos: Position) -> Iterable[Tuple[Position, Move]]:
    for move in MOVE_ORDER:
        nxt = _step(grid, pos, move)
        if nxt != pos:
            yield nxt, move


def _legal_moves(grid: Grid, pos: Position, include_stay: bool = True) -> List[Move]:
    moves = [move for _, move in _neighbors(grid, pos)]
    if include_stay:
        moves.append(Move.STAY)
    return moves


def _apply_steps(grid: Grid, pos: Position, move: Move, steps: int) -> Position:
    cur = pos
    for _ in range(max(1, int(steps))):
        nxt = _step(grid, cur, move)
        if nxt == cur:
            break
        cur = nxt
    return cur


def _max_valid_steps(grid: Grid, pos: Position, move: Move, max_steps: int) -> int:
    if move == Move.STAY:
        return 1
    cur = pos
    count = 0
    for _ in range(max(1, int(max_steps))):
        nxt = _step(grid, cur, move)
        if nxt == cur:
            break
        cur = nxt
        count += 1
    return count


class SearchBoard:
    """Small cached search helper for one observed arena map."""

    _cache: Dict[Grid, "SearchBoard"] = {}

    def __new__(cls, grid: Grid):
        cached = cls._cache.get(grid)
        if cached is None:
            cached = super().__new__(cls)
            cls._cache[grid] = cached
            cached._initialized = False
            if len(cls._cache) > 4:
                cls._cache.pop(next(iter(cls._cache)))
        return cached

    def __init__(self, grid: Grid):
        if self._initialized:
            return
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0
        self.passable = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if grid[r][c] == 0
        ]
        self._dist_cache: Dict[Position, List[List[int]]] = {}
        self._degree_cache: Optional[List[List[int]]] = None
        self._dead_cache: Optional[List[List[int]]] = None
        self._initialized = True

    def distance_map(self, source: Optional[Position]) -> List[List[int]]:
        if source is None or not _is_open(self.grid, source):
            return [[INF] * self.cols for _ in range(self.rows)]
        cached = self._dist_cache.get(source)
        if cached is not None:
            return cached

        dist = [[INF] * self.cols for _ in range(self.rows)]
        q = deque([source])
        dist[source[0]][source[1]] = 0
        while q:
            pos = q.popleft()
            nd = dist[pos[0]][pos[1]] + 1
            for nxt, _ in _neighbors(self.grid, pos):
                if dist[nxt[0]][nxt[1]] == INF:
                    dist[nxt[0]][nxt[1]] = nd
                    q.append(nxt)
        self._dist_cache[source] = dist
        return dist

    def distance(self, a: Optional[Position], b: Optional[Position]) -> int:
        if a is None or b is None:
            return INF
        if not _is_open(self.grid, a) or not _is_open(self.grid, b):
            return _manhattan(a, b)
        d = self.distance_map(a)[b[0]][b[1]]
        return d if d < INF else _manhattan(a, b) + 20

    def astar_path(self, start: Position, goal: Position) -> List[Position]:
        if not _is_open(self.grid, start) or not _is_open(self.grid, goal):
            return []

        heap: List[Tuple[int, int, Position]] = []
        heapq.heappush(heap, (_manhattan(start, goal), 0, start))
        came_from: Dict[Position, Position] = {}
        g_score = {start: 0}
        closed = set()

        while heap:
            _, cost, pos = heapq.heappop(heap)
            if pos in closed:
                continue
            if pos == goal:
                path = []
                cur = goal
                while cur != start:
                    path.append(cur)
                    cur = came_from[cur]
                path.reverse()
                return path
            closed.add(pos)

            for nxt, _ in _neighbors(self.grid, pos):
                tentative = cost + 1
                if tentative < g_score.get(nxt, INF):
                    came_from[nxt] = pos
                    g_score[nxt] = tentative
                    heapq.heappush(heap, (tentative + _manhattan(nxt, goal), tentative, nxt))
        return []

    def degree_map(self) -> List[List[int]]:
        if self._degree_cache is not None:
            return self._degree_cache
        degree = [[0] * self.cols for _ in range(self.rows)]
        for pos in self.passable:
            degree[pos[0]][pos[1]] = sum(1 for _ in _neighbors(self.grid, pos))
        self._degree_cache = degree
        return degree

    def dead_end_distance_map(self) -> List[List[int]]:
        if self._dead_cache is not None:
            return self._dead_cache
        degree = self.degree_map()
        dist = [[INF] * self.cols for _ in range(self.rows)]
        q = deque()
        for pos in self.passable:
            if degree[pos[0]][pos[1]] <= 1:
                dist[pos[0]][pos[1]] = 0
                q.append(pos)
        while q:
            pos = q.popleft()
            nd = dist[pos[0]][pos[1]] + 1
            for nxt, _ in _neighbors(self.grid, pos):
                if dist[nxt[0]][nxt[1]] == INF:
                    dist[nxt[0]][nxt[1]] = nd
                    q.append(nxt)
        self._dead_cache = dist
        return dist

    def reachable_area(self, start: Position, max_depth: int = 18) -> int:
        if not _is_open(self.grid, start):
            return 0
        q = deque([(start, 0)])
        seen = {start}
        while q:
            pos, depth = q.popleft()
            if depth >= max_depth:
                continue
            for nxt, _ in _neighbors(self.grid, pos):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, depth + 1))
        return len(seen)

    def safe_area(self, ghost: Position, pacman: Optional[Position], max_depth: int = 18, pacman_speed: int = 2) -> int:
        if pacman is None or not _is_open(self.grid, ghost):
            return self.reachable_area(ghost, max_depth)
        pac_dist = self.distance_map(pacman)
        q = deque([(ghost, 0)])
        seen = {ghost}
        safe = 0
        while q:
            pos, depth = q.popleft()
            pac_turns = math.ceil(pac_dist[pos[0]][pos[1]] / max(1, pacman_speed))
            if pac_turns <= depth + 1:
                continue
            safe += 1
            if depth >= max_depth:
                continue
            for nxt, _ in _neighbors(self.grid, pos):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, depth + 1))
        return safe

    def first_move_toward(self, start: Position, goal: Position) -> Move:
        path = self.astar_path(start, goal)
        if not path:
            return Move.STAY
        nxt = path[0]
        dr = nxt[0] - start[0]
        dc = nxt[1] - start[1]
        for move in MOVE_ORDER:
            if move.value == (dr, dc):
                return move
        return Move.STAY


class TimedAgentMixin:
    def _begin_step(self, seconds: float = 0.86) -> None:
        self._deadline = time.perf_counter() + seconds

    def _time_left(self) -> float:
        return self._deadline - time.perf_counter()


class PacmanAgent(BasePacmanAgent, TimedAgentMixin):
    """Pacman/seeker: use A*, prediction, and shallow adversarial scoring."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Search Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos: Optional[Position] = None
        self.visit_count: Dict[Position, int] = {}

    def step(self, map_state, my_position, enemy_position, step_number):
        self._begin_step()
        grid = _normalize_map(map_state)
        board = SearchBoard(grid)
        my_position = tuple(my_position)
        self.visit_count[my_position] = self.visit_count.get(my_position, 0) + 1

        if enemy_position is not None:
            self.last_known_enemy_pos = tuple(enemy_position)
        target = self.last_known_enemy_pos

        if target is None:
            return self._explore_action(board, my_position)

        direct = self._direct_capture_action(board, my_position, target)
        if direct is not None:
            return direct

        candidates = self._pacman_actions(board, my_position)
        if not candidates:
            return (Move.STAY, 1)

        ghost_dist = board.distance_map(target)
        ghost_escape = self._predicted_ghost_moves(board, target, my_position)
        best_action = candidates[0]
        best_key = (-INF, 0, 0, 0)

        for move, steps, new_pos in candidates:
            score = self._score_pacman_candidate(board, new_pos, target, ghost_dist, ghost_escape, step_number)
            if self._time_left() > 0.08 and enemy_position is not None:
                score += 0.35 * self._minimax_pacman(board, new_pos, target, depth=2, alpha=-INF, beta=INF)

            non_stay = 0 if move == Move.STAY else 1
            key = (score, non_stay, steps, -GHOST_MOVE_ORDER.index(move) if move in GHOST_MOVE_ORDER else -99)
            if key > best_key:
                best_key = key
                best_action = (move, steps, new_pos)

        move, steps, _ = best_action
        return (move, max(1, min(self.pacman_speed, steps)))

    def _pacman_actions(self, board: SearchBoard, pos: Position) -> List[Tuple[Move, int, Position]]:
        actions = []
        for move in GHOST_MOVE_ORDER:
            limit = self.pacman_speed if move != Move.STAY else 1
            max_steps = _max_valid_steps(board.grid, pos, move, limit)
            if move == Move.STAY:
                actions.append((move, 1, pos))
            else:
                for steps in range(1, max_steps + 1):
                    actions.append((move, steps, _apply_steps(board.grid, pos, move, steps)))
        return actions

    def _direct_capture_action(self, board: SearchBoard, pacman: Position, ghost: Position):
        if pacman[0] == ghost[0] or pacman[1] == ghost[1]:
            for move in MOVE_ORDER:
                dr, dc = move.value
                aligned = (dr != 0 and pacman[1] == ghost[1]) or (dc != 0 and pacman[0] == ghost[0])
                if not aligned:
                    continue
                dist = _manhattan(pacman, ghost)
                if dist <= self.pacman_speed:
                    final = _apply_steps(board.grid, pacman, move, dist)
                    if final == ghost or _manhattan(final, ghost) < 2:
                        return (move, max(1, dist))
        return None

    def _predicted_ghost_moves(self, board: SearchBoard, ghost: Position, pacman: Position) -> List[Position]:
        scored = []
        dead = board.dead_end_distance_map()
        for move in GHOST_MOVE_ORDER:
            nxt = _step(board.grid, ghost, move)
            if move != Move.STAY and nxt == ghost:
                continue
            dist = board.distance(pacman, nxt)
            area = board.safe_area(nxt, pacman, max_depth=10, pacman_speed=self.pacman_speed)
            score = 11.0 * dist + 1.3 * area + 2.0 * min(dead[nxt[0]][nxt[1]], 8)
            scored.append((score, nxt))
        scored.sort(reverse=True)
        return [pos for _, pos in scored[:3]] or [ghost]

    def _score_pacman_candidate(
        self,
        board: SearchBoard,
        pacman: Position,
        ghost: Position,
        ghost_dist: List[List[int]],
        ghost_escape: Sequence[Position],
        step_number: int,
    ) -> float:
        dist = ghost_dist[pacman[0]][pacman[1]]
        if dist >= INF:
            dist = board.distance(pacman, ghost)
        nearest_escape = min(board.distance(pacman, escape) for escape in ghost_escape)
        area_after = board.safe_area(ghost, pacman, max_depth=10, pacman_speed=self.pacman_speed)
        degree = board.degree_map()[pacman[0]][pacman[1]]
        repeat_penalty = self.visit_count.get(pacman, 0)
        capture_bonus = 500.0 if _manhattan(pacman, ghost) < 2 else 0.0
        return (
            capture_bonus
            - 18.0 * dist
            - 8.0 * nearest_escape
            - 1.6 * area_after
            + 2.0 * degree
            - 1.5 * repeat_penalty
            - 0.02 * step_number
        )

    def _minimax_pacman(self, board: SearchBoard, pacman: Position, ghost: Position, depth: int, alpha: float, beta: float) -> float:
        if _manhattan(pacman, ghost) < 2:
            return 10000.0
        if depth <= 0 or self._time_left() < 0.035:
            return -20.0 * board.distance(pacman, ghost) - board.safe_area(ghost, pacman, 8, self.pacman_speed)

        value = INF
        for ghost_move in GHOST_MOVE_ORDER:
            g2 = _step(board.grid, ghost, ghost_move)
            if ghost_move != Move.STAY and g2 == ghost:
                continue
            child = -INF
            for move, steps, p2 in self._pacman_actions(board, pacman)[:10]:
                score = self._minimax_pacman(board, p2, g2, depth - 1, alpha, beta)
                child = max(child, score)
                alpha = max(alpha, child)
                if beta <= alpha or self._time_left() < 0.035:
                    break
            value = min(value, child)
            beta = min(beta, value)
            if beta <= alpha or self._time_left() < 0.035:
                break
        return value

    def _explore_action(self, board: SearchBoard, pos: Position):
        best = (Move.STAY, 1, -INF)
        dead = board.dead_end_distance_map()
        for move, steps, nxt in self._pacman_actions(board, pos):
            score = board.reachable_area(nxt, 12) + min(dead[nxt[0]][nxt[1]], 8) - 3 * self.visit_count.get(nxt, 0)
            if move == Move.STAY:
                score -= 8
            if score > best[2]:
                best = (move, steps, score)
        return (best[0], best[1])


class GhostAgent(BaseGhostAgent, TimedAgentMixin):
    """Ghost/hider: maximize survival with BFS safety and alpha-beta search."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Search Ghost"
        self.last_known_enemy_pos: Optional[Position] = None
        self.visit_count: Dict[Position, int] = {}
        self.pacman_speed = 2

    def step(self, map_state, my_position, enemy_position, step_number) -> Move:
        self._begin_step()
        grid = _normalize_map(map_state)
        board = SearchBoard(grid)
        my_position = tuple(my_position)
        self.visit_count[my_position] = self.visit_count.get(my_position, 0) + 1

        if enemy_position is not None:
            self.last_known_enemy_pos = tuple(enemy_position)
        pacman = self.last_known_enemy_pos

        actions = _legal_moves(grid, my_position, include_stay=True)
        if not actions:
            return Move.STAY

        best_move = Move.STAY
        best_key = (-INF, 0, 0)
        for move in actions:
            nxt = _step(grid, my_position, move)
            score = self._score_ghost_candidate(board, nxt, pacman, step_number)
            if pacman is not None and self._time_left() > 0.08:
                score += 0.4 * self._minimax_ghost(board, nxt, pacman, depth=3, alpha=-INF, beta=INF)
            non_stay = 0 if move == Move.STAY else 1
            order = -GHOST_MOVE_ORDER.index(move)
            key = (score, non_stay, order)
            if key > best_key:
                best_key = key
                best_move = move

        return best_move

    def _score_ghost_candidate(self, board: SearchBoard, ghost: Position, pacman: Optional[Position], step_number: int) -> float:
        degree = board.degree_map()[ghost[0]][ghost[1]]
        dead = board.dead_end_distance_map()[ghost[0]][ghost[1]]
        repeat = self.visit_count.get(ghost, 0)

        if pacman is None:
            return 2.8 * board.reachable_area(ghost, 14) + 4.0 * degree + min(dead, 10) - 4.0 * repeat

        dist = board.distance(pacman, ghost)
        safe = board.safe_area(ghost, pacman, max_depth=18, pacman_speed=self.pacman_speed)
        local = board.reachable_area(ghost, max_depth=8)
        danger = 140.0 if _manhattan(pacman, ghost) < 2 else 0.0
        corridor_penalty = 18.0 if degree <= 2 and dist <= 6 else 0.0
        dead_penalty = max(0, 7 - min(dead, 7)) * (10.0 if dist <= 8 else 2.0)
        return (
            20.0 * dist
            + 2.6 * safe
            + 0.6 * local
            + 4.0 * degree
            + 3.0 * min(dead, 10)
            - danger
            - corridor_penalty
            - dead_penalty
            - 2.0 * repeat
            + 0.04 * step_number
        )

    def _pacman_responses(self, board: SearchBoard, pacman: Position, ghost: Position) -> List[Position]:
        responses = []
        path_move = board.first_move_toward(pacman, ghost)
        for move in GHOST_MOVE_ORDER:
            if move == Move.STAY:
                steps_options = [1]
            else:
                max_steps = _max_valid_steps(board.grid, pacman, move, self.pacman_speed)
                steps_options = range(1, max_steps + 1)
            for steps in steps_options:
                pos = _apply_steps(board.grid, pacman, move, steps)
                chase_bonus = 0 if move == path_move else 1
                responses.append((board.distance(pos, ghost), chase_bonus, pos))
        responses.sort()
        return [pos for _, _, pos in responses[:6]] or [pacman]

    def _minimax_ghost(self, board: SearchBoard, ghost: Position, pacman: Position, depth: int, alpha: float, beta: float) -> float:
        if _manhattan(pacman, ghost) < 2:
            return -10000.0
        if depth <= 0 or self._time_left() < 0.035:
            return self._score_ghost_candidate(board, ghost, pacman, 0)

        value = INF
        for p2 in self._pacman_responses(board, pacman, ghost):
            if _manhattan(p2, ghost) < 2:
                child = -10000.0
            else:
                child = -INF
                moves = _legal_moves(board.grid, ghost, include_stay=True)
                moves.sort(key=lambda m: -self._score_ghost_candidate(board, _step(board.grid, ghost, m), p2, 0))
                for move in moves[:5]:
                    g2 = _step(board.grid, ghost, move)
                    score = self._minimax_ghost(board, g2, p2, depth - 1, alpha, beta)
                    child = max(child, score)
                    alpha = max(alpha, child)
                    if beta <= alpha or self._time_left() < 0.035:
                        break
            value = min(value, child)
            beta = min(beta, value)
            if beta <= alpha or self._time_left() < 0.035:
                break
        return value
