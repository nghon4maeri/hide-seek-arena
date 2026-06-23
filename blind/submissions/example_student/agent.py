"""
Example student submission for Blind Adversary (Lab 2).
Demonstrates required interface with partial observability support.

Key differences from Lab 1 (Perfect Information):
- map_state: cells outside vision are marked -1 (UNSEEN)
- enemy_position: may be None when outside observation radius
- Agents must maintain internal memory to track last known positions
"""

import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random


class PacmanAgent(BasePacmanAgent):
    """
    Blind Pacman agent — must seek under limited vision.

    Strategy:
    - If enemy visible: A* or greedy toward enemy
    - If enemy not visible: navigate to last known position
    - If no prior sighting: explore using frontier-based exploration
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Blind Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.last_known_enemy_pos = None
        self._memory_map = None          # Accumulated map knowledge
        self._visited = set()            # Cells known to be visited

    def step(self, map_state, my_position, enemy_position, step_number):
        my_position = tuple(my_position)

        # Maintain accumulated memory map
        self._update_memory(map_state)

        # Track enemy
        if enemy_position is not None:
            self.last_known_enemy_pos = tuple(enemy_position)

        # Decide target
        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            return self._explore(my_position)

        if my_position == target:
            self.last_known_enemy_pos = None
            return self._explore(my_position)

        # A* pathfinding toward target on memory map
        path = self._astar(self._memory_map, my_position, target)
        if path:
            return self._path_to_move(path, my_position)

        return self._explore(my_position)

    # ------------------------------------------------------------------
    # Memory: accumulate map knowledge across steps
    # ------------------------------------------------------------------
    def _update_memory(self, map_state):
        if self._memory_map is None:
            self._memory_map = np.full_like(map_state, -1, dtype=int)
        # Reveal cells that are now visible (not -1 in current observation)
        visible_mask = (map_state != -1)
        self._memory_map[visible_mask] = map_state[visible_mask]

    # ------------------------------------------------------------------
    # A* pathfinding (on memory map)
    # ------------------------------------------------------------------
    def _astar(self, ms, start, goal):
        if goal is None or start == goal:
            return []
        h, w = ms.shape
        if not (0 <= start[0] < h and 0 <= start[1] < w):
            return []
        if not (0 <= goal[0] < h and 0 <= goal[1] < w):
            return []

        import heapq
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        moves = [(-1, 0, Move.UP), (1, 0, Move.DOWN),
                 (0, -1, Move.LEFT), (0, 1, Move.RIGHT)]

        open_set = [(heuristic(start, goal), 0, start)]
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
                    current, move = came_from[current]
                    path.append(move)
                path.reverse()
                return path

            for dr, dc, move in moves:
                nr, nc = current[0] + dr, current[1] + dc
                nxt = (nr, nc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                # Treat unseen cells (-1) as potentially traversable
                if ms[nr, nc] == 1 or nxt in closed:
                    continue
                ng = g + 1
                if nxt not in g_score or ng < g_score[nxt]:
                    g_score[nxt] = ng
                    came_from[nxt] = (current, move)
                    heapq.heappush(open_set, (ng + heuristic(nxt, goal), ng, nxt))
        return []

    # ------------------------------------------------------------------
    # Convert A* path to (Move, steps) action
    # ------------------------------------------------------------------
    def _path_to_move(self, path, my_position):
        if not path:
            return (Move.STAY, 1)
        first_move = path[0]
        steps = 1
        for m in path[1:]:
            if m == first_move and steps < self.pacman_speed:
                steps += 1
            else:
                break
        return (first_move, steps)

    # ------------------------------------------------------------------
    # Exploration: pick direction toward nearest frontier (known→unknown)
    # ------------------------------------------------------------------
    def _explore(self, my_position):
        if self._memory_map is None:
            moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            random.shuffle(moves)
            return (moves[0], 1)

        # Find nearest frontier cell (known=0 adjacent to unknown=-1)
        h, w = self._memory_map.shape
        target = None
        best_dist = float("inf")

        for r in range(h):
            for c in range(w):
                if self._memory_map[r, c] != 0:
                    continue
                # Check if adjacent to unknown
                has_unknown = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and self._memory_map[nr, nc] == -1:
                        has_unknown = True
                        break
                if has_unknown:
                    d = abs(r - my_position[0]) + abs(c - my_position[1])
                    if d < best_dist:
                        best_dist = d
                        target = (r, c)

        if target is None:
            moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            random.shuffle(moves)
            return (moves[0], 1)

        path = self._astar(self._memory_map, my_position, target)
        if path:
            return self._path_to_move(path, my_position)

        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)
        return (moves[0], 1)


class GhostAgent(BaseGhostAgent):
    """
    Blind Ghost agent — must evade under limited vision.

    Strategy:
    - If enemy visible: maximize distance (A* to farthest safe cell)
    - If enemy not visible: use last known direction + exploration
    - Maintain memory map for pathfinding
    - Anti-oscillation: avoid revisiting recent cells
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Blind Ghost"
        self.last_known_enemy_pos = None
        self._memory_map = None
        self._history = []               # Recent positions for anti-oscillation
        self._history_limit = 6

    def step(self, map_state, my_position, enemy_position, step_number):
        my_position = tuple(my_position)

        # Maintain accumulated memory map
        self._update_memory(map_state)

        # Track enemy
        if enemy_position is not None:
            self.last_known_enemy_pos = tuple(enemy_position)

        # Track history (anti-oscillation)
        if not self._history or self._history[-1] != my_position:
            self._history.append(my_position)
        if len(self._history) > self._history_limit * 3:
            self._history = self._history[-self._history_limit * 2:]

        # Get candidates with anti-oscillation filter
        candidates = self._get_legal_moves(my_position)
        safe_candidates = self._filter_history(my_position, candidates)

        threat = enemy_position or self.last_known_enemy_pos

        if threat is None:
            return self._explore(my_position, candidates)

        # Strategic evasion: find move maximizing distance from threat
        best_move = self._strategic_evasion(my_position, threat, safe_candidates or candidates)

        return best_move or self._explore(my_position, candidates)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    def _update_memory(self, map_state):
        if self._memory_map is None:
            self._memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = (map_state != -1)
        self._memory_map[visible_mask] = map_state[visible_mask]

    # ------------------------------------------------------------------
    # Legal moves (treat -1 as potentially valid)
    # ------------------------------------------------------------------
    def _get_legal_moves(self, pos):
        candidates = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nr, nc = pos[0] + dr, pos[1] + dc
            h, w = (21, 21)
            if self._memory_map is not None:
                h, w = self._memory_map.shape
            if 0 <= nr < h and 0 <= nc < w:
                # Valid if known-empty (0) or unknown (-1); wall (1) blocks
                if self._memory_map is not None:
                    cell = self._memory_map[nr, nc]
                    if cell != 1:  # 0 or -1 both OK
                        candidates.append(move)
                else:
                    candidates.append(move)  # No memory yet, assume valid
        return candidates

    # ------------------------------------------------------------------
    # Anti-oscillation
    # ------------------------------------------------------------------
    def _filter_history(self, pos, candidates):
        if not self._history or len(self._history) < 2:
            return candidates
        recent = set(self._history[-self._history_limit:])
        filtered = []
        for m in candidates:
            dr, dc = m.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt not in recent:
                filtered.append(m)
        return filtered if filtered else candidates

    # ------------------------------------------------------------------
    # Strategic evasion: move away from threat
    # ------------------------------------------------------------------
    def _strategic_evasion(self, my_position, threat, candidates):
        if not candidates:
            return None

        best_move = None
        best_score = float("-inf")

        for m in candidates:
            dr, dc = m.value
            nxt = (my_position[0] + dr, my_position[1] + dc)

            # Avoid moving into threat
            dist = abs(nxt[0] - threat[0]) + abs(nxt[1] - threat[1])
            if dist < 2:
                continue

            # Prefer more open areas (higher degree)
            exits = self._count_exits(nxt)
            junction_bonus = 30 if exits >= 3 else 0
            dead_end_penalty = -50 if exits <= 1 else 0

            score = dist * 10 + junction_bonus + dead_end_penalty

            if score > best_score:
                best_score = score
                best_move = m

        return best_move

    def _count_exits(self, pos):
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if self._memory_map is not None:
                h, w = self._memory_map.shape
                if 0 <= nr < h and 0 <= nc < w:
                    if self._memory_map[nr, nc] != 1:
                        count += 1
        return count

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------
    def _explore(self, pos, candidates):
        if not candidates:
            return Move.STAY
        # Prefer moves toward open areas
        best = candidates[0]
        best_exits = -1
        for m in candidates:
            dr, dc = m.value
            nxt = (pos[0] + dr, pos[1] + dc)
            exits = self._count_exits(nxt)
            if exits > best_exits:
                best_exits = exits
                best = m
        return best
