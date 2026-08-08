from environment import Move  
from agent_interface import GhostAgent as BaseGhostAgent  
from agent_interface import PacmanAgent as BasePacmanAgent  
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))
from collections import deque
import random
import numpy as np

DIRECTIONS = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]


class PacmanAgent(BasePacmanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Predator Pacman"

        self.global_map = None
        self.last_known_enemy_pos = None

    def _update_global_map(self, map_state: np.ndarray):
        if self.global_map is None:
            self.global_map = np.copy(map_state)
        else:
            visible_mask = map_state != -1
            self.global_map[visible_mask] = map_state[visible_mask]

    def _bfs_find_path(self, start, targets):
        """Shortest path (list of Moves) from start to any cell in targets."""
        if start in targets:
            return []

        queue = deque([(start, [])])
        visited = {start}
        height, width = self.global_map.shape

        while queue:
            current, path = queue.popleft()

            for move in DIRECTIONS:
                dr, dc = move.value
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)

                if 0 <= nr < height and 0 <= nc < width:
                    if neighbor not in visited and self.global_map[nr, nc] != 1:
                        new_path = path + [move]
                        if neighbor in targets:
                            return new_path
                        visited.add(neighbor)
                        queue.append((neighbor, new_path))
        return None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):

        self._update_global_map(map_state)

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        elif self.last_known_enemy_pos == my_position or \
                (self.last_known_enemy_pos is not None and
                 map_state[self.last_known_enemy_pos[0], self.last_known_enemy_pos[1]] != -1):
            # We're at the last spot we saw it, or we can now see that cell
            # and it's empty - the sighting is stale, drop it.
            self.last_known_enemy_pos = None

        target_path = None

        threat_pos = enemy_position or self.last_known_enemy_pos
        if threat_pos is not None:
            target_path = self._bfs_find_path(my_position, {threat_pos})
            
        if target_path is None or len(target_path) == 0:
            if threat_pos is not None and abs(my_position[0] - threat_pos[0]) + abs(my_position[1] - threat_pos[1]) <= 3:
                for m in DIRECTIONS:
                    nr, nc = my_position[0] + m.value[0], my_position[1] + m.value[1]
                    if 0 <= nr < self.global_map.shape[0] and 0 <= nc < self.global_map.shape[1]:
                        if self.global_map[nr, nc] == 0:
                            return (m, 1)

        if target_path is None or len(target_path) == 0:
            unseen_tiles = []
            height, width = self.global_map.shape
            for r in range(height):
                for c in range(width):
                    if self.global_map[r, c] == -1:
                        has_open_neighbor = False
                        for m in DIRECTIONS:
                            tr, tc = r + m.value[0], c + m.value[1]
                            if 0 <= tr < height and 0 <= tc < width and self.global_map[tr, tc] == 0:
                                has_open_neighbor = True
                                break
                        if has_open_neighbor:
                            unseen_tiles.append((r, c))
            
            if unseen_tiles:
                unseen_tiles.sort(key=lambda pos: abs(pos[0] - my_position[0]) + abs(pos[1] - my_position[1]))
                for utile in unseen_tiles[:10]: # Thử 10 ô gần nhất
                    target_path = self._bfs_find_path(my_position, {utile})
                    if target_path:
                        break

        if target_path and len(target_path) > 0:
            first_move = target_path[0]
            steps_to_take = 1
            current_sim_pos = my_position

            for move in target_path[1:self.pacman_speed]:
                if move != first_move:
                    break
                dr, dc = move.value
                current_sim_pos = (
                    current_sim_pos[0] + dr, current_sim_pos[1] + dc)
                # Only chain multiple steps through confirmed-empty cells,
                # never blind into fog.
                if self.global_map[current_sim_pos[0], current_sim_pos[1]] == 0:
                    steps_to_take += 1
                else:
                    break

            return (first_move, steps_to_take)

        # Nothing to chase, nothing to explore, no path found - take any
        # open step so we're not just sitting there.
        fallback = DIRECTIONS[:]
        random.shuffle(fallback)
        for move in fallback:
            dr, dc = move.value
            nr, nc = my_position[0] + dr, my_position[1] + dc
            if 0 <= nr < self.global_map.shape[0] and 0 <= nc < self.global_map.shape[1]:
                if self.global_map[nr, nc] == 0:
                    return (move, 1)

        for move in fallback:
            dr, dc = move.value
            nr, nc = my_position[0] + dr, my_position[1] + dc
            if 0 <= nr < self.global_map.shape[0] and 0 <= nc < self.global_map.shape[1]:
                if self.global_map[nr, nc] != 1:
                    return (move, 1)

        return (Move.STAY, 1)


class GhostAgent(BaseGhostAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_map = None
        self.last_known_enemy_pos = None

        # Where we're headed when we have no idea where Pacman is.
        self.wander_target = None
        self.wander_recalc_at = 0
        self.history_positions = deque(maxlen=6)
        self.last_move = None

    def _update_global_map(self, map_state: np.ndarray):
        if self.global_map is None:
            self.global_map = np.copy(map_state)
        else:
            visible_mask = map_state != -1
            self.global_map[visible_mask] = map_state[visible_mask]

    def _get_distance_map(self, start, max_depth=float('inf')):
        """BFS distances from start to every reachable cell"""
        distances = {start: 0}
        queue = deque([start])
        height, width = self.global_map.shape

        while queue:
            current = queue.popleft()
            current_dist = distances[current]

            if current_dist >= max_depth:
                continue

            for move in DIRECTIONS:
                dr, dc = move.value
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)

                if 0 <= nr < height and 0 <= nc < width:
                    if neighbor not in distances and self.global_map[nr, nc] != 1:
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def _bfs_path_to(self, start, goal):
        if start == goal:
            return []
        queue = deque([(start, [])])
        visited = {start}
        height, width = self.global_map.shape

        while queue:
            current, path = queue.popleft()
            for move in DIRECTIONS:
                dr, dc = move.value
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)
                if 0 <= nr < height and 0 <= nc < width and neighbor not in visited \
                        and self.global_map[nr, nc] != 1:  
                    new_path = path + [move]
                    if neighbor == goal:
                        return new_path
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        return None

    def _pick_wander_target(self, my_position, step_number):
        distances = self._get_distance_map(my_position)
        if len(distances) <= 1:
            self.wander_target = None
            return

        max_dist = max(distances.values())
        # A handful of the farthest cells, not just the single farthest one -
        # keeps us from always making a beeline for the exact same corner.
        far_cells = [c for c, d in distances.items() if d >=
                     max_dist - 2 and d > 0]
        self.wander_target = random.choice(far_cells) if far_cells else None
        self.wander_recalc_at = step_number + 20

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:

        self._update_global_map(map_state)

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        elif self.last_known_enemy_pos == my_position or \
                (self.last_known_enemy_pos is not None and
                 map_state[self.last_known_enemy_pos[0], self.last_known_enemy_pos[1]] != -1):
            self.last_known_enemy_pos = None

        threat_pos = enemy_position or self.last_known_enemy_pos

        if threat_pos is None:
            need_new_target = (
                self.wander_target is None
                or my_position == self.wander_target
                or step_number >= self.wander_recalc_at
                or self.global_map[self.wander_target[0], self.wander_target[1]] == 1
            )
            if need_new_target:
                self._pick_wander_target(my_position, step_number)

            if self.wander_target is not None:
                path = self._bfs_path_to(my_position, self.wander_target)
                if path:
                    self.last_move = path[0]
                    return path[0]
            return Move.STAY

        pacman_dists = self._get_distance_map(threat_pos)

        best_move = Move.STAY
        best_score = -float('inf')
        height, width = self.global_map.shape

        candidates = DIRECTIONS + [Move.STAY]
        random.shuffle(candidates)
        opposite_moves = {
            Move.UP: Move.DOWN,
            Move.DOWN: Move.UP,
            Move.LEFT: Move.RIGHT,
            Move.RIGHT: Move.LEFT,
            Move.STAY: Move.STAY
        }

        for move in candidates:
            if move == Move.STAY:
                dr, dc = 0, 0
            else:
                dr, dc = move.value

            nr, nc = my_position[0] + dr, my_position[1] + dc
            next_pos = (nr, nc)

            if 0 <= nr < height and 0 <= nc < width and self.global_map[nr, nc] != 1:
                dist_to_pacman = pacman_dists.get(next_pos, 0)

                ghost_dists = self._get_distance_map(next_pos, max_depth=6)
                safe_area_count = sum(
                    1 for cell, g_dist in ghost_dists.items()
                    if g_dist < pacman_dists.get(cell, 0)
                )

                score = (dist_to_pacman * 1000) + safe_area_count
                
                if next_pos in self.history_positions:
                    score -= 10000
                
                if self.last_move and move == opposite_moves.get(self.last_move):
                    score -= 500

                if score > best_score:
                    best_score = score
                    best_move = move

        self.last_move = best_move
        self.history_positions.append(my_position)
        return best_move
