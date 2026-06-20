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


class GhostAgent(BaseGhostAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(
        self,
        map_state,
        my_position,
        enemy_position,
        step_number
    ):
        return Move.STAY