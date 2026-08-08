

from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


MOVES = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
GHOST_MOVES = MOVES + (Move.STAY,)
INF = 10_000
FIXED_PACMAN_START = (15, 10)
FIXED_GHOST_START = (9, 10)
GHOST_MODELS = ("stay", "inertia", "flee", "junction")
MAX_MODEL_CONFIDENCE = 0.65
PACMAN_MODELS = ("stay", "inertia", "chase", "intercept")
PACMAN_PREDICTOR_MIN_UPDATES = 2
PACMAN_PREDICTOR_MIN_STREAK = 3
PACMAN_PREDICTOR_CONFIDENCE = 0.55
PACMAN_PREDICTOR_UNIFORM_MIX = 0.20



PACMAN_ACTIVE_SENSING_ENABLED = False
GHOST_OPENING = Move.RIGHT
CAMP_SEARCH_STEP = 9
CAMP_SEARCH_DELAY_AFTER_SIGHTING = 8
GHOST_FEINT_STEPS = (13,)
ROBUST_HIDE_ROUTE = tuple(
    Move[name]
    for name in """
    RIGHT STAY RIGHT RIGHT RIGHT RIGHT UP UP UP UP
    UP UP LEFT LEFT LEFT LEFT LEFT LEFT UP UP
    STAY STAY LEFT RIGHT DOWN DOWN LEFT LEFT DOWN DOWN
    RIGHT RIGHT DOWN DOWN RIGHT RIGHT UP STAY STAY STAY
    STAY STAY STAY STAY STAY STAY STAY UP RIGHT LEFT
    DOWN STAY DOWN LEFT STAY STAY STAY STAY LEFT STAY
    STAY STAY LEFT STAY LEFT DOWN STAY STAY STAY STAY
    STAY STAY UP RIGHT RIGHT UP UP LEFT LEFT UP
    UP RIGHT RIGHT UP DOWN UP DOWN RIGHT RIGHT UP
    STAY DOWN RIGHT RIGHT RIGHT STAY RIGHT RIGHT RIGHT RIGHT
    RIGHT UP UP LEFT LEFT LEFT STAY STAY STAY STAY
    STAY STAY STAY STAY STAY STAY STAY LEFT DOWN DOWN
    RIGHT RIGHT RIGHT STAY STAY STAY STAY RIGHT UP DOWN
    LEFT STAY STAY STAY STAY STAY STAY STAY STAY STAY
    STAY STAY STAY STAY STAY STAY STAY STAY STAY STAY
    RIGHT UP UP LEFT LEFT LEFT STAY STAY STAY STAY
    STAY STAY STAY STAY STAY STAY STAY STAY STAY STAY
    STAY STAY STAY STAY STAY STAY STAY STAY STAY STAY
    STAY LEFT DOWN DOWN LEFT LEFT DOWN STAY STAY STAY
    STAY STAY STAY STAY STAY STAY STAY STAY STAY DOWN
    """.split()
)
HIDDEN_ROUTE = "HIDDEN_ROUTE"
VISIBLE_EVASION = "VISIBLE_EVASION"
BELIEF_FALLBACK = "BELIEF_FALLBACK"



BELIEF_SHIELD_STAGE = 0
BELIEF_SHIELD_LAMBDA = 25_000.0
BELIEF_RISK_THRESHOLD = 0.25


class _MapModel:
    def __init__(
        self,
        observation: np.ndarray,
        pacman_speed: int = 2,
        exclude_unreachable=False,
    ):
        self.height, self.width = observation.shape



        self.walls = observation == 1
        raw_cells = tuple(
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if not self.walls[r, c]
        )
        raw_cell_set = set(raw_cells)
        fixed_starts = [
            point
            for point in (FIXED_PACMAN_START, FIXED_GHOST_START)
            if point in raw_cell_set
        ]
        if exclude_unreachable and fixed_starts:
            reachable = set(fixed_starts)
            queue = deque(fixed_starts)
            while queue:
                point = queue.popleft()
                for move in MOVES:
                    neighbor = (
                        point[0] + move.value[0],
                        point[1] + move.value[1],
                    )
                    if neighbor in raw_cell_set and neighbor not in reachable:
                        reachable.add(neighbor)
                        queue.append(neighbor)
            self.cells = tuple(
                point for point in raw_cells if point in reachable
            )
        else:
            self.cells = raw_cells
        self.cell_set = set(self.cells)
        self.pacman_speed = max(1, int(pacman_speed))
        self.neighbors = {p: self._neighbors(p, include_stay=False) for p in self.cells}
        self.ghost_endpoints = {
            p: tuple(dict.fromkeys(q for _, q in self._ghost_actions(p))) for p in self.cells
        }
        self.pacman_actions = {p: self._pacman_actions(p) for p in self.cells}
        self.dist = {p: self._bfs(p) for p in self.cells}
        self.turn_dist = {p: self._turn_bfs(p) for p in self.cells}
        self.degree = {p: len(self.neighbors[p]) for p in self.cells}
        self.visibility = {p: self._visibility(p, 5) for p in self.cells}
        self.dead_end_depth = self._dead_end_depths()

    def valid(self, p):
        return p in self.cell_set

    def _neighbors(self, p, include_stay=False):
        result = [(Move.STAY, p)] if include_stay else []
        for move in MOVES:
            q = (p[0] + move.value[0], p[1] + move.value[1])
            if q in self.cell_set:
                result.append((move, q))
        return tuple(result)

    def _ghost_actions(self, p):
        return self._neighbors(p, include_stay=True)

    def _pacman_actions(self, p):
        actions = [(Move.STAY, 1, p)]
        for move in MOVES:
            current = p
            for steps in range(1, self.pacman_speed + 1):
                q = (current[0] + move.value[0], current[1] + move.value[1])
                if q not in self.cell_set:
                    break
                actions.append((move, steps, q))
                current = q
        return tuple(actions)

    def _bfs(self, start):
        result = {start: 0}
        queue = deque([start])
        while queue:
            p = queue.popleft()
            for _, q in self.neighbors[p]:
                if q not in result:
                    result[q] = result[p] + 1
                    queue.append(q)
        return result

    def distance(self, a, b):
        return self.dist.get(a, {}).get(b, INF)

    def _turn_bfs(self, start):
        result = {start: 0}
        queue = deque([start])
        while queue:
            p = queue.popleft()
            for _, _, q in self.pacman_actions[p]:
                if q not in result:
                    result[q] = result[p] + 1
                    queue.append(q)
        return result

    def turn_distance(self, a, b):
        return self.turn_dist.get(a, {}).get(b, INF)

    def _visibility(self, p, radius):
        visible = {p}
        for move in MOVES:
            for distance in range(1, radius + 1):
                q = (p[0] + move.value[0] * distance, p[1] + move.value[1] * distance)
                if not (0 <= q[0] < self.height and 0 <= q[1] < self.width):
                    break
                if self.walls[q]:
                    break
                visible.add(q)
        return frozenset(visible)

    def _dead_end_depths(self):

        degree = dict(self.degree)
        queue = deque(p for p, d in degree.items() if d <= 1)
        removed = set()
        while queue:
            p = queue.popleft()
            if p in removed:
                continue
            removed.add(p)
            for _, q in self.neighbors[p]:
                if q in removed:
                    continue
                degree[q] -= 1
                if degree[q] == 1:
                    queue.append(q)
        core = set(self.cells) - removed
        if not core:
            return {p: 0 for p in self.cells}
        depth = {p: 0 for p in core}
        queue = deque(core)
        while queue:
            p = queue.popleft()
            for _, q in self.neighbors[p]:
                if q not in depth:
                    depth[q] = depth[p] + 1
                    queue.append(q)
        return depth

    def first_action_toward(self, start, target, prefer_speed=True):
        if start == target or target not in self.cell_set:
            return Move.STAY, 1
        best = None
        for move, steps, q in self.pacman_actions[start]:
            if move == Move.STAY:
                continue
            score = self.turn_distance(q, target)
            key = (score, -steps if prefer_speed else steps, MOVES.index(move))
            if best is None or key < best[0]:
                best = (key, move, steps)
        return (best[1], best[2]) if best else (Move.STAY, 1)


class _Belief:
    def __init__(self, model: _MapModel, start, pacman=False):
        self.model = model
        self.pacman = pacman
        if start in model.cell_set:
            self.prob = {start: 1.0}
        else:
            weight = 1.0 / max(1, len(model.cells))
            self.prob = {p: weight for p in model.cells}

    def predict(self, target=None):
        new = defaultdict(float)
        for p, mass in self.prob.items():
            if self.pacman:
                endpoints = tuple(dict.fromkeys(q for _, _, q in self.model.pacman_actions[p]))
                weights = self._pacman_weights(p, endpoints, target)
            else:
                endpoints = self.model.ghost_endpoints[p]
                weights = self._ghost_weights(p, endpoints, target)
            total = sum(weights)
            for q, weight in zip(endpoints, weights):
                new[q] += mass * weight / total
        self.prob = dict(new)

    def observe(self, observer, enemy):
        if enemy is not None:
            self.prob = {enemy: 1.0}
            return
        visible = self.model.visibility.get(observer, frozenset())
        self.prob = {p: mass for p, mass in self.prob.items() if p not in visible}
        self._normalize_or_reset(visible)

    def _normalize_or_reset(self, excluded=frozenset()):
        total = sum(self.prob.values())
        if total <= 1e-12:
            candidates = [p for p in self.model.cells if p not in excluded]
            weight = 1.0 / max(1, len(candidates))
            self.prob = {p: weight for p in candidates}
        else:
            self.prob = {p: mass / total for p, mass in self.prob.items()}

    def _ghost_weights(self, p, endpoints, pacman):
        if pacman not in self.model.cell_set:
            return [1.0] * len(endpoints)
        distances = [self.model.turn_distance(pacman, q) for q in endpoints]
        least = min(distances)
        return [1.0 + 2.5 * (d > least) + 1.5 * (d == max(distances)) for d in distances]

    def _pacman_weights(self, p, endpoints, ghost):
        if ghost not in self.model.cell_set:
            return [1.0] * len(endpoints)
        distances = [self.model.turn_distance(q, ghost) for q in endpoints]
        best = min(distances)
        return [1.0 + 4.0 * (d == best) + 1.5 / (1 + d) for d in distances]

    def likely(self, limit=24):
        return sorted(self.prob.items(), key=lambda item: (-item[1], item[0]))[:limit]


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.model = None
        self.belief = None
        self.last_step = 0
        self.last_seen = None
        self.previous_seen = None
        self.last_seen_step = 0
        self.ghost_model_weights = {
            name: 1.0 / len(GHOST_MODELS) for name in GHOST_MODELS
        }
        self.ghost_model_updates = 0
        self.pending_ghost_predictions = None
        self.pending_prediction_step = 0
        self.visit_count = defaultdict(int)
        self.camp_anchors = ()
        self.cleared_camps = set()

    def step(self, map_state, my_position, enemy_position, step_number):
        if self.model is None:
            self.model = _MapModel(map_state, self.pacman_speed)
            self.belief = _Belief(self.model, FIXED_GHOST_START, pacman=False)
            self.camp_anchors = tuple(
                p for p in ((19, 3), (19, 17)) if p in self.model.cell_set
            )

        if step_number > self.last_step + 0 and self.last_step > 0:
            self.belief.predict(target=my_position)
        self.belief.observe(my_position, enemy_position)
        self.last_step = step_number
        self.visit_count[my_position] += 1

        if enemy_position is None:
            self.cleared_camps.update(
                anchor
                for anchor in self.camp_anchors
                if anchor in self.model.visibility[my_position]
            )

        if enemy_position is not None:
            previous = self.last_seen if self.last_seen_step == step_number - 1 else None
            self._update_ghost_models(
                my_position, enemy_position, step_number, previous
            )
            self.previous_seen, self.last_seen = previous, enemy_position
            self.last_seen_step = step_number
            return self._visible_action(my_position, enemy_position)
        self.pending_ghost_predictions = None
        self.pending_prediction_step = 0
        if step_number == 1 and Move.RIGHT in [move for move, _, _ in self.model.pacman_actions[my_position]]:
            return Move.RIGHT, 1
        return self._hidden_action(my_position)

    def _visible_action(self, pacman, ghost):
        ghost_options = self.model.ghost_endpoints[ghost]
        endpoint_mass = self._ghost_endpoint_distribution(ghost_options)
        predicted = max(
            ghost_options,
            key=lambda q: (
                endpoint_mass[q],
                q == self._predicted_ghost_endpoint(pacman, ghost, ghost_options),
            ),
        )
        best = None
        for move, steps, p_next in self.model.pacman_actions[pacman]:
            captures = [abs(p_next[0] - g[0]) + abs(p_next[1] - g[1]) < 2 for g in ghost_options]
            guaranteed = int(all(captures))
            expected = sum(endpoint_mass[g] for g, capture in zip(ghost_options, captures) if capture)
            worst_graph = max(self.model.turn_distance(p_next, g) for g in ghost_options)
            predicted_dist = self.model.turn_distance(p_next, predicted)
            reveals = sum(g in self.model.visibility[p_next] for g in ghost_options)

            score = (
                guaranteed * 1_000_000
                + expected * 20_000
                + reveals * 80
                - worst_graph * 55
                - predicted_dist * 25
                + steps * 3
            )
            key = (score, -worst_graph, steps, -MOVES.index(move) if move in MOVES else -9)
            if best is None or key > best[0]:
                best = (key, move, steps)
        return best[1], best[2]

    def _update_ghost_models(self, pacman, ghost, step_number, previous_ghost):

        if (
            self.pending_ghost_predictions is not None
            and self.pending_prediction_step == step_number - 1
        ):
            weighted = {}
            for name, prediction in self.pending_ghost_predictions.items():
                error = self.model.distance(prediction, ghost)
                likelihood = 1.0 if error == 0 else (0.18 if error == 1 else 0.03)
                weighted[name] = self.ghost_model_weights[name] * likelihood
            total = sum(weighted.values())
            if total > 0:


                uniform = 1.0 / len(GHOST_MODELS)
                self.ghost_model_weights = {
                    name: 0.85 * weighted[name] / total + 0.15 * uniform
                    for name in GHOST_MODELS
                }
                self.ghost_model_updates += 1

        self.pending_ghost_predictions = self._ghost_model_predictions(
            pacman, ghost, previous_ghost
        )
        self.pending_prediction_step = step_number

    def _ghost_model_predictions(self, pacman, ghost, previous_ghost=None):
        options = self.model.ghost_endpoints[ghost]

        inertia = None
        if previous_ghost is not None:
            velocity = (
                ghost[0] - previous_ghost[0],
                ghost[1] - previous_ghost[1],
            )
            continuation = (ghost[0] + velocity[0], ghost[1] + velocity[1])
            if continuation in options:
                inertia = continuation
        if inertia is None:
            inertia = max(
                options,
                key=lambda q: (
                    self.model.distance(q, pacman),
                    q != ghost,
                    self.model.degree[q],
                ),
            )

        flee = max(
            options,
            key=lambda q: (
                self.model.turn_distance(pacman, q),
                self.model.degree[q],
                -self.model.dead_end_depth[q],
                q != ghost,
            ),
        )
        junction = max(
            options,
            key=lambda q: (
                self.model.degree[q],
                -self.model.dead_end_depth[q],
                self.model.distance(q, pacman),
                q != ghost,
            ),
        )
        return {
            "stay": ghost,
            "inertia": inertia,
            "flee": flee,
            "junction": junction,
        }

    def _ghost_endpoint_distribution(self, ghost_options):
        uniform = 1.0 / len(ghost_options)
        if not self.pending_ghost_predictions or self.ghost_model_updates <= 0:
            return {q: uniform for q in ghost_options}

        learned = {q: 0.0 for q in ghost_options}
        for name, prediction in self.pending_ghost_predictions.items():
            learned[prediction] += self.ghost_model_weights[name]



        confidence = min(MAX_MODEL_CONFIDENCE, 0.16 * self.ghost_model_updates)
        return {
            q: (1.0 - confidence) * uniform + confidence * learned[q]
            for q in ghost_options
        }

    def _predicted_ghost_endpoint(self, pacman, ghost, options):
        if self.previous_seen is not None:
            velocity = (ghost[0] - self.previous_seen[0], ghost[1] - self.previous_seen[1])
            continuation = (ghost[0] + velocity[0], ghost[1] + velocity[1])
            if continuation in options:
                return continuation
        return max(
            options,
            key=lambda q: (
                self.model.distance(q, pacman),
                self.model.degree[q],
                -self.model.dead_end_depth[q],
            ),
        )

    def _hidden_action(self, pacman):




        if self._should_search_camps():
            remaining_camps = [
                anchor for anchor in self.camp_anchors
                if anchor not in self.cleared_camps
            ]
            if remaining_camps:
                target = min(
                    remaining_camps,
                    key=lambda p: (self.model.turn_distance(pacman, p), p),
                )
                return self.model.first_action_toward(pacman, target)

        candidates = self.belief.likely(40)
        best = None
        for move, steps, p_next in self.model.pacman_actions[pacman]:
            if move == Move.STAY and len(self.model.pacman_actions[pacman]) > 1:
                continue
            visible_mass = 0.0
            capture_mass = 0.0
            expected_distance = 0.0
            worst_distance = 0
            for ghost, mass in candidates:
                d = self.model.turn_distance(p_next, ghost)
                expected_distance += mass * d
                worst_distance = max(worst_distance, d)
                if ghost in self.model.visibility[p_next]:
                    visible_mass += mass
                if abs(p_next[0] - ghost[0]) + abs(p_next[1] - ghost[1]) < 2:
                    capture_mass += mass
            revisit_penalty = 320 * self.visit_count[p_next] + (40 if p_next == pacman else 0)
            score = capture_mass * 50_000 + visible_mass * 1_300 - expected_distance * 28 - worst_distance - revisit_penalty
            key = (score, visible_mass, -expected_distance, steps)
            if best is None or key > best[0]:
                best = (key, move, steps)
        return (best[1], best[2]) if best else (Move.STAY, 1)

    def _should_search_camps(self):
        if self.last_step < CAMP_SEARCH_STEP:
            return False
        if self.last_seen_step <= 0:
            return True
        return (
            self.last_step - self.last_seen_step
            >= CAMP_SEARCH_DELAY_AFTER_SIGHTING
        )


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.belief = None
        self.last_step = 0
        self.last_move = Move.STAY
        self.last_seen = None
        self.robust_route_active = True
        self.controller_state = HIDDEN_ROUTE
        self.shield_override_count = 0
        self.route_deactivation_reason = None
        self.hide_route = ROBUST_HIDE_ROUTE
        self.pacman_model_weights = {
            name: 1.0 / len(PACMAN_MODELS) for name in PACMAN_MODELS
        }
        self.pacman_model_updates = 0
        self.visible_pacman_streak = 0
        self.previous_visible_pacman = None
        self.previous_visible_step = 0
        self.inferred_pacman_action = None
        self.pending_pacman_predictions = None
        self.pending_pacman_prediction_step = 0

    def _deactivate_route(self, reason):
        self.robust_route_active = False
        self.route_deactivation_reason = reason
        if self.controller_state == HIDDEN_ROUTE:
            self.controller_state = BELIEF_FALLBACK

    def step(self, map_state, my_position, enemy_position, step_number):
        if self.model is None:
            self.model = _MapModel(
                map_state, pacman_speed=2, exclude_unreachable=True
            )
            self.belief = _Belief(self.model, FIXED_PACMAN_START, pacman=True)

        route_contiguous = (
            step_number == 1
            if self.last_step == 0
            else step_number == self.last_step + 1
        )
        if not route_contiguous:
            self._deactivate_route("discontinuous_step")
        if PACMAN_ACTIVE_SENSING_ENABLED:
            self._update_pacman_predictor(
                my_position, enemy_position, step_number
            )
        if step_number > self.last_step + 0 and self.last_step > 0:
            self.belief.predict(target=my_position)
        self.belief.observe(my_position, enemy_position)
        self.last_step = step_number

        proposal, route_move = self._propose_action(
            my_position, enemy_position, step_number
        )
        move = self._apply_belief_risk_shield(
            my_position, enemy_position, proposal
        )
        if enemy_position is not None:
            self._deactivate_route("direct_sighting")
            self.controller_state = VISIBLE_EVASION
        elif route_move is not None and move != route_move:
            self.shield_override_count += 1
            self._deactivate_route("shield_override")
        if move not in {candidate for candidate, _ in self.model._ghost_actions(my_position)}:
            self._deactivate_route("illegal_final_action")
            move = self._choose_action(my_position, enemy_position)
        if move not in {candidate for candidate, _ in self.model._ghost_actions(my_position)}:
            move = Move.STAY
        self.last_move = move
        if enemy_position is not None:
            self.last_seen = enemy_position
        return move

    def _propose_action(self, ghost, visible_pacman, step_number):
        legal_moves = {
            candidate for candidate, _ in self.model._ghost_actions(ghost)
        }
        route_move = None
        if (
            self.robust_route_active
            and visible_pacman is None
            and step_number <= len(self.hide_route)
        ):
            route_move = self.hide_route[step_number - 1]
            if route_move not in legal_moves:
                self._deactivate_route("illegal_route_action")
                route_move = None
        if route_move is not None:
            self.controller_state = HIDDEN_ROUTE
            return route_move, route_move

        if visible_pacman is not None:
            self.controller_state = VISIBLE_EVASION
            if PACMAN_ACTIVE_SENSING_ENABLED:
                predicted = self._choose_observation_contingent_action(
                    ghost, visible_pacman
                )
                if predicted is not None:
                    return predicted, None
            return self._choose_action(ghost, visible_pacman), None

        self.controller_state = BELIEF_FALLBACK
        if step_number == 1 and GHOST_OPENING in legal_moves:
            return GHOST_OPENING, None
        if step_number in GHOST_FEINT_STEPS:
            feint = {
                Move.UP: Move.DOWN,
                Move.DOWN: Move.UP,
                Move.LEFT: Move.RIGHT,
                Move.RIGHT: Move.LEFT,
            }.get(self.last_move, Move.RIGHT)
            if feint in legal_moves:
                return feint, None
        return self._choose_action(ghost, None), None

    def _update_pacman_predictor(self, ghost, visible_pacman, step_number):
        consecutive = (
            visible_pacman is not None
            and self.previous_visible_pacman is not None
            and self.previous_visible_step == step_number - 1
            and self.pending_pacman_predictions is not None
            and self.pending_pacman_prediction_step == step_number - 1
        )
        if visible_pacman is None:
            self.visible_pacman_streak = 0
            self.previous_visible_pacman = None
            self.previous_visible_step = 0
            self.inferred_pacman_action = None
            self.pending_pacman_predictions = None
            self.pending_pacman_prediction_step = 0
            return

        if consecutive:
            inferred_action = self._infer_pacman_action(
                self.previous_visible_pacman, visible_pacman
            )
            if inferred_action is None:
                self.visible_pacman_streak = 1
                self.inferred_pacman_action = None
                self.pending_pacman_predictions = (
                    self._pacman_model_predictions(
                        visible_pacman, ghost, None
                    )
                )
                self.pending_pacman_prediction_step = step_number
                self.previous_visible_pacman = visible_pacman
                self.previous_visible_step = step_number
                return
            likelihoods = {}
            for name, prediction in self.pending_pacman_predictions.items():
                error = self.model.turn_distance(prediction, visible_pacman)
                likelihoods[name] = 1.0 / ((1.0 + error) ** 2)
            posterior = {
                name: self.pacman_model_weights[name] * likelihoods[name]
                for name in PACMAN_MODELS
            }
            total = sum(posterior.values())
            uniform = 1.0 / len(PACMAN_MODELS)
            self.pacman_model_weights = {
                name: 0.85 * posterior[name] / total + 0.15 * uniform
                for name in PACMAN_MODELS
            }
            self.pacman_model_updates += 1
            self.visible_pacman_streak += 1
            self.inferred_pacman_action = inferred_action
        else:
            self.visible_pacman_streak = 1
            self.inferred_pacman_action = None

        self.pending_pacman_predictions = self._pacman_model_predictions(
            visible_pacman, ghost, self.inferred_pacman_action
        )
        self.pending_pacman_prediction_step = step_number
        self.previous_visible_pacman = visible_pacman
        self.previous_visible_step = step_number

    def _infer_pacman_action(self, previous, current):
        dr = current[0] - previous[0]
        dc = current[1] - previous[1]
        if dr != 0 and dc != 0:
            return None
        steps = abs(dr) + abs(dc)
        if steps > self.model.pacman_speed:
            return None
        if steps == 0:
            return Move.STAY, 1
        delta = (dr // steps, dc // steps)
        move = next(
            (candidate for candidate in MOVES if candidate.value == delta),
            None,
        )
        if move is None:
            return None
        legal = {
            (candidate, candidate_steps, endpoint)
            for candidate, candidate_steps, endpoint
            in self.model.pacman_actions[previous]
        }
        action = (move, steps, current)
        return (move, steps) if action in legal else None

    def _pacman_model_predictions(self, pacman, ghost, inferred_action):
        options = self.model.pacman_actions[pacman]
        endpoints = tuple(dict.fromkeys(endpoint for _, _, endpoint in options))
        stay = pacman
        inertia = stay
        if inferred_action is not None:
            move, steps = inferred_action
            inertia = next(
                (
                    endpoint
                    for candidate, candidate_steps, endpoint in options
                    if candidate == move and candidate_steps == steps
                ),
                stay,
            )
        chase = min(
            endpoints,
            key=lambda point: (
                self.model.turn_distance(point, ghost),
                abs(point[0] - ghost[0]) + abs(point[1] - ghost[1]),
                point,
            ),
        )
        ghost_options = dict(self.model._ghost_actions(ghost))
        ghost_projection = ghost_options.get(self.last_move, ghost)
        intercept = min(
            endpoints,
            key=lambda point: (
                self.model.turn_distance(point, ghost_projection),
                abs(point[0] - ghost_projection[0])
                + abs(point[1] - ghost_projection[1]),
                point,
            ),
        )
        return {
            "stay": stay,
            "inertia": inertia,
            "chase": chase,
            "intercept": intercept,
        }

    def _prediction_threats(self, visible_pacman):
        if (
            self.pending_pacman_predictions is None
            or self.pending_pacman_prediction_step != self.last_step
            or self.pacman_model_updates < PACMAN_PREDICTOR_MIN_UPDATES
            or self.visible_pacman_streak < PACMAN_PREDICTOR_MIN_STREAK
        ):
            return None
        learned = defaultdict(float)
        for name, endpoint in self.pending_pacman_predictions.items():
            learned[endpoint] += self.pacman_model_weights[name]

        endpoints = tuple(dict.fromkeys(
            endpoint
            for _, _, endpoint in self.model.pacman_actions[visible_pacman]
        ))
        uniform = PACMAN_PREDICTOR_UNIFORM_MIX / max(1, len(endpoints))
        threats = {
            endpoint: uniform for endpoint in endpoints
        }
        learned_mass = 1.0 - PACMAN_PREDICTOR_UNIFORM_MIX
        for endpoint, mass in learned.items():
            threats[endpoint] += learned_mass * mass
        confidence = max(threats.values(), default=0.0)
        if confidence < PACMAN_PREDICTOR_CONFIDENCE:
            return None
        return list(threats.items())

    def _choose_observation_contingent_action(self, ghost, visible_pacman):
        threats = self._prediction_threats(visible_pacman)
        if threats is None:
            return None
        options = self.model._ghost_actions(ghost)
        best = None
        for move, endpoint in options:
            score = self._score_action(
                ghost, visible_pacman, move, endpoint, threats=threats
            )
            risk = self._risk_from_threats(endpoint, threats)
            key = (score, -risk, move == self.last_move, move != Move.STAY)
            if best is None or key > best[0]:
                best = (key, move)
        return best[1] if best else None

    def _threats(self, visible_pacman):
        if visible_pacman is not None:
            endpoints = dict.fromkeys(
                end for _, _, end in self.model.pacman_actions[visible_pacman]
            )
            mass = 1.0 / max(1, len(endpoints))
            return [(point, mass) for point in endpoints]

        return list(self.belief.prob.items())

    def _belief_support_risk(self, ghost_next, visible_pacman):
        threats = self._threats(visible_pacman)
        return self._risk_from_threats(ghost_next, threats)

    @staticmethod
    def _risk_from_threats(ghost_next, threats):
        return sum(
            mass
            for point, mass in threats
            if abs(ghost_next[0] - point[0]) + abs(ghost_next[1] - point[1]) < 2
        )

    def _score_action(
        self, ghost, visible_pacman, move, g_next, threats=None
    ):
        if threats is None:
            threats = self._threats(visible_pacman)
        reverse = {
            Move.UP: Move.DOWN,
            Move.DOWN: Move.UP,
            Move.LEFT: Move.RIGHT,
            Move.RIGHT: Move.LEFT,
        }.get(self.last_move)
        immediate_distances = [
            abs(g_next[0] - p[0]) + abs(g_next[1] - p[1])
            for p, _ in threats
        ]
        unsafe_mass = sum(
            mass
            for (p, mass), distance in zip(threats, immediate_distances)
            if distance < 2
        )
        worst_graph = min(
            self.model.turn_distance(p, g_next) for p, _ in threats
        )
        expected_graph = sum(
            mass * self.model.turn_distance(p, g_next) for p, mass in threats
        )
        seen_mass = sum(
            mass for p, mass in threats if g_next in self.model.visibility[p]
        )
        exits = self.model.degree[g_next]
        dead_depth = self.model.dead_end_depth[g_next]
        reverse_penalty = 1 if move == reverse else 0
        stay_penalty = 1 if move == Move.STAY else 0
        inertia_bonus = 1 if move == self.last_move and move != Move.STAY else 0
        return (
            -unsafe_mass * 1_000_000
            + worst_graph * 2_200
            + expected_graph * 90
            - seen_mass * 1_100
            + exits * 170
            - dead_depth * 3_000
            - reverse_penalty * (6_000 if visible_pacman is None else 220)
            - stay_penalty * (2_500 if visible_pacman is None else 80)
            + inertia_bonus * (1_200 if visible_pacman is None else 140)
        )

    def _apply_belief_risk_shield(self, ghost, visible_pacman, proposal):
        options = self.model._ghost_actions(ghost)
        legal = {move for move, _ in options}
        if proposal not in legal:
            proposal = Move.STAY if Move.STAY in legal else options[0][0]
        if BELIEF_SHIELD_STAGE == 0:
            return proposal
        scored = []
        for move, endpoint in options:
            risk = self._belief_support_risk(endpoint, visible_pacman)
            base = self._score_action(ghost, visible_pacman, move, endpoint)
            scored.append((move, endpoint, risk, base))
        safe = [row for row in scored if row[2] <= BELIEF_RISK_THRESHOLD]
        if safe:
            chosen = max(
                safe,
                key=lambda row: row[3] - BELIEF_SHIELD_LAMBDA * row[2],
            )
            return chosen[0]


        return max(scored, key=lambda row: (-row[2], row[3]))[0]

    def _choose_action(self, ghost, visible_pacman):
        options = self.model._ghost_actions(ghost)
        best = None
        for move, g_next in options:
            score = self._score_action(
                ghost, visible_pacman, move, g_next
            )
            risk = self._belief_support_risk(g_next, visible_pacman)
            key = (score, -risk, move == self.last_move, move != Move.STAY)
            if best is None or key > best[0]:
                best = (key, move)
        return best[1] if best else Move.STAY
