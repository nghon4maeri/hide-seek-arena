from __future__ import annotations

from typing import Dict, Tuple

from src.agents.base_agent import BaseAgent
from src.core.constants import GHOST_SPEED, INF
from src.core.game_state import parse_state
from src.core.map_utils import manhattan
from src.core.types import Action, GameState, SearchTrace
from src.evaluation.features import danger_cells, dead_end_cells
from src.evaluation.hide_eval import evaluate_hide
from src.search.alpha_beta import should_prune
from src.search.bfs import SearchToolkit
from src.search.minimax import is_capture


class HideAgent(BaseAgent):
    """Pacman agent: maximize survival with heuristic search and alpha-beta."""

    def __init__(self, time_limit: float = 0.85):
        super().__init__(time_limit)
        self.ghost_speed = GHOST_SPEED

    def get_action(self, *args, **kwargs):
        return_trace = bool(kwargs.pop("return_trace", kwargs.pop("debug", False)))
        state = parse_state(*args, **kwargs)
        self._start_timer()
        search = SearchToolkit(state.grid)
        trace = SearchTrace(algorithm="HideAgent") if return_trace else None
        action = self._choose_action(search, state, trace)
        result = self._action_name(action)
        if trace is not None:
            trace.chosen_action = result
            return result, trace
        return result

    step = get_action
    act = get_action

    def _choose_action(self, search: SearchToolkit, state: GameState, trace: SearchTrace | None = None) -> Action:
        actions = search.legal_actions(state.my_position, include_stay=True)
        ghost_dist = search.bfs_distance_map(state.enemy_position)
        candidate_scores: Dict[str, float] = {}

        def ordered_key(action: Action) -> Tuple[float, str]:
            p = search.step(state.my_position, action)
            score = self._hide_eval(search, p, state.enemy_position, state.step_number + 1)
            candidate_scores[action.name] = score
            return -score, action.name

        actions.sort(key=ordered_key)
        best = actions[0]
        best_score = -INF
        depth = 4 if len(actions) <= 3 else 3

        for action in actions:
            p = search.step(state.my_position, action)
            if ghost_dist[p[0]][p[1]] < 2:
                score = -50000
            else:
                score = self._minimax_hide(
                    search,
                    p,
                    state.enemy_position,
                    depth - 1,
                    -INF,
                    INF,
                    state.step_number + 1,
                )
            candidate_scores[action.name] = score
            if score > best_score:
                best_score = score
                best = action
            if self._time_left() < 0.04:
                break

        if trace is not None:
            trace.start = state.my_position
            trace.goal = state.enemy_position
            trace.candidate_actions = [action.name for action in actions]
            trace.evaluation_scores = candidate_scores
            trace.danger_cells = danger_cells(search, state.enemy_position, radius=5)
            trace.safe_area = search.safe_reachable_cells(state.my_position, state.enemy_position, max_depth=15)
            trace.dead_end_cells = dead_end_cells(search)
            search.bfs_distance_map(state.enemy_position, trace)
        return best

    def _minimax_hide(
        self,
        search: SearchToolkit,
        pacman,
        ghost,
        depth: int,
        alpha: float,
        beta: float,
        step: int,
    ) -> float:
        if is_capture(pacman, ghost):
            return -100000 + step
        if depth <= 0 or self._time_left() < 0.025:
            return self._hide_eval(search, pacman, ghost, step)

        ghost_actions = search.legal_actions(ghost, include_stay=False)
        ghost_actions.sort(key=lambda a: search.distance(search.step(ghost, a), pacman))
        value = INF
        for ga in ghost_actions[:4]:
            g2 = search.step(ghost, ga)
            if is_capture(pacman, g2):
                return -100000 + step

            p_actions = search.legal_actions(pacman, include_stay=True)
            p_actions.sort(key=lambda a: -self._hide_eval(search, search.step(pacman, a), g2, step + 1))
            child = -INF
            for pa in p_actions[:4]:
                p2 = search.step(pacman, pa)
                score = self._minimax_hide(search, p2, g2, depth - 1, alpha, beta, step + 1)
                child = max(child, score)
                alpha = max(alpha, child)
                if should_prune(alpha, beta) or self._time_left() < 0.025:
                    break
            value = min(value, child)
            beta = min(beta, value)
            if should_prune(alpha, beta) or self._time_left() < 0.025:
                break
        return value

    def _hide_eval(self, search: SearchToolkit, pacman, ghost, step: int) -> float:
        return evaluate_hide(search, pacman, ghost, step, ghost_speed=self.ghost_speed)

