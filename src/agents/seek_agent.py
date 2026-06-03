from __future__ import annotations

from typing import Dict, Tuple

from src.agents.base_agent import BaseAgent
from src.agents.hide_agent import HideAgent
from src.core.constants import ACTIONS, INF
from src.core.game_state import parse_state
from src.core.types import Action, GameState, SearchTrace
from src.evaluation.features import danger_cells, dead_end_cells
from src.evaluation.seek_eval import evaluate_seek
from src.search.alpha_beta import should_prune
from src.search.bfs import SearchToolkit
from src.search.minimax import is_capture


class SeekAgent(BaseAgent):
    """Ghost agent: catch Pacman with pursuit, interception, and trap pressure."""

    def get_action(self, *args, **kwargs):
        return_trace = bool(kwargs.pop("return_trace", kwargs.pop("debug", False)))
        state = parse_state(*args, **kwargs)
        self._start_timer()
        search = SearchToolkit(state.grid)
        trace = SearchTrace(algorithm="SeekAgent") if return_trace else None
        action = self._choose_action(search, state, trace)
        result = self._action_name(action)
        if trace is not None:
            trace.chosen_action = result
            return result, trace
        return result

    step = get_action
    act = get_action

    def _choose_action(self, search: SearchToolkit, state: GameState, trace: SearchTrace | None = None) -> Action:
        actions = search.legal_actions(state.my_position, include_stay=False)
        if not actions:
            return ACTIONS[-1]

        astar_trace = SearchTrace(algorithm="A*") if trace is not None else None
        path = search.astar_path(state.my_position, state.enemy_position, astar_trace)
        pursuit_next = path[0] if path else None
        candidate_scores: Dict[str, float] = {}

        def ordered_key(action: Action) -> Tuple[float, str]:
            g = search.step(state.my_position, action)
            pursuit_bonus = -5 if pursuit_next == g else 0
            return search.distance(g, state.enemy_position) + pursuit_bonus, action.name

        actions.sort(key=ordered_key)
        best = actions[0]
        best_score = -INF
        depth = 4 if len(actions) <= 3 else 3

        for action in actions:
            ghost = search.step(state.my_position, action)
            score = self._minimax_seek(
                search,
                state.enemy_position,
                ghost,
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
            trace.final_path = path
            trace.explored_nodes = astar_trace.explored_nodes if astar_trace is not None else []
            trace.frontier_snapshots = astar_trace.frontier_snapshots if astar_trace is not None else []
            trace.candidate_actions = [action.name for action in actions]
            trace.evaluation_scores = candidate_scores
            trace.danger_cells = danger_cells(search, state.my_position, radius=5)
            trace.safe_area = search.safe_reachable_cells(state.enemy_position, state.my_position, max_depth=14)
            trace.dead_end_cells = dead_end_cells(search)
        return best

    def _minimax_seek(
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
            return 100000 - 40 * step
        if depth <= 0 or self._time_left() < 0.025:
            return self._seek_eval(search, pacman, ghost, step)

        pac_actions = search.legal_actions(pacman, include_stay=True)
        hide_model = HideAgent(time_limit=max(0.03, self._time_left()))
        pac_actions.sort(key=lambda a: -hide_model._hide_eval(search, search.step(pacman, a), ghost, step))

        value = INF
        for pa in pac_actions[:4]:
            p2 = search.step(pacman, pa)
            if is_capture(p2, ghost):
                child = 100000 - 40 * step
            else:
                ghost_actions = search.legal_actions(ghost, include_stay=False)
                ghost_actions.sort(key=lambda a: search.distance(search.step(ghost, a), p2))
                child = -INF
                for ga in ghost_actions[:4]:
                    g2 = search.step(ghost, ga)
                    score = self._minimax_seek(search, p2, g2, depth - 1, alpha, beta, step + 1)
                    child = max(child, score)
                    alpha = max(alpha, child)
                    if should_prune(alpha, beta) or self._time_left() < 0.025:
                        break
            value = min(value, child)
            beta = min(beta, value)
            if should_prune(alpha, beta) or self._time_left() < 0.025:
                break
        return value

    def _seek_eval(self, search: SearchToolkit, pacman, ghost, step: int) -> float:
        return evaluate_seek(search, pacman, ghost, step)

