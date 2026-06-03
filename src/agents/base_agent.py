from __future__ import annotations

from time import perf_counter

from src.core.constants import DEFAULT_TIME_LIMIT
from src.core.types import Action


class BaseAgent:
    def __init__(self, time_limit: float = DEFAULT_TIME_LIMIT):
        self.time_limit = time_limit
        self._deadline = 0.0

    def _start_timer(self) -> None:
        self._deadline = perf_counter() + self.time_limit

    def _time_left(self) -> float:
        return self._deadline - perf_counter()

    @staticmethod
    def _action_name(action: Action) -> str:
        return action.name

