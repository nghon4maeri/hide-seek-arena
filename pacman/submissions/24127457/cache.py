"""LRU cache for paths and distance maps.

Used by both PacmanAgent (path cache) and GhostAgent (distance-map cache)
to avoid redundant recomputation across steps.
"""

from collections import OrderedDict
from typing import Any, Optional


class LRUCache:
    """Bounded Least-Recently-Used cache with O(1) get/put."""

    def __init__(self, maxsize: int = 64):
        self._maxsize = max(1, maxsize)
        self._store: OrderedDict = OrderedDict()

    def get(self, key: Any) -> Optional[Any]:
        """Return cached value, moving key to most-recently-used end."""
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: Any, value: Any) -> None:
        """Insert or update key, evicting oldest if at capacity."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: Any) -> bool:
        return key in self._store
