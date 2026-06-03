from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from src.core.types import SearchTrace


def trace_to_dict(trace: SearchTrace) -> Dict[str, Any]:
    return asdict(trace)


def empty_trace(algorithm: str = "agent") -> SearchTrace:
    return SearchTrace(algorithm=algorithm)

