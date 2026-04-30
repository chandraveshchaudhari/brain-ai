from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Action:
    """Action scaffold representing one pipeline choice."""

    granularity: Optional[str] = None
    fusion: Optional[str] = None
    model: Optional[str] = None
