from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class State:
    """RL state scaffold for pipeline decision context."""

    metadata: Dict[str, Any]
