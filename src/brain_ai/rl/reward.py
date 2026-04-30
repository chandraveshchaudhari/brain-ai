from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RewardConfig:
    lambda_time: float = 0.01
    lambda_complexity: float = 0.01


def compute_reward(
    performance: float,
    time_taken: float,
    complexity: float,
    lam_time: float = 0.01,
    lam_complexity: float = 0.01,
) -> float:
    """Reward = performance - lambda_time * time - lambda_complexity * complexity."""
    return performance - lam_time * time_taken - lam_complexity * complexity


class RewardFunction:
    """Scaffold class for future learnable reward shaping."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def __call__(self, performance: float, time_taken: float, complexity: float) -> float:
        return compute_reward(
            performance=performance,
            time_taken=time_taken,
            complexity=complexity,
            lam_time=self.config.lambda_time,
            lam_complexity=self.config.lambda_complexity,
        )
