from .state import State
from .reward import RewardFunction


class Environment:
    """Placeholder RL environment representing pipeline search space."""

    def __init__(self, dataset_metadata=None):
        self.metadata = dataset_metadata or {}
        self.reward_fn = RewardFunction()

    def reset(self):
        return State(metadata=self.metadata)

    def step(self, action):
        # returns (state, reward, done, info)
        reward = self.reward_fn(performance=0.0, time_taken=0.0, complexity=0.0)
        return State(metadata=self.metadata), reward, True, {"action": action}
