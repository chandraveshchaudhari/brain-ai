import random
from .action import Action


class RandomPolicy:
    """Simple random policy scaffold for RL action selection."""

    def __init__(self, action_space=None):
        self.action_space = action_space or {}

    def select(self, state):
        # returns an Action
        return Action(
            granularity=random.choice(self.action_space.get("granularity", [None])),
            fusion=random.choice(self.action_space.get("fusion", [None])),
            model=random.choice(self.action_space.get("model", [None])),
        )
