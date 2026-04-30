class Trainer:
    """Placeholder Trainer - no RL training implemented (scaffold only)."""

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    def train(self, episodes: int = 1):
        # scaffold: iterate episodes and collect traces
        traces = []
        for _ in range(episodes):
            state = self.env.reset()
            action = self.policy.select(state)
            next_state, reward, done, info = self.env.step(action)
            traces.append((state, action, reward, next_state))
        return traces
