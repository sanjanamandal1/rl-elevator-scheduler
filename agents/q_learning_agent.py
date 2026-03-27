import numpy as np
import config

class QLearningAgent:
    """Off-policy TD control (Q-Learning)."""
    def __init__(self, state_size, action_size):
        self.q_table = {}   # sparse dict; use discretized states
        self.alpha   = config.ALPHA
        self.gamma   = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.action_size = action_size

    def _discretize(self, state):
        """Bin continuous state to a hashable key."""
        return tuple(np.round(state, 1))

    def get_action(self, state):
        key = self._discretize(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        q_vals = self.q_table.get(key, np.zeros(self.action_size))
        return int(np.argmax(q_vals))

    def update(self, state, action, reward, next_state, done):
        key      = self._discretize(state)
        next_key = self._discretize(next_state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)

        # Q-Learning: bootstrap from greedy next action
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_key])

        self.q_table[key][action] += self.alpha * (target - self.q_table[key][action])

    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_END,
                           self.epsilon * config.EPSILON_DECAY)