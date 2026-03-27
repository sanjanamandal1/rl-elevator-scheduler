from agents.dqn_agent import DQNAgent
import config

class MultiAgentDQN:
    """One independent DQN per elevator — decentralized execution."""
    def __init__(self, state_size, action_size_per_agent=4):
        self.agents = [
            DQNAgent(state_size, action_size_per_agent)
            for _ in range(config.NUM_ELEVATORS)
        ]
        self.action_size_per_agent = action_size_per_agent

    def get_action(self, state):
        """Each agent picks its own action; combine into joint action int."""
        joint = 0
        for i, agent in enumerate(self.agents):
            a = agent.get_action(state)
            joint |= (a << (2 * i))
        return joint

    def update(self, state, joint_action, reward, next_state, done):
        """Each agent learns from the shared reward signal."""
        for i, agent in enumerate(self.agents):
            individual_action = (joint_action >> (2 * i)) & 3
            agent.update(state, individual_action, reward, next_state, done)

    def decay_epsilon(self):
        for agent in self.agents:
            agent.decay_epsilon()

    @property
    def epsilon(self):
        return self.agents[0].epsilon