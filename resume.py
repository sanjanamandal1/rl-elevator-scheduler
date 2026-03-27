# resume.py  — run this instead of main.py
from training.train import resume_dqn
from training.evaluate import evaluate_agent
from env.elevator_env import ElevatorEnv
from agents.baseline_agent import SCANAgent
import config

# Resume DQN from checkpoint
print("=== Resuming DQN training ===")
dqn_agent, new_rewards = resume_dqn("dqn_model.pth", extra_episodes=2000)

# Evaluate the resumed agent
print("\n=== Evaluation after resume ===")
env = ElevatorEnv()
evaluate_agent(dqn_agent, env, agent_name="DQN (resumed)")

class SCANWrapper:
    def __init__(self, e):
        self.scan = SCANAgent(config.NUM_FLOORS, config.NUM_ELEVATORS)
        self.env  = e
    def get_action(self, state):
        return self.scan.get_action(state, self.env.elevators, self.env.building)

evaluate_agent(SCANWrapper(env), env, agent_name="SCAN Baseline")