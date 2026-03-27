# run_dashboard.py
import sys
from env.elevator_env import ElevatorEnv
from agents.dqn_agent import DQNAgent
from agents.multi_agent_dqn import MultiAgentDQN
from agents.baseline_agent import SCANAgent
from utils.visualize import ElevatorDashboard
import config

def make_scan_agent(env):
    """Wraps SCAN so it has a get_action(state) interface."""
    class SCANWrapper:
        def __init__(self):
            self.scan    = SCANAgent(config.NUM_FLOORS, config.NUM_ELEVATORS)
            self.epsilon = 0.0
        def get_action(self, state):
            return self.scan.get_action(state, env.elevators, env.building)
    return SCANWrapper()

if __name__ == "__main__":
    # Pick agent via command line:  python run_dashboard.py dqn / madqn / scan
    mode = sys.argv[1] if len(sys.argv) > 1 else "dqn"

    env = ElevatorEnv()
    obs, _ = env.reset()
    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n

    if mode == "madqn":
        agent = MultiAgentDQN(state_size, action_size_per_agent=4)
        # load each sub-agent if you saved them
        # agent.agents[0].load("dqn_agent0.pth")
        name  = "Multi-Agent DQN"

    elif mode == "scan":
        agent = make_scan_agent(env)
        name  = "SCAN Baseline"

    else:  # default: dqn
        agent = DQNAgent(state_size, action_size)
        agent.load("dqn_model.pth")        # loads your trained weights
        agent.epsilon = 0.05               # almost greedy during demo
        name  = "DQN (trained)"

    print(f"Launching dashboard — agent: {name}")
    print("Controls:  SPACE = toggle speed   R = reset episode   ESC = quit")

    dash = ElevatorDashboard(env, agent, agent_name=name, fps=10)
    dash.run(max_steps=10000)