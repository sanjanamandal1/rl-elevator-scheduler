import numpy as np
import matplotlib.pyplot as plt
from training.train import train, train_dqn
from training.evaluate import evaluate_agent, plot_training_curves
from env.elevator_env import ElevatorEnv
from agents.baseline_agent import SCANAgent
from utils.visualize import ElevatorDashboard
import config

def plot_all_curves(ql_r, sarsa_r, dqn_r, madqn_r):
    fig, ax = plt.subplots(figsize=(12, 5))
    window  = 50

    def smooth(x):
        return np.convolve(x, np.ones(window) / window, mode='valid')

    ax.plot(smooth(ql_r),    label="Q-Learning",       color="#185FA5", alpha=0.8)
    ax.plot(smooth(sarsa_r), label="SARSA",             color="#0F6E56", alpha=0.8)
    ax.plot(smooth(dqn_r),   label="DQN",               color="#BA7517", linewidth=2)
    ax.plot(smooth(madqn_r), label="Multi-Agent DQN",   color="#993556", linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward (smoothed 50-ep)")
    ax.set_title("All agents — training convergence")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("all_training_curves.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    # --- Train all agents ---
    print("=== Q-Learning ===")
    ql_agent, ql_rewards = train("q_learning")

    print("\n=== SARSA ===")
    sarsa_agent, sarsa_rewards = train("sarsa")

    print("\n=== DQN ===")
    dqn_agent, dqn_rewards = train_dqn(use_multi_agent=False)

    print("\n=== Multi-Agent DQN ===")
    madqn_agent, madqn_rewards = train_dqn(use_multi_agent=True)

    # --- Evaluate ---
    print("\n=== Evaluation ===")
    env = ElevatorEnv()

    evaluate_agent(ql_agent,    env, agent_name="Q-Learning")
    evaluate_agent(sarsa_agent, env, agent_name="SARSA")
    evaluate_agent(dqn_agent,   env, agent_name="DQN")
    evaluate_agent(madqn_agent, env, agent_name="Multi-Agent DQN")

    class SCANWrapper:
        def __init__(self, e):
            self.scan = SCANAgent(config.NUM_FLOORS, config.NUM_ELEVATORS)
            self.env  = e
        def get_action(self, state):
            return self.scan.get_action(state, self.env.elevators, self.env.building)

    evaluate_agent(SCANWrapper(env), env, agent_name="SCAN Baseline")

    # --- Plot training curves ---
    plot_all_curves(ql_rewards, sarsa_rewards, dqn_rewards, madqn_rewards)

    # --- Live Pygame Dashboard (best agent) ---
    print("\n=== Launching live dashboard (DQN agent) ===")
    print("SPACE = toggle speed | ESC = quit")
    vis_env = ElevatorEnv()
    dashboard = ElevatorDashboard(vis_env, dqn_agent, fps=10)
    dashboard.run(max_steps=3000)