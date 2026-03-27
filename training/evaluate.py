import numpy as np
import matplotlib.pyplot as plt

def evaluate_agent(agent, env, episodes=50, agent_name="Agent"):
    rewards, wait_times = [], []
    for _ in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        for _ in range(500):
            action = agent.get_action(state)
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
            if done: break
        rewards.append(ep_reward)
    print(f"{agent_name} | Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    return rewards

def plot_training_curves(ql_rewards, sarsa_rewards):
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 50
    def smooth(x): return np.convolve(x, np.ones(window)/window, mode='valid')
    ax.plot(smooth(ql_rewards),    label="Q-Learning",  color="#185FA5")
    ax.plot(smooth(sarsa_rewards), label="SARSA",       color="#0F6E56")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward (smoothed)")
    ax.set_title("Training convergence — RL elevator scheduler")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()