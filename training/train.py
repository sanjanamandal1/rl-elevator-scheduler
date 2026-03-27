from env.elevator_env import ElevatorEnv
from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SARSAAgent
from agents.baseline_agent import SCANAgent
import config
import numpy as np
from tqdm import tqdm
from agents.dqn_agent import DQNAgent
from agents.multi_agent_dqn import MultiAgentDQN

def train_dqn(use_multi_agent=False):
    env = ElevatorEnv()
    obs, _ = env.reset()
    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n

    if use_multi_agent:
        agent = MultiAgentDQN(state_size, action_size_per_agent=4)
        label = "Multi-Agent DQN"
    else:
        agent = DQNAgent(state_size, action_size)
        label = "DQN"

    episode_rewards = []
    episode_losses  = []

    for ep in tqdm(range(config.DQN_EPISODES), desc=f"Training {label}"):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(config.EPISODE_STEPS):
            action     = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state       = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if ep % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"  Ep {ep:4d} | ε={agent.epsilon:.3f} | Avg reward={avg:.1f}")

    # Save model
    if not use_multi_agent:
        agent.save("dqn_model.pth")

    return agent, episode_rewards

def train(agent_type="q_learning"):
    env = ElevatorEnv()
    obs, _ = env.reset()
    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n

    if agent_type == "q_learning":
        agent = QLearningAgent(state_size, action_size)
    else:
        agent = SARSAAgent(state_size, action_size)

    episode_rewards = []

    for ep in tqdm(range(config.NUM_EPISODES), desc=f"Training {agent_type}"):
        state, _ = env.reset()
        total_reward = 0

        if agent_type == "sarsa":
            action = agent.get_action(state)

        for _ in range(config.EPISODE_STEPS):
            if agent_type == "q_learning":
                action = agent.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
            else:  # SARSA
                next_state, reward, done, _, _ = env.step(action)
                next_action = agent.get_action(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
                action = next_action

            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

    return agent, episode_rewards

def resume_dqn(checkpoint_path="dqn_model.pth", extra_episodes=2000):
    """Load saved DQN weights and continue training."""
    env = ElevatorEnv()
    obs, _ = env.reset()
    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    agent.load(checkpoint_path)         # loads weights you already trained
    agent.epsilon = 0.3                 # resume with some exploration still left
    print(f"Resumed from {checkpoint_path} | ε set to 0.3")

    episode_rewards = []

    for ep in tqdm(range(extra_episodes), desc="Resuming DQN"):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(config.EPISODE_STEPS):
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if ep % 200 == 0:
            avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"  Ep {ep:4d} | ε={agent.epsilon:.3f} | Avg={avg:.1f}")

    agent.save("dqn_model_resumed.pth")
    return agent, episode_rewards