# config.py
NUM_FLOORS = 10
NUM_ELEVATORS = 2
MAX_CAPACITY = 8
EPISODE_STEPS = 500
NUM_EPISODES = 2000

# RL hyperparameters
ALPHA = 0.1          # Learning rate
GAMMA = 0.95         # Discount factor
EPSILON_START = 1.0  # Exploration rate
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# Reward weights
REWARD_WAIT_PENALTY = -1.0
REWARD_DELIVERY = +10.0
REWARD_ENERGY_PENALTY = -0.5
REWARD_IDLE_PENALTY = -0.1

# Extended config
NUM_FLOORS       = 20          # Scale up for DQN
BASE_ARRIVAL_RATE = 0.4
DQN_EPISODES     = 3000
BATCH_SIZE       = 64
REPLAY_BUFFER    = 20000
TARGET_UPDATE    = 100

REWARD_WAIT_PENALTY   = -0.05    # was -1.0
REWARD_DELIVERY       = +20.0    # was +10.0
REWARD_ENERGY_PENALTY = -0.1     # was -0.5
EPISODE_STEPS         = 300 