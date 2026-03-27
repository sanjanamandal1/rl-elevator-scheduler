import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.building import Building
from env.elevator import Elevator
import config

class ElevatorEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_floors = config.NUM_FLOORS
        self.num_elevators = config.NUM_ELEVATORS

        # Action: for each elevator → {move up, move down, idle, open doors}
        # Flattened: 4^num_elevators joint actions (or per-elevator for simplicity)
        self.action_space = spaces.Discrete(4 * self.num_elevators)

        # State: [floor, direction, #passengers] × num_elevators + [hall_calls per floor × 2]
        obs_size = (3 * self.num_elevators) + (2 * self.num_floors)
        self.observation_space = spaces.Box(
            low=0, high=self.num_floors, shape=(obs_size,), dtype=np.float32
        )

    def reset(self, seed=None):
        self.building = Building(self.num_floors)
        self.elevators = [Elevator(i, self.num_floors) for i in range(self.num_elevators)]
        self.step_count = 0
        self.total_delivered = 0
        self.total_wait = 0
        return self._get_obs(), {}

    def _get_obs(self):
        elev_states = []
        for e in self.elevators:
            elev_states += [e.current_floor / self.num_floors,
                            (e.direction + 1) / 2,
                            len(e.passengers) / config.MAX_CAPACITY]
        hall_calls_up   = [1.0 if self.building.waiting[f] else 0.0
                           for f in range(self.num_floors)]
        hall_calls_down = [1.0 if any(p.destination < f
                           for p in self.building.waiting[f])
                           else 0.0 for f in range(self.num_floors)]
        return np.array(elev_states + hall_calls_up + hall_calls_down, dtype=np.float32)

    def step(self, action):
        self.building.step()
        reward = 0.0

        # Decode action per elevator
        for i, elev in enumerate(self.elevators):
            elev_action = (action >> (2 * i)) & 3   # 2-bit per elevator
            if elev_action == 0:   elev.direction = 1    # up
            elif elev_action == 1: elev.direction = -1   # down
            elif elev_action == 2: elev.direction = 0    # idle
            elif elev_action == 3:                       # open doors
                delivered = elev.unload_passengers()
                boarded = elev.load_passengers(self.building.waiting[elev.current_floor])
                for p in boarded:
                    reward += config.REWARD_DELIVERY * p.reward_multiplier
                    reward += config.REWARD_WAIT_PENALTY * p.wait_time * 0.1
                    self.building.waiting[elev.current_floor].remove(p)
                self.total_delivered += delivered
            elev.move()

        # Penalties
        reward += config.REWARD_WAIT_PENALTY * self.building.total_waiting() * 0.01
        for elev in self.elevators:
            if elev.direction != 0:
                reward += config.REWARD_ENERGY_PENALTY

        self.step_count += 1
        done = self.step_count >= config.EPISODE_STEPS
        return self._get_obs(), reward, done, False, {}