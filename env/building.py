import numpy as np
from dataclasses import dataclass, field
import config

@dataclass
class Passenger:
    origin: int
    destination: int
    arrival_time: int
    priority: str = "normal"    # "normal", "vip", "emergency"
    wait_time: int = 0

    @property
    def reward_multiplier(self):
        return {"normal": 1.0, "vip": 2.5, "emergency": 5.0}[self.priority]


class Building:
    def __init__(self, num_floors, arrival_rate=None):
        self.num_floors   = num_floors
        self.base_rate    = arrival_rate or config.BASE_ARRIVAL_RATE
        self.waiting      = [[] for _ in range(num_floors)]
        self.time         = 0

    def _current_arrival_rate(self):
        """Sinusoidal traffic pattern: peak at t=150 (morning rush) and t=400."""
        t = self.time
        base  = self.base_rate
        rush1 = 0.6 * np.sin(np.pi * t / 300) ** 2   # morning rush
        rush2 = 0.4 * np.sin(np.pi * (t - 250) / 300) ** 2  # afternoon
        return base + rush1 + rush2

    def step(self):
        self.time += 1
        rate = self._current_arrival_rate()

        for floor in range(self.num_floors):
            if np.random.random() < rate / self.num_floors:
                dest = np.random.choice(
                    [f for f in range(self.num_floors) if f != floor]
                )
                # Priority distribution: 80% normal, 15% VIP, 5% emergency
                priority = np.random.choice(
                    ["normal", "vip", "emergency"], p=[0.80, 0.15, 0.05]
                )
                self.waiting[floor].append(
                    Passenger(origin=floor, destination=dest,
                              arrival_time=self.time, priority=priority)
                )

        for floor_queue in self.waiting:
            for p in floor_queue:
                p.wait_time += 1

    def total_waiting(self):
        return sum(len(q) for q in self.waiting)

    def priority_waiting(self):
        """Returns weighted waiting count — emergencies count more."""
        total = 0
        for q in self.waiting:
            for p in q:
                total += p.reward_multiplier * p.wait_time
        return total