class SCANAgent:
    """Classic SCAN (elevator algorithm) — serves requests in current direction first."""
    def __init__(self, num_floors, num_elevators):
        self.num_floors = num_floors
        self.num_elevators = num_elevators

    def get_action(self, state, elevators, building):
        """Return an integer action following SCAN logic."""
        action = 0
        for i, elev in enumerate(elevators):
            calls_above = any(len(building.waiting[f]) > 0
                              for f in range(elev.current_floor + 1, self.num_floors))
            calls_below = any(len(building.waiting[f]) > 0
                              for f in range(0, elev.current_floor))
            has_dest_above = any(d > elev.current_floor for d in elev.passengers)
            has_dest_below = any(d < elev.current_floor for d in elev.passengers)

            if building.waiting[elev.current_floor]:
                elev_action = 3   # open doors
            elif (elev.direction >= 0) and (calls_above or has_dest_above):
                elev_action = 0   # move up
            elif calls_below or has_dest_below:
                elev_action = 1   # move down
            else:
                elev_action = 2   # idle

            action |= (elev_action << (2 * i))
        return action