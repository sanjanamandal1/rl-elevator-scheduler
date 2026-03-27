class Elevator:
    def __init__(self, elevator_id, num_floors):
        self.id = elevator_id
        self.current_floor = 0
        self.direction = 0       # -1 down, 0 idle, 1 up
        self.passengers = []     # list of destination floors
        self.door_open = False
        self.num_floors = num_floors

    def move(self):
        """Move one step in current direction."""
        if self.direction == 1 and self.current_floor < self.num_floors - 1:
            self.current_floor += 1
        elif self.direction == -1 and self.current_floor > 0:
            self.current_floor -= 1

    def load_passengers(self, waiting):
        """Board passengers waiting at current floor heading same direction."""
        boarded = []
        for p in waiting:
            if len(self.passengers) < 8:
                self.passengers.append(p.destination)
                boarded.append(p)
        return boarded

    def unload_passengers(self):
        """Unload passengers whose destination is current floor."""
        delivered = self.passengers.count(self.current_floor)
        self.passengers = [d for d in self.passengers if d != self.current_floor]
        return delivered