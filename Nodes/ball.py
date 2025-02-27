from prototype.simulation_node import SimulationNode

class BallNode(SimulationNode):
    def __init__(self, name: str, timestep: float, state: dict = None):
        super().__init__(name, timestep, state)
        self.state.setdefault("position", 0.0)
        self.state.setdefault("velocity", 0.0)
        self.state.setdefault("elasticity", 0.8)  # Default elasticity

    def update(self):
        """Physics update: gravity and movement."""
        position = self.state["position"]
        velocity = self.state["velocity"]
        gravity = -9.81  # m/s^2

        velocity += gravity * self.delta_t
        position += velocity * self.delta_t

        self.state["position"] = position
        self.state["velocity"] = velocity

