from prototype.simulation_node import SimulationNode

class TableNode(SimulationNode):
    def __init__(self, name: str, timestep: float, state: dict = None):
        super().__init__(name, timestep, state)
        self.state.setdefault("position", 0)  # Table height

    def update(self):
        """Static table, does not move."""
        pass
