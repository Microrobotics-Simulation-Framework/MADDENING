import jax
import jax.numpy as jnp
from abc import ABC

class SimulationNode(ABC):
    def __init__(self, name: str, timestep: float, state: dict = None):
        """
        Initialize a node in the simulation.

        :param name: Unique identifier for this node.
        :param timestep: The timestep of a simulation in seconds.
        :param state: A dictionary containing the node's state.
        """
        self.name = name
        self.delta_t = timestep
        self.state = state if state is not None else {}
        self.callbacks = []  # Stores functions that update the node's boundary conditions

    def register_boundary_condition(self, callback):
        """
        Register a callback function that updates boundary conditions.

        :param callback: A function that takes `self.state` as input and modifies it.
        """
        if not callable(callback):
            raise TypeError("Callback must be a callable function.")
        self.callbacks.append(callback)

    def update_boundary_conditions(self):
        """
        Execute all registered callbacks to update boundary conditions.
        """
        for callback in self.callbacks:
            callback(self.state)

    def update(self):
        """
        This method must be implemented by subclasses to define their physics logic.
        """
        raise NotImplementedError("The update method must be implemented by subclasses.")

    def step(self):
        """
        First updates boundary conditions via callbacks, then updates physics.
        """
        self.update_boundary_conditions()
        self.update()
