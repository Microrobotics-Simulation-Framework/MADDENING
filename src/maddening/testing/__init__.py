"""Testing utilities for MADDENING nodes.

Property-based testing harnesses for
:class:`~maddening.core.node.SimulationNode` subclasses, built on
Hypothesis.

Install the ``[verify]`` extra to use this module::

    pip install maddening[verify]

Two sub-modules:

- :mod:`maddening.testing.verification` — ``verify_node`` battery
  (finite outputs, preserved structure, determinism, jit/eager
  agreement, finite gradients, plus opt-in bounds / energy / custom
  invariants), each failure shrunk to a minimal counterexample.
- :mod:`maddening.testing.strategies` — Hypothesis strategies that
  generate states, boundary inputs and timesteps from a node's declared
  interface, for writing your own ``@given`` tests.
"""
