"""Testing utilities for MADDENING nodes.

Property-based testing harnesses for
:class:`~maddening.core.node.SimulationNode` subclasses, built on
Hypothesis.

Install the ``[verify]`` extra to use this module::

    pip install maddening[verify]

Five sub-modules:

- :mod:`maddening.testing.verification` — ``verify_node`` battery
  (finite outputs, preserved structure, determinism, jit/eager
  agreement, finite gradients, plus opt-in bounds / energy / custom
  invariants), each failure shrunk to a minimal counterexample.
- :mod:`maddening.testing.mms` — ``verify_node_order`` /
  ``assert_node_order_verified``: the Method of Manufactured Solutions,
  measuring the observed order of convergence against the order the
  node declares.  The battery above compares the code to itself and so
  cannot see a wrong discretisation; this one compares it to the
  mathematics.
- :mod:`maddening.testing.mapping` — ``verify_mapping`` /
  ``assert_mapping_verified``: the edge between two nodes.  A battery
  for an interface mapping, registered or not, or a whole ``EdgeSpec``:
  linearity, the consistency or conservation it claims, the adjoint
  identity, the derivative with respect to positions, the save/load
  round trip.  Experimental.
- :mod:`maddening.testing.coupled` — ``verify_graph_order`` /
  ``verify_graph_gci``: the order of a coupled graph, with a guard that
  the coupling's iteration error is far below the discretisation error
  being measured.  Experimental.
- :mod:`maddening.testing.strategies` — Hypothesis strategies that
  generate states, boundary inputs and timesteps from a node's declared
  interface, for writing your own ``@given`` tests.
"""
