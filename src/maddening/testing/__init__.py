"""Testing and verification utilities for MADDENING nodes.

This module provides reusable harnesses for formally verifying and
property-testing :class:`~maddening.core.node.SimulationNode`
subclasses.

Install the ``[verify]`` extra to use this module::

    pip install maddening[verify]

Two sub-modules:

- :mod:`maddening.testing.verification` — stelling-based formal
  verification (proves properties over ALL inputs in a declared
  envelope).
- :mod:`maddening.testing.strategies` — Hypothesis strategies for
  property-based testing (samples from a rich input space and shrinks
  counterexamples).
"""
