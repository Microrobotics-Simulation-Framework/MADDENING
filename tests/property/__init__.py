"""Property-based tests for round-trip-shaped invariants.

``tests/verification/hypothesis/`` covers the physics core (integrators,
coupling, determinism, multirate, sysid).  This package covers the
*serialisation* surfaces added around it -- config dicts, USD stages,
checkpoints -- and the ``AdaptiveNode`` contract, none of which carried a
``@given`` before v0.4.0.
"""
