"""Local configuration for the numerics property suite.

Nothing is configured here any more.  The Hypothesis profiles this suite
used to register for itself (``deadline=None``, ``print_blob=True``) now
live in the ROOT ``tests/conftest.py``, so that property tests elsewhere
in the tree -- ``tests/fmi/``, ``tests/core/`` -- get the same settings
and the same example database instead of hand-rolling them per module.
Select a profile with ``MADDENING_HYPOTHESIS_PROFILE``; see
``docs/developer_guide/testing_standards.md``.

This file is kept as a signpost: it is the first place anyone adding a
property test under this directory looks for the settings.
"""
