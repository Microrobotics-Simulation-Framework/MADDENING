"""Shared fixtures and global Hypothesis configuration for MADDENING tests.

Hypothesis profiles are registered *here*, in the root conftest, so that
every property test in the tree gets them -- not just the numerics suite
under ``tests/verification/hypothesis/``.  pytest imports the rootdir
conftest before any plugin's ``pytest_configure`` runs, so the selected
profile is in force for collection and for every test module, with or
without the Hypothesis pytest plugin (the suite runs with
``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`` almost everywhere, which means the
plugin and its ``--hypothesis-profile`` flag are *not* available).

Select a profile with the ``MADDENING_HYPOTHESIS_PROFILE`` environment
variable; see ``docs/developer_guide/testing_standards.md``.
"""

import os
# Force CPU backend for tests.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import pytest
import jax.numpy as jnp

from hypothesis import HealthCheck, settings
from hypothesis.database import DirectoryBasedExampleDatabase

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode

# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------
# One example database for the whole repository, at a fixed absolute path
# derived from this file rather than from the working directory, so a run
# from any cwd (and the ``actions/cache`` step in the ``verify-hypothesis``
# CI job) reads and writes the same directory.  ``.hypothesis/`` is
# git-ignored.  A failing example found once is replayed first on the next
# run, which is what turns a one-in-a-thousand flake into a hard failure.
_REPO_ROOT = Path(__file__).resolve().parent.parent
HYPOTHESIS_DATABASE_DIR = _REPO_ROOT / ".hypothesis" / "examples"

_COMMON = dict(
    # JAX JIT compilation makes the first execution of any example take
    # hundreds of milliseconds, which no wall-clock deadline survives.
    deadline=None,
    # Print the ``@reproduce_failure`` blob so a CI failure can be replayed
    # verbatim on a laptop that does not share the example database.
    print_blob=True,
    database=DirectoryBasedExampleDatabase(HYPOTHESIS_DATABASE_DIR),
    # Tracing + compiling a graph per example is legitimately slow; the
    # deadline is already off, the too_slow *health check* would still
    # abort otherwise-healthy physics properties.
    suppress_health_check=[HealthCheck.too_slow],
)

# ``dev`` -- the default for a local run.  Deliberately shallow: the full
# suite is ~50 minutes already and a property test you never run is worth
# nothing.
settings.register_profile("dev", max_examples=50, **_COMMON)
# ``ci`` -- what the ``verify-hypothesis`` job runs.  4x the examples; the
# shared example database makes anything it finds replayable.
settings.register_profile("ci", max_examples=200, **_COMMON)

HYPOTHESIS_PROFILES = ("dev", "ci")
HYPOTHESIS_PROFILE = os.environ.get("MADDENING_HYPOTHESIS_PROFILE", "dev")
if HYPOTHESIS_PROFILE not in HYPOTHESIS_PROFILES:
    raise RuntimeError(
        f"MADDENING_HYPOTHESIS_PROFILE={HYPOTHESIS_PROFILE!r} is not a known "
        f"Hypothesis profile; expected one of {list(HYPOTHESIS_PROFILES)}"
    )
settings.load_profile(HYPOTHESIS_PROFILE)


# ---------------------------------------------------------------------------
# Depth tiers
# ---------------------------------------------------------------------------
# A per-test ``@settings(max_examples=N)`` *overrides* the profile, so a
# tree full of hand-picked counts leaves ``ci`` no deeper than ``dev``.
# Measured before these tiers existed, ``tests/verification/hypothesis/``
# took 707 s under ``dev`` and 721 s under ``ci`` -- 2% apart, for a
# profile that asks for four times the search.
#
# A property that genuinely needs its own cap therefore names a *tier*
# instead of a number.  A tier is resolved once, here, from whatever
# profile is active, and is named for what one example COSTS -- which is
# the only thing the test author can judge.  How deep to search at that
# cost is the profile's call, not the test's.
#
# * ``EXAMPLES_CHEAP`` -- a pure function on scalars or small arrays: no
#   JAX trace, no graph build, no fresh compile.  A few milliseconds an
#   example at most, so search several times wider than the baseline.
# * ``EXAMPLES_STANDARD`` -- the profile's own depth, for the middle
#   ground: an eager node update on a fixed shape, a call into an
#   already-compiled step, a constrain/unconstrain round trip.
# * ``EXAMPLES_COSTLY`` -- a fresh JAX trace and compile, a dense solve
#   or a ``vjp`` per draw, an optimiser loop, a multi-device
#   ``shard_map``, a full rollout, a graph built and compiled per draw.
#   Tens of milliseconds an example and up, so search a fraction as wide.
#
# Absolute values remain legal where the number encodes a real
# constraint -- an exhausted search space, or an example measured in
# seconds -- and, as before, need a comment saying why.  The house rule's
# floor of 20 binds the tiers too: ``EXAMPLES_COSTLY`` never resolves
# below it, however shallow a future profile is.  See
# ``docs/developer_guide/testing_standards.md``.
#
# The tiers are read from the profile loaded above, i.e. from
# ``MADDENING_HYPOTHESIS_PROFILE``.  A ``--hypothesis-profile`` flag (only
# available when the Hypothesis plugin is loaded, which the suite
# normally disables) changes the profile but not the already-resolved
# tiers, so the report header below prints both.
EXAMPLES_FLOOR = 20

_PROFILE_EXAMPLES = settings.default.max_examples

#: Cheap per example -- four times the profile (``dev`` 200, ``ci`` 800).
EXAMPLES_CHEAP = 4 * _PROFILE_EXAMPLES
#: The profile's own depth (``dev`` 50, ``ci`` 200).
EXAMPLES_STANDARD = _PROFILE_EXAMPLES
#: Costly per example -- two fifths of the profile, never below the
#: house floor (``dev`` 20, ``ci`` 80).
EXAMPLES_COSTLY = max(EXAMPLES_FLOOR, (2 * _PROFILE_EXAMPLES) // 5)


def pytest_configure(config):
    """CI-only plugins, each switched on by its environment variable.

    ``MADDENING_TEST_SHARD=i/N`` keeps one CI shard's test files
    (``tests/_sharding.py``).  ``MADDENING_TEST_JAX_TIMING=1`` records each
    test's JAX trace / lower / compile times into the JUnit XML
    (``tests/_jax_timing.py``), which ``scripts/report_test_durations.py``
    reads.
    """
    spec = os.environ.get("MADDENING_TEST_SHARD")
    if spec:
        from tests import _sharding
        _sharding.register(config, spec)
    if os.environ.get("MADDENING_TEST_JAX_TIMING") == "1":
        from tests import _jax_timing
        _jax_timing.register(config)


def pytest_report_header(config):
    """Report the active profile even when the Hypothesis plugin is off.

    ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`` suppresses Hypothesis's own
    header, and "which profile ran?" is the first question asked of any
    property-test failure.
    """
    # Read the live profile rather than the environment variable: when the
    # plugin *is* loaded, an explicit ``--hypothesis-profile`` wins, and the
    # header has to say what actually ran.
    return (
        f"hypothesis profile: {settings.get_current_profile_name()} "
        f"(max_examples={settings.default.max_examples}, "
        f"tiers cheap/standard/costly="
        f"{EXAMPLES_CHEAP}/{EXAMPLES_STANDARD}/{EXAMPLES_COSTLY}, "
        f"database={HYPOTHESIS_DATABASE_DIR})"
    )


@pytest.fixture
def ball_node():
    """A ball starting at height 5 with zero velocity."""
    return BallNode(name="ball", timestep=0.01, initial_position=5.0,
                    initial_velocity=0.0, elasticity=0.7)


@pytest.fixture
def table_node():
    """A table at height 0."""
    return TableNode(name="table", timestep=0.01, position=0.0)


@pytest.fixture
def bouncing_ball_graph(ball_node, table_node):
    """A compiled bouncing-ball graph (table -> ball)."""
    gm = GraphManager()
    gm.add_node(table_node)
    gm.add_node(ball_node)
    gm.add_edge(source="table", target="ball",
                source_field="position", target_field="table_position")
    gm.compile()
    return gm
