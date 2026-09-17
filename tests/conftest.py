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
