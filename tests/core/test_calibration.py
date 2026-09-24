"""Tests for Phase 2a: Coupling parameter auto-tuning.

Verifies that tune_coupling_params:
1. Returns a TuneResult with expected structure
2. Finds configurations that pass accuracy threshold
3. Prefers lower iteration count among passing configs
4. Handles edge cases (all fail, single config)
"""

import warnings

import jax.numpy as jnp
import pytest

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.simulation.calibration import TuneResult, tune_coupling_params
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


def _build_springs(**coupling_kwargs):
    """Build bidirectional springs with given coupling params."""
    gm = GraphManager()
    a = SpringDamperNode(
        name="spring_a", timestep=0.001,
        stiffness=50.0, damping=1.0, mass=1.0,
        rest_length=1.0, initial_position=0.0,
    )
    b = SpringDamperNode(
        name="spring_b", timestep=0.001,
        stiffness=50.0, damping=1.0, mass=1.0,
        rest_length=1.0, initial_position=2.0,
    )
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
    gm.add_edge("spring_b", "spring_a", "position", "anchor_position")
    gm.add_coupling_group(
        ["spring_a", "spring_b"],
        diagnostics=True,
        **coupling_kwargs,
    )
    return gm


class TestTuneCouplingParamsDeprecation:
    """`tune_coupling_params` is deprecated in favour of `maddening.sysid.fit`."""

    def test_calling_tune_coupling_params_warns_and_names_its_replacement(self):
        """A caller who finds the grid search first is told, at the call
        site, which tool replaces it and when this one goes away.

        The warning is issued before anything is built, so the graph
        builder here stops the search at its first call: running the
        grid compiled two graphs for a check that needs none (4 s on the
        CI runner).  That the deprecated search still runs is
        :class:`TestTuneCouplingParams`'s to show.
        """
        class _Stop(Exception):
            pass

        def _stop(**_group_kw):
            raise _Stop

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            with pytest.raises(_Stop):
                tune_coupling_params(
                    build_graph_fn=_stop,
                    param_grid={"tolerance": [1e-6], "max_iterations": [5]},
                    n_steps=2,
                )

        deprecations = [w for w in record
                        if issubclass(w.category, DeprecationWarning)]
        assert deprecations, "tune_coupling_params no longer warns"
        message = str(deprecations[0].message)
        assert "maddening.sysid.fit" in message
        assert "0.5.0" in message

    def test_tune_coupling_params_is_tagged_deprecated_in_the_registry(self):
        assert tune_coupling_params._stability_level is StabilityLevel.DEPRECATED


# Slow-marked (still run by slow-tests.yml): every test runs a grid search,
# i.e. builds and compiles a graph per configuration plus a reference -- up
# to 15 s each on the CI runner -- over an API deprecated for removal in
# 0.5.0.  The deprecation warning itself is checked on every push above.
@pytest.mark.slow
class TestTuneCouplingParams:
    """Tests for the tune_coupling_params utility."""

    def test_the_deprecated_grid_search_still_runs(self):
        """Deprecated is not broken: it still searches the grid."""
        with pytest.warns(DeprecationWarning, match="maddening.sysid.fit"):
            result = tune_coupling_params(
                build_graph_fn=_build_springs,
                param_grid={"tolerance": [1e-6], "max_iterations": [5]},
                n_steps=2,
            )
        assert isinstance(result, TuneResult)
        assert len(result.all_trials) == 1

    def test_returns_tune_result(self):
        """Should return a TuneResult dataclass."""
        result = tune_coupling_params(
            build_graph_fn=_build_springs,
            param_grid={
                "tolerance": [1e-4, 1e-6],
                "max_iterations": [5, 10],
            },
            n_steps=10,
        )
        assert isinstance(result, TuneResult)
        assert isinstance(result.best_params, dict)
        assert isinstance(result.all_trials, list)
        assert len(result.all_trials) == 4  # 2 x 2 grid

    def test_finds_passing_config(self):
        """Should find at least one config that passes accuracy check."""
        result = tune_coupling_params(
            build_graph_fn=_build_springs,
            param_grid={
                "tolerance": [1e-4, 1e-8],
                "max_iterations": [10, 20],
            },
            n_steps=20,
            accuracy_threshold=0.1,  # generous threshold
        )
        passing = [t for t in result.all_trials if t["passed"]]
        assert len(passing) > 0

    def test_prefers_fewer_iterations(self):
        """Among passing configs, should prefer lowest iteration count."""
        result = tune_coupling_params(
            build_graph_fn=_build_springs,
            param_grid={
                "tolerance": [1e-4, 1e-6, 1e-8],
                "max_iterations": [5, 10, 20],
            },
            n_steps=20,
            accuracy_threshold=0.1,
        )
        passing = [t for t in result.all_trials if t["passed"]]
        if len(passing) > 1:
            min_iters = min(t["total_iters"] for t in passing)
            assert result.best_total_iters == min_iters

    def test_single_config(self):
        """Should work with a single config."""
        result = tune_coupling_params(
            build_graph_fn=_build_springs,
            param_grid={
                "tolerance": [1e-6],
                "max_iterations": [10],
            },
            n_steps=10,
        )
        assert len(result.all_trials) == 1

    def test_custom_metric(self):
        """Should work with a custom state metric."""
        def my_metric(state):
            return float(state["spring_a"]["position"])

        result = tune_coupling_params(
            build_graph_fn=_build_springs,
            param_grid={
                "tolerance": [1e-4, 1e-8],
                "max_iterations": [10],
            },
            n_steps=10,
            state_metric=my_metric,
        )
        assert isinstance(result, TuneResult)
        assert result.best_max_error >= 0.0

    def test_acceleration_grid(self):
        """Should search over acceleration methods."""
        result = tune_coupling_params(
            build_graph_fn=_build_springs,
            param_grid={
                "tolerance": [1e-6],
                "max_iterations": [10],
                "acceleration": ["none", "aitken"],
            },
            n_steps=20,
            accuracy_threshold=0.01,
        )
        assert len(result.all_trials) == 2
        assert result.best_params["acceleration"] in ["none", "aitken"]

    def test_reference_params_override(self):
        """Should accept explicit reference parameters."""
        result = tune_coupling_params(
            build_graph_fn=_build_springs,
            param_grid={
                "tolerance": [1e-4, 1e-6],
                "max_iterations": [10],
            },
            n_steps=10,
            reference_params={
                "tolerance": 1e-10,
                "max_iterations": 30,
            },
        )
        assert isinstance(result, TuneResult)
