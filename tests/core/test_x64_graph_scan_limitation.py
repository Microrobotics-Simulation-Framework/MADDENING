"""MADD-ANO-017: ``jax_enable_x64`` does not reach ``GraphManager``'s scans.

An anomaly entry asserts that a limitation exists.  Nothing else in the tree
executes a graph scan under ``jax_enable_x64`` -- ``JAX_ENABLE_X64`` appears
in no CI workflow, in no ``conftest.py`` and nowhere in ``pyproject.toml`` --
so without this module MADD-ANO-017 is an unpinned claim, of exactly the kind
this release spent its time removing.

These tests fail in **both** directions, which is the point:

* if x64 starts reaching the scan paths, the ``pytest.raises`` blocks fail and
  say that MADD-ANO-017 can be closed;
* if the failure changes shape -- a different exception type, or a message
  that no longer describes a carry dtype change -- the shape assertions fail
  and say that MADD-ANO-017's recorded mechanism is stale;
* if the underlying dtype asymmetry is repaired on either side (params
  narrowed to float32, or the state seed widened to the canonical dtype),
  :class:`TestTheDtypeAsymmetryThatCausesIt` fails and names which side moved.

The node-dependence MADD-ANO-017 records is pinned here too, and it is not
what it looks like.  The discriminator is *whether a parameter reaches an
output field at all*, not how the node computes: ``TableNode.update`` returns
its state unchanged and reads no parameter, so it scans under x64; every node
that consumes a parameter promotes its carry and raises, the spring included.

``jax.config.update`` is process-global, so every test here runs inside
:func:`_x64`, which restores the previous setting the way the ``_float64``
fixture in ``tests/verification/test_mms_order_ode_nodes.py`` does.
``test_the_x64_context_manager_restores_the_previous_setting`` is defined last
in the module so that it observes the state the rest of the module left.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

#: The setting the session was in when this module was imported.  Asserted
#: against rather than a hard ``False`` so that a deliberate
#: ``JAX_ENABLE_X64=1`` run -- the run that found MADD-ANO-017 -- still
#: reports a leak rather than a spurious failure.
_X64_AT_IMPORT = jax.config.jax_enable_x64

#: Fix the ID in one place so a failure message cannot drift from the entry.
_ANOMALY = "MADD-ANO-017"

_UPDATE_THE_ENTRY = (
    f"Update {_ANOMALY} in docs/validation/known_anomalies.yaml (and this "
    f"module, which is its only verification entry) in the same commit."
)


@contextlib.contextmanager
def _x64():
    """Run the body in double precision, restoring the global setting."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _compiled(node) -> GraphManager:
    """A freshly compiled single-node graph.

    Fresh per measurement, deliberately.  ``step()`` stores the promoted
    float64 state (see :class:`TestTheSilentHalf`), so a graph that has been
    stepped once no longer exhibits the defect -- reusing one graph across
    these tests would make half of them pass for the wrong reason.
    """
    gm = GraphManager()
    gm.add_node(node)
    gm.compile()
    return gm


def _dtypes(tree) -> set[str]:
    return {str(jnp.asarray(leaf).dtype) for leaf in jax.tree.leaves(tree)}


# ---------------------------------------------------------------------------
# The fixture has to be able to express the defect
# ---------------------------------------------------------------------------

def test_the_context_manager_actually_puts_jax_in_double_precision():
    """Without this, every ``TableNode`` assertion below is vacuous.

    ``TestANodeThatReachesNoParameterStillScans`` asserts that something
    *works* under x64.  If ``_x64`` silently failed to enable x64 those
    assertions would pass on the float32 default and pin nothing at all, and
    the ``raises`` tests would be the only live checks in the module.
    """
    with _x64():
        assert jax.config.jax_enable_x64 is True
        assert jnp.result_type(float) == jnp.float64
        assert jnp.zeros(()).dtype == jnp.float64


# ---------------------------------------------------------------------------
# The mechanism: params at the canonical dtype, state pinned to float32
# ---------------------------------------------------------------------------

class TestTheDtypeAsymmetryThatCausesIt:
    """``params_pytree`` follows ``jax_enable_x64``; ``initial_state`` does not.

    Pinned separately from the ``TypeError`` because it is the part a fix
    would move.  A fix on either side makes exactly one of these two fail,
    which tells the next reader which direction the repair came from.
    """

    @pytest.mark.parametrize(
        "make_node",
        [
            pytest.param(
                lambda: BallNode(name="ball", timestep=0.01,
                                 initial_position=5.0),
                id="ball",
            ),
            pytest.param(
                lambda: SpringDamperNode(name="spring", timestep=0.01,
                                         rest_length=0.3,
                                         initial_position=0.5),
                id="spring",
            ),
        ],
    )
    def test_the_params_pytree_is_float64_and_the_seed_state_is_float32(
        self, make_node
    ):
        with _x64():
            gm = _compiled(make_node())
            params_dtypes = _dtypes(gm.params)
            state_dtypes = _dtypes(
                {n: gm.get_node_state(n) for n in gm.node_names}
            )

        assert params_dtypes, (
            "this graph has no parameter leaves, so it cannot express "
            f"{_ANOMALY}'s mechanism -- pick a node with parameters"
        )
        assert params_dtypes == {"float64"}, (
            f"{_ANOMALY} records `gm.params` as float64 under x64 "
            f"(`SimulationNode.params_pytree` places a Python float at "
            f"`jnp.zeros(()).dtype`); this graph's params are "
            f"{sorted(params_dtypes)}.  If the params side has been narrowed "
            f"back to float32, the scan below stops raising but x64 is being "
            f"ignored rather than honoured, which is the pre-0.4.0 behaviour "
            f"and is NOT a fix.  {_UPDATE_THE_ENTRY}"
        )
        assert state_dtypes == {"float32"}, (
            f"{_ANOMALY} records the seed state as float32 under x64 (every "
            f"node's `initial_state()` writes `dtype=jnp.float32`); this "
            f"graph's seed state is {sorted(state_dtypes)}.  If the state "
            f"seed now follows the canonical float dtype, that IS the fix "
            f"{_ANOMALY} asks for and the entry should be resolved.  "
            f"{_UPDATE_THE_ENTRY}"
        )


# ---------------------------------------------------------------------------
# The limitation itself
# ---------------------------------------------------------------------------

class TestAFreshlyCompiledGraphRefusesToScanUnderX64:
    """The loud half, on both scan entry points and on both sample nodes.

    The spring is here rather than in the "still works" class on purpose.
    MADD-ANO-017 was first written up with the spring as the node that
    narrows; it does not.  That reading came from a harness that passed
    explicitly float32 params (``_with_params`` in
    ``tests/property/test_sysid_contract.py`` does exactly this), which is the
    entry's workaround (b) and not a property of the node.
    """

    _NODES = [
        pytest.param(
            lambda: BallNode(name="ball", timestep=0.01, initial_position=5.0),
            "ball",
            id="ball",
        ),
        pytest.param(
            lambda: SpringDamperNode(name="spring", timestep=0.01,
                                     rest_length=0.3, initial_position=0.5),
            "spring",
            id="spring",
        ),
    ]

    @staticmethod
    def _refusal_message(call, entry_point: str) -> str:
        """Call ``call`` and return the ``TypeError`` it must raise.

        Written out rather than ``pytest.raises(TypeError)`` because the
        no-raise branch is the one that matters most here: a bare
        ``Failed: DID NOT RAISE <class 'TypeError'>`` is the single most
        likely message a future maintainer will see from this module, and it
        has to say what to do about it.
        """
        try:
            call()
        except TypeError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - shape change, report it
            pytest.fail(
                f"{entry_point} raised {type(exc).__name__} under x64, but "
                f"{_ANOMALY} records a TypeError from `lax.scan` about the "
                f"carry dtype.  The mechanism has changed.  "
                f"{_UPDATE_THE_ENTRY}\n\n{exc}"
            )
        pytest.fail(
            f"{entry_point} now succeeds under `jax_enable_x64` on a freshly "
            f"compiled graph.  {_ANOMALY} records that it cannot: the params "
            f"pytree is float64 and the state seed is float32, so `lax.scan` "
            f"refuses the carry.  If x64 now reaches the graph scan paths, "
            f"that limitation is gone and the entry should be resolved -- "
            f"check `TestTheDtypeAsymmetryThatCausesIt` first, because "
            f"narrowing the params back to float32 also makes this pass and "
            f"is NOT a fix.  {_UPDATE_THE_ENTRY}"
        )

    @staticmethod
    def _assert_is_the_recorded_failure(message: str, node_name: str) -> None:
        """The failure's *shape*, not JAX's exact wording.

        Deliberately three tokens rather than the whole sentence: the
        sentence is jaxlib's and this repository runs two different jaxlibs
        (0.11.0 locally, 0.10.2 in CI), so pinning it verbatim would pin the
        dependency instead of the defect.  ``carry``, ``float32`` and
        ``float64`` together say "the scan carry changed precision", which is
        the mechanism the entry records and the thing that must not change
        quietly.
        """
        lowered = message.lower()
        for token in ("carry", "float32", "float64", node_name):
            assert token in lowered, (
                f"the scan failed under x64, but not in the shape "
                f"{_ANOMALY} records: the message does not mention "
                f"{token!r}.  {_ANOMALY} says the carry changes from float32 "
                f"to float64 in a named node's state.  Either the mechanism "
                f"has changed or jaxlib has reworded the error.  "
                f"{_UPDATE_THE_ENTRY}\n\nfull message:\n{message}"
            )

    @pytest.mark.parametrize("make_node, node_name", _NODES)
    def test_run_scan_raises_on_the_carry_dtype(self, make_node, node_name):
        with _x64():
            gm = _compiled(make_node())
            message = self._refusal_message(
                lambda: gm.run_scan(4), "GraphManager.run_scan"
            )
        self._assert_is_the_recorded_failure(message, node_name)

    @pytest.mark.parametrize("make_node, node_name", _NODES)
    def test_run_scan_with_history_raises_the_same_way(
        self, make_node, node_name
    ):
        with _x64():
            gm = _compiled(make_node())
            message = self._refusal_message(
                lambda: gm.run_scan_with_history(4),
                "GraphManager.run_scan_with_history",
            )
        self._assert_is_the_recorded_failure(message, node_name)

    def test_the_same_graphs_scan_without_x64(self):
        """The control.

        Without it, a graph broken for some unrelated reason would satisfy
        every ``raises`` above and the module would report MADD-ANO-017 while
        measuring something else entirely.
        """
        assert jax.config.jax_enable_x64 is _X64_AT_IMPORT
        if jax.config.jax_enable_x64:
            pytest.skip(
                "this session was started with x64 already enabled "
                "(JAX_ENABLE_X64=1), so there is no float32 control to take"
            )
        for param in self._NODES:
            make_node, _ = param.values
            gm = _compiled(make_node())
            out = gm.run_scan(4)
            assert _dtypes(out) == {"float32"}


class TestANodeThatReachesNoParameterStillScans:
    """MADD-ANO-017's node-dependence, pinned rather than asserted.

    ``TableNode.update`` returns its state unchanged and reads nothing out of
    ``params``, so no float64 leaf reaches its output and the carry stays
    float32.  If this ever starts raising, the defect is wider than the entry
    says -- it would no longer be about parameters reaching outputs -- and the
    entry's mechanism paragraph is wrong.
    """

    def test_a_table_graph_scans_under_x64_and_stays_float32(self):
        with _x64():
            gm = _compiled(TableNode(name="table", timestep=0.01, position=0.0))
            assert _dtypes(gm.params) == {"float64"}, (
                "TableNode has no float64 params under x64, so this graph "
                f"cannot distinguish {_ANOMALY}'s 'a param must reach an "
                "output' mechanism from 'this node has no params' -- the "
                "test would pass for the wrong reason"
            )
            try:
                out = gm.run_scan(4)
            except TypeError as exc:  # pragma: no cover - the failing branch
                pytest.fail(
                    f"TableNode now fails to scan under x64 too, so "
                    f"{_ANOMALY} is not node-dependent in the way it "
                    f"records: it says a node whose `update()` reaches no "
                    f"parameter keeps a float32 carry and scans.  "
                    f"{_UPDATE_THE_ENTRY}\n\n{exc}"
                )
            assert _dtypes(out) == {"float32"}, (
                f"TableNode's carry is no longer float32 under x64 "
                f"({sorted(_dtypes(out))}).  {_ANOMALY}'s mechanism turns on "
                f"the carry dtype, so whichever way this moved, the entry "
                f"needs re-deriving.  {_UPDATE_THE_ENTRY}"
            )


class TestTheSilentHalf:
    """``step()`` does not raise -- it promotes and stores the promotion.

    This is the half MADD-ANO-017 rates ``context_dependent`` for, and the
    reason the entry is not ``enhancement``: a user who enables x64 for
    precision gets double precision from the second step onward on a seed
    that was silently rounded to single.  It is also why every graph here is
    built fresh; a stepped graph no longer reproduces the scan failure.
    """

    def test_step_promotes_the_stored_state_instead_of_raising(self):
        with _x64():
            gm = _compiled(
                BallNode(name="ball", timestep=0.01, initial_position=5.0)
            )
            assert _dtypes(
                {n: gm.get_node_state(n) for n in gm.node_names}
            ) == {"float32"}
            out = gm.step()
            assert _dtypes(out) == {"float64"}, (
                f"{_ANOMALY} records `step()` as silently promoting the carry "
                f"to float64 while the seed stays float32; it returned "
                f"{sorted(_dtypes(out))}.  If `step()` now raises, or now "
                f"stays float32, the entry's severity and safety judgement "
                f"both rest on behaviour that has changed.  "
                f"{_UPDATE_THE_ENTRY}"
            )
            assert _dtypes(
                {n: gm.get_node_state(n) for n in gm.node_names}
            ) == {"float64"}, (
                f"{_ANOMALY} records that the promoted state is *stored*, "
                f"which is what makes a later `run_scan` succeed and what "
                f"makes the defect a property of a fresh graph.  "
                f"{_UPDATE_THE_ENTRY}"
            )

    def test_a_scan_after_a_step_succeeds_because_the_seed_was_promoted(self):
        with _x64():
            gm = _compiled(
                BallNode(name="ball", timestep=0.01, initial_position=5.0)
            )
            gm.step()
            out = gm.run_scan(4)
            assert _dtypes(out) == {"float64"}, (
                f"{_ANOMALY} records that one `step()` unblocks the scan by "
                f"promoting the stored seed.  {_UPDATE_THE_ENTRY}"
            )


class TestTheRecordedWorkarounds:
    """Both workarounds the entry offers, so neither can rot unnoticed.

    A workaround in an IEC 62304 known-anomalies list is an instruction to a
    user; an untested one is a claim.
    """

    def test_promoting_the_seed_state_gives_a_genuine_float64_scan(self):
        """Workaround (a) -- the one to use when precision was the point."""
        with _x64():
            gm = _compiled(
                BallNode(name="ball", timestep=0.01, initial_position=5.0)
            )
            for name in gm.node_names:
                gm.set_node_state(name, {
                    field: jnp.asarray(value, jnp.float64)
                    for field, value in gm.get_node_state(name).items()
                })
            out = gm.run_scan(4)
            assert _dtypes(out) == {"float64"}, (
                f"{_ANOMALY}'s workaround (a) no longer produces a float64 "
                f"scan ({sorted(_dtypes(out))}).  {_UPDATE_THE_ENTRY}"
            )

    def test_narrowing_the_params_scans_in_float32_not_float64(self):
        """Workaround (b) -- and the reason the entry warns about it.

        It makes the scan run; it does not give double precision.  Asserting
        the float32 result is what stops (b) being read as a fix.
        """
        with _x64():
            gm = _compiled(
                BallNode(name="ball", timestep=0.01, initial_position=5.0)
            )
            narrowed = jax.tree.map(
                lambda leaf: jnp.asarray(leaf, jnp.float32), gm.params
            )
            out = gm.run_scan(4, params=narrowed)
            assert _dtypes(out) == {"float32"}, (
                f"{_ANOMALY}'s workaround (b) is recorded as scanning in "
                f"float32 -- pre-0.4.0 behaviour, not double precision.  It "
                f"produced {sorted(_dtypes(out))}.  {_UPDATE_THE_ENTRY}"
            )


# ---------------------------------------------------------------------------
# Defined last: everything above has already run by the time this executes
# ---------------------------------------------------------------------------

def test_the_x64_context_manager_restores_the_previous_setting():
    """``jax.config.update`` is global; a leak would silently re-type the
    rest of the session, and the symptom would surface in an unrelated file.
    """
    assert jax.config.jax_enable_x64 is _X64_AT_IMPORT, (
        "this module left jax_enable_x64 set to "
        f"{jax.config.jax_enable_x64!r}, but the session was in "
        f"{_X64_AT_IMPORT!r} when it was imported.  Every test here must run "
        "inside the _x64() context manager."
    )
    assert jnp.zeros(()).dtype == (
        jnp.float64 if _X64_AT_IMPORT else jnp.float32
    )
