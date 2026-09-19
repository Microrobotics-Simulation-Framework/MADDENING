"""The precision contract of :meth:`SimulationNode.params_pytree`.

The pytree is the entry point for every parameter the framework
differentiates, serialises or fits, so the dtype it chooses is the dtype
a user's constant *has* from then on.  Two things must hold: one value
must not acquire two dtypes depending on how it was spelled, and nothing
may be narrowed below the precision the rest of the graph is working in.

``float`` was pinned to ``float32`` outright, and ``numpy.float64`` is a
subclass of ``float``, so under ``jax_enable_x64`` a Python float, an
``np.float64`` scalar and a list of floats all came back float32 while
the same number written as an array came back float64 -- a silent 1.7e-8
shift for three spellings out of six.  Reproducers:
``benchmarks/results/audit_040_final/params-io/repro/r13_dtype_narrowing.py``
and ``r12_dtype_x64.py``; ``.../adaptive/repro_params_float32_under_x64.py``
for the same bug reached from an ``AdaptiveNode``.
"""

import contextlib
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.spring import SpringDamperNode
from tests.conftest import EXAMPLES_STANDARD

#: A value no float32 can hold exactly, so a narrowing shows up in the
#: value and not only in the dtype.
VALUE = 0.12345678901234568


@contextlib.contextmanager
def x64(enabled: bool):
    """``jax_enable_x64`` for the duration of the block.

    Same shape as ``tests/property/test_sysid_contract.py``: a context
    manager rather than a fixture, so no other test in the module is
    dragged into float64.
    """
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


class _Probe(SimulationNode):
    """A node whose only job is to carry one parameter into the pytree."""

    def initial_state(self):
        return {"x": jnp.zeros(())}

    def update(self, s, bi, dt, params):
        return {"x": s["x"] + params["gain"]}


def _spellings(value):
    """The same number, written the six ways a user might write it."""
    return {
        "python float": float(value),
        "np.float64 scalar": np.float64(value),
        "np.float64 0-d array": np.array(value, dtype=np.float64),
        "np.float64 1-d array": np.array([value], dtype=np.float64),
        "list of python floats": [float(value)],
        # No explicit ``dtype=``: JAX itself warns (and this suite errors)
        # when an explicit float64 request cannot be honoured, which is
        # exactly the loudness the pytree entry was missing.
        "jnp array": jnp.asarray(np.array([value], dtype=np.float64)),
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_one_value_reaches_the_pytree_as_one_dtype_however_it_is_spelled(enabled):
    with x64(enabled):
        got = {}
        for name, spelled in _spellings(VALUE).items():
            leaf = _Probe("n", 0.01, gain=spelled).params_pytree()["gain"]
            got[name] = (leaf.dtype, float(np.asarray(leaf).reshape(-1)[0]))
        dtypes = {d for d, _ in got.values()}
        values = {v for _, v in got.values()}
        assert len(dtypes) == 1, got
        assert len(values) == 1, got


def test_a_parameter_keeps_float64_when_the_user_asked_for_x64():
    """The narrowing that mattered: ``jax_enable_x64`` is an explicit
    request for double precision, and the pytree used to ignore it for
    every spelling that is not already an array."""
    with x64(True):
        for name, spelled in _spellings(VALUE).items():
            leaf = _Probe("n", 0.01, gain=spelled).params_pytree()["gain"]
            assert leaf.dtype == np.dtype("float64"), name
            assert float(np.asarray(leaf).reshape(-1)[0]) == VALUE, name


def test_the_default_precision_is_unchanged_without_x64():
    """float32 is the working precision when x64 is off, and the pytree
    still places every spelling there -- this change is not a widening."""
    with x64(False):
        for name, spelled in _spellings(VALUE).items():
            leaf = _Probe("n", 0.01, gain=spelled).params_pytree()["gain"]
            assert leaf.dtype == np.dtype("float32"), name


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_an_array_parameter_keeps_the_floating_dtype_it_was_given(dtype):
    """A value that states a precision keeps it; only a value that
    states none (a Python float, a list) takes the working one."""
    with x64(True):
        arr = jnp.asarray([1.5, 2.5], dtype=dtype)
        leaf = _Probe("n", 0.01, gain=arr).params_pytree()["gain"]
        assert leaf.dtype == jnp.dtype(dtype)


def test_structural_entries_are_still_excluded():
    """Ints, bools, strings, dicts, ``None`` and empty arrays change
    shapes or the trace, so they stay on the recompile path."""
    node = _Probe("n", 0.01, gain=1.0, n_cells=8, flag=True, label="x",
                  nested={"a": 1.0}, nothing=None,
                  empty=np.zeros((0,), dtype=np.float64),
                  ints=[1, 2, 3])
    assert set(node.params_pytree()) == {"gain"}


def test_a_compiled_graph_carries_the_working_precision_into_gm_params():
    """End to end: the dtype a node resolves is the dtype the fitters,
    the serialisers and ``jax.grad`` see."""
    with x64(True):
        gm = GraphManager()
        gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0))
        gm.compile()
        dtypes = {k: jnp.asarray(v).dtype
                  for k, v in gm.params["nodes"]["s"].items()}
        assert set(dtypes.values()) == {np.dtype("float64")}, dtypes


# ---------------------------------------------------------------------------
# The same statement as a property, over arbitrary values
# ---------------------------------------------------------------------------


@given(value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False,
                       allow_infinity=False, allow_subnormal=False),
       enabled=st.booleans())
@settings(max_examples=EXAMPLES_STANDARD, deadline=None)
def test_a_value_survives_params_pytree_at_the_working_precision(value, enabled):
    """One value, one dtype, one number -- and that number is the
    working precision's nearest representation of what the user wrote,
    not something narrower."""
    with x64(enabled):
        leaves = [_Probe("n", 0.01, gain=spelled).params_pytree()["gain"]
                  for spelled in _spellings(value).values()]
        dtypes = {leaf.dtype for leaf in leaves}
        assert len(dtypes) == 1, dtypes
        working = dtypes.pop()
        assert working == np.dtype("float64" if enabled else "float32")
        expected = np.asarray(value).astype(working).item()
        for leaf in leaves:
            assert np.asarray(leaf).reshape(-1)[0].item() == expected
