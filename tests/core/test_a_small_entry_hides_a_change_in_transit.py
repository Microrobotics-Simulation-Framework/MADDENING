"""A small entry of a field hides a change in transit, at the default ``atol = 0``.

MADD-ANO-254 is a change that has to cross a field inside the dead band
(``atol > 0``): the field is not in the residual, so the change is not
seen until it reaches a field the norm keeps, and the loop accepts on the
first residual it measures.  It takes two lagged reads in series between
a kept quantity and itself.  At ``atol = 0`` every field is measured
against its own largest magnitude, so an ENTRY far below its field's
largest one is just as invisible, and the same thing happens with no
band declared (MADD-ANO-297, open):

* ONE member, one field ``y`` of two entries, one edge from the member to
  itself (``y -> u``)::

      y = [b_n + (K / S) u[1],  S (1 + C u[0])]

  with ``S = 1e-8``, ``C = 0.5``, ``K = 0.8`` and ``b_n = 1 + n / 2`` at
  step ``n`` (an integer step counter in the state; integers are in no
  norm).  ``y[0]``, of order one, reads ``y[1]``, of order ``1e-8`` and
  invisible against ``max|y|``, which reads ``y[0]``: two lagged reads.
  The fixed point of ``y[0]`` at step ``n`` is closed form,
  ``(b_n + K) / (1 - K C)``.

On every other step the first pass moves ``y[0]`` towards the new fixed
point, the second pass moves only ``y[1]``, whose change is below what
the norm resolves against ``y[0]``, and the group reports
``iterations=1, converged=True`` thousands of tolerances from the fixed
point.  The same map with the small entry in a field of its own holds:
that is the workaround, and the control here.

Measured (CPU, jax 0.10.2, 0.11.0 and 0.11.2, the same digits; eight
steps): the one-pass steps 3.4e3 to 7.1e3 tolerances off in float32 at
``rtol = 1e-5`` and 3.4e7 to 7.1e7 in float64 at ``rtol = 1e-9``, under
``"mixed"``, ``"l2"`` and ``"interface"`` alike, with no acceleration and
with IQN-ILS; the split control 0.36 to 0.79 tolerances with no
acceleration.  ``strict_convergence`` does not raise, and neither
``validate()`` nor ``compile()`` says anything.  v0.1.0 to v0.3.1 held it
under ``"mixed"``, whose scale was then each entry's own (at most 2.9
tolerances): the per-field scale is 0.4.0's.

What is held here:

* the defect stays visible: a strict xfail of the claim "a converged step
  leaves ``y[0]`` within ``PROMISE`` tolerances of its fixed point", and
  its premise stated as what happens today;
* nothing warns: no advisory from ``validate()``, no warning from
  ``compile()``;
* the workaround holds: the small quantity in a field of its own.
"""

from __future__ import annotations

import functools
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.sparse_mapping_support import x64

#: The small entry's size, against the order-one entry beside it.
S = 1e-8
#: How the small entry responds to the large one.
C = 0.5
#: The large entry's response to the small one, in units of the small one.
K = 0.8
#: ``rtol`` (``tolerance`` under ``"l2"``) in each dtype.
RTOL = {"float32": 1e-5, "float64": 1e-9}
#: Two one-pass acceptances (steps 2 and 4): 7.1e3 and 5.3e3 tolerances
#: off in float32.
STEPS = 4
NORMS = ("mixed", "l2", "interface")
#: The norm the per-push tests run (the three give the same digits here).
PER_PUSH_NORM = "mixed"

#: The claim, in tolerances: a converged step leaves ``y[0]`` this close to
#: its fixed point.  The split control measured 0.36 to 0.79.
PROMISE = 10.0

#: What the packed field is pinned at: more than this many tolerances off
#: on some converged one-pass step.  Measured 5.3e3 and 7.1e3 over these
#: four steps in float32 (5.3e7 and 7.1e7 in float64).  1e3 leaves a factor
#: of five for another jax or platform, and is still 100 promises.
PINNED_OFF = 1e3

_LABEL = "MADD-ANO-297"

_XFAIL = pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    f"{_LABEL} (open): at atol = 0 a field whose entries are 1 and 1e-8 hides a change in "
    "transit through its small entry, and the group accepts after one pass 5.3e3 to 7.1e3 "
    "tolerances off"))


class Packed(SimulationNode):
    """``y = [b_n + (K / S) u[1], S (1 + C u[0])]``: both quantities in one field."""

    def __init__(self, name: str, dtype: str):
        super().__init__(name, 1.0)
        self._dtype = dtype

    def initial_state(self):
        return {"y": jnp.zeros(2, self._dtype), "n": jnp.zeros((), jnp.int32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=jnp.dtype(self._dtype),
                                       default=jnp.zeros(2, self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        u, n = boundary_inputs["u"], state["n"] + 1
        large = (1.0 + 0.5 * n.astype(self._dtype)) + jnp.asarray(K / S, self._dtype) * u[1]
        small = jnp.asarray(S, self._dtype) * (1.0 + jnp.asarray(C, self._dtype) * u[0])
        return {"y": jnp.stack([large, small]), "n": n}


class Split(SimulationNode):
    """The same map with the small quantity in a field of its own, ``z``."""

    def __init__(self, name: str, dtype: str):
        super().__init__(name, 1.0)
        self._dtype = dtype

    def initial_state(self):
        return {"y": jnp.zeros(1, self._dtype), "z": jnp.zeros(1, self._dtype),
                "n": jnp.zeros((), jnp.int32)}

    def boundary_input_spec(self):
        zero = jnp.zeros(1, self._dtype)
        return {"uy": BoundaryInputSpec(shape=(1,), dtype=jnp.dtype(self._dtype), default=zero),
                "uz": BoundaryInputSpec(shape=(1,), dtype=jnp.dtype(self._dtype), default=zero)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        n = state["n"] + 1
        y = ((1.0 + 0.5 * n.astype(self._dtype))
             + jnp.asarray(K / S, self._dtype) * boundary_inputs["uz"])
        z = jnp.asarray(S, self._dtype) * (1.0 + jnp.asarray(C, self._dtype) * boundary_inputs["uy"])
        return {"y": y, "z": z, "n": n}


def graph(shape: str, norm: str, *, acceleration: str = "none",
          dtype: str = "float32") -> GraphManager:
    """The group, not compiled, at the default ``atol``."""
    gm = GraphManager()
    if shape == "packed":
        gm.add_node(Packed("A", dtype))
        gm.add_edge("A", "A", "y", "u")
    else:
        gm.add_node(Split("A", dtype))
        gm.add_edge("A", "A", "y", "uy")
        gm.add_edge("A", "A", "z", "uz")
    tolerance = {"tolerance": RTOL[dtype]} if norm == "l2" else {"rtol": RTOL[dtype]}
    gm.add_coupling_group(["A"], convergence_norm=norm, max_iterations=200, solver="ift",
                          acceleration=acceleration, **tolerance)
    return gm


def _fixed_point(n: int) -> float:
    """``y[0]`` at step *n*: ``y0 = b_n + K (1 + C y0)``."""
    return (1.0 + 0.5 * n + K) / (1.0 - K * C)


@functools.lru_cache(maxsize=None)
def run(shape: str, norm: str, acceleration: str = "none", dtype: str = "float32") -> tuple:
    """``(iterations, converged, distance of y[0] in tolerances)`` for each
    of ``STEPS`` steps of a freshly built graph.  The numbers only are
    kept: no compiled graph outlives the call."""
    with x64(dtype == "float64"):
        gm = graph(shape, norm, acceleration=acceleration, dtype=dtype)
        gm.compile()
        out = []
        for n in range(1, STEPS + 1):
            gm.step()
            (report,) = gm.coupling_diagnostics().values()
            y0 = float(np.asarray(gm.get_node_state("A")["y"], np.float64)[0])
            star = _fixed_point(n)
            out.append((int(report["iterations"]), bool(report["converged"]),
                        abs(y0 - star) / (RTOL[dtype] * abs(star))))
    return tuple(out)


def _held(steps: tuple) -> bool:
    """Every converged step left ``y[0]`` within the promise."""
    return all(off < PROMISE for _iterations, converged, off in steps if converged)


def _pinned(steps: tuple) -> bool:
    """Some converged step accepted after ONE pass far outside the promise."""
    return any(converged and iterations == 1 and off > PINNED_OFF
               for iterations, converged, off in steps)


# ---------------------------------------------------------------------------
# The claim that does not hold today, and its premise
# ---------------------------------------------------------------------------

@_XFAIL
def test_a_converged_step_of_a_packed_field_holds_its_large_entry():
    """What CPL-011's oracle asks of ``y[0]`` at ``atol = 0``, and today
    does not hold."""
    steps = run("packed", PER_PUSH_NORM)
    assert all(converged for _i, converged, _off in steps), steps
    assert _held(steps), steps


def test_the_packed_field_accepts_after_one_pass_far_from_its_fixed_point():
    """The premise of the strict xfail, as what happens today (a
    characterisation of MADD-ANO-297).
    When this stops being true the xfail above turns into a pass and both
    are to be rewritten."""
    steps = run("packed", PER_PUSH_NORM)
    assert all(converged for _i, converged, _off in steps), steps
    assert _pinned(steps), steps


def test_nothing_warns_about_the_packed_field():
    """Silent: at the default ``atol`` no advisory from ``validate()`` and
    no warning from ``compile()``.  The condition depends on the values
    the entries take, which no static check sees."""
    gm = graph("packed", PER_PUSH_NORM)
    assert not [issue for issue in gm.validate() if issue.startswith("WARNING")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm.compile()
    assert [str(w.message) for w in caught] == []


def test_the_workaround_the_small_quantity_in_its_own_field_is_held():
    """The same map with the small quantity in a field of its own: the
    norm measures it against itself, and every step holds."""
    steps = run("split", PER_PUSH_NORM)
    assert all(converged for _i, converged, _off in steps), steps
    assert _held(steps), steps
    assert all(iterations > 1 for iterations, _c, _off in steps), steps


# ---------------------------------------------------------------------------
# The other norms, float64 and IQN-ILS
# ---------------------------------------------------------------------------

# Per push: tests/core/test_a_small_entry_hides_a_change_in_transit.py::test_a_converged_step_of_a_packed_field_holds_its_large_entry
@pytest.mark.slow
@_XFAIL
@pytest.mark.parametrize("norm", [n for n in NORMS if n != PER_PUSH_NORM])
def test_a_converged_step_of_a_packed_field_holds_under_the_other_norms(norm):
    """The same claim under ``"l2"`` and ``"interface"``: neither holds."""
    steps = run("packed", norm)
    assert all(converged for _i, converged, _off in steps), steps
    assert _held(steps), steps


# Per push: tests/core/test_a_small_entry_hides_a_change_in_transit.py::test_the_packed_field_accepts_after_one_pass_far_from_its_fixed_point
@pytest.mark.slow
@pytest.mark.parametrize("acceleration", ["none", "iqn-ils"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("norm", NORMS)
def test_the_packed_field_accepts_after_one_pass_under_every_norm(norm, dtype, acceleration):
    """The premise in every norm, in float64 at ``rtol = 1e-9`` (the
    distance is the same, the tolerance 1e4 times smaller) and with
    IQN-ILS, which accepts after one pass as far off as the plain loop."""
    steps = run("packed", norm, acceleration, dtype)
    assert all(converged for _i, converged, _off in steps), steps
    assert _pinned(steps), steps
    if dtype == "float64":
        assert max(off for _i, _c, off in steps) > 1e4 * PINNED_OFF, steps


# Per push: tests/core/test_a_small_entry_hides_a_change_in_transit.py::test_the_workaround_the_small_quantity_in_its_own_field_is_held
@pytest.mark.slow
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("norm", NORMS)
def test_the_workaround_holds_under_every_norm(norm, dtype):
    """The split control under every norm and in both dtypes, with no
    acceleration.  (Under IQN-ILS the split map reaches its iteration cap
    and reports ``converged=False``: MADD-ANO-285, loud, not this.)"""
    steps = run("split", norm, "none", dtype)
    assert all(converged for _i, converged, _off in steps), steps
    assert _held(steps), steps
