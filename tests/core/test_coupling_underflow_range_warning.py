"""A coupled group whose fields sit in the subnormal range says so, once.

Below ``finfo(dtype).tiny / finfo(dtype).eps`` (about ``9.9e-32`` in float32)
a change of one ulp of a field is smaller than the smallest normal number,
which XLA's CPU backend flushes to zero.  MADDENING's own coupling arithmetic
works in power-of-two frames and keeps its resolution down to ``tiny``; a
node's ``update`` does not, and below ``tiny`` the group's norm reads the field
as zero.  ``GraphManager`` checks each coupled group on the first step after
``compile()`` and warns once per group with the field, its magnitude and the
remedy.  An exactly zero field -- zero is not a small unit -- never warns.
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.coupling._reports import _underflow_range_fields
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.warnings import PrecisionLimitWarning, UnderflowRangeWarning

F32 = jnp.finfo(jnp.float32)
THRESHOLD = float(F32.tiny) / float(F32.eps)        # 2**-103, about 9.9e-32


class _Relay(SimulationNode):
    """``x <- g * u + c``, plus a field ``z`` that stays exactly zero."""

    def __init__(self, name, g, c, dtype=jnp.float32):
        super().__init__(name, 1.0, c=jnp.asarray(c, dtype))
        self._g, self._dtype = g, dtype

    def initial_state(self):
        return {"x": jnp.zeros(1, self._dtype), "z": jnp.zeros(3, self._dtype),
                "k": jnp.zeros((), jnp.int32)}

    def state_fields(self):
        return ["x", "z", "k"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(1,), dtype=self._dtype, description="u")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        x = (self._g * boundary_inputs["u"] + p["c"]).astype(self._dtype)
        return {"x": x, "z": state["z"], "k": state["k"] + 1}


def _pair(c, dtype=jnp.float32):
    gm = GraphManager()
    gm.add_node(_Relay("a", 0.5, c, dtype))
    gm.add_node(_Relay("b", 1.0, 0.0, dtype))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], max_iterations=60, tolerance=1e-5)
    gm.compile()
    return gm


def _recorded(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [w for w in caught if issubclass(w.category, UnderflowRangeWarning)]


def test_a_group_in_the_subnormal_range_warns_once_with_the_field_and_the_remedy():
    gm = _pair(1e-33)                     # fixed point 2e-33, below 9.9e-32
    first = _recorded(gm.step)
    assert len(first) == 1
    msg = str(first[0].message)
    assert "'a+b'" in msg and "a.x" in msg and "Rescale" in msg and "float32" in msg
    assert "1 more field(s)" in msg      # b.x is there too; z (zero) is not counted
    assert issubclass(UnderflowRangeWarning, PrecisionLimitWarning)
    # Once per group: later steps, and a recompile, do not repeat it.
    assert _recorded(lambda: [gm.step() for _ in range(3)]) == []
    gm.compile()
    assert _recorded(gm.step) == []


def test_an_exactly_zero_group_never_warns():
    gm = _pair(0.0)
    assert _recorded(lambda: [gm.step() for _ in range(3)]) == []
    assert float(jnp.max(jnp.abs(gm.get_node_state("a")["x"]))) == 0.0


@pytest.mark.parametrize("c", [1.0, 1e-20, 2.0 ** -100])
def test_a_group_above_the_threshold_does_not_warn(c):
    gm = _pair(c)                         # fixed point 2c, at or above 2**-99
    assert _recorded(gm.step) == []


def test_the_warning_waits_for_a_step_outside_a_transform():
    gm = _pair(1e-33)

    def loss(p):
        out = gm.run_scan(1, params=p)
        return jnp.sum(out["a"]["x"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jax.grad(loss)(gm.params)         # stores tracers: no check, still pending
    assert len(_recorded(gm.step)) == 1


def test_the_field_scan_lists_exactly_the_nonzero_finite_fields_below_tiny_over_eps():
    class _G:
        nodes = ("n",)

    def listed(**fields):
        state = {"n": {k: jnp.asarray(v[0], v[1]) for k, v in fields.items()}}
        return [(f, d.name if hasattr(d, "name") else str(d))
                for _n, f, _m, d in _underflow_range_fields([_G()], state).get("n", [])]

    f32_thr = THRESHOLD
    assert listed(a=([0.5 * f32_thr, 0.0], jnp.float32)) == [("a", "float32")]
    assert listed(a=([2.0 * f32_thr], jnp.float32)) == []
    assert listed(a=([0.0, 0.0], jnp.float32)) == []
    assert listed(a=([np.nan, 1e-35], jnp.float32), b=([np.inf], jnp.float32)) == []
    assert listed(a=([3], jnp.int32)) == []
    # A typed PRNG key cannot become a numpy array; it is skipped, not read.
    state = {"n": {"key": jax.random.key(0), "x": jnp.asarray([0.5 * f32_thr], jnp.float32)}}
    assert [f for _n, f, _m, _d in _underflow_range_fields([_G()], state)["n"]] == ["x"]
    # Each field at its own dtype's threshold: float16's is 0.0625.
    assert listed(h=([0.01], jnp.float16), g=([0.1], jnp.float16)) == [("h", "float16")]
    bf_thr = float(jnp.finfo(jnp.bfloat16).tiny) / float(jnp.finfo(jnp.bfloat16).eps)
    assert listed(b=([0.5 * bf_thr], jnp.bfloat16)) == [("b", "bfloat16")]
    assert listed(b=([4.0 * bf_thr], jnp.bfloat16)) == []


# ---------------------------------------------------------------------------
# The check is due again after a state write, and reads the state a step
# starts from as well as the one it leaves
# ---------------------------------------------------------------------------
# It used to be asked once per compile, of the first stepped state alone: a
# state written into the range afterwards (set_node_state, a loaded
# checkpoint, reset_state) stepped unwarned (a float32 pair written at 1e-37
# returned 25% off with converged=True), and a state already below ``tiny``
# at compile was flushed to exactly zero by the first step, which never warns.


def _scaled(scale):
    """``[1, 2, 3] * scale`` in float32, multiplied on the host: XLA would
    flush a subnormal product to zero before the state existed."""
    return jnp.asarray(np.asarray([1.0, 2.0, 3.0]) * float(scale), jnp.float32)


class _Leaky(SimulationNode):
    """``x <- x / 4 + u / 4``: linear in its state, so a small state stays small."""

    def __init__(self, name, x0=0.0):
        super().__init__(name, 1.0, x0=float(x0))

    def initial_state(self):
        return {"x": _scaled(self.params["x0"])}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(3,), dtype=jnp.float32, description="u")}

    def update(self, state, boundary_inputs, dt):
        return {"x": jnp.float32(0.25) * state["x"] + jnp.float32(0.25) * boundary_inputs["u"]}


def _leaky_pair(x0):
    gm = GraphManager()
    gm.add_node(_Leaky("a", x0))
    gm.add_node(_Leaky("b", x0))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], max_iterations=60, tolerance=1e-5)
    gm.compile()
    return gm


def _write(gm, scale):
    for name in ("a", "b"):
        gm.set_node_state(name, {"x": _scaled(scale)})


def _set_node_state(gm, scale, tmp_path):
    _write(gm, scale)


def _load_state(gm, scale, tmp_path):
    donor = _leaky_pair(1.0)
    donor.step()
    _write(donor, scale)
    gm.load_state(donor.save_state(tmp_path / "written.npz"))


_WRITES = {"set_node_state": _set_node_state, "load_state": _load_state}


@pytest.mark.parametrize("scale", [1e-33, 1e-37])
@pytest.mark.parametrize("door", sorted(_WRITES))
def test_a_state_written_into_the_range_after_the_first_step_warns_at_the_next_step(
        door, scale, tmp_path):
    gm = _leaky_pair(1.0)
    assert _recorded(gm.step) == []                    # of order one: nothing to say
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnderflowRangeWarning)   # the donor's own steps
        _WRITES[door](gm, scale, tmp_path)
    raised = _recorded(gm.step)
    assert len(raised) == 1
    msg = str(raised[0].message)
    assert "'a+b'" in msg and "float32 subnormal range" in msg and "Rescale" in msg
    # Still once per group: neither a later step nor a later write repeats it.
    assert _recorded(gm.step) == []
    _write(gm, scale)
    assert _recorded(gm.step) == []


def test_reset_state_onto_an_initial_state_in_the_range_warns_at_the_next_step():
    gm = _leaky_pair(1e-33)
    _write(gm, 1.0)                                     # the first step starts of order one
    assert _recorded(gm.step) == []
    gm.reset_state()
    assert len(_recorded(gm.step)) == 1


@pytest.mark.parametrize("scale", [1.0, 2.0 ** -90, 0.0])
def test_a_written_state_outside_the_range_does_not_warn(scale):
    gm = _leaky_pair(1.0)
    gm.step()
    _write(gm, scale)
    assert _recorded(lambda: [gm.step() for _ in range(2)]) == []


@pytest.mark.parametrize("scale", [1e-38, 1e-41])
def test_a_state_below_tiny_at_compile_warns_although_the_first_step_flushes_it(scale):
    """The state the first step starts from is read as well as the one it
    leaves: below ``tiny`` the nodes' products are flushed to exactly zero
    (XLA's CPU backend), and an exactly zero field never warns."""
    gm = _leaky_pair(scale)
    # On the host: XLA reads a subnormal as zero.
    before = float(np.max(np.abs(np.asarray(gm.get_node_state("a")["x"]))))
    assert 0.0 < before < THRESHOLD
    raised = _recorded(gm.step)
    assert len(raised) == 1
    assert f"has magnitude {before:.3g}, inside the float32 subnormal range" in str(
        raised[0].message)


def test_a_write_made_under_a_transform_leaves_the_check_pending():
    gm = _leaky_pair(1.0)
    gm.step()

    def loss(x):
        gm.set_node_state("a", {"x": x})
        return jnp.sum(gm.run_scan(1)["a"]["x"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jax.grad(loss)(_scaled(1e-33))
        gm.get_node_state("a")                          # the graph is put back, untraced
    assert _recorded(gm.step) == []                     # ... to a state of order one
    _write(gm, 1e-33)
    assert len(_recorded(gm.step)) == 1


def test_the_state_a_step_replaces_is_not_read_while_it_holds_a_tracer():
    """A stepper that stores an untraced state over a traced one (the
    check pending) reads the stored state only: the replaced one cannot be
    brought to the host."""
    gm = _leaky_pair(1.0)
    gm.step()
    stepped = {name: dict(fields) for name, fields in gm._state.items()}   # noqa: SLF001
    stepped["a"]["x"] = _scaled(1e-33)
    seen = []

    def loss(x):
        gm.set_node_state("a", {"x": x})                # traced, and the check is due
        seen.extend(_recorded(lambda: gm._store_state(stepped)))            # noqa: SLF001
        return jnp.sum(x)

    jax.grad(loss)(_scaled(1.0))
    assert len(seen) == 1 and "a.x" in str(seen[0].message)


# ---------------------------------------------------------------------------
# A loop that writes a state before every step is not asked at every step
# ---------------------------------------------------------------------------
# The check reads every field of every group twice on the host, which costs
# several times a small graph's step (a three-entry pair's loop of
# set_node_state and step: 45 microseconds without the check, 170 with it at
# every step).  A write, or a few, is asked at the next step every time; in an
# unbroken run of steps that each follow a write (or compile()) the check is
# made at the first eight, at each power of two after that and at every 128th
# from there on.


_ONE = _scaled(1.0)


def _counted_reads(monkeypatch):
    """How many times the groups' fields are read on the host from here on
    (``GraphManager`` reads the function from ``_reports`` at each call)."""
    from maddening.core.coupling import _reports

    reads = []
    real = _reports._underflow_range_fields

    def counted(groups, state):
        reads.append(len(groups))
        return real(groups, state)

    monkeypatch.setattr(_reports, "_underflow_range_fields", counted)
    return reads


def test_the_positions_of_a_run_at_which_the_check_is_made():
    from maddening.core.coupling._reports import _underflow_check_due

    asked = [run for run in range(1, 700) if _underflow_check_due(run)]
    assert asked == [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 384, 512, 640]


def test_a_loop_that_writes_before_every_step_is_asked_a_bounded_number_of_times(monkeypatch):
    gm = _leaky_pair(1.0)
    reads = _counted_reads(monkeypatch)
    gm.step()
    assert len(reads) == 2                              # the first step after compile()
    del reads[:]
    for _ in range(300):
        gm.set_node_state("a", {"x": _ONE})
        gm.step()
    # The step after compile() was the run's first.  Two reads (the state
    # the step starts from, the one it leaves) at steps 2 to 8, 16, 32,
    # 64, 128 and 256 of the run: 12 of these 300.
    assert len(reads) == 2 * 12
    # A run of steps without a write reads nothing at all.
    del reads[:]
    for _ in range(5):
        gm.step()
    assert reads == []


def test_a_step_that_follows_no_write_ends_the_run():
    """After a plain step, the next write is the first of a new run and is
    asked at the step that follows it, however long the run before."""
    gm = _leaky_pair(1.0)
    gm.step()
    for _ in range(20):                                 # the run stands at 20: not asked
        _write(gm, 1.0)
        assert _recorded(gm.step) == []
    assert _recorded(gm.step) == []                     # no write before this one
    _write(gm, 1e-33)
    assert len(_recorded(gm.step)) == 1


def test_a_write_into_the_range_inside_a_long_run_is_warned_of_at_the_next_asked_step():
    """What the bound costs: a state written into the range at a step of
    the run that is not asked (the 9th to the 15th) is warned of when the
    run reaches the next asked one (the 16th), if it is still there."""
    gm = _leaky_pair(1.0)
    gm.step()                                           # the run's first step
    for _ in range(9):
        _write(gm, 1.0)
        assert _recorded(gm.step) == []
    for position in range(11, 16):
        _write(gm, 1e-33)
        assert _recorded(gm.step) == [], position
    _write(gm, 1e-33)
    assert len(_recorded(gm.step)) == 1                 # the 16th


def test_compile_starts_a_new_run():
    """The first step after every ``compile()`` is asked, wherever a run
    of writes stood when the graph was compiled again."""
    gm = _leaky_pair(1.0)
    gm.step()
    for _ in range(12):
        _write(gm, 1.0)
        gm.step()
    _write(gm, 1e-33)
    gm._dirty = True                                    # noqa: SLF001
    gm.compile()
    assert len(_recorded(gm.step)) == 1


def test_a_group_already_warned_of_is_not_read_again(monkeypatch):
    gm = _leaky_pair(1.0)
    gm.step()
    _write(gm, 1e-33)
    assert len(_recorded(gm.step)) == 1
    reads = _counted_reads(monkeypatch)
    for _ in range(3):
        _write(gm, 1e-33)
        assert _recorded(gm.step) == []
    assert reads == []
