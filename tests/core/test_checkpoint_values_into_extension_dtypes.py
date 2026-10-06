"""A checkpoint value is checked against a bfloat16 leaf as against any other.

``load_state`` casts each member of a checkpoint to the live leaf's dtype
and refuses a value the cast would lose: a finite one that overflows to
``inf``, a non-zero one flushed to ``+-0`` (MADD-ANO-138).  Which rule
applies was chosen by the leaf's ``dtype.kind`` -- and NumPy reports
bfloat16, like every dtype it carries as an extension type, as kind ``"V"``
(void).  A bfloat16 leaf therefore got **no** rule: a float32 checkpoint
loaded into a bfloat16 graph stored ``1e-44`` as ``0.0`` and
``3.4028235e38`` as ``inf`` with nothing said (MADD-ANO-207).  The rule is
now chosen from JAX's own dtype lattice, so every floating or integer
dtype JAX has gets the one for its kind, and a dtype with no rule is
refused, not assumed lossless.

One thing here is not fixed, and is pinned as it stands: a bfloat16 leaf
cannot be restored from a checkpoint of its own graph at all
(MADD-ANO-206, open).  ``save_state`` writes it as two raw bytes per
element with no dtype name -- NumPy's ``.npy`` format has none for an
extension type -- and ``load_state`` refuses raw bytes.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.simulation.checkpoint import _checked_cast

BFLOAT16 = np.dtype(ml_dtypes.bfloat16)
N = 4


class Holder(SimulationNode):
    """One field ``x`` and one parameter ``k``, both of the dtype named."""

    def __init__(self, name, timestep, dtype="float32", k=0.5):
        super().__init__(name, timestep, dtype=dtype, k=k)

    def initial_state(self):
        return {"x": jnp.ones(N, self.params["dtype"])}

    def params_pytree(self):
        return {"k": jnp.asarray(self.params["k"], self.params["dtype"])}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": state["x"] * params["k"].astype(state["x"].dtype)}


def _graph(dtype: str) -> GraphManager:
    gm = GraphManager()
    gm.add_node(Holder("n", 0.1, dtype=dtype))
    gm.compile()
    gm.step()
    return gm


def _checkpoint(tmp_path, member: str, values: np.ndarray):
    """A checkpoint of the float32 graph with *member* replaced by
    *values* -- of whatever dtype they are."""
    source = _graph("float32")
    path = source.save_state(tmp_path / "saved")
    with np.load(path) as archive:
        members = {key: archive[key] for key in archive.files}
    assert member in members, sorted(members)
    members[member] = np.asarray(values).reshape(members[member].shape)
    np.savez(path, **members)
    return path


def _x(gm: GraphManager) -> np.ndarray:
    return np.asarray(gm.get_node_state("n")["x"])


#: A value the cast to bfloat16 would lose, as (member dtype, value, what
#: it would be stored as).
LOST = [
    (np.float32, 1e-44, "0.0"),              # a float32 subnormal: flushed to zero
    (np.float32, -1e-44, "-0.0"),
    (np.float32, 3.4028235e38, "inf"),       # float32's largest: past bfloat16's
    (np.float64, 1e39, "inf"),
    (np.float64, -1e-50, "-0.0"),
]


@pytest.mark.parametrize("member_dtype, value, stored", LOST,
                         ids=[f"{np.dtype(d).name}:{v!r}" for d, v, _ in LOST])
def test_a_value_a_bfloat16_field_cannot_hold_is_refused(tmp_path, member_dtype, value, stored):
    """The premise first -- the bare cast does lose the value -- and then
    the load: a ``ValueError`` naming the field, the value and what it
    would have loaded as, the state left as it was."""
    with np.errstate(all="ignore"):
        bare = np.asarray([value], member_dtype).astype(BFLOAT16)
    assert repr(float(bare[0])) == stored, "premise: bfloat16 cannot hold this value"
    values = np.full(N, 1.5, member_dtype)
    values[2] = value
    path = _checkpoint(tmp_path, "n/x", values)
    gm = _graph("bfloat16")
    before = _x(gm).copy()
    with pytest.raises(ValueError, match=r"field 'n/x' holds .* which this graph's bfloat16 "
                                         r"cannot hold: it would load as") as refusal:
        gm.load_state(path)
    assert stored in str(refusal.value) and "Nothing was loaded" in str(refusal.value)
    assert _x(gm).dtype == BFLOAT16 and np.array_equal(_x(gm), before)


@pytest.mark.parametrize("member_dtype", [np.float32, np.float64, np.float16])
def test_a_value_a_bfloat16_field_can_hold_loads_as_the_cast_rounds_it(tmp_path, member_dtype):
    """What must still load: a value bfloat16 holds exactly, one it
    rounds, one that rounds to a subnormal (it keeps its sign and
    magnitude), and ``inf`` and ``NaN``, which load as they were."""
    with np.errstate(all="ignore"):
        values = np.asarray([1.5, 0.1, -1e-39, np.inf], np.float64).astype(member_dtype)
        expected = values.astype(BFLOAT16)
    path = _checkpoint(tmp_path, "n/x", values)
    gm = _graph("bfloat16")
    gm.load_state(path)
    loaded = _x(gm)
    assert loaded.dtype == BFLOAT16
    assert loaded.tobytes() == expected.tobytes()
    assert float(loaded[0]) == 1.5 and float(loaded[1]) != 0.1


@pytest.mark.parametrize("member_dtype, value, stored", [
    (np.float32, 1e-9, "0.0"), (np.float32, 1e5, "inf"), (np.float64, -1e-50, "-0.0"),
], ids=["float32:1e-09", "float32:100000.0", "float64:-1e-50"])
def test_a_value_a_float16_field_cannot_hold_is_refused(tmp_path, member_dtype, value, stored):
    """The other 16-bit float, which NumPy does give a kind: it had the
    rule already, and keeps it."""
    with np.errstate(all="ignore"):
        bare = np.asarray([value], member_dtype).astype(np.float16)
    assert repr(float(bare[0])) == stored, "premise: float16 cannot hold this value"
    values = np.full(N, 1.5, member_dtype)
    values[2] = value
    gm = _graph("float16")
    before = _x(gm).copy()
    with pytest.raises(ValueError, match="which this graph's float16 cannot hold"):
        gm.load_state(_checkpoint(tmp_path, "n/x", values))
    assert _x(gm).dtype == np.float16 and np.array_equal(_x(gm), before)


def test_a_nan_loads_into_a_bfloat16_field_as_nan(tmp_path):
    values = np.asarray([1.0, np.nan, 2.0, -np.inf], np.float32)
    gm = _graph("bfloat16")
    gm.load_state(_checkpoint(tmp_path, "n/x", values))
    assert np.isnan(float(_x(gm)[1])) and float(_x(gm)[3]) == -np.inf


@pytest.mark.parametrize("member_dtype, value", [(np.float32, 1e-44), (np.float64, 1e39)])
def test_a_parameter_leaf_of_bfloat16_is_held_to_the_same_rule(tmp_path, member_dtype, value):
    """The params section of a checkpoint goes through the same cast."""
    path = _checkpoint(tmp_path, "_params/n/k", np.asarray(value, member_dtype))
    gm = _graph("bfloat16")
    before = np.asarray(gm.params["nodes"]["n"]["k"]).copy()
    with pytest.raises(ValueError, match="which this graph's bfloat16 cannot hold"):
        gm.load_state(path)
    assert np.array_equal(np.asarray(gm.params["nodes"]["n"]["k"]), before)


def test_the_same_checkpoint_still_loads_into_the_graph_that_wrote_it(tmp_path):
    """The fixture can express the defect and nothing else moved: a
    float32 subnormal is a float32 value, and loads into a float32 leaf."""
    values = np.asarray([1.5, 1e-44, 3.4028235e38, 0.1], np.float32)
    gm = _graph("float32")
    gm.load_state(_checkpoint(tmp_path, "n/x", values))
    assert _x(gm).tobytes() == values.tobytes()


#: The dtypes NumPy carries as extension types (kind "V", but for one
#: 8-bit float), by the kind of number each is to a JAX that has it.
EXTENSION_KINDS = {
    "bfloat16": "f", "float8_e4m3fn": "f", "float8_e4m3": "f", "float8_e3m4": "f",
    "float6_e2m3fn": "f", "float6_e3m2fn": "f", "float4_e2m1fn": "f",
    "int4": "i", "int2": "i", "uint4": "u", "uint2": "u",
}


def test_the_right_dtypes_are_told_as_numbers_on_this_jax():
    """Whatever else this JAX has, bfloat16 and the 8-bit floats are
    floats and the 4-bit integers are integers: the loop below, which
    allows for a dtype a JAX does not have, cannot pass by finding none."""
    from maddening.core.simulation.checkpoint import _number_kind  # noqa: PLC0415

    assert _number_kind(BFLOAT16) == "f"
    assert _number_kind(np.dtype(ml_dtypes.float8_e4m3fn)) == "f"
    assert _number_kind(np.dtype(ml_dtypes.int4)) == "i"
    assert _number_kind(np.dtype(ml_dtypes.uint4)) == "u"
    assert jnp.zeros(2, BFLOAT16).dtype == BFLOAT16, "premise: a leaf can be bfloat16"


@pytest.mark.parametrize("name", sorted(EXTENSION_KINDS))
def test_an_extension_dtype_is_told_as_the_kind_of_number_it_is(name):
    from maddening.core.simulation.checkpoint import _number_kind  # noqa: PLC0415

    # The kind it is, or -- on a JAX whose lattice does not have the dtype
    # -- "V", which the cast check refuses: never another number's kind.
    dtype = np.dtype(getattr(ml_dtypes, name))
    assert _number_kind(dtype) in (EXTENSION_KINDS[name], "V")


@pytest.mark.parametrize("dtype, kind", [
    (np.float32, "f"), (np.float16, "f"), (np.float64, "f"), (np.complex64, "c"),
    (np.int32, "i"), (np.uint8, "u"), (np.bool_, "b"), ("U4", "U"), ("V2", "V"),
    ([("a", "<u2")], "V"), ([("bfloat16", "<u2")], "V"), (("<f4", (2,)), "V"),
])
def test_a_numpy_dtype_keeps_its_own_kind(dtype, kind):
    """Raw bytes, a record and a sub-array stay ``"V"``: none is a number
    (JAX's lattice is asked about each, and says so)."""
    from maddening.core.simulation.checkpoint import _number_kind  # noqa: PLC0415

    assert _number_kind(np.dtype(dtype)) == kind


@pytest.mark.parametrize("target, values", [
    ("bfloat16", np.asarray([1.5, 1e-44], np.float32)),               # flushed to zero
    ("float8_e4m3fn", np.asarray([1.5, 1e6], np.float32)),            # no inf: NaN
    ("float8_e5m2", np.asarray([1.5, 1e-9], np.float32)),             # flushed to zero
    ("int4", np.asarray([3, 100], np.int64)),                         # wraps
    ("uint4", np.asarray([3, -1], np.int64)),                         # wraps
    ("int4", np.asarray([3.0, 2.5], np.float32)),                     # truncates
    ("int4", np.asarray([3.0, np.inf], np.float32)),                  # not a number of ints
], ids=lambda v: v if isinstance(v, str) else f"{v.dtype}:{v[1]!r}")
def test_the_cast_check_holds_for_every_kind_of_extension_dtype(target, values):
    """The rule by kind, on the cast itself: the first value is kept, the
    second refused, for a small float, a float without ``inf`` and the
    small integers."""
    dtype = np.dtype(getattr(ml_dtypes, target))
    kept = _checked_cast(values[:1], dtype, "leaf 'probe'")
    assert kept.dtype == dtype and float(kept[0]) == float(values[0])
    with pytest.raises(ValueError, match="leaf 'probe' holds .* cannot hold"):
        _checked_cast(values, dtype, "leaf 'probe'")


#: A float wider than float64, where the platform has one (x86's 80-bit
#: extended; elsewhere ``longdouble`` is float64 and the case cannot be
#: written).
_WIDER_THAN_FLOAT64 = np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant


@pytest.mark.parametrize("target, values", [
    # A complex value for a real leaf: the cast drops the imaginary part.
    # The rule read the NumPy kind of the leaf, which is void for these.
    ("bfloat16", np.asarray([1.5 + 0j, 1.5 + 1j], np.complex64)),
    ("float8_e4m3fn", np.asarray([1.5 + 0j, 1.5 + 1j], np.complex128)),
    ("int4", np.asarray([3 + 0j, 3 + 1j], np.complex64)),
    ("uint4", np.asarray([3 + 0j, 3 + 2j], np.complex128)),
    ("int4", np.asarray([3 + 0j, 2.5 + 0j], np.complex64)),           # real, not whole
    # The other signedness, which a cast there and back cannot see.
    ("uint4", np.asarray([3, -1], np.int8)),
    ("int4", np.asarray([3, 200], np.uint8)),
    ("int4", np.asarray([3, np.iinfo(np.uint64).max], np.uint64)),
    ("uint4", np.asarray([3, np.iinfo(np.int64).min], np.int64)),
    ("int4", np.asarray([3.0, np.nan], np.float64)),
    *([("int4", np.asarray([3, np.longdouble(7) + np.longdouble(2) ** -60], np.longdouble)),
       ("uint4", np.asarray([3, np.longdouble(15) + np.longdouble(2) ** -59], np.longdouble))]
      if _WIDER_THAN_FLOAT64 else []),                                # whole only as a float64
], ids=lambda v: v if isinstance(v, str) else f"{v.dtype}:{v[1]!r}")
def test_an_extension_leaf_is_held_to_the_rule_a_numpy_leaf_of_its_kind_is(target, values):
    """What ``load_state`` refuses for a float32 or an int32 leaf, it
    refuses for a bfloat16 or a 4-bit one: an imaginary part, an integer
    of the other signedness, a value that is not whole."""
    dtype = np.dtype(getattr(ml_dtypes, target))
    kept = _checked_cast(values[:1], dtype, "leaf 'probe'")
    assert kept.dtype == dtype and float(kept[0]) == float(np.real(values[0]))
    with pytest.raises(ValueError, match="leaf 'probe' holds .* cannot hold"):
        _checked_cast(values, dtype, "leaf 'probe'")


@pytest.mark.parametrize("source", [np.int8, np.int64, np.float32, np.float64, np.complex64])
@pytest.mark.parametrize("target", ["int4", "uint4"])
def test_a_four_bit_leaf_takes_its_own_extremes_and_nothing_past_them(target, source):
    """NumPy has no ``iinfo`` for JAX's small integers, so the rule's
    bounds for them are JAX's: each extreme loads, and one past it is
    refused where the cast would wrap it."""
    dtype = np.dtype(getattr(ml_dtypes, target))
    low, high = (-8, 7) if target == "int4" else (0, 15)
    assert (int(jnp.iinfo(dtype).min), int(jnp.iinfo(dtype).max)) == (low, high)
    kept = _checked_cast(np.asarray([low, high], source), dtype, "leaf 'probe'")
    assert kept.dtype == dtype and [int(v) for v in kept] == [low, high]
    for past in (low - 1, high + 1):
        with pytest.raises(ValueError, match="leaf 'probe' holds .* cannot hold"):
            _checked_cast(np.asarray([low, past], source), dtype, "leaf 'probe'")


def test_a_leaf_dtype_with_no_rule_is_refused_not_assumed_lossless():
    """A cast into a dtype that is neither a float, an integer nor a
    boolean used to be taken as it came."""
    with pytest.raises(ValueError, match="cannot be checked against"):
        _checked_cast(np.asarray([1.0], np.float32), np.dtype("V4"), "leaf 'probe'")


@pytest.mark.xfail(strict=True, raises=ValueError, reason=(
    "MADD-ANO-206: save_state writes a bfloat16 leaf as two raw bytes per element "
    "with no dtype name (the .npy format has none for a NumPy extension dtype) and "
    "load_state refuses raw bytes, so a bfloat16 graph cannot be restored from its "
    "own checkpoint; open, deferred to 0.5.0"))
def test_a_bfloat16_state_survives_a_checkpoint_restart(tmp_path):
    """``save_state`` persists "all node states": a bfloat16 graph saved
    after a step and loaded into a fresh one must hold the same bits.
    float16, float32 and every other NumPy dtype do
    (``test_every_numpy_float_state_survives_a_checkpoint_restart``)."""
    gm = _graph("bfloat16")
    path = gm.save_state(tmp_path / "bf16")
    fresh = GraphManager()
    fresh.add_node(Holder("n", 0.1, dtype="bfloat16"))
    fresh.compile()
    fresh.load_state(path)
    assert _x(fresh).dtype == BFLOAT16 and _x(fresh).tobytes() == _x(gm).tobytes()


def test_what_a_bfloat16_checkpoint_holds_today_is_the_leafs_bits_without_its_name(tmp_path):
    """The other half of MADD-ANO-206, so the entry's description stays
    true: the file is written, its member is ``|V2``, and those bytes are
    the leaf's -- nothing is lost on disk, only the name."""
    gm = _graph("bfloat16")
    path = gm.save_state(tmp_path / "bf16")
    with np.load(path) as archive:
        member = archive["n/x"]
    assert member.dtype.str == "|V2"
    assert member.tobytes() == _x(gm).tobytes()


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_every_numpy_float_state_survives_a_checkpoint_restart(tmp_path, dtype):
    gm = _graph(dtype)
    path = gm.save_state(tmp_path / dtype)
    fresh = GraphManager()
    fresh.add_node(Holder("n", 0.1, dtype=dtype))
    fresh.compile()
    fresh.load_state(path)
    assert _x(fresh).dtype == np.dtype(dtype) and _x(fresh).tobytes() == _x(gm).tobytes()
