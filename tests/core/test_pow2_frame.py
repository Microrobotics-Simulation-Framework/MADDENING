"""``maddening.core._pow2_frame``: the one power-of-two helper.

Every frame in the coupling runtime (the accelerators' state frame, Aitken's
and IQN's dot products and least squares, the relaxed step, the IFT solve's
right-hand side, the report's residuals, curvature step, secant and JVP
tangent lift, the norm's underflow rescue), the sharded solvers' and
``ift_linear_solve``'s right-hand-side frame, and the Adam fitters' gradient
frame come from it.  These tests pin what each mode returns; the call sites'
scale-invariance is pinned where each one is tested
(``test_coupling_accelerators_are_units_invariant.py``,
``test_coupling_bounds_in_any_units.py``,
``test_coupling_non_finite_state.py``,
``test_coupling_ift_gradient_in_any_units.py``,
``test_linear_solvers_in_any_units.py``,
``test_sysid_adam_in_any_units.py``).
"""

import ast
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core._pow2_frame import pow2_frame, pow2_rescue

REPO = Path(__file__).resolve().parents[2]
DTYPES = (jnp.float32, jnp.bfloat16, jnp.float16)
#: A spread of binades for each dtype, every one normal.
EXPONENTS = (-120, -60, -30, -14, -3, -1, 0, 1, 7, 13, 60, 120)


def _normal_exponents(dtype):
    info = jnp.finfo(dtype)
    return [e for e in EXPONENTS if int(info.minexp) + 2 <= e <= int(info.maxexp) - 2]


def _is_power_of_two(x) -> bool:
    m, _ = np.frexp(np.asarray(x, np.float64))
    return bool(np.all(m == 0.5))


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_common_frame_brings_the_largest_entry_into_half_to_one(dtype):
    for e in _normal_exponents(dtype):
        a = jnp.asarray([0.3 * 2.0 ** e, -0.7 * 2.0 ** e], dtype)
        b = jnp.asarray([0.01 * 2.0 ** e], dtype)
        p = pow2_frame(a, b)
        assert p.dtype == dtype and p.shape == ()
        assert _is_power_of_two(p)
        top = float(jnp.max(jnp.abs(a)) * p)
        assert 0.5 <= top < 1.0, (dtype, e, top)


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_entrywise_frame_frames_each_entry_by_its_own_size(dtype):
    exps = _normal_exponents(dtype)
    x = jnp.asarray([0.75 * 2.0 ** e for e in exps], dtype)
    y = jnp.asarray([-0.3 * 2.0 ** e for e in exps], dtype)
    k = pow2_frame(x, y, mode="entrywise")
    assert k.shape == x.shape and k.dtype == dtype
    framed = np.asarray(jnp.maximum(jnp.abs(x), jnp.abs(y)) * k, np.float64)
    assert np.all((framed >= 0.5) & (framed < 1.0)), framed
    assert _is_power_of_two(k)


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_lift_is_one_at_every_ordinary_magnitude_and_lifts_below_tiny_over_eps(dtype):
    info = jnp.finfo(dtype)
    threshold = float(info.tiny) / float(info.eps)
    for e in _normal_exponents(dtype):
        v = jnp.asarray([0.75 * 2.0 ** e], dtype)
        lift = float(pow2_frame(v, mode="lift"))
        assert _is_power_of_two(lift)
        if 0.75 * 2.0 ** e >= threshold:
            assert lift == 1.0, (dtype, e)
        else:
            lifted = 0.75 * 2.0 ** e * lift
            assert threshold <= lifted < 2 * threshold, (dtype, e, lifted)


@pytest.mark.parametrize("mode", ["common", "entrywise", "lift"])
def test_a_zero_or_non_finite_input_is_framed_by_one(mode):
    for bad in (0.0, np.inf, -np.inf, np.nan):
        f = pow2_frame(jnp.asarray([bad, bad], jnp.float32), mode=mode)
        assert np.all(np.asarray(f) == 1.0), (mode, bad, f)


def test_the_frame_itself_is_a_normal_number_at_both_ends_of_the_range():
    info = jnp.finfo(jnp.float32)
    for v in (float(info.max), 1e-40, 1e-45):      # top of the range, two subnormals
        p = float(pow2_frame(jnp.asarray([v], jnp.float32)))
        assert float(info.tiny) <= p <= float(info.max), (v, p)
        k = float(pow2_frame(jnp.asarray([v], jnp.float32), mode="entrywise")[0])
        assert float(info.tiny) <= k <= float(info.max), (v, k)


def test_a_framed_difference_is_the_bare_difference_to_the_bit_between_ordinary_numbers():
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal(4096) * 10.0 ** rng.uniform(-6, 6, 4096), jnp.float32)
    b = jnp.asarray(np.asarray(a) * (1 + rng.uniform(-1e-3, 1e-3, 4096)), jnp.float32)
    k = pow2_frame(a, b, mode="entrywise")
    framed = (a * k - b * k) / k
    assert np.array_equal(np.asarray(framed), np.asarray(a - b))
    p = pow2_frame(a, b)
    assert np.array_equal(np.asarray(jnp.dot(a * p, b * p) / (p * p)), np.asarray(jnp.dot(a, b)))


def test_a_framed_difference_survives_where_the_bare_one_flushes():
    a = jnp.asarray([3.0e-35, 1.0], jnp.float32)
    b = jnp.asarray([3.0000002e-35, 1.0], jnp.float32)
    k = pow2_frame(a, b, mode="entrywise")
    assert float((a * k - b * k)[0]) != 0.0


def test_the_rescue_factor_lifts_the_smallest_normal_to_one():
    for dtype in DTYPES:
        r = pow2_rescue(dtype)
        assert float(r) * float(jnp.finfo(dtype).tiny) == 1.0
        assert _is_power_of_two(float(r))


def test_an_unknown_mode_and_a_lift_of_two_arrays_are_refused():
    with pytest.raises(ValueError, match="mode"):
        pow2_frame(jnp.ones(2), mode="per-field")
    with pytest.raises(ValueError, match="one array"):
        pow2_frame(jnp.ones(2), jnp.ones(2), mode="lift")


#: The modules whose power-of-two frames must come from the one helper.
_FRAMED_SCOPE = ("src/maddening/core", "src/maddening/sysid.py",
                 "src/maddening/cloud/multigpu")


def _frexp_ldexp_calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("frexp", "ldexp"):
            yield node.lineno, node.func.attr


def test_no_module_outside_the_helper_builds_its_own_power_of_two_frame():
    """One helper: a ``frexp`` / ``ldexp`` anywhere else in the numerical core
    is a second copy of the normaliser, with its own docstring and its own
    chance to drift from the one the bit-identity proof covers."""
    found, scanned = [], 0
    for entry in _FRAMED_SCOPE:
        root = REPO / entry
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for f in files:
            scanned += 1
            if f.name == "_pow2_frame.py":
                continue
            found += [f"{f.relative_to(REPO)}:{line} {name}" for line, name in _frexp_ldexp_calls(f)]
    assert scanned > 40, f"the scan read {scanned} files: its scope no longer exists"
    assert found == [], "build the frame with maddening.core._pow2_frame.pow2_frame:\n" + "\n".join(found)
