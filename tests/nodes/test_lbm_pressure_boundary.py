"""The Zou-He pressure boundary of ``LBMNode`` imposes the pressure it is given.

``_zou_he_pressure_face`` rebuilds the populations that stream into the
domain through an inlet/outlet face so that the face carries a prescribed
density ``rho_p = p / cs2`` and zero tangential velocity (Zou & He 1997
for D2Q9, Hecht & Harting 2010 for D3Q19).  Until 0.4.0 the closure for
the face-normal velocity counted the outgoing ("known") populations once
instead of twice, so the face came out at ``rho_p + S_K`` -- 15% above the
prescribed pressure on a unit-density lattice -- on every face of both
lattices (MADD-ANO-020).  The in-tree Poiseuille test normalised both
velocity profiles and could not see it.

Every test here reads a *magnitude*: the density of the face, its
tangential momentum, the populations the closure may and may not touch,
the node's pressure field, and the pressure it reports to a coupled
node.  The steady-flow consequences (pressure drop, centreline velocity)
are measured in ``tests/verification/test_lbm_pressure_poiseuille.py``.
"""

import os
import zlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from maddening.nodes.lbm import (  # noqa: E402
    _FACE_MAP,
    LatticeDescriptor,
    LBMNode,
    _classify_directions,
    _equilibrium,
    _zou_he_face_closure,
    _zou_he_pressure_face,
    d2q9,
    d3q19,
)

CS2 = 1.0 / 3.0

# Every face each supported lattice has.
_CASES = [
    ("D2Q9", d2q9, (7, 6), face) for face in ("x_min", "x_max", "y_min", "y_max")
] + [
    ("D3Q19", d3q19, (5, 4, 6), face)
    for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
]
_IDS = [f"{name}-{face}" for name, _, _, face in _CASES]


def _face_slice(ndim, axis, side):
    sl = [slice(None)] * ndim
    sl[axis] = 0 if side == "min" else -1
    return tuple(sl)


def _random_f(lat, shape, seed):
    """A non-equilibrium population field: equilibrium at a random density
    and velocity, plus a random non-equilibrium part.  The face is far from
    the prescribed state, so the closure has real work to do."""
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.8, 1.2, shape)
    u = rng.uniform(-0.05, 0.05, shape + (lat.D,))
    feq = np.asarray(_equilibrium(jnp.asarray(rho), jnp.asarray(u), lat.e, lat.w, lat.cs2))
    noise = rng.uniform(-0.01, 0.01, feq.shape) * lat.w
    return jnp.asarray(feq + noise, jnp.float32)


def _face_moments(f_face, e):
    rho = f_face.sum(-1)
    mom = f_face @ e.astype(np.float64)
    return rho, mom


@pytest.mark.parametrize("name,factory,shape,face", _CASES, ids=_IDS)
@pytest.mark.parametrize("p_prescribed", [0.30, 1.0 / 3.0, 0.36])
def test_the_face_carries_exactly_the_prescribed_density(name, factory, shape, face, p_prescribed):
    """rho(face) == p / cs2 to float32 rounding, cell by cell.

    The defect gave rho_p + S_K here: 15.4% high at p = 0.36 on a
    unit-density lattice, on every face of both lattices."""
    lat = factory()
    axis, side = _FACE_MAP[face]
    f = _random_f(lat, shape, seed=zlib.crc32(f"{name}-{face}".encode()))
    rho_p = p_prescribed / CS2
    out = _zou_he_pressure_face(
        f, jnp.float32(rho_p), lat.e, lat.w, lat.cs2, axis, side,
        jnp.zeros(shape, bool),
    )
    f_face = np.asarray(out, np.float64)[_face_slice(lat.D, axis, side)]
    rho, _ = _face_moments(f_face, lat.e)
    np.testing.assert_allclose(rho, rho_p, rtol=2e-6, atol=0.0)


@pytest.mark.parametrize("name,factory,shape,face", _CASES, ids=_IDS)
def test_the_face_carries_zero_tangential_velocity(name, factory, shape, face):
    """The transverse-momentum correction: every tangential component of the
    face's momentum is zero after the closure, from a face that started with
    tangential momentum of order 0.05 rho."""
    lat = factory()
    axis, side = _FACE_MAP[face]
    f = _random_f(lat, shape, seed=1 + zlib.crc32(f"{name}-{face}".encode()))
    out = _zou_he_pressure_face(
        f, jnp.float32(1.05), lat.e, lat.w, lat.cs2, axis, side,
        jnp.zeros(shape, bool),
    )
    f_face_in = np.asarray(f, np.float64)[_face_slice(lat.D, axis, side)]
    f_face = np.asarray(out, np.float64)[_face_slice(lat.D, axis, side)]
    _, mom_in = _face_moments(f_face_in, lat.e)
    _, mom = _face_moments(f_face, lat.e)
    for t in range(lat.D):
        if t == axis:
            continue
        # The input really had tangential momentum to remove.
        assert np.max(np.abs(mom_in[..., t])) > 1e-3
        np.testing.assert_allclose(mom[..., t], 0.0, atol=2e-7)


@pytest.mark.parametrize("name,factory,shape,face", _CASES, ids=_IDS)
def test_only_the_incoming_populations_of_fluid_face_cells_are_rebuilt(name, factory, shape, face):
    """Known and tangential populations are untouched everywhere, the
    interior is untouched, and a wall cell on the face keeps all of its
    populations."""
    lat = factory()
    axis, side = _FACE_MAP[face]
    f = _random_f(lat, shape, seed=2 + zlib.crc32(f"{name}-{face}".encode()))
    wall = np.zeros(shape, bool)
    face_sl = _face_slice(lat.D, axis, side)
    wall_face = np.zeros(wall[face_sl].shape, bool)
    wall_face.flat[::3] = True
    wall[face_sl] = wall_face
    out = np.asarray(_zou_he_pressure_face(
        f, jnp.float32(0.95), lat.e, lat.w, lat.cs2, axis, side, jnp.asarray(wall),
    ))
    f = np.asarray(f)
    known, unknown, tangential = _classify_directions(lat.e, axis, side)
    interior = np.ones(shape, bool)
    interior[face_sl] = False
    np.testing.assert_array_equal(out[interior], f[interior])
    np.testing.assert_array_equal(out[..., known + tangential], f[..., known + tangential])
    np.testing.assert_array_equal(out[wall], f[wall])
    # ...and the unknowns of the fluid face cells did change.
    fluid_face = ~wall[face_sl]
    assert np.all(out[face_sl][fluid_face][:, unknown] != f[face_sl][fluid_face][:, unknown])


@pytest.mark.parametrize("name,factory,shape,face", _CASES, ids=_IDS)
def test_a_face_already_at_the_prescribed_equilibrium_is_a_fixed_point(name, factory, shape, face):
    """Consistency: if the face populations are the equilibrium at the
    prescribed density and a face-normal velocity, the closure returns
    them unchanged (it recovers that very velocity from the moments)."""
    lat = factory()
    axis, side = _FACE_MAP[face]
    rho_p = 1.02
    u = np.zeros(shape + (lat.D,))
    u[..., axis] = 0.03
    f = _equilibrium(jnp.full(shape, rho_p), jnp.asarray(u), lat.e, lat.w, lat.cs2)
    f = jnp.asarray(f, jnp.float32)
    out = _zou_he_pressure_face(
        f, jnp.float32(rho_p), lat.e, lat.w, lat.cs2, axis, side,
        jnp.zeros(shape, bool),
    )
    np.testing.assert_allclose(np.asarray(out), np.asarray(f), rtol=0.0, atol=2e-7)


def test_d2q9_west_face_matches_zou_and_he_1997_as_published():
    """The closure written out by hand from Zou & He (1997), for D2Q9 at a
    west (x_min) pressure boundary with u_y = 0, in this module's numbering
    (1:+x 2:-x 3:+y 4:-y 5:+x+y 6:-x+y 7:+x-y 8:-x-y):

        u   = 1 - (f0 + f3 + f4 + 2 (f2 + f6 + f8)) / rho
        f1  = f2 + 2/3 rho u
        f5  = f8 + 1/6 rho u - 1/2 (f3 - f4)
        f7  = f6 + 1/6 rho u + 1/2 (f3 - f4)

    Redundant with the moment tests above on purpose: a sign slip that
    happened to preserve the moments would still fail here."""
    lat = d2q9()
    f = np.asarray(_random_f(lat, (3, 4), seed=7), np.float64)
    rho = 1.07
    out = np.asarray(_zou_he_pressure_face(
        jnp.asarray(f), rho, lat.e, lat.w, lat.cs2, 0, "min", jnp.zeros((3, 4), bool),
    ), np.float64)
    g = f[0]
    u = 1.0 - (g[:, 0] + g[:, 3] + g[:, 4] + 2.0 * (g[:, 2] + g[:, 6] + g[:, 8])) / rho
    np.testing.assert_allclose(out[0, :, 1], g[:, 2] + 2.0 / 3.0 * rho * u, rtol=1e-5)
    np.testing.assert_allclose(out[0, :, 5], g[:, 8] + rho * u / 6.0 - 0.5 * (g[:, 3] - g[:, 4]), rtol=1e-5)
    np.testing.assert_allclose(out[0, :, 7], g[:, 6] + rho * u / 6.0 + 0.5 * (g[:, 3] - g[:, 4]), rtol=1e-5)


def test_d2q9_east_face_matches_zou_and_he_1997_as_published():
    """The same at an east (x_max) boundary, where the incoming set is
    2, 6, 8 and the normal velocity changes sign:

        u   = -1 + (f0 + f3 + f4 + 2 (f1 + f5 + f7)) / rho
        f2  = f1 - 2/3 rho u
        f6  = f7 - 1/6 rho u - 1/2 (f3 - f4)
        f8  = f5 - 1/6 rho u + 1/2 (f3 - f4)
    """
    lat = d2q9()
    f = np.asarray(_random_f(lat, (3, 4), seed=8), np.float64)
    rho = 0.93
    out = np.asarray(_zou_he_pressure_face(
        jnp.asarray(f), rho, lat.e, lat.w, lat.cs2, 0, "max", jnp.zeros((3, 4), bool),
    ), np.float64)
    g = f[-1]
    u = -1.0 + (g[:, 0] + g[:, 3] + g[:, 4] + 2.0 * (g[:, 1] + g[:, 5] + g[:, 7])) / rho
    np.testing.assert_allclose(out[-1, :, 2], g[:, 1] - 2.0 / 3.0 * rho * u, rtol=1e-5)
    np.testing.assert_allclose(out[-1, :, 6], g[:, 7] - rho * u / 6.0 - 0.5 * (g[:, 3] - g[:, 4]), rtol=1e-5)
    np.testing.assert_allclose(out[-1, :, 8], g[:, 5] - rho * u / 6.0 + 0.5 * (g[:, 3] - g[:, 4]), rtol=1e-5)


def test_a_velocity_set_the_closure_cannot_close_is_refused():
    """D2Q5 has no incoming direction with a tangential component, so no
    choice of the incoming populations can make the tangential velocity
    zero.  The closure says so instead of imposing a different condition
    under the same name."""
    e = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], np.int32)
    w = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
    with pytest.raises(ValueError, match="does not apply to this velocity set"):
        _zou_he_face_closure(e, w, CS2, 0, "min")
    # And a D2Q9 whose weights break the density identity.
    lat = d2q9()
    bad_w = lat.w.copy()
    bad_w[[1, 2]] = 0.2
    with pytest.raises(ValueError, match="w_q e_qn"):
        _zou_he_face_closure(lat.e, bad_w, CS2, 0, "min")


@pytest.mark.parametrize("lattice,shape", [("D2Q9", (10, 6)), ("D3Q19", (8, 5, 5))])
@pytest.mark.parametrize("inlet_face,outlet_face", [("x_min", "x_max"), ("y_max", "y_min")])
def test_update_leaves_every_fluid_face_cell_at_the_prescribed_pressure(
    lattice, shape, inlet_face, outlet_face,
):
    """Through the public node: after every step the node's own
    ``pressure`` field equals the prescribed inlet/outlet pressure on every
    fluid cell of the face, with walls touching both faces.  Measured
    before the fix: 0.4156 for 0.36 after one step."""
    # Walls along an axis orthogonal to the faces, so each face has fluid
    # cells and wall cells.
    wall_axis = 1 if _FACE_MAP[inlet_face][0] == 0 else 0
    wall = np.zeros(shape, bool)
    wall[_face_slice(len(shape), wall_axis, "min")] = True
    wall[_face_slice(len(shape), wall_axis, "max")] = True
    node = LBMNode("l", 1.0, grid_shape=shape, viscosity=0.1, lattice=lattice,
                   wall_mask=wall, inlet_face=inlet_face, outlet_face=outlet_face)
    p_in, p_out = 0.36, 0.30
    bi = {"inlet_pressure": jnp.float32(p_in), "outlet_pressure": jnp.float32(p_out)}
    st = node.initial_state()
    for _ in range(5):
        st = node.update(st, bi, 1.0)
        p = np.asarray(st["pressure"])
        for face, target in ((inlet_face, p_in), (outlet_face, p_out)):
            axis, side = _FACE_MAP[face]
            sl = _face_slice(len(shape), axis, side)
            fluid = ~wall[sl]
            np.testing.assert_allclose(p[sl][fluid], target, rtol=2e-6)
        assert np.all(np.isfinite(p))


def test_update_padded_imposes_the_same_face_pressure():
    """The sharded path calls the same closure; its face is exact too."""
    grid = (8, 4, 4)
    node = LBMNode("l", 1.0, grid_shape=grid, viscosity=0.1, lattice="D3Q19")
    st = node.initial_state()
    padded = {k: jnp.pad(v, [(1, 1)] * 3 + [(0, 0)] * (v.ndim - 3), mode="wrap")
              for k, v in st.items()}
    bi = {"inlet_pressure": jnp.float32(0.36), "outlet_pressure": jnp.float32(0.30)}
    out = node.update_padded(padded, bi, 1.0)
    p = np.asarray(out["pressure"])[1:-1, 1:-1, 1:-1]
    np.testing.assert_allclose(p[0], 0.36, rtol=2e-6)
    np.testing.assert_allclose(p[-1], 0.30, rtol=2e-6)


@pytest.mark.parametrize("lattice,shape", [("D2Q9", (10, 6)), ("D3Q19", (8, 5, 5))])
def test_the_reported_outlet_pressure_is_the_imposed_one(lattice, shape):
    """``compute_boundary_fluxes()["outlet_pressure_avg"]`` is what a coupled
    node (a Windkessel's backpressure, say) reads.  It equals the imposed
    outlet pressure; before the fix it reported 0.3556 for 0.30 after one
    step and 0.4014 after 400."""
    node = LBMNode("l", 1.0, grid_shape=shape, viscosity=0.1, lattice=lattice)
    bi = {"inlet_pressure": jnp.float32(0.36), "outlet_pressure": jnp.float32(0.30)}
    st = node.initial_state()
    for _ in range(3):
        st = node.update(st, bi, 1.0)
        flux = float(node.compute_boundary_fluxes(st, bi, 1.0)["outlet_pressure_avg"])
        assert flux == pytest.approx(0.30, rel=2e-6)


@pytest.mark.parametrize("via", ["boundary_input", "state"])
def test_the_reported_outlet_pressure_averages_over_the_runtime_wall_mask(via):
    """``update`` applies the runtime wall mask (``wall_mask_update`` or the
    ``wall_mask`` state field); the reported outlet average must exclude
    the same cells.  With a runtime wall over half the outlet face the
    fluid cells sit exactly at the imposed pressure and the wall cells do
    not, so averaging over the constructor's (empty) mask reported a value
    that no cell carries."""
    shape = (8, 6)
    node = LBMNode("l", 1.0, grid_shape=shape, viscosity=0.1, lattice="D2Q9")
    wall = np.zeros(shape, bool)
    wall[-1, :3] = True
    bi = {"inlet_pressure": jnp.float32(0.36), "outlet_pressure": jnp.float32(0.33)}
    st = node.initial_state()
    if via == "boundary_input":
        bi["wall_mask_update"] = jnp.asarray(wall)
    else:
        st = {**st, "wall_mask": jnp.asarray(wall, jnp.uint8)}
    for _ in range(20):
        st = node.update(st, bi, 1.0)
    p_face = np.asarray(st["pressure"])[-1]
    # The fixture can express the defect: wall and fluid cells differ.
    assert abs(p_face[:3].mean() - p_face[3:].mean()) > 1e-3
    flux = float(node.compute_boundary_fluxes(st, bi, 1.0)["outlet_pressure_avg"])
    assert flux == pytest.approx(0.33, rel=2e-6)
    assert flux == pytest.approx(float(p_face[~wall[-1]].mean()), rel=1e-6)


def test_the_closure_is_checked_for_both_supported_lattices_on_every_face():
    """Every face of both lattices passes the closure's lattice checks with
    the transverse coefficient 1/2 (Zou & He; Hecht & Harting)."""
    for lat in (d2q9(), d3q19()):
        assert isinstance(lat, LatticeDescriptor)
        for face, (axis, side) in _FACE_MAP.items():
            if axis >= lat.D:
                continue
            *_, tang_weight, _ = _zou_he_face_closure(lat.e, lat.w, lat.cs2, axis, side)
            assert set(tang_weight) == {a for a in range(lat.D) if a != axis}
            assert all(v == 0.5 for v in tang_weight.values()), (lat.name, face)
