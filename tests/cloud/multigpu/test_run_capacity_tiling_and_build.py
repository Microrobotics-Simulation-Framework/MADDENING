"""``run_capacity.py``: the tiling it allows, the field it builds, and the
comparison it makes -- each on its own, on CPU virtual devices.

The capacity test's reference is a tile's own run, tiled.  That proves a
large sharded run only if (1) the tiling lets a wrong halo show -- every
shard at a different phase of the tile -- (2) the field really is the tile
repeated, built one device's block at a time and never whole, and (3) the
comparison sees a single wrong cell and counts every cell once.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.cloud.multigpu import run_capacity_support as S

rc = S.module()
TILE = S.TILE

#: (mesh, tile, counts, a phrase of the reason) the script must refuse.
REFUSED = [
    # A count commensurate with the devices on a split axis.
    ("2x2", TILE, (2, 1, 1), "must be odd"),
    ("2x2", TILE, (1, 2, 1), "must be odd"),
    ("2x2", TILE, (3, 4, 1), "must be odd"),
    ("4x1", TILE, (2, 1, 1), "must be odd"),
    ("4x1", TILE, (4, 1, 1), "must be odd"),
    ("4x1", TILE, (6, 3, 1), "must be odd"),
    ("3x1", (6, 4, 8), (3, 1, 1), "share no factor"),     # odd, and a multiple of the devices
    ("3x1", (6, 4, 8), (2, 1, 1), "must be odd"),
    # A tile the devices do not divide.
    ("2x2", (7, 4, 6), (1, 1, 1), "do not divide"),
    ("2x2", (8, 5, 6), (1, 1, 1), "do not divide"),
    ("4x1", (10, 4, 6), (1, 1, 1), "do not divide"),
    # A tile with one row per device.
    ("4x1", (4, 6, 5), (1, 1, 1), "less than twice"),
    ("2x2", (8, 2, 6), (1, 1, 1), "less than twice"),
    # Two axes of the grid alike.
    ("2x2", (8, 4, 8), (1, 1, 1), "same extent"),
    ("2x2", TILE, (1, 3, 2), "same extent"),               # 8 x 12 x 12
    ("2x2", TILE, (0, 1, 1), "at least 1"),
]
ALLOWED = [
    ("2x2", TILE, (1, 1, 1)), ("2x2", TILE, (3, 1, 1)), ("2x2", TILE, (3, 3, 1)),
    ("2x2", TILE, (5, 7, 4)), ("4x1", TILE, (3, 2, 1)), ("4x1", TILE, (5, 4, 3)),
    ("3x1", (6, 4, 8), (5, 1, 1)), ("2x2", rc.DEFAULT_TILE, (1, 1, 1)),
    ("4x1", rc.DEFAULT_TILE, (1, 1, 1)),
]


def _per_axis(mesh: str) -> tuple:
    return rc.devices_per_axis(rc.parse_mesh(mesh))


@pytest.mark.parametrize("mesh, tile, k, words", REFUSED)
def test_a_tiling_on_which_a_wrong_halo_could_hide_is_refused_with_the_reason(mesh, tile, k,
                                                                             words):
    why = rc.tiling_problem(tile, k, _per_axis(mesh))
    assert why is not None and words in why, why


@pytest.mark.parametrize("mesh, tile, k", ALLOWED)
def test_an_allowed_tiling_starts_every_shard_at_its_own_phase_of_the_tile(mesh, tile, k):
    """What the rules are for, as arithmetic: along a split axis the D
    shards start on D different rows of the tile, and the row a halo must
    hold differs from the row each wrong source would deliver."""
    per_axis = _per_axis(mesh)
    assert rc.tiling_problem(tile, k, per_axis) is None
    for t, n, d in zip(tile, k, per_axis):
        if d == 1:
            continue
        block = t * n // d
        starts = [(shard * block) % t for shard in range(d)]
        assert len(set(starts)) == d, starts
        for shard in range(d):
            right = (starts[shard] + block) % t            # the right neighbour's first row
            assert right != starts[shard]                  # the shard's own first row
            assert right != (starts[shard] - 1) % t        # the left neighbour's last row
            for other in range(d):
                if other != (shard + 1) % d:
                    assert right != starts[other] % t, (shard, other)
            if d > 2:
                # The right edge of the neighbour on the other side.
                assert (starts[shard] - 1) % t != (starts[(shard + 2) % d] - 1) % t


@pytest.mark.parametrize("mesh, k", [("4x1", (2, 1, 1)), ("2x2", (2, 1, 1)), ("4x1", (4, 1, 1))])
def test_a_refused_count_is_one_that_makes_two_shards_alike(mesh, k):
    """The other direction: each refused count puts two shards on the same
    rows of the tile, which is the reason given."""
    d = _per_axis(mesh)[0]
    block = TILE[0] * k[0] // d
    starts = [(shard * block) % TILE[0] for shard in range(d)]
    assert len(set(starts)) < d, starts
    assert "two shards hold the same rows" in rc.tiling_problem(TILE, k, _per_axis(mesh))


@pytest.mark.parametrize("mesh", ["2x2", "4x1"])
@pytest.mark.parametrize("cells", [192, 200, 1000, 5000, 123_457, 10 ** 6, 3 * 10 ** 8])
def test_the_tiling_chosen_for_a_size_is_allowed_and_never_larger(mesh, cells):
    for tile in (TILE, rc.DEFAULT_TILE):
        if cells < math.prod(tile):
            with pytest.raises(ValueError, match="hold no allowed tiling"):
                rc.tiling_for_cells(cells, tile, _per_axis(mesh))
            continue
        k = rc.tiling_for_cells(cells, tile, _per_axis(mesh))
        assert rc.tiling_problem(tile, k, _per_axis(mesh)) is None
        got = math.prod(rc.grid_shape(tile, k))
        assert got <= cells
        if cells >= 10 ** 6:
            # At the session's sizes the grid is close to the size asked
            # for and close to a cube.
            shape = rc.grid_shape(tile, k)
            assert got >= 0.9 * cells and max(shape) <= 1.5 * min(shape), (k, shape)


def _refusing_launcher(argv, timeout_s, stderr_path):
    raise AssertionError(f"a refused run started a rung: {argv}")


@pytest.mark.parametrize("argv, words", [
    (["--mesh", "4x1", *S.TILE_ARGS, "--k", "2", "1", "1"], "must be odd"),
    (["--mesh", "2x2", *S.TILE_ARGS, "--k", "1", "2", "1"], "must be odd"),
    (["--mesh", "2x2", "--tile", "7", "4", "6", "--cells", "5000"], "do not divide"),
    (["--mesh", "4x1", "--tile", "4", "6", "5", "--cells", "5000"], "less than twice"),
    (["--mesh", "2x2", *S.TILE_ARGS, "--cells", "100"], "hold no allowed tiling"),
    (["--mesh", "1x4", *S.TILE_ARGS, "--cells", "5000"], "at least two devices"),
    (["--mesh", "pencil", *S.TILE_ARGS, "--cells", "5000"], "expected AxB"),
    ([*S.TILE_ARGS, "--ramp", "0.5"], "--device-memory-gb"),
    ([*S.TILE_ARGS, "--ramp", "0.5", "0.4", "--device-memory-gb", "24"], "must rise"),
    ([*S.TILE_ARGS, "--ramp", "0.5", "0.99", "--device-memory-gb", "24"], "must be in (0, 0.95]"),
    ([*S.TILE_ARGS, "--ramp", "0.25", "--device-memory-gb", "1e-9"], "hold no allowed tiling"),
    ([*S.TILE_ARGS, "--cells", "5000", "--steps", "0"], "--steps 0"),
    ([*S.TILE_ARGS, "--cells", "5000", "--soak-at", "1.5"], "--soak-at"),
])
def test_the_command_line_refuses_it_with_exit_2_before_anything_runs(argv, words, tmp_path,
                                                                    capsys):
    out = tmp_path / "out"
    status = rc.exit_status(["--dry-run", "--out", str(out), *argv], _refusing_launcher)
    assert status == rc.EXIT_REFUSED == 2
    assert words in capsys.readouterr().err
    assert not out.exists()


def test_an_abbreviated_dry_run_is_refused(tmp_path, capsys):
    """The CPU pin reads the literal ``--dry-run`` before the options are
    parsed: an accepted ``--dry`` would be a "dry run" on whatever
    accelerator is there."""
    with pytest.raises(SystemExit) as refused:
        rc.exit_status(["--dry", "--out", str(tmp_path / "out"), *S.TILE_ARGS, "--cells", "5000"],
                       _refusing_launcher)
    assert refused.value.code == rc.EXIT_REFUSED
    assert "unrecognized arguments: --dry" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


# --- the field, built one device's block at a time ----------------------------


@pytest.fixture(scope="module")
def backend():
    import jax

    if len(jax.devices()) < 4:
        pytest.skip("needs >= 4 devices")
    rc._load_backend()
    return jax


def _grid(mesh: str):
    """``(k, shape, mesh, axis_map)`` of the tests' grid on ``mesh``: 24 x
    12 x 6, a block of which is larger than the tile on either mesh."""
    k = (3, 3, 1)
    built, axis_map = rc.build_mesh(rc.parse_mesh(mesh))
    return k, rc.grid_shape(TILE, k), built, axis_map


def _tile_state():
    node = rc.LBMNode("tile", 1.0, grid_shape=TILE, viscosity=rc.VISCOSITY, lattice=rc.LATTICE)
    return rc.tile_start(TILE, node.lattice)


def _tiled(field: np.ndarray, k) -> np.ndarray:
    return np.tile(field, tuple(k) + (1,) * (field.ndim - 3))


def test_the_tile_differs_on_every_plane_and_holds_the_states_bytes(backend):
    state = _tile_state()
    assert set(state) == set(rc.STATE_FIELDS)
    assert sum(v.dtype.itemsize * math.prod(v.shape[3:]) for v in state.values()) \
        == rc.STATE_BYTES_PER_CELL == 97
    scale = float(np.max(state["f"]))
    for tile, tile_state in ((TILE, state), (rc.DEFAULT_TILE, rc.tile_start(
            rc.DEFAULT_TILE, rc.LBMNode("t", 1.0, grid_shape=rc.DEFAULT_TILE,
                                        lattice=rc.LATTICE).lattice))):
        for axis in range(3):
            apart = rc.least_plane_difference(tile_state["f"], axis) / scale
            assert apart > rc.PLANES_APART * rc.LIMIT, (tile, axis, apart)
    # A velocity far above the populations' float32 rounding: the
    # comparison is relative to it.
    assert float(np.max(np.abs(state["velocity"]))) > 0.05
    assert rc.least_plane_difference(np.zeros((4, 3, 2)), 0) == 0.0


@pytest.mark.parametrize("mesh", ["2x2", "4x1"])
def test_the_built_field_is_the_tile_repeated_with_each_block_on_its_own_device(backend, mesh):
    k, shape, built, axis_map = _grid(mesh)
    tile_state = _tile_state()
    state = rc.build_state(tile_state, TILE, shape, built, axis_map)
    devices = list(built.devices.flat)
    for name, array in state.items():
        want = _tiled(tile_state[name], k)
        assert array.shape == want.shape and array.dtype == want.dtype
        np.testing.assert_array_equal(np.asarray(array), want)
        assert array.sharding.is_equivalent_to(
            rc.field_sharding(built, axis_map, array.ndim), array.ndim)
        shards = array.addressable_shards
        assert sorted(str(s.device) for s in shards) == sorted(map(str, devices))
        for shard in shards:
            assert shard.data.devices() == {shard.device}
            np.testing.assert_array_equal(np.asarray(shard.data), want[shard.index])
    # No two devices hold the same block of the populations: the reason
    # for the tiling rules.
    blocks = [np.asarray(s.data) for s in state["f"].addressable_shards]
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            assert not np.array_equal(blocks[i], blocks[j]), (i, j)


@pytest.mark.parametrize("mesh", ["2x2", "4x1"])
def test_the_builder_is_never_asked_for_more_than_one_shards_block(backend, mesh, monkeypatch):
    """The whole field never exists on one device or on the host: each
    device is asked once, for its own block, and nothing larger than the
    tile leaves the host."""
    k, shape, built, axis_map = _grid(mesh)
    cells, n_devices = math.prod(shape), built.devices.size
    asked, sent = [], []
    real_block, real_put = rc.tiled_block, backend.device_put

    def recording_block(tile_state, ranges, tile, device, *, most_cells):
        asked.append((str(device), [tuple(r) for r in ranges], most_cells))
        return real_block(tile_state, ranges, tile, device, most_cells=most_cells)

    def recording_put(value, device=None, **kwargs):
        if isinstance(value, np.ndarray):
            sent.append(value.size)
        return real_put(value, device, **kwargs)

    monkeypatch.setattr(rc, "tiled_block", recording_block)
    monkeypatch.setattr(backend, "device_put", recording_put)
    requests: list = []
    tile_state = _tile_state()
    state = rc.build_state(tile_state, TILE, shape, built, axis_map, requests)
    backend.block_until_ready(state)

    assert len(asked) == n_devices == len(requests)             # the patch took effect
    assert len({device for device, _, _ in asked}) == n_devices
    covered = np.zeros(shape, np.int32)
    for _device, ranges, most_cells in asked:
        block_cells = math.prod(stop - start for start, stop in ranges)
        assert block_cells == most_cells == cells // n_devices
        covered[tuple(slice(*r) for r in ranges)] += 1
    assert np.all(covered == 1)                                 # every cell, once
    # From the host: the tile's fields and index vectors, nothing of a
    # block's size.
    assert sent and max(sent) <= max(v.size for v in tile_state.values())
    assert max(sent) < cells // n_devices * rc.LATTICE_Q


def test_the_builder_refuses_a_block_larger_than_a_shard(backend):
    k, shape, built, axis_map = _grid("2x2")
    whole = [(0, n) for n in shape]
    with pytest.raises(RuntimeError, match="more than one shard's"):
        rc.tiled_block(_tile_state(), whole, TILE, built.devices.flat[0],
                       most_cells=math.prod(shape) // built.devices.size)


# --- the comparison -----------------------------------------------------------


def _plane_bytes(mesh: str, shape) -> int:
    a, b = rc.parse_mesh(mesh)
    return (shape[1] // b) * shape[2] * rc.STATE_BYTES_PER_CELL


@pytest.mark.parametrize("mesh", ["2x2", "4x1"])
@pytest.mark.parametrize("planes", [1, 4, 5, 1000])
def test_the_comparison_reads_zero_for_the_tiled_field_and_counts_every_cell_once(
        backend, mesh, planes):
    """Whatever the chunk -- one plane, a chunk that divides the block, one
    that does not (its last chunk starts early and repeats planes), one
    larger than the block -- every cell is compared once and the mass is
    the tile's times the number of tiles."""
    k, shape, built, axis_map = _grid(mesh)
    cells = math.prod(shape)
    tile_state = _tile_state()
    state = rc.build_state(tile_state, TILE, shape, built, axis_map)
    block_planes = shape[0] // rc.parse_mesh(mesh)[0]
    got = rc.compare_with_tiling(state, tile_state, TILE, shape,
                                 planes * _plane_bytes(mesh, shape))
    used = min(planes, block_planes)
    assert got["chunk_planes"] == used
    assert got["chunk_bytes"] == used * _plane_bytes(mesh, shape)
    assert got["chunks"] == built.devices.size * math.ceil(block_planes / used)
    for name, entry in got["fields"].items():
        assert entry["max_abs"] == 0.0 and entry["exact"] and entry["finite"], name
        assert entry["cells_compared"] == cells, name
        assert entry["reference_scale"] == pytest.approx(
            float(np.max(np.abs(tile_state[name]))), rel=1e-6), name
    want = float(np.sum(tile_state["f"], dtype=np.float64)) * math.prod(k)
    assert got["mass"] == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("mesh", ["2x2", "4x1"])
def test_one_wrong_cell_on_one_device_is_found_in_its_field_alone(backend, mesh):
    """One population of one cell of the last device's block, off by 1e-3:
    the comparison reports that difference for ``f`` and none elsewhere --
    and a NaN there as not finite, with an infinite difference (never
    zero: a reduction's ``max`` may drop a NaN)."""
    k, shape, built, axis_map = _grid(mesh)
    tile_state = _tile_state()
    state = rc.build_state(tile_state, TILE, shape, built, axis_map)
    cell = (shape[0] - 1, shape[1] - 2, 3, 7)
    for wrong, finite in ((np.float32(1e-3), True), (np.float32(np.nan), False)):
        f = np.asarray(state["f"]).copy()
        f[cell] += wrong
        changed = dict(state, f=backend.device_put(f, state["f"].sharding))
        got = rc.compare_with_tiling(changed, tile_state, TILE, shape, 4 * _plane_bytes(mesh, shape))
        entry = got["fields"]["f"]
        assert entry["finite"] is finite and not entry["exact"]
        if finite:
            assert entry["max_abs"] == pytest.approx(1e-3, rel=1e-3)
            assert entry["max_rel"] == pytest.approx(1e-3 / float(np.max(tile_state["f"])),
                                                     rel=1e-3)
            assert entry["max_rel"] > rc.LIMIT
        else:
            assert math.isinf(entry["max_abs"]) and math.isinf(entry["max_rel"])
            assert rc.check("f", entry["max_rel"], rc.LIMIT)["passed"] is False
        for name in ("density", "velocity", "pressure", "wall_mask"):
            assert got["fields"][name]["exact"] and got["fields"][name]["finite"], name


def test_the_limit_is_the_runners_forward_limit():
    spec_runner = rc._RUNNER
    assert rc.LIMIT == spec_runner.LIMITS["forward"] == 1e-5
    assert rc.RUNNER_GOALS == tuple(spec_runner.ALL_GOALS) and len(rc.RUNNER_GOALS) == 8
