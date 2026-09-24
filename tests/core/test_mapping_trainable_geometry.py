"""A mapped edge on a grid derived from a trainable parameter keeps its
constructor geometry when that parameter is calibrated.

MADD-ANO-022.  A ``{"node": ..., "field": ...}`` point reference resolves
its coordinates from the node's ``static_data`` once, when the mapping
is built, and the weights are snapshotted into ``gm.params["mappings"]``
at compile time.  On a *uniform* ``HeatNode`` the ``grid_x`` static is
built in ``__init__`` from ``length``, and ``length`` is trainable: the
node's own step reads the traced ``length`` (``dx = length / n_cells``)
and never ``grid_x``, which is why ``static_data_deps`` names nothing
and ``compile()`` accepts the graph.  So calibrating ``length`` through
``gm.params`` moves the rod and leaves the mapping's interpolation
weights at the constructor's geometry.  Nothing refuses it and nothing
warns.

The reference throughout is the graph the user means: the same graph
*constructed* at the calibrated length, so the mapping is built from the
calibrated grid.  Every measurement uses a freshly built graph, because
``run_scan`` advances the graph's own state.

The strict xfail is the acceptance criterion and changes state when the
fix being scoped for 0.5.0 lands, whichever form it takes: a mapping that
follows the parameter makes it pass, and a refusal makes it error
instead of failing an assertion.  The plain tests pin what a user
observes today, and the documented workaround.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening import sysid
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.heat import HeatNode
from maddening.serialization import config as cfg

N_SRC, N_DST = 8, 16
ALPHA, DT, STEPS = 0.05, 1e-3, 40
L_CONSTRUCTED, L_CALIBRATED = 1.0, 1.25
L_DST = L_CALIBRATED               # the target rod spans the true source length
EDGE = "src.temperature->dst.heat_source"

#: A declarative mapped edge: the points come from the two rods' own
#: ``grid_x`` statics through node-field references.  Thin-plate spline
#: with the linear polynomial keeps the operator well conditioned
#: (max|H| about 2), so the effect measured is the geometry, not an
#: ill-conditioned extrapolation.
SPEC = {"kind": "rbf", "kernel": "thin_plate_spline", "epsilon": 2.0,
        "mode": "consistent",
        "points": {"source_points": {"node": "src", "field": "grid_x"},
                   "target_points": {"node": "dst", "field": "grid_x"}}}


def _build(length, *, mapped=True):
    gm = GraphManager()
    gm.add_node(HeatNode("src", DT, n_cells=N_SRC, length=length,
                         thermal_diffusivity=ALPHA, initial_temperature=0.0))
    gm.add_node(HeatNode("dst", DT, n_cells=N_DST, length=L_DST,
                         thermal_diffusivity=ALPHA, initial_temperature=0.0))
    if mapped:
        gm.add_edge("src", "dst", "temperature", "heat_source", mapping=dict(SPEC))
    with warnings.catch_warnings():
        # An unmapped control graph has two disconnected nodes, on purpose.
        warnings.filterwarnings("ignore", message=".*is disconnected.*")
        gm.compile()
    # The same curved profile in cell index whatever the length, so the
    # length enters through the physics and the mapping only.
    i = jnp.arange(N_SRC, dtype=jnp.float32)
    gm.set_node_state(
        "src", {"temperature": 1.0 + 0.5 * jnp.sin(i / (N_SRC - 1) * jnp.pi)})
    return gm


def _with_length(gm, length):
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["src"]["length"] = jnp.asarray(length, jnp.float32)
    return p


def _dst_after(gm, params=None):
    return np.asarray(gm.run_scan(STEPS, params=params)["dst"]["temperature"])


def _sum_dst_injected(length):
    """``sum(dst T)`` after STEPS steps, ``length`` injected through params,
    differentiable (the step function driven directly, as sysid does)."""
    gm = _build(L_CONSTRUCTED)
    step = gm._build_step_fn()                           # noqa: SLF001
    ext = gm._resolve_external_inputs(None)              # noqa: SLF001
    p = _with_length(gm, length)
    state = jax.lax.fori_loop(0, STEPS, lambda _, s: step(s, ext, p), gm._state)  # noqa: SLF001
    return jnp.sum(state["dst"]["temperature"])


def _sum_dst_constructed(length):
    return float(np.sum(_dst_after(_build(length))))


@pytest.fixture(scope="module")
def runs():
    """Final target temperatures, each from its own freshly built graph."""
    injected_graph = _build(L_CONSTRUCTED)
    return {
        "injected": _dst_after(injected_graph, _with_length(injected_graph, L_CALIBRATED)),
        "constructed": _dst_after(_build(L_CALIBRATED)),
        "uncalibrated": _dst_after(_build(L_CONSTRUCTED)),
    }


def test_the_fixture_can_express_the_defect():
    """The mapping really depends on the length: the operator built at the
    calibrated length differs from the constructor's by O(1)."""
    h0 = np.asarray(_build(L_CONSTRUCTED).params["mappings"][EDGE]["H"])
    h1 = np.asarray(_build(L_CALIBRATED).params["mappings"][EDGE]["H"])
    assert np.max(np.abs(h0)) < 3.0 and np.max(np.abs(h1)) < 3.0
    assert np.max(np.abs(h1 - h0)) > 1.0


def test_the_source_rod_itself_honours_the_calibrated_length():
    """Control: without the mapped edge, injecting ``length`` is exactly
    constructing at it.  The discrepancy below is the mapping's alone."""
    a = _build(L_CONSTRUCTED, mapped=False)
    injected = a.run_scan(STEPS, params=_with_length(a, L_CALIBRATED))["src"]["temperature"]
    constructed = _build(L_CALIBRATED, mapped=False).run_scan(STEPS)["src"]["temperature"]
    np.testing.assert_array_equal(np.asarray(injected), np.asarray(constructed))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "MADD-ANO-022: a {'node', 'field'} point reference resolves grid_x "
        "from static_data once, built from the constructor's length; "
        "calibrating the trainable length through gm.params moves the rod "
        "and leaves the mapping's weights at the constructor geometry.  A fix "
        "is being scoped for 0.5.0: a mapping that follows the parameter "
        "turns this into an XPASS, a refusal turns it into an error.  Either "
        "way, update MADD-ANO-022 and the pins below."
    ),
)
def test_a_calibrated_length_reaches_the_mapped_edge(runs):
    np.testing.assert_allclose(runs["injected"], runs["constructed"],
                               rtol=1e-4, atol=1e-5)


def test_the_calibration_barely_moves_the_mapped_target(runs):
    """What a user sees today.  Constructed at the calibrated length, the
    target moves by about 1.2e-2; calibrated through ``gm.params``, by
    about 4e-4 -- under a tenth of it -- and ends about 1.2e-2 from where
    it should be."""
    effect_meant = np.max(np.abs(runs["constructed"] - runs["uncalibrated"]))
    effect_got = np.max(np.abs(runs["injected"] - runs["uncalibrated"]))
    error = np.max(np.abs(runs["injected"] - runs["constructed"]))
    assert effect_meant > 1e-2
    assert effect_got < 0.1 * effect_meant
    assert error > 0.9 * effect_meant


def test_the_weights_stay_the_constructor_weights_after_calibration():
    gm = _build(L_CONSTRUCTED)
    before = np.asarray(gm.params["mappings"][EDGE]["H"])
    p = _with_length(gm, L_CALIBRATED)
    gm.run_scan(STEPS, params=p)
    np.testing.assert_array_equal(np.asarray(p["mappings"][EDGE]["H"]), before)
    np.testing.assert_array_equal(np.asarray(gm.params["mappings"][EDGE]["H"]), before)


def test_the_length_gradient_through_the_edge_has_the_wrong_sign():
    """``jax.grad`` is exact for the model as wired (it matches a finite
    difference of the injected path), but that model has frozen weights:
    against a finite difference over graphs constructed at each length it
    has the opposite sign (about -6.4e-3 against +2.7e-1)."""
    g = float(jax.grad(_sum_dst_injected)(jnp.float32(L_CONSTRUCTED)))
    h = 1e-2
    fd_injected = (float(_sum_dst_injected(jnp.float32(L_CONSTRUCTED + h)))
                   - float(_sum_dst_injected(jnp.float32(L_CONSTRUCTED - h)))) / (2 * h)
    fd_constructed = (_sum_dst_constructed(L_CONSTRUCTED + h)
                      - _sum_dst_constructed(L_CONSTRUCTED - h)) / (2 * h)
    assert g == pytest.approx(fd_injected, rel=1e-2)
    assert fd_constructed > 0.1
    assert g < 0.0


def test_the_model_cannot_fit_its_own_data_at_the_true_length():
    """Data generated by the graph constructed at the true length, a
    windowed loss on the target rod: through ``gm.params`` the loss at the
    true length is not zero (about 1e-2), so a fit of ``length`` from the
    mapped target converges somewhere else."""
    truth = _build(L_CALIBRATED)
    init = {n: truth.get_node_state(n) for n in truth.node_names}
    _, hist = truth.run_scan_with_history(STEPS)
    obs = sysid.observations_from_history(init, {n: hist[n] for n in init})
    obs_fn = lambda s: s["dst"]["temperature"]          # noqa: E731
    reference = _build(L_CALIBRATED)
    at_truth_constructed = float(sysid.windowed_loss(
        reference, reference.params, obs, obs_fn=obs_fn, window=STEPS))
    fit_graph = _build(L_CONSTRUCTED)
    at_truth_injected = float(sysid.windowed_loss(
        fit_graph, _with_length(fit_graph, L_CALIBRATED), obs, obs_fn=obs_fn,
        window=STEPS))
    assert at_truth_constructed == 0.0
    assert at_truth_injected > 1e-3


def test_a_fit_of_the_length_through_the_mapped_edge_lands_far_from_the_truth():
    """The symptom a user meets.  Fitting ``length`` (only) with
    ``fit_lm`` from the target rod's data, starting at 1.0 with the truth
    at 1.25, stops at about 0.22 with a small loss (about 1e-5), so the
    fit looks successful.  The same fit from the *source* rod's own data
    recovers 1.25: the rod's physics honours the calibrated length, the
    mapped edge does not."""
    truth = _build(L_CALIBRATED)
    init = {n: truth.get_node_state(n) for n in truth.node_names}
    _, hist = truth.run_scan_with_history(STEPS)
    obs = sysid.observations_from_history(init, {n: hist[n] for n in init})

    def fitted(observed):
        gm = _build(L_CONSTRUCTED)
        mask = jax.tree.map(lambda _: False, gm.params)
        mask["nodes"]["src"]["length"] = True
        res = sysid.fit_lm(
            gm,
            lambda p: sysid.windowed_loss(
                gm, p, obs, obs_fn=lambda s: s[observed]["temperature"], window=STEPS),
            mask=mask, n_iter=20)
        return float(res.params["nodes"]["src"]["length"]), float(res.losses[-1])

    through_edge, loss_edge = fitted("dst")
    from_rod, _ = fitted("src")
    assert from_rod == pytest.approx(L_CALIBRATED, abs=1e-3)
    assert through_edge < 0.5
    assert loss_edge < 1e-4


def test_nothing_refuses_or_warns_about_the_stale_geometry():
    """The silence the registry records: the length is trainable by
    default, ``compile()`` accepts the edge, a run with a calibrated
    length warns nothing, and ``to_dict`` writes the node references
    (their hashes are of the constructor grid, which is unchanged)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm = _build(L_CONSTRUCTED)
        assert gm.trainable_mask()["nodes"]["src"]["length"] is True
        gm.run_scan(STEPS, params=_with_length(gm, L_CALIBRATED))
        d = cfg.to_dict(gm)
    assert d["edges"][0]["mapping"]["points"]["source_points"]["node"] == "src"
    ours = [w for w in caught if "maddening" in (w.filename or "")]
    assert ours == [], [str(w.message) for w in ours]


def test_the_workaround_freezing_the_length_keeps_a_fit_off_it():
    """Declaring ``length`` non-trainable removes it from the default mask,
    and a mask that tries to widen back onto it is refused."""
    gm = _build(L_CONSTRUCTED)
    gm.set_param_spec("src", "length", ParamSpec(trainable=False))
    assert gm.trainable_mask()["nodes"]["src"]["length"] is False
    mask = jax.tree.map(lambda _: False, gm.params)
    mask["nodes"]["src"]["length"] = True
    with pytest.raises(ValueError, match="trainable"):
        sysid.fit(gm, lambda p: jnp.sum(p["nodes"]["src"]["length"]), mask=mask, n_iter=1)
