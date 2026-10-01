"""Every built-in node, as a generator of valid constructions and of writes.

The differential state-and-I/O harness (``test_differential_*.py`` in this
directory, ``tests/usd/test_usd_differential_round_trip.py`` and
``tests/cloud/multigpu/test_property_sharded_state_io_differential.py``)
compares two paths that must agree -- a write and a reload, a checkpoint and
an uninterrupted run, a graph and its serialisation, three FMU transports --
over *every* built-in node rather than the five scalar kinds of
``strategies.py``.  This module is the one place that knows, per node:

* how to construct it with **valid** arguments (:attr:`Kind.kwargs`), drawn
  so that a few steps stay finite and every constructor check passes;
* which boundary inputs to declare as external inputs, so that a parameter
  read only through an input (a ball's ``elasticity`` through
  ``table_position``) is live;
* which **writes** to propose (:func:`writes`), in named categories:
  ``valid``, ``boundary``, ``invalid``, ``non_finite``, ``oversized``,
  ``wrong_type``, ``unknown``, ``initial``, ``structural``, ``cross``,
  ``multi`` and ``mixed`` (a valid key beside a refused one).  A write is a ``{key: JSON value}`` request body; the oracle
  decides what the paths must do with it, the generator only proposes.

``cheap`` kinds compile and step in well under a second on a CI runner and
run in the per-push properties; the others (the LBM nodes, the wavelet
node) run in the slow lane.

``test_every_built_in_node_has_a_catalogue_entry`` in
``test_differential_serialisation.py`` fails closed when a node class is
exported without an entry here.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.nodes import (
    BallNode,
    HealthCheckNode,
    HeartPumpNode,
    HeatNode,
    LBMNode,
    LBMPipeNode,
    RigidBody2DNode,
    RigidBodyNode,
    SpringDamperNode,
    TableNode,
)
from maddening.nodes.adaptive import WaveletAdaptiveNode


def f32(value: float) -> float:
    """``value`` rounded to the nearest float32, as a Python float.

    Every node constant is held as float32, so a float64 literal would be
    rounded on its way in and an exact comparison would then depend on the
    rounding rather than on the path under test.
    """
    return float(np.float32(value))


def floats32(lo: float, hi: float) -> st.SearchStrategy[float]:
    """Finite float32-exact values in ``[lo, hi]``."""
    return st.floats(min_value=f32(lo), max_value=f32(hi), allow_nan=False,
                     allow_infinity=False, allow_subnormal=False, width=32)


def vec32(lo: float, hi: float, n: int) -> st.SearchStrategy[list]:
    return st.lists(floats32(lo, hi), min_size=n, max_size=n)


# ---------------------------------------------------------------------------
# A kind
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Kind:
    """One built-in node configuration family.

    Attributes
    ----------
    name : str
        Catalogue key (a node may have several families: a uniform and a
        non-uniform rod).
    cls : type
        The node class.
    timestep : float
        Its timestep; fixed per family so a family is one compiled shape
        modulo its drawn constants.
    kwargs : SearchStrategy[dict]
        Valid constructor arguments.
    inputs : callable
        ``kwargs -> {input name: shape}``: boundary inputs to declare as
        external inputs (left at zero).
    safe : dict
        ``{params key: (lo, hi)}``: the range a ``valid`` write of that
        key is drawn from.  Inside every ``ParamSpec`` bound and every
        constructor check, and stable over a handful of steps.
    structural : callable
        ``kwargs -> SearchStrategy[dict]`` of writes to constructor
        arguments that are not leaves of the params pytree (cell counts,
        grids, lattices, check dicts); ``None`` when there are none.
    cross : callable
        ``kwargs -> SearchStrategy[dict]`` of multi-key writes whose keys
        constrain each other; ``None`` when there are none.
    cheap : bool
        Compiles and steps in well under a second: runs per push.
    """

    name: str
    cls: type
    timestep: float
    kwargs: st.SearchStrategy
    inputs: Callable[[dict], dict[str, tuple]] = lambda kw: {}
    safe: dict = field(default_factory=dict)
    structural: Optional[Callable[[dict], st.SearchStrategy]] = None
    cross: Optional[Callable[[dict], st.SearchStrategy]] = None
    cheap: bool = True

    def build_node(self, kwargs: dict, name: Optional[str] = None):
        with warnings.catch_warnings():
            # RigidBody2DNode's deprecation; pyproject filters it too.
            warnings.simplefilter("ignore", DeprecationWarning)
            return self.cls(name or self.name, self.timestep, **kwargs)

    def graph(self, kwargs: dict, *, name: Optional[str] = None,
              compile: bool = True) -> GraphManager:
        """A one-node graph with every :attr:`inputs` declared."""
        node = self.build_node(kwargs, name)
        gm = GraphManager()
        gm.add_node(node)
        for field_name, shape in self.inputs(kwargs).items():
            gm.add_external_input(node.name, field_name, shape=shape)
        if compile:
            gm.compile()
        return gm


# ---------------------------------------------------------------------------
# Constructions
# ---------------------------------------------------------------------------

def _ball_kwargs():
    return st.fixed_dictionaries({
        "initial_position": floats32(0.0, 3.0),
        "initial_velocity": floats32(-1.0, 1.0),
        "elasticity": floats32(0.0, 1.0),
        "gravity": floats32(-12.0, -1.0),
    })


def _spring_kwargs():
    return st.fixed_dictionaries({
        "stiffness": floats32(1.0, 50.0),
        "damping": floats32(0.0, 2.0),
        "mass": floats32(0.5, 2.0),
        # Never exactly the initial position, or the one-step residual is
        # 0 whatever the stiffness (the common brief's fixture trap).
        "rest_length": floats32(0.25, 2.0),
        "initial_position": floats32(-1.0, 0.0),
        "initial_velocity": floats32(-1.0, 1.0),
    })


def _heart_kwargs():
    return st.fixed_dictionaries({
        "resistance": floats32(0.5, 2.0),
        "compliance": floats32(0.5, 2.0),
        "heart_rate": floats32(50.0, 90.0),
        "stroke_volume": floats32(40.0, 80.0),
        "venous_pressure": floats32(0.0, 5.0),
        "systole_fraction": floats32(0.2, 0.5),
        "initial_pressure": floats32(60.0, 90.0),
    })


def _health_kwargs():
    bound = st.fixed_dictionaries({
        "finite": st.booleans(),
        "min": floats32(-5.0, -1.0),
        "max": floats32(1.0, 5.0),
    })
    return st.fixed_dictionaries({
        "checks": st.dictionaries(st.sampled_from(["x", "y", "flow"]), bound,
                                  min_size=1, max_size=2),
    })


@st.composite
def _heat_kwargs(draw, *, nonuniform: bool, order: Optional[int] = None):
    """A rod well inside its Fourier limit at ``dt = 0.01``."""
    n = draw(st.integers(min_value=5, max_value=8))
    length = draw(st.sampled_from([1.0, 2.0]))
    stencil = order if order is not None else draw(st.sampled_from([2, 4]))
    # Fourier number <= 0.1 on the finest spacing either grid can have.
    dx = length / n
    alpha_hi = 0.1 * (dx * dx) / 0.01
    kw: dict[str, Any] = {
        "n_cells": n,
        "length": length,
        "thermal_diffusivity": draw(floats32(1e-3, min(1e-2, alpha_hi))),
        "stencil_order": stencil,
    }
    if draw(st.booleans()):
        kw["initial_temperature"] = draw(vec32(0.0, 5.0, n))
    else:
        kw["initial_temperature"] = draw(floats32(0.0, 5.0))
    if nonuniform:
        # Strictly increasing, cell widths at least half the uniform one.
        gaps = draw(st.lists(floats32(0.5, 1.5), min_size=n, max_size=n))
        xs = np.cumsum(np.asarray(gaps, np.float64)) * (length / n)
        kw["grid_points"] = [f32(x) for x in xs]
        kw["thermal_diffusivity"] = draw(floats32(1e-4, 1e-3))
    return kw


def _heat_inputs(kw: dict) -> dict[str, tuple]:
    return {"left_temperature": (), "right_temperature": (),
            "heat_source": (int(kw["n_cells"]),)}


def _lbm_kwargs():
    @st.composite
    def build(draw):
        nx = draw(st.integers(min_value=6, max_value=8))
        ny = draw(st.integers(min_value=4, max_value=5))
        kw: dict[str, Any] = {
            "grid_shape": (nx, ny),
            "lattice": "D2Q9",
            "viscosity": draw(floats32(0.05, 0.2)),
        }
        if draw(st.booleans()):
            wall = np.zeros((nx, ny), bool)
            wall[:, 0] = wall[:, -1] = True
            if draw(st.booleans()):
                wall[nx // 2, ny // 2] = True
            kw["wall_mask"] = wall.tolist()
        return kw
    return build()


def _pipe_kwargs():
    @st.composite
    def build(draw):
        multiphase = draw(st.booleans())
        kw: dict[str, Any] = {
            "nx": 6, "ny": 6, "nz": 6,
            "tau": draw(floats32(0.6, 1.0)),
            "pipe_radius": draw(st.sampled_from([0.7, 0.8])),
            "propeller_x": 2,
            "propeller_strength": draw(floats32(0.0, 1e-3)),
        }
        if multiphase:
            kw.update(G=draw(floats32(-5.0, -3.0)), rho_liquid=2.0, rho_gas=0.2,
                      fill_fraction=draw(st.sampled_from([0.5, 0.75])))
        return kw
    return build()


def _rigid_kwargs():
    return st.fixed_dictionaries({
        "mass": floats32(0.5, 2.0),
        "inertia": vec32(0.5, 2.0, 3),
        "gravity": vec32(-2.0, 2.0, 3),
        "constraints": st.sampled_from([{}, {"z": 0.0}, {"rx": 0.0, "ry": 0.0}]),
        "initial_position": vec32(-1.0, 1.0, 3),
        "initial_velocity": vec32(-1.0, 1.0, 3),
        "initial_angular_velocity": vec32(-0.5, 0.5, 3),
    })


def _rigid2d_kwargs():
    return st.fixed_dictionaries({
        "mass": floats32(0.5, 2.0),
        "inertia": floats32(0.5, 2.0),
        "gravity": vec32(-2.0, 2.0, 2),
        "initial_x": floats32(-1.0, 1.0),
        "initial_y": floats32(-1.0, 1.0),
        "initial_vx": floats32(-1.0, 1.0),
        "initial_vy": floats32(-1.0, 1.0),
        "initial_angle": floats32(-1.0, 1.0),
        "initial_omega": floats32(-1.0, 1.0),
    })


def _wavelet_kwargs():
    return st.fixed_dictionaries({
        "n_levels": st.just(4),
        "n_coarse": st.just(2),
        "theta": floats32(0.3, 0.5),
        "sigma": floats32(0.05, 0.15),
    })


# ---------------------------------------------------------------------------
# Structural and cross-parameter writes
# ---------------------------------------------------------------------------

def _heat_structural(kw):
    n = int(kw["n_cells"])
    return st.one_of(
        st.fixed_dictionaries({"n_cells": st.sampled_from([n, n + 1, 3, 0, -2])}),
        st.fixed_dictionaries({"stencil_order": st.sampled_from([2, 3, 4])}),
        st.fixed_dictionaries({"grid_points": st.one_of(
            st.none(),
            vec32(0.0, 2.0, n).map(sorted),
            vec32(0.0, 2.0, n + 1).map(sorted),
        )}),
        st.fixed_dictionaries({"geometry_source": st.sampled_from([None, "rod.usd"])}),
    )


def _heat_cross(kw):
    n = int(kw["n_cells"])
    return st.fixed_dictionaries({
        "thermal_diffusivity": floats32(1e-4, 5e-2),
        "length": floats32(0.2, 3.0),
    }) | st.fixed_dictionaries({
        "n_cells": st.just(n + 1),
        "initial_temperature": vec32(0.0, 5.0, n + 1),
    })


def _lbm_structural(kw):
    nx, ny = kw["grid_shape"]
    return st.one_of(
        st.fixed_dictionaries({"grid_shape": st.sampled_from(
            [[nx, ny], [nx + 1, ny], [nx, ny, 4]])}),
        st.fixed_dictionaries({"lattice": st.sampled_from(["D2Q9", "D3Q19", "D1Q3"])}),
        st.fixed_dictionaries({"inlet_face": st.sampled_from(["x_min", "y_min", "q"])}),
        st.fixed_dictionaries({"wall_mask": st.sampled_from(
            [None, np.zeros((nx, ny), bool).tolist()])}),
    )


def _pipe_structural(kw):
    return st.one_of(
        st.fixed_dictionaries({"nx": st.sampled_from([6, 7, 100_000])}),
        st.fixed_dictionaries({"propeller_x": st.sampled_from([2, 3, 99])}),
    )


def _pipe_cross(kw):
    return st.fixed_dictionaries({
        "rho_liquid": floats32(0.5, 3.0),
        "rho_gas": floats32(0.05, 3.0),
    }) | st.fixed_dictionaries({"G": st.sampled_from([0.0, -4.5, -5.0, 1.0])})


def _rigid_structural(kw):
    return st.fixed_dictionaries({"constraints": st.sampled_from(
        [{}, {"z": 0.0}, {"x": 1.0}, {"rx": 0.0, "ry": 0.0}])})


def _health_structural(kw):
    return st.fixed_dictionaries({"checks": st.sampled_from([
        {}, {"x": {"finite": True}}, {"x": {"min": 0.0, "max": 1.0}},
        {"x": {"finite": True, "min": -1.0}}, "not a dict",
    ])})


def _spring_cross(kw):
    return st.fixed_dictionaries({
        "stiffness": floats32(0.5, 100.0),
        "mass": floats32(-1.0, 3.0),
    })


def _heart_cross(kw):
    return st.fixed_dictionaries({
        "heart_rate": floats32(-10.0, 120.0),
        "systole_fraction": floats32(-0.5, 1.5),
    })


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

KINDS: dict[str, Kind] = {k.name: k for k in (
    Kind("ball", BallNode, 0.01, _ball_kwargs(),
         inputs=lambda kw: {"table_position": ()},
         safe={"gravity": (-20.0, 0.0), "elasticity": (0.0, 1.0)}),
    Kind("table", TableNode, 0.01,
         st.fixed_dictionaries({"position": floats32(-1.0, 1.0)}),
         safe={}),
    Kind("spring", SpringDamperNode, 0.01, _spring_kwargs(),
         inputs=lambda kw: {"anchor_position": ()},
         safe={"stiffness": (0.5, 100.0), "damping": (0.0, 5.0),
               "mass": (0.5, 10.0), "rest_length": (0.0, 3.0)},
         cross=_spring_cross),
    Kind("heart_pump", HeartPumpNode, 0.01, _heart_kwargs(),
         safe={"resistance": (0.5, 2.0), "compliance": (0.5, 2.0),
               "heart_rate": (40.0, 100.0), "stroke_volume": (30.0, 90.0),
               "venous_pressure": (-5.0, 5.0), "systole_fraction": (0.2, 0.6)},
         cross=_heart_cross),
    Kind("health_check", HealthCheckNode, 0.01, _health_kwargs(),
         structural=_health_structural),
    Kind("heat_uniform", HeatNode, 0.01, _heat_kwargs(nonuniform=False),
         inputs=_heat_inputs,
         safe={"thermal_diffusivity": (1e-4, 2e-3), "length": (1.0, 3.0)},
         structural=_heat_structural, cross=_heat_cross),
    Kind("heat_nonuniform", HeatNode, 0.01,
         _heat_kwargs(nonuniform=True, order=2),
         inputs=_heat_inputs,
         safe={"thermal_diffusivity": (1e-5, 5e-4)},
         structural=_heat_structural, cross=_heat_cross),
    Kind("rigid_body", RigidBodyNode, 0.01, _rigid_kwargs(),
         inputs=lambda kw: {"force": (3,), "torque": (3,)},
         safe={"mass": (0.5, 3.0), "inertia": (0.5, 3.0), "gravity": (-5.0, 5.0)},
         structural=_rigid_structural),
    Kind("rigid_body_2d", RigidBody2DNode, 0.01, _rigid2d_kwargs(),
         inputs=lambda kw: {"force": (2,), "torque": ()},
         safe={"mass": (0.5, 3.0), "inertia": (0.5, 3.0), "gravity": (-5.0, 5.0)}),
    Kind("lbm", LBMNode, 1.0, _lbm_kwargs(),
         safe={"viscosity": (0.03, 0.3)},
         structural=_lbm_structural, cheap=False),
    Kind("lbm_pipe", LBMPipeNode, 1.0, _pipe_kwargs(),
         safe={"tau": (0.55, 1.2), "propeller_strength": (-1e-3, 1e-3),
               "gravity": (-1e-4, 1e-4), "rho_0": (0.8, 1.2)},
         structural=_pipe_structural, cross=_pipe_cross, cheap=False),
    Kind("wavelet", WaveletAdaptiveNode, 1.0, _wavelet_kwargs(),
         safe={"theta": (0.3, 0.5), "sigma": (0.05, 0.15)}, cheap=False),
)}

CHEAP_KINDS = tuple(k for k, v in KINDS.items() if v.cheap)
COSTLY_KINDS = tuple(k for k, v in KINDS.items() if not v.cheap)
REGISTRY: dict[str, type] = {k.cls.__name__: k.cls for k in KINDS.values()}


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Write:
    """One ``PUT /graph/params/{node}`` body (or ``gm.params`` assignment)."""

    category: str
    params: dict

    def body(self) -> str:
        """The request body as JSON text, NaN and Infinity spelled out the
        way Python's encoder (and a careless client) spells them."""
        return json.dumps({"params": self.params}, allow_nan=True)


def _bounded(spec) -> tuple[Optional[float], Optional[float], bool]:
    lo, hi = spec.bounds
    strict = spec.transform in ("log", "logit")
    if lo is None and spec.transform == "log":
        lo = 0.0
    return lo, hi, strict


def _as_shape(value: float, like: Any) -> Any:
    shape = np.shape(like)
    return value if shape == () else np.full(shape, value).tolist()


@st.composite
def writes(draw, kind: Kind, kwargs: dict, *,
           categories: Optional[tuple[str, ...]] = None) -> Write:
    """One proposed write to a node of ``kind`` built with ``kwargs``.

    ``categories`` narrows the draw (default: every category the node has).
    """
    node = kind.build_node(kwargs)
    pytree = {k: np.asarray(v) for k, v in node.params_pytree().items()}
    specs = node.param_specs()
    leaves = sorted(pytree)
    safe = {k: r for k, r in kind.safe.items() if k in pytree}
    initial = sorted(k for k in node.params if k.startswith("initial_") and k in pytree)

    options: dict[str, Callable[[], dict]] = {}
    if safe:
        def valid():
            key = draw(st.sampled_from(sorted(safe)))
            lo, hi = safe[key]
            shape = np.shape(pytree[key])
            if shape == ():
                return {key: draw(floats32(lo, hi))}
            return {key: draw(vec32(lo, hi, int(np.prod(shape))))}
        options["valid"] = valid

        def multi():
            keys = draw(st.lists(st.sampled_from(sorted(safe)), min_size=2,
                                 max_size=min(3, len(safe)), unique=True)) \
                if len(safe) >= 2 else sorted(safe)
            out = {}
            for key in keys:
                lo, hi = safe[key]
                v = draw(floats32(lo, hi))
                out[key] = _as_shape(v, pytree[key])
            return out
        options["multi"] = multi

    bounded = [k for k in leaves if k in specs and any(
        b is not None for b in _bounded(specs[k])[:2])]
    if bounded:
        def boundary():
            key = draw(st.sampled_from(bounded))
            lo, hi, _ = _bounded(specs[key])
            v = draw(st.sampled_from([b for b in (lo, hi) if b is not None]))
            return {key: _as_shape(v, pytree[key])}
        options["boundary"] = boundary

        def invalid():
            key = draw(st.sampled_from(bounded))
            lo, hi, _ = _bounded(specs[key])
            picks = []
            if lo is not None:
                picks += [lo - 1.0, lo - 1e-3]
            if hi is not None:
                picks += [hi + 1e-3, hi + 1.0]
            return {key: _as_shape(draw(st.sampled_from(picks)), pytree[key])}
        options["invalid"] = invalid

    any_key = sorted(set(leaves) | {k for k, v in node.params.items()
                                    if isinstance(v, (int, float)) and not isinstance(v, bool)})
    if any_key:
        def non_finite():
            key = draw(st.sampled_from(any_key))
            v = draw(st.sampled_from([math.nan, math.inf, -math.inf]))
            like = pytree.get(key, 0.0)
            if np.shape(like) != () and draw(st.booleans()):
                out = np.asarray(like, np.float64).copy()
                out.flat[draw(st.integers(0, out.size - 1))] = v
                return {key: out.tolist()}
            return {key: _as_shape(v, like)}
        options["non_finite"] = non_finite

    float_leaves = [k for k in leaves if pytree[k].dtype == np.float32]
    if float_leaves:
        def oversized():
            key = draw(st.sampled_from(float_leaves))
            v = draw(st.sampled_from([1e39, -1e39, 3.5e38, 1e300]))
            return {key: _as_shape(v, pytree[key])}
        options["oversized"] = oversized

        def wrong_type():
            key = draw(st.sampled_from(float_leaves))
            shape = np.shape(pytree[key])
            bad = draw(st.sampled_from(
                ["a string", True, None, {"x": 1.0}]
                + ([[1.0, 2.0]] if shape == () else
                   [[1.0] * (int(np.prod(shape)) + 1), 1.0 if len(shape) else []])))
            return {key: bad}
        options["wrong_type"] = wrong_type

    options["unknown"] = lambda: {draw(st.sampled_from(
        ["no_such_param", "Gravity", "params", ""])): 1.0}

    if initial:
        def initial_write():
            key = draw(st.sampled_from(initial))
            like = pytree[key]
            if np.shape(like) == ():
                return {key: draw(floats32(-1.0, 1.0))}
            return {key: draw(vec32(-1.0, 1.0, int(np.size(like))))}
        options["initial"] = initial_write

    if safe and bounded:
        def mixed():
            # A valid key and a refused one in one request, in either order:
            # "refused whole" means the valid one is not written either.
            ok = options["valid"]()
            bad = options["invalid"]()
            if set(ok) & set(bad):
                return bad
            pairs = [*ok.items(), *bad.items()]
            if draw(st.booleans()):
                pairs.reverse()
            return dict(pairs)
        options["mixed"] = mixed

    if kind.structural is not None:
        options["structural"] = lambda: draw(kind.structural(kwargs))
    if kind.cross is not None:
        options["cross"] = lambda: draw(kind.cross(kwargs))

    names = sorted(options) if categories is None else [
        c for c in categories if c in options]
    if not names:
        names = ["unknown"]
    category = draw(st.sampled_from(names))
    return Write(category, options[category]())
