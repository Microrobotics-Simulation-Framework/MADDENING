"""A numerical float64 reference for one coupling group, with no closed form.

``test_coupling_targeted_search.py`` scores ``coupling_diagnostics()``
against :class:`~tests.property.coupled_topologies.LinearModel`: the fixed
point and the pass's Jacobian of a group of *linear* relays, in closed
form.  A group with a nonlinear node, a dtype-widening transform or a
mapping that reads a moving geometry has no closed form.  This module
computes the same answers numerically, for any small group (up to a few
dozen scalars) built from any nodes and edges.

**How the pass map is obtained** (:class:`PassReference`).  The caller
builds an *x64 twin* of the graph under test -- the same nodes, edges,
schedule and sub-cycling, every float in float64 -- whose group runs
``max_iterations=1`` with no acceleration and ``predictor="linear"``
(:func:`twin_knobs`).  One step of that twin is one coupling pass and is
differentiated straight through (there is no fixed point to apply the
implicit function theorem at).  The pass starts from the predictor's
extrapolation ``2 pred_0 - pred_1`` of the last two results, which the
step keeps in three ``_meta`` slots of the state; writing the iterate
``x`` into both and ``2`` into the count makes the step compute

    ``P(x; pre) = one pass of the group from the iterate x, each member
    integrating from the pre-step state pre``

with *pre* the members' fields of the state handed in.  The map is the
graph's own compiled step (``gm._raw_step_fn``, the callable ``step()``
runs) called on a state: nothing of the estimator is called, and the only
thing written that ``set_node_state`` could not write is those three
slots.  Their layout is not assumed either: :meth:`PassReference.of`
steps the twin once and finds the order of nodes whose flattened floating
fields equal the stored ``pred_0`` bit for bit.

From ``P``, in float64 (``jax.jacfwd`` for every derivative, so every
dependence the pass has on the iterate is in it: transforms, mappings,
and a geometry field where a member holds one):

* :meth:`PassReference.fixed_point`: Picard passes, then Newton on
  ``P(x) - x`` with the dense Jacobian, to a residual of a few float64
  ``eps`` of the fields; the residual history is returned as the evidence;
* :meth:`PassReference.jacobian`: the dense ``dP/dx`` at any iterate, and
  :func:`radius` of it;
* :meth:`PassReference.distance`: the distance from a returned iterate
  to the fixed point in the norm ``spectral_error_bound`` documents (each
  field of the norm's reading over its own largest magnitude at the
  returned state; a root mean square under the relative norms);
* :meth:`PassReference.gradient_error`: the relative error of the
  implicit derivative ``(I - dP/dx)^{-1} dP/dc`` taken at the returned
  iterate against the same dense solve at the fixed point, for every
  scalar constant ``c`` (the form ``gradient_relative_error_bound`` is
  read in) or for a scalar loss ``w . x``;
* :meth:`PassReference.residual`: the exact residual ``P(x) - x`` of a
  returned state in the group's norm;
* :meth:`PassReference.finite_difference_gap`: a Jacobian-vector product
  (the pass's, or a reading's) against a central difference;
* :meth:`PassReference.nonlinearity`: ``||(I - J(x))^{-1} (J_mean -
  J(x))||`` with ``J_mean`` the mean Jacobian over the segment from the
  fixed point to ``x`` -- the exact factor by which a linear bound taken
  at ``x`` can miss on a nonlinear map (see the method).

**What it catches.**  Anything by which the estimator's numbers disagree
with the map the solve iterates: a term of the pass missing from the
estimator's Jacobian (a mapping, a transform, a geometry dependence), a
reading taken at another time level than the pass takes it, a Jacobian
taken at another state than the returned one, a bound in another norm
than the one documented, a float32 analysis that lost the quantity.

**What it does not catch.**

* A defect in the pass itself (the solve iterating a wrong map): the
  reference inherits it.  The closed-form and time-level references are
  what hold the pass.
* A difference between the single-pass branch (``max_iterations=1``) and
  the pass an iterating group runs.  :func:`passes_compose` checks that on
  a twin run for several passes; the search's validation runs it.
* A difference between the float32 graph's pass and its float64 twin's
  beyond rounding (a node that branches on its dtype).
* A graph with more than one coupling group (the others would run their
  own single pass), or a group whose members hold non-float state the
  pass map depends on.

**Cost.**  Per twin: the twin's own build, and two compiles (the pass
with its Jacobian, one program; the constants' sensitivities, only where
a gradient is scored), a second or two each for a group of a dozen
scalars.  Per example: a few dozen jitted calls, a few milliseconds.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import math
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

#: The float64 rounding of one evaluation.
EPS64 = float(np.finfo(np.float64).eps)
#: A fixed point is accepted where the pass moves no entry by more than
#: this many float64 ``eps`` of its field's magnitude (one pass of a few
#: dozen operations rounds at a handful; a resolvent of 1e3 leaves the
#: Newton iterate that far from a stationary float).
FIXED_POINT_ULPS = 2.0 ** 12


@contextlib.contextmanager
def x64():
    """Run the body under ``jax_enable_x64``, restoring the setting."""
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


def twin_knobs(knobs: dict) -> dict:
    """The x64 twin's group configuration for a group under *knobs*: the
    schedule and the sub-cycling of the group under test (what decides the
    pass), one pass, and the predictor the iterate is handed in through."""
    keep = ("iteration_mode", "subcycling", "boundary_interpolation")
    return {**{k: knobs[k] for k in keep if k in knobs},
            "max_iterations": 1, "predictor": "linear"}


def radius(J: np.ndarray) -> float:
    """The spectral radius of *J*."""
    J = np.asarray(J, np.float64)
    return float(np.max(np.abs(np.linalg.eigvals(J)))) if J.size else 0.0


@dataclasses.dataclass(frozen=True)
class Norm:
    """The norm a report states its bound in.

    ``fields(x)`` returns the norm's reading of a flat iterate as a list
    of arrays (JAX or NumPy): a member's field under ``"l2"`` and
    ``"mixed"``; under ``"interface"`` one field per internal edge, on the
    side the norm reads it (the caller states it, by
    ``coupled_topologies.interface_side_of``: this module knows no edge).
    Each is weighted by ``1 / (rtol max|field|)`` at the returned state
    (``rtol`` 1 under ``"l2"``); ``rms`` divides the sum of squares by the
    number of entries.
    """

    fields: Callable
    rtol: float = 1.0
    rms: bool = False

    def weights(self, x_returned, also=None) -> list:
        """One weight per field; *also* is a second iterate whose magnitude
        counts too (the residual's own weights read both)."""
        out = []
        second = self.fields(also) if also is not None else None
        for k, f in enumerate(self.fields(x_returned)):
            ref = float(np.max(np.abs(np.asarray(f)))) if np.size(f) else 0.0
            if second is not None and np.size(second[k]):
                ref = max(ref, float(np.max(np.abs(np.asarray(second[k])))))
            out.append(1.0 / (self.rtol * ref) if ref > 0 else 0.0)
        return out

    def of_difference(self, a, b, weights) -> float:
        """``||w (fields(a) - fields(b))||``, a root mean square under ``rms``."""
        total, count = 0.0, 0
        for w, fa, fb in zip(weights, self.fields(a), self.fields(b)):
            d = np.asarray(fa, np.float64) - np.asarray(fb, np.float64)
            total += float(np.sum((w * d) ** 2))
            count += d.size
        return math.sqrt(total / count if self.rms and count else total)


@dataclasses.dataclass(frozen=True)
class FixedPoint:
    """A fixed point and the evidence it is one."""

    x: np.ndarray
    #: ``max |P(x) - x|`` over each field's magnitude, in float64 ``eps``.
    ulps: float
    converged: bool
    #: The same measure after every Picard pass and Newton step taken.
    history: tuple
    picard: int
    newton: int


class PassReference:
    """The one-pass map of an x64 twin's coupling group and what follows from it.

    Build with :meth:`of`; :meth:`at` binds a pre-step state and the
    constants (both ride as arguments of the jitted map, so one twin
    serves every draw of its cell).
    """

    def __init__(self, gm, key, layout, constants_of):
        self.gm = gm
        self.key = key
        #: ``((node, field, shape, start, stop), ...)`` of the flat iterate.
        self.layout = layout
        self.size = layout[-1][4] if layout else 0
        self._constants_of = constants_of
        self._slots = tuple(f"coupling_{key}_pred_{s}" for s in ("0", "1", "count"))
        self._ext = gm._default_external_inputs()        # noqa: SLF001
        self._step = gm._raw_step_fn                     # noqa: SLF001
        self._jit_both = jax.jit(lambda x, pre, params: (
            self._pass(x, pre, params), jax.jacfwd(self._pass)(x, pre, params)))
        self._jit_sens = None
        self._pre = None
        self._params = None

    # -- construction -------------------------------------------------------

    @classmethod
    def of(cls, gm, key: Optional[str] = None, *, params: Optional[dict] = None,
           constants: Optional[Callable] = None) -> "PassReference":
        """The reference of compiled x64 twin *gm* (built under
        :func:`twin_knobs`) for its group *key* (its only one by default).

        The layout of the predictor's slots is found by stepping the twin
        once from the state it holds, with *params* (``gm.params`` by
        default): give it a state and constants under which no two members
        end the step with equal fields (a drawn example; not the zeros a
        fresh graph holds), or the stored vector does not say which member
        is where and this raises.

        *constants* maps ``gm.params`` to the sub-tree of constants the
        gradient is taken with respect to; by default every floating leaf
        of the members' parameters and of the mappings.
        """
        with x64():
            groups = {"+".join(sorted(g.nodes)): g for g in gm._coupling_groups}  # noqa: SLF001
            assert len(groups) == 1, (
                f"the reference handles a graph with one coupling group, not {sorted(groups)}")
            key = next(iter(groups)) if key is None else key
            group = groups[key]
            assert group.max_iterations == 1 and group.predictor == "linear", (
                "the twin's group runs one pass under the linear predictor (twin_knobs)")
            state = gm._state                            # noqa: SLF001
            slots = tuple(f"coupling_{key}_pred_{s}" for s in ("0", "1", "count"))
            assert all(s in state["_meta"] for s in slots), sorted(state["_meta"])
            ext = gm._default_external_inputs()          # noqa: SLF001
            after = gm._raw_step_fn(state, ext, gm.params if params is None else params)  # noqa: SLF001
            stored = np.asarray(after["_meta"][slots[0]])
            members = sorted(group.nodes)

            def fields_of(name):
                return [f for f in sorted(state[name])
                        if jnp.issubdtype(jnp.asarray(state[name][f]).dtype, jnp.floating)]

            orders = [tuple(members), tuple(n for n in gm.node_names if n in group.nodes),
                      tuple(group.nodes)]
            if len(members) <= 6:
                orders += list(itertools.permutations(members))
            matches = []
            for order in dict.fromkeys(orders):
                flat = np.concatenate([np.ravel(np.asarray(after[n][f]))
                                       for n in order for f in fields_of(n)])
                if flat.shape == stored.shape and np.array_equal(flat, stored):
                    matches.append(order)
            assert matches, (
                f"no order of {members} flattens the stepped state to the stored pred_0")
            assert len(matches) == 1, (
                f"the stored pred_0 does not determine the layout (it is the stepped state "
                f"under each of {matches[:4]}): give the twin a state and constants under "
                f"which the members' fields differ after a step")
            layout, at = [], 0
            for n in matches[0]:
                for f in fields_of(n):
                    shape = tuple(np.shape(state[n][f]))
                    size = int(np.prod(shape, dtype=np.int64))
                    assert jnp.asarray(state[n][f]).dtype == jnp.float64, (
                        f"{n}.{f} is {jnp.asarray(state[n][f]).dtype} in the twin, not float64")
                    layout.append((n, f, shape, at, at + size))
                    at += size

            def default_constants(params):
                def floats(tree):
                    return {k: v for k, v in tree.items()
                            if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)}
                return {"nodes": {n: floats(params["nodes"].get(n, {})) for n in members},
                        "mappings": {k: floats(v)
                                     for k, v in params.get("mappings", {}).items()}}

            return cls(gm, key, tuple(layout), constants or default_constants)

    def at(self, pre_state: dict, params: dict) -> "PassReference":
        """Bind the pre-step state (``gm._state`` after the caller set it:
        the members' fields are *pre*; outside nodes are read as the step
        reads them) and the constants.  Returns ``self``."""
        self._pre, self._params = pre_state, params
        return self

    # -- the flat iterate ---------------------------------------------------

    def flat(self, state: dict) -> np.ndarray:
        """``{node: {field: array}}`` (any float dtype) as the flat float64 iterate."""
        return np.concatenate([np.ravel(np.asarray(state[n][f], np.float64))
                               for n, f, _shape, _a, _b in self.layout])

    def field(self, x, node: str, field: str):
        """One field of flat iterate *x* (NumPy or JAX), in its shape."""
        for n, f, shape, a, b in self.layout:
            if (n, f) == (node, field):
                return x[a:b].reshape(shape)
        raise KeyError((node, field))

    def member_fields(self, x) -> list:
        """Every floating field of every member: the reading of ``"l2"``
        and ``"mixed"``."""
        return [x[a:b] for _n, _f, _shape, a, b in self.layout]

    def _flat_of(self, state):
        return jnp.concatenate([jnp.ravel(state[n][f]) for n, f, _s, _a, _b in self.layout])

    def _pass(self, x, pre, params):
        p0, p1, count = self._slots
        meta = {**pre["_meta"], p0: x, p1: x, count: jnp.asarray(2, pre["_meta"][count].dtype)}
        return self._flat_of(self._step({**pre, "_meta": meta}, self._ext, params))

    # -- the map and its derivatives ---------------------------------------

    def apply(self, x) -> np.ndarray:
        """``P(x; pre)`` in float64."""
        return self._both(x)[0]

    def jacobian(self, x) -> np.ndarray:
        """The dense ``dP/dx`` at iterate *x*."""
        return self._both(x)[1]

    def _both(self, x):
        with x64():
            P, J = self._jit_both(jnp.asarray(x, jnp.float64), self._pre, self._params)
        return np.asarray(P), np.asarray(J)

    def sensitivities(self, x) -> np.ndarray:
        """``dP/dc`` at iterate *x*: one column per scalar constant
        (:meth:`constant_names` names them)."""
        with x64():
            theta, restore = ravel_pytree(self._constants_of(self._params))
            if self._jit_sens is None:
                def moved(x_, theta_, pre, params):
                    c = restore(theta_)
                    nodes = {n: {**params["nodes"].get(n, {}), **c["nodes"].get(n, {})}
                             for n in params["nodes"]}
                    mappings = {k: {**v, **c.get("mappings", {}).get(k, {})}
                                for k, v in params.get("mappings", {}).items()}
                    return self._pass(x_, pre, {**params, "nodes": nodes, "mappings": mappings})
                self._jit_sens = jax.jit(jax.jacfwd(moved, argnums=1))
            return np.asarray(self._jit_sens(jnp.asarray(x, jnp.float64), theta, self._pre,
                                             self._params))

    def pass_responses(self, x, norm: Norm) -> np.ndarray:
        """How far one pass from iterate *x* moves, in *norm* at *x*'s
        weights, when each scalar constant is moved by its own magnitude
        (a zero entry by its constant's largest magnitude, or by 1 for an
        all-zero constant): the size ``gradient_relative_error_bound``
        probes a constant at, and the quantity it compares with the
        residual's float floor to say whether the pass resolves the
        constant at all.  One entry per column of :meth:`sensitivities`."""
        x = np.asarray(x, np.float64)
        with x64():
            theta, _restore = ravel_pytree(self._constants_of(self._params))
        theta = np.abs(np.asarray(theta, np.float64))
        names = self.constant_names()
        tops: dict = {}
        for name, value in zip(names, theta):
            leaf = name.rsplit("[", 1)[0]
            tops[leaf] = max(tops.get(leaf, 0.0), float(value))
        scale = np.asarray([value if value > 0 else (tops[name.rsplit("[", 1)[0]] or 1.0)
                            for name, value in zip(names, theta)])
        moved = self.sensitivities(x) * scale[None, :]
        weights, zero = norm.weights(x), np.zeros(self.size)
        return np.asarray([norm.of_difference(moved[:, c], zero, weights)
                           for c in range(moved.shape[1])])

    def constant_names(self) -> list:
        """``node.leaf[i]`` / ``mapping:key.leaf[i]`` per column of :meth:`sensitivities`."""
        with x64():
            tree = self._constants_of(self._params)
            _flat, restore = ravel_pytree(tree)
            index = restore(jnp.arange(_flat.size, dtype=jnp.float64))
        names = [None] * int(_flat.size)
        for kind in ("nodes", "mappings"):
            for owner, leaves in index.get(kind, {}).items():
                for leaf, where in leaves.items():
                    for local, k in enumerate(np.ravel(np.asarray(where)).astype(int)):
                        prefix = "" if kind == "nodes" else "mapping:"
                        names[k] = f"{prefix}{owner}.{leaf}[{local}]"
        return names

    # -- the fixed point ----------------------------------------------------

    def _ulps(self, x, P) -> float:
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(P))):
            return math.inf
        worst = 0.0
        for _n, _f, _shape, a, b in self.layout:
            scale = max(float(np.max(np.abs(x[a:b]))), float(np.max(np.abs(P[a:b]))),
                        float(np.finfo(np.float64).tiny))
            with np.errstate(over="ignore"):
                worst = max(worst, float(np.max(np.abs(P[a:b] - x[a:b])) / (EPS64 * scale)))
        return worst

    def fixed_point(self, start, *, picard: int = 12, newton: int = 40) -> FixedPoint:
        """The fixed point of ``P(.; pre)`` the iteration from *start* (the
        returned iterate) leads to: *picard* passes of the map itself,
        then Newton on ``P(x) - x`` with the dense Jacobian until the pass
        moves no entry by more than :data:`FIXED_POINT_ULPS` float64 ``eps``
        of its field, and then while it still improves."""
        x = np.asarray(start, np.float64)
        history = []
        for _ in range(picard):
            P = self.apply(x)
            history.append(self._ulps(x, P))
            if not math.isfinite(history[-1]):
                break           # the map left float range: Newton from the last finite iterate
            x = P
            if history[-1] <= 1.0:
                break
        n_picard = len(history)
        steps = 0
        best, best_ulps = x, math.inf
        eye = np.eye(self.size)
        for _ in range(newton):
            P, J = self._both(x)
            if not (np.all(np.isfinite(P)) and np.all(np.isfinite(J))):
                break
            now = self._ulps(x, P)
            history.append(now)
            if now < best_ulps:
                best, best_ulps = x, now
            elif best_ulps <= FIXED_POINT_ULPS:
                break           # no longer improving, and already a fixed point
            if now <= 1.0:
                break
            try:
                x = x + np.linalg.solve(eye - J, P - x)
            except np.linalg.LinAlgError:
                break
            steps += 1
        return FixedPoint(best, best_ulps, best_ulps <= FIXED_POINT_ULPS, tuple(history),
                          n_picard, steps)

    # -- what a report is scored against -----------------------------------

    def norm(self, kind: str, rtol: float = 1e-6, fields: Optional[Callable] = None) -> Norm:
        """The group's norm: ``"l2"`` (weights ``1 / max|field|``, a plain
        2-norm), ``"mixed"`` or ``"interface"`` (``1 / (rtol max|field|)``,
        a root mean square).  *fields* is the reading; the members'
        floating fields by default, which ``"interface"`` is not."""
        assert kind in ("l2", "mixed", "interface"), kind
        assert fields is not None or kind != "interface", (
            "the interface norm reads the internal edges, each on its side: pass fields=")
        return Norm(fields or self.member_fields, 1.0 if kind == "l2" else float(rtol),
                    kind != "l2")

    def distance(self, returned, fixed: FixedPoint, norm: Norm) -> float:
        """The distance from *returned* to the fixed point in *norm* at the
        returned state's weights."""
        x = np.asarray(returned, np.float64)
        return norm.of_difference(x, fixed.x, norm.weights(x))

    def residual(self, returned, norm: Norm) -> float:
        """``||P(x) - x||`` of the returned state in *norm*, each field over
        the larger of its magnitude before and after the pass (the weights
        the step's own residual uses)."""
        x = np.asarray(returned, np.float64)
        P = self.apply(x)
        return norm.of_difference(P, x, norm.weights(x, also=P))

    def implicit_derivatives(self, x) -> np.ndarray:
        """``(I - dP/dx)^{-1} dP/dc`` at iterate *x*: the implicit-function
        derivative of the fixed point with respect to every scalar
        constant, as taken at *x* (one column per constant)."""
        x = np.asarray(x, np.float64)
        return np.linalg.solve(np.eye(self.size) - self.jacobian(x), self.sensitivities(x))

    def gradient_errors(self, returned, fixed: FixedPoint, norm: Norm, *,
                        loss: Optional[Sequence[float]] = None) -> tuple:
        """``(miss, at_iterate, at_fixed_point)``, one entry per scalar
        constant: the size of the implicit derivative taken at the returned
        iterate, of the one at the fixed point, and of their difference.

        With *loss* (a cotangent ``w``: the loss is ``w . x``) the sizes
        are of the scalar ``d loss / d c``.  Without, of the vector ``d x /
        d c`` in *norm* (linear fields: selections of the iterate) at the
        returned state's weights, the form
        ``gradient_relative_error_bound`` documents.
        """
        x = np.asarray(returned, np.float64)
        g_k, g_star = self.implicit_derivatives(x), self.implicit_derivatives(fixed.x)
        if loss is not None:
            w = np.asarray(loss, np.float64)
            return np.abs(w @ (g_k - g_star)), np.abs(w @ g_k), np.abs(w @ g_star)
        weights = norm.weights(x)
        zero = np.zeros(self.size)
        columns = range(g_k.shape[1])
        return (np.asarray([norm.of_difference(g_k[:, c], g_star[:, c], weights) for c in columns]),
                np.asarray([norm.of_difference(g_k[:, c], zero, weights) for c in columns]),
                np.asarray([norm.of_difference(g_star[:, c], zero, weights) for c in columns]))

    def gradient_error(self, returned, fixed: FixedPoint, norm: Norm, *,
                       loss: Optional[Sequence[float]] = None,
                       columns: Optional[Sequence[bool]] = None):
        """``(worst, column)``: the worst relative error ``|g_k - g*| /
        |g_k|`` of :meth:`gradient_errors` over the scalar constants
        (those *columns* marks, by the order of :meth:`constant_names`;
        all of them by default)."""
        miss, at_iterate, _at_fixed = self.gradient_errors(returned, fixed, norm, loss=loss)
        worst, column = 0.0, None
        for c in range(len(miss)):
            if at_iterate[c] <= 0 or (columns is not None and not columns[c]):
                continue
            if miss[c] / at_iterate[c] > worst:
                worst, column = float(miss[c] / at_iterate[c]), c
        return worst, column

    def gradient_resolution(self, returned, norm: Norm) -> float:
        """What this reference cannot resolve of a relative gradient error:
        2**10 float64 ``eps`` times the resolvent's norm at the returned
        iterate and the spread of *norm*'s weights (a derivative is a
        dense solve, and a field a thousandth of another weighs a
        thousand times as much)."""
        x = np.asarray(returned, np.float64)
        weights = [w for w in norm.weights(x) if w > 0]
        spread = max(weights) / min(weights) if weights else 1.0
        resolvent = float(np.linalg.norm(np.linalg.inv(np.eye(self.size) - self.jacobian(x)), 2))
        return 2.0 ** 10 * EPS64 * spread * max(resolvent, 1.0)

    def nonlinearity(self, returned, fixed: FixedPoint, norm: Norm, *, nodes: int = 8) -> float:
        """``h = ||T (I - J(x))^{-1} (J_mean - J(x)) T^+||`` with ``T`` the
        weighted (linear) reading of *norm* and ``J_mean`` the mean of
        ``dP/dx`` over the segment from the fixed point to ``x``.

        ``x - x* = (I - J_mean)^{-1} (x - P(x))`` exactly (the mean value
        theorem in integral form), so the error a *linear* analysis at
        ``x`` computes, ``e = (I - J(x))^{-1} (x - P(x))``, misses the true
        one by ``(I - E)^{-1}`` with ``E = (I - J(x))^{-1} (J_mean -
        J(x))``: the true distance is at most ``1 / (1 - h)`` times the
        linear one where ``h < 1``.  Zero on an affine map.  *nodes*: the
        Gauss-Legendre points the mean is taken with.
        """
        x = np.asarray(returned, np.float64)
        J = self.jacobian(x)
        t, w = np.polynomial.legendre.leggauss(nodes)
        mean = sum(wk * self.jacobian(fixed.x + 0.5 * (tk + 1.0) * (x - fixed.x))
                   for tk, wk in zip(t, w)) / 2.0
        T = self.reading_matrix(norm, x)
        E = np.linalg.solve(np.eye(self.size) - J, mean - J)
        return float(np.linalg.norm(T @ E @ np.linalg.pinv(T), 2))

    def finite_difference_gap(self, x, reading: Optional[Callable] = None, *,
                              directions: int = 4, step: float = 1e-6, seed: int = 0) -> float:
        """How far a Jacobian-vector product is from a central difference,
        relative to the product: the worst over *directions* seeded
        directions of ``|J v - (f(x + h v) - f(x - h v)) / 2h| / |J v|``.

        ``f`` is the pass ``P`` and ``J`` its :meth:`jacobian`; with
        *reading* (a JAX function of a flat iterate, differentiated here by
        ``jax.jvp``) it is that reading and its own product -- the check a
        run-time self-test of a reading's Jacobian would make.  ``h`` is
        *step* times the iterate's size, so the difference is good to
        about ``step ** 2`` on a smooth map.
        """
        x = np.asarray(x, np.float64)
        rng = np.random.default_rng(seed)
        h = step * max(float(np.max(np.abs(x))), 1.0)
        J = None if reading is not None else self.jacobian(x)
        worst = 0.0
        for _ in range(directions):
            v = rng.normal(size=x.shape)
            if reading is None:
                product = J @ v
                difference = (self.apply(x + h * v) - self.apply(x - h * v)) / (2.0 * h)
            else:
                with x64():
                    product = np.asarray(jax.jvp(
                        lambda z: jnp.concatenate([jnp.ravel(f) for f in reading(z)]),
                        (jnp.asarray(x),), (jnp.asarray(v),))[1])
                    difference = (np.concatenate([np.ravel(np.asarray(f)) for f in reading(
                        jnp.asarray(x + h * v))]) - np.concatenate([
                            np.ravel(np.asarray(f)) for f in reading(jnp.asarray(x - h * v))])
                    ) / (2.0 * h)
            size = float(np.linalg.norm(product))
            if size > 0:
                worst = max(worst, float(np.linalg.norm(product - difference)) / size)
        return worst

    def reading_matrix(self, norm: Norm, returned) -> np.ndarray:
        """The weighted reading of a *linear* norm as a matrix: its rows
        applied to a flat iterate are ``w fields(x)``."""
        x = np.asarray(returned, np.float64)
        weights = norm.weights(x)
        zero = [np.asarray(f, np.float64) for f in norm.fields(np.zeros(self.size))]
        columns = []
        for k in range(self.size):
            unit = np.zeros(self.size)
            unit[k] = 1.0
            columns.append(np.concatenate([
                w * (np.ravel(np.asarray(f, np.float64)) - np.ravel(z))
                for w, f, z in zip(weights, norm.fields(unit), zero)]))
        return np.stack(columns, axis=1) if columns else np.zeros((0, 0))


def passes_compose(reference: PassReference, several, passes: int, start) -> float:
    """How far *passes* passes of an iterating twin are from *passes*
    compositions of the reference's single pass, in float64 ``eps`` of
    each field.

    *several* is a second x64 twin of the same graph whose group runs
    ``max_iterations=passes`` with no acceleration, no predictor and a
    tolerance it cannot meet, holding the same pre-step state and
    constants the reference is bound to; *start* is that state's flat
    iterate.  The single-pass branch is another branch of the step than
    the iteration (``CouplingGroup.max_iterations``): this is the check
    that both run the same pass.
    """
    with x64():
        after = several._raw_step_fn(several._state, several._default_external_inputs(),  # noqa: SLF001
                                     reference._params)                                   # noqa: SLF001
    got = reference.flat({n: {f: np.asarray(v) for f, v in after[n].items()}
                          for n, _f, _s, _a, _b in reference.layout})
    x = np.asarray(start, np.float64)
    for _ in range(passes):
        x = reference.apply(x)
    return reference._ulps(got, x)                       # noqa: SLF001
