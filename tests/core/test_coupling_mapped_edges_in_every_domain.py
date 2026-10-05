"""The interface norm and its bound on mapped edges, in every numeric domain.

Under ``convergence_norm="interface"`` a coupling group measures what its
internal edges *deliver*: each edge's source field through the edge's
interface mapping, with the weights the step ran with, and then its
transform (CPL-041).  The spectral bound is taken in that reading (CPL-088).
Those rows' own tests run float32 pairs; here both claims are stated once,
on a pair whose two internal edges each carry a mapping, and run in every
domain of :mod:`tests.core.coupling_domains` -- float64, a float32 member
beside a float64 one under x64 (where a float64 matrix applied to a float32
field delivers float64 numbers that carry float32 rounding), bfloat16 and
float16, a ``jax.vmap`` of the step, a multi-rate graph, a sub-cycled
group, a predictor, a checkpoint restart, (slow) ``run_adaptive`` and, from
``tests/cloud/multigpu``, a member sharded over four devices.

**The fixture.**  ``x_a <- g_a (H_ba x_b) + c_a``, ``x_b <- g_b (H_ab x_a)
+ c_b`` on two entries a member, memoryless, so every domain solves the
same fixed point.  Each member's first entry is a constant 32 the mapping
on its outgoing edge weights by a few thousandths, so what an edge delivers
is of order one and moves with its source's second entry, while the
source field's magnitude is the 32: read on the source fields, the same
change is an order of magnitude smaller -- more than every dtype here
resolves, bfloat16's eight bits included.  The mapping *objects* hold a decoy matrix
and the real matrices reach each step through ``params["mappings"]``, as a
fitted or a per-step weight does: a reading taken with a mapping's own
weights is a reading of another edge.

**The oracles** are in float64 NumPy from the parameters and matrices as
each dtype stores them, and call nothing of the library's reading:

* CPL-041: a group stopped at its cap unconverged reports the residual of
  the state it returns, ``||F(x) - x||`` with ``F`` one Gauss-Seidel pass
  -- here the RMS, over every entry the two edges deliver, of the change
  of the delivered value over ``rtol`` times its own largest magnitude;
* CPL-088: where ``spectral_usable`` is True, ``spectral_error_bound`` is
  at least the distance to the exact fixed point in the same norm at the
  returned state.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tests.core import coupling_domains as cd

#: Every domain, ``run_adaptive`` slow (it compiles its step on every call).
EVERY = list(cd.EVERY) + [pytest.param(cd.ADAPTIVE, marks=pytest.mark.slow)]

#: The matrices the steps run with: ``b -> a`` and ``a -> b``.  Neither is a
#: selection, a scale or symmetric, and the two differ.  Every number here
#: is exact in bfloat16.
H_BA = np.array([[2.0 ** -8, 1.0], [-(2.0 ** -8), 0.875]])
H_AB = np.array([[2.0 ** -8, 0.5], [2.0 ** -7, 0.875]])
#: What the mapping objects hold instead.
DECOY = np.array([[0.125, 0.25], [0.5, -0.125]])
#: A member's first entry has no gain: it stays at its forcing, 32.  The
#: second entries close the loop at a gain of 0.875 ** 2.
GAINS = ((0.0, 1.0), (0.0, 1.0))
FORCING = ((32.0, -0.5), (32.0, 0.75))

_GRAPHS: dict = {}


def _sixteen(domain) -> bool:
    return jnp.dtype(domain.coarsest).itemsize == 2


def _rtol(domain) -> float:
    """A tolerance the dtype resolves, far below the pair's distance at the cap."""
    return 0.02 if _sixteen(domain) else 1e-5


def _graph(label):
    """The mapped pair in *label*'s domain: two passes with the report's
    analysis on, compiled once per module."""
    if label not in _GRAPHS:
        d = cd.DOMAINS[label]
        with cd.entered(d):
            _GRAPHS[label] = cd.pair(
                d, n=2, g=GAINS, c=FORCING, mappings=(DECOY, DECOY),
                convergence_norm="interface", rtol=_rtol(d), max_iterations=2,
                diagnostics=True)
    return _GRAPHS[label]


def _scenarios(domain, gm) -> list:
    """Steps' parameters: the real matrices, and the forcing moving from
    step to step where the domain runs a sequence."""
    moves = (0.0, 0.25, -0.125, 0.5)
    seq = [cd.params_with(gm, g=GAINS, mappings=(H_BA, H_AB),
                          c=tuple(np.asarray(c) * (1.0 + m) for c in FORCING)) for m in moves]
    return [seq] if (domain.predictor or domain.restart) else [seq[0]]


def _solves(label) -> list:
    d = cd.DOMAINS[label]
    gm = _graph(label)
    with cd.entered(d):
        out = []
        for scenario in _scenarios(d, gm):
            if isinstance(scenario, list):
                out.extend(cd.run_sequence(d, gm, scenario))
            else:
                out.extend(cd.run(d, gm, [scenario]))
        cd.assert_in_domain(d, gm, out)
    return out


def _delivered(s, state) -> list:
    """What the two edges deliver at *state* (``{"a": x_a, "b": x_b}``), float64."""
    h_ba, h_ab = s.mapping_matrices()
    return [h_ba @ state["b"], h_ab @ state["a"]]


def _one_pass(gm, s, x) -> dict:
    """One Gauss-Seidel pass of the pair from *x*, in the group's sweep order."""
    ga, gb = (np.asarray(g, np.float64) for g in s.gains())
    ca, cb = s.forcing()
    h_ba, h_ab = s.mapping_matrices()
    cur = dict(x)
    for name in [n for n in gm.schedule if n in ("a", "b")]:
        if name == "a":
            cur["a"] = ga * (h_ba @ cur["b"]) + ca
        else:
            cur["b"] = gb * (h_ab @ cur["a"]) + cb
    return cur


def _interface_norm(new: list, old: list, rtol, weights_from_new=False):
    """``(norm, conditioning)``: the RMS over every delivered entry of its
    change over ``rtol`` times the field's largest magnitude (over the pair,
    or at *new* alone), and the largest ``magnitude / change`` of a field --
    how many times a relative rounding of the values grows in the norm."""
    terms, conditioning = [], 1.0
    for a, b in zip(new, old):
        ref = np.max(np.abs(a)) if weights_from_new else max(np.max(np.abs(a)),
                                                             np.max(np.abs(b)))
        terms.append(np.abs(a - b) / (rtol * ref))
        conditioning = max(conditioning, float(ref / np.max(np.abs(a - b))))
    return float(np.sqrt(np.mean(np.concatenate(terms) ** 2))), conditioning


def _state(s) -> dict:
    return {n: np.asarray(s.x(n), np.float64) for n in ("a", "b")}


# Per push: tests/core/test_coupling_mapped_edges_in_every_domain.py::test_the_interface_norm_measures_what_mapped_edges_deliver
# (every domain but run_adaptive, the same check)
@pytest.mark.parametrize("label", EVERY)
def test_the_interface_norm_measures_what_mapped_edges_deliver(label):
    """CPL-041 with a mapping on each internal edge.

    At its cap of two passes the group is far from converged, so the
    residual it reports is that of the state it returns: the change, over
    one more pass, of what the two edges deliver with the matrices the step
    ran with.  Read on the source fields -- or with the mapping objects' own
    weights -- the same state gives another number.
    """
    d = cd.DOMAINS[label]
    gm = _graph(label)
    eps = float(cd.finfo(d.coarsest).eps)
    for s in _solves(label):
        r = s.report
        x = _state(s)
        fx = _one_pass(gm, s, x)
        want, conditioning = _interface_norm(_delivered(s, fx), _delivered(s, x), _rtol(d))
        # run_adaptive's controller repeats the capped solve on shrinking
        # steps, so its state nears the fixed point; the residual is checked
        # all the same.
        assert not r["converged"] or d.adaptive, (label, r)
        # The one more pass and the delivered values are evaluated in the
        # members' dtype: a few eps of each value, ``conditioning`` times
        # that in its change.
        tolerance = 8 * eps * conditioning
        assert abs(r["residual"] - want) <= tolerance * want, (
            f"{label}: reported {r['residual']:.6e}, the delivered values moved {want:.6e} "
            f"(tolerance {tolerance:.2e})")
        # The fixture premise: this dtype tells the two readings apart (the
        # source fields' reading is outside the band the residual is held to).
        raw, _ = _interface_norm([fx["b"], fx["a"]], [x["b"], x["a"]], _rtol(d))
        assert tolerance < 0.5 and abs(raw - want) > tolerance * want or d.adaptive, (
            f"{label}: the source fields read {raw:.4e}, within the tolerance of {want:.4e}")


# Per push: tests/core/test_coupling_mapped_edges_in_every_domain.py::test_a_usable_bound_covers_the_distance_in_what_mapped_edges_deliver
# (every domain but run_adaptive, the same check)
@pytest.mark.parametrize("label", EVERY)
def test_a_usable_bound_covers_the_distance_in_what_mapped_edges_deliver(label):
    """CPL-088 with a mapping on each internal edge.

    The spectrum is taken on the reading the residual is in, so the bound
    covers the distance to the exact fixed point measured in what the edges
    deliver, each over its own magnitude at the returned state.
    """
    d = cd.DOMAINS[label]
    for s in _solves(label):
        r = s.report
        x = _state(s)
        xa_star, xb_star = cd.mapped_fixed_point(s)
        dist, _ = _interface_norm(_delivered(s, x), _delivered(s, {"a": xa_star, "b": xb_star}),
                                  _rtol(d), weights_from_new=True)
        assert dist > 1.0 or d.adaptive, (
            f"{label}: fixture premise: outside the tolerance at the cap ({dist})")
        assert r["spectral_usable"] is True, (label, r)
        assert r["spectral_error_bound"] >= dist, (
            f"{label}: bound {r['spectral_error_bound']:.4e} under the distance {dist:.4e} "
            f"in the delivered values ({r})")
