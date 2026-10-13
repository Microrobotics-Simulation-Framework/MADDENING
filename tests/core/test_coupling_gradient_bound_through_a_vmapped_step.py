"""``gradient_relative_error_bound`` read through a vmapped step, on every push.

``tests/core/test_coupling_gradient_bound_under_vmap.py`` holds the bound a
vmapped call of a compiled multi-rate group reports to the unbatched step's
and to the true error, and is slow-marked for compiling the spectral and
gradient-bound machinery twice (17 s on three cores).  Without this module
no push would read the bound through ``jax.vmap`` at all.

This is one compile: ``jax.vmap`` of the compiled step of the two-node
group ``u <- a + g u**2`` (``_Curved`` "square" and its relay, from
``test_coupling_gradient_error_bound.py``), over two initial states, capped
at four passes so each stops at its own distance from the fixed point.  The
map's fixed point and its gradient are closed-form, so for each member of
the batch:

* the returned iterate is the float64 iteration of the map from that
  member's start (the forward is batched correctly);
* the bound is finite and differs from the other member's (it is the
  member's own, not one member's broadcast);
* it bounds the true relative error of ``d x / d a`` at that member's
  iterate, ``|t_k - t*| / |t_k|`` with ``t = 1 / (1 - 2 g x)``, and sits
  within the recorded ratio of it (``_RECORDED_RATIO``, the same band as
  the unbatched sweep).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from tests.core.test_coupling_gradient_error_bound import (
    _BAND, _CURVED, _RECORDED_RATIO, _Curved, _Relay, _analytic_curved)

KEY = "a+b"
BOUND = f"coupling_{KEY}_gradient_relative_error_bound"
PASSES = 4
#: Two starts that stop at visibly different distances from the fixed point.
STARTS = (0.0, 0.5)


def test_the_gradient_bound_holds_for_each_member_of_a_vmapped_batch():
    a, g = _CURVED["square"]
    gm = GraphManager()
    gm.add_node(_Curved("a", "square", a, g))
    gm.add_node(_Relay("b"))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], diagnostics=True, max_iterations=PASSES,
                          tolerance=1e-7)
    gm.compile()
    state0 = gm._state  # noqa: SLF001 -- the step the graph compiled, batched
    batch = jax.tree.map(lambda leaf: jnp.stack([leaf, leaf]), state0)
    for node in ("a", "b"):
        batch[node]["x"] = jnp.asarray(STARTS, jnp.float32)
    step = gm._build_step_fn()  # noqa: SLF001
    out = jax.jit(jax.vmap(step, in_axes=(0, None, None)))(
        batch, gm._default_external_inputs(), gm.params)  # noqa: SLF001

    u_star, exact = _analytic_curved("square")
    bounds = np.asarray(out["_meta"][BOUND], np.float64)
    iterations = np.asarray(out["_meta"][f"coupling_{KEY}_iterations"])
    assert bounds.shape == (2,) and np.all(np.isfinite(bounds)), bounds
    assert bounds[0] != bounds[1], "one member's bound broadcast to the batch"
    for i, start in enumerate(STARTS):
        x = float(out["a"]["x"][i])
        want = start
        for _ in range(PASSES):
            want = a + g * want * want
        assert x == np.float32(want) or abs(x - want) <= 4e-7 * abs(want), (i, x, want)
        # Premise: the forward stopped early, visibly short of the fixed point.
        assert int(iterations[i]) == PASSES and abs(x - u_star) / u_star > 1e-2, (i, x)
        t_k = 1.0 / (1.0 - 2.0 * g * x)
        true = abs(t_k - exact["a"]) / abs(t_k)
        ratio = bounds[i] / true
        rec = _RECORDED_RATIO[("square", "a")]
        assert max(1.0, rec / _BAND) <= ratio <= rec * _BAND, (
            f"member {i} (start {start}): bound {bounds[i]:.3e}, true relative error of "
            f"d x/d a {true:.3e}, ratio {ratio:.3f}; recorded ~{rec} (band x{_BAND})")
