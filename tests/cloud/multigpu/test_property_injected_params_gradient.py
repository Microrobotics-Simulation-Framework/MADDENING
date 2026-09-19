"""An injected parameter reaches the inner node, and so does its gradient.

``ShardedStencilNode`` probes its inner ``update_padded`` once, at
construction, for three keyword arguments.  Two of the probes accepted a
``**kwargs`` signature and the third, ``params``, did not, so a node
written as ``update_padded(self, state_padded, boundary_inputs, dt,
**kwargs)`` received ``static_padded`` and ``shard_info`` but never
``params``.  It silently fell back to its constructor constant: the
physics was wrong by the difference between the two values, and
``d(loss)/d(rate)`` came back **exactly zero**, because the injected leaf
never entered the trace at all.  No shipped node uses a var-keyword
``update_padded`` -- the trap was for user-authored nodes, and it was
silent.

The invariant is the same for every signature the wrapper accepts: the
sharded gradient is the unsharded gradient.  Stated as a property over
the signature styles and the device counts so a fourth probe, or a
fourth signature style, is covered by adding an entry.

From the cloud/sharding audit of 2026-09-19
(``benchmarks/results/audit_040_final/cloud-sharding/``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from tests.cloud.multigpu.property_support import (
    StencilDiffusion1D,
    cells_for,
    device_counts,
    param_values,
)
from tests.conftest import EXAMPLES_COSTLY


class _VarKeywordDiffusion(StencilDiffusion1D):
    """The same physics behind ``update_padded(..., **kwargs)``.

    A signature the wrapper visibly supports for its other two injected
    keywords, so a node author has every reason to expect it to work for
    ``params`` too.
    """

    def update_padded(self, state_padded, boundary_inputs, dt, **kwargs):
        return super().update_padded(state_padded, boundary_inputs, dt, **kwargs)


#: Label -> inner-node class, one per ``update_padded`` signature style.
SIGNATURE_STYLES = {
    "explicit_keywords": StencilDiffusion1D,
    "var_keyword": _VarKeywordDiffusion,
}


def _case(style: str, n_devices: int, n_cells: int):
    inner = SIGNATURE_STYLES[style](name="diff", n_cells=n_cells)
    wrapped = ShardedStencilNode(
        SIGNATURE_STYLES[style](name="diff", n_cells=n_cells),
        create_device_mesh(shape=(n_devices,)),
        axis_map={"devices": 0}, boundary="periodic",
    )
    boundary_inputs = {
        "source": jnp.asarray(
            np.cos(np.linspace(0.0, 4 * np.pi, n_cells, endpoint=False),
                   dtype=np.float32)),
        "gain": jnp.float32(0.75),
    }
    return inner, wrapped, boundary_inputs


def _loss_fn(node, boundary_inputs, steps: int):
    """``sum(f**2)`` after *steps* updates with ``params={"rate": r}``."""
    def loss(rate):
        state = node.initial_state()
        for _ in range(steps):
            state = node.update(state, boundary_inputs, node.delta_t,
                                params={"rate": rate})
        return jnp.sum(state["f"] ** 2)
    return loss


@given(style=st.sampled_from(sorted(SIGNATURE_STYLES)),
       n_devices=device_counts(), rate=param_values())
@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
def test_the_gradient_of_an_injected_parameter_survives_the_sharded_wrapper(
        style, n_devices, rate):
    """d(loss)/d(rate) through the wrapper equals the unsharded adjoint.

    And is not zero: a dropped ``params`` kwarg does not raise, it
    detaches the leaf from the trace, and the only visible symptom is a
    gradient of exactly 0.0.
    """
    n_cells = cells_for(n_devices, at_least=16)
    inner, wrapped, boundary_inputs = _case(style, n_devices, n_cells)
    rate = jnp.float32(rate)

    reference = jax.grad(_loss_fn(inner, boundary_inputs, 2))(rate)
    sharded = jax.grad(_loss_fn(wrapped, boundary_inputs, 2))(rate)

    assert float(reference) != 0.0, "the reference gradient is degenerate"
    assert float(sharded) != 0.0, (
        f"{style}: the sharded gradient is exactly zero -- the injected "
        "parameter never reached the inner node")
    np.testing.assert_allclose(np.asarray(sharded), np.asarray(reference),
                               rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("style", sorted(SIGNATURE_STYLES))
def test_an_injected_parameter_changes_the_sharded_trajectory(style):
    """The value itself lands, not only its derivative.

    The physics ran on the constructor constant, which is a wrong answer
    rather than a missing one -- worth its own assertion, since a
    gradient test alone would pass on a node that happened to be
    constructed with the injected value.
    """
    n_devices = min(2, len(jax.devices()))   # 1 on the single-device policy run
    n_cells = cells_for(n_devices, at_least=16)
    inner, wrapped, boundary_inputs = _case(style, n_devices, n_cells)
    injected = {"rate": jnp.float32(0.9)}

    def step(node, params):
        state = node.initial_state()
        return np.asarray(jax.device_get(
            node.update(state, boundary_inputs, node.delta_t,
                        params=params)["f"]))

    np.testing.assert_allclose(step(wrapped, injected), step(inner, injected),
                               rtol=1e-6, atol=1e-7)
    assert not np.allclose(step(wrapped, injected), step(wrapped, None)), (
        f"{style}: injecting a parameter changed nothing")
