"""``convergence_norm="interface"`` on a mapped edge: the value the edge delivers.

The step applies an edge's interface mapping and then its transform
(``_apply_edge``).  ``coupling_residual_interface`` and
``residual_precision_floor`` read the edge's *source field* through the
transform alone, so on a mapped edge inside a coupling group the norm is
taken on the source field's scale, not on the delivered value's
(MADD-ANO-177, open; mapped edges are new in 0.4.0, so no release carried
it).  The unmapped case is the control and passes.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import (
    coupling_residual_interface,
    residual_precision_floor,
)
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.edge import EdgeSpec

F32 = jnp.float32
RTOL = 1e-2
#: The mapping delivers the field's second entry only.
SELECT = np.array([[0.0, 1.0]], np.float32)
NEW = {"a": {"x": jnp.asarray([1000.0, 1.1], F32)}, "d": {"x": jnp.asarray([1.1], F32)}}
OLD = {"a": {"x": jnp.asarray([1000.0, 1.0], F32)}, "d": {"x": jnp.asarray([1.0], F32)}}


def _delivered_norm() -> float:
    """The norm of an unmapped edge that carries the delivered value itself."""
    return float(coupling_residual_interface(
        NEW, OLD, [EdgeSpec("d", "b", "x", "u")], atol=0.0, rtol=RTOL))


def test_the_norm_of_an_unmapped_edge_is_the_scaled_change_of_what_it_delivers():
    """0.1 of 1.1, in units of rtol: the control for the mapped case below."""
    assert _delivered_norm() == pytest.approx(0.1 / (RTOL * 1.1), rel=1e-5)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "MADD-ANO-177: the interface norm and the precision floor read a mapped edge's "
    "source field through the transform, without its interface mapping, so they "
    "measure on the source field's scale and not what the edge delivers; open"))
def test_the_norm_of_a_mapped_edge_is_the_scaled_change_of_what_it_delivers():
    """The mapped edge delivers exactly what edge ``d -> b`` delivers, so the
    two norms, and the two floors, must be equal.  Today the mapped one is
    measured against the source field's 1000: 0.0071 where it is 9.09."""
    mapped = [EdgeSpec("a", "b", "x", "u", mapping=matrix_mapping(SELECT))]
    direct = [EdgeSpec("d", "b", "x", "u")]
    got = float(coupling_residual_interface(NEW, OLD, mapped, atol=0.0, rtol=RTOL))
    assert got == pytest.approx(_delivered_norm(), rel=1e-5)
    floors = [float(residual_precision_floor(NEW, ["a", "d"], "interface", 0.0, RTOL, edges))
              for edges in (mapped, direct)]
    assert floors[0] == pytest.approx(floors[1], rel=1e-5)
