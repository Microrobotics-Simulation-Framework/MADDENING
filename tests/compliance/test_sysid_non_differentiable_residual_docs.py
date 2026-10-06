"""The warning ``fit_lm`` gives beside a jump names the guide that explains it.

Split from ``tests/core/test_sysid_non_differentiable_residual.py``: the
assertion names a documentation path, and a docs-only pull request runs
``tests/compliance`` and no other test (see ``test_ci_workflows.py``).  The
warning is read from real fits on the bouncing ball, the same two starts of
that file's grid that end beside a jump, so what is checked is the text a
user is shown, not the constant it is built from.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: F401  (loads jax before maddening, as the sibling does)

from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode, TableNode
from maddening.sysid import fit_lm
from tests.core.test_sysid_non_differentiable_residual import (
    BALL_STARTS,
    DT,
    JUMP,
    _messages,
    _started,
    _twin,
)

GUIDE = "docs/user_guide/parameters.md"

#: Indices into ``BALL_STARTS`` whose fit ends beside a jump (``converged=False``
#: with the warning); the ones the sibling file's grid shows ending that way
#: and that cover both the smallest and the largest loss it records there.
_STOPPED_BESIDE_A_JUMP = (0, 3)


def test_the_warning_beside_a_jump_names_the_guide():
    gm = GraphManager()
    gm.add_node(TableNode("table", DT))
    for name in ("ball", "record"):
        gm.add_node(BallNode(name, DT, initial_position=1.0, elasticity=0.7))
        gm.add_edge("table", name, "position", "table_position")
    gm.compile()
    residual = _twin(gm, "ball", 200, "position")

    warned = []
    for i in _STOPPED_BESIDE_A_JUMP:
        elasticity, gravity = BALL_STARTS[i]
        start = _started(gm, "ball", elasticity=elasticity, gravity=-9.81 * gravity)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = fit_lm(gm, residual, params=start)
        assert not res.converged, (i, res.best_loss)
        warned.append(_messages(caught, JUMP))
    assert all(len(texts) == 1 for texts in warned), warned
    for (text,) in warned:
        assert GUIDE in text
