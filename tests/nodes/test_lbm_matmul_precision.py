"""The LBM nodes compute their lattice products at full float32 precision.

``LBMNode`` and ``LBMPipeNode`` form float32 matrix products whose inner
dimension is the lattice size: ``velocity @ e.T`` (inner size 2 or 3) and
``f @ e`` (inner size 9 or 19).  On NVIDIA GPUs, JAX's default
``jax_default_matmul_precision`` runs such products at reduced precision
(TF32-class passes): on an RTX A2000 a ``(1e5, 3) @ (3, 19)`` product was off
by 3.4e-4 relative to float64 at the default and by 5.9e-8 at ``highest``, and
the runner's stencil goal measured the LBM gradients at 1.6e-3 relative where
its limit is 1e-5.  Each product therefore passes
``precision=jax.lax.Precision.HIGHEST`` itself, whatever the global setting.

No CPU test can see the defect itself: XLA's CPU backend computes these
products in full float32 whatever precision is requested, so the patched and
unpatched steps are bit-identical here.  These tests pin the cause instead, at
two levels:

* the sources of ``lbm.py`` and ``lbm_pipe.py`` hold no ``@`` operator, and
  every matrix-product call in them passes ``precision=`` equal to
  ``jax.lax.Precision.HIGHEST`` (or ``"highest"``);
* every ``dot_general`` in the traced step of each node, and in its gradient,
  carries ``Precision.HIGHEST``, traced with the global default set to
  ``bfloat16`` so that a product which does not pin its own precision cannot
  pass by inheriting a global ``highest``.

The sources are located through the imported package, not the repository
layout, so a scan reads the code that actually runs.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.lbm import LBMNode
from maddening.nodes.lbm_pipe import LBMPipeNode

_MODULES = ("maddening.nodes.lbm", "maddening.nodes.lbm_pipe")

# Call names that form a matrix product (``jnp.<name>``, ``jax.lax.<name>``
# or an array method of that name).  Wider than the three the nodes use, so a
# product written another way is held to the same rule.
_PRODUCT_NAMES = frozenset({
    "matmul", "dot", "einsum", "tensordot", "inner", "vdot", "dot_general",
    "batch_matmul", "multi_dot", "matvec", "vecmat", "vecdot",
})
# Host-side NumPy runs in float64 at construction time and is not affected.
_HOST_ROOTS = frozenset({"np", "numpy"})

# The helper functions that hold a lattice product today.  Held equal, so the
# scan cannot pass by recognising no call at all.
_EXPECTED_PRODUCT_HOLDERS = {
    "maddening.nodes.lbm": {
        "_equilibrium", "_compute_macroscopic", "_guo_forcing",
    },
    "maddening.nodes.lbm_pipe": {
        "_equilibrium", "_compute_macroscopic", "_guo_forcing",
        "_tracer_equilibrium",
    },
}


def _source_tree(module: str) -> tuple[Path, ast.Module]:
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin is not None, module
    path = Path(spec.origin)
    return path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dotted(node: ast.expr) -> list[str]:
    """``a.b.c`` -> ``["a", "b", "c"]``; anything else ends the chain."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):
        parts.append("<call>")
    else:
        parts.append("<expr>")
    return parts[::-1]


def _is_highest(value: ast.expr) -> bool:
    if isinstance(value, ast.Constant):
        return isinstance(value.value, str) and value.value == "highest"
    if isinstance(value, ast.Attribute):
        chain = _dotted(value)
        return len(chain) >= 2 and chain[-2:] == ["Precision", "HIGHEST"]
    if isinstance(value, (ast.Tuple, ast.List)):
        return bool(value.elts) and all(_is_highest(e) for e in value.elts)
    return False


def _product_calls(tree: ast.Module):
    """Yield ``(enclosing function, call)`` for every matrix-product call."""
    def visit(node: ast.AST, owner: str):
        for child in ast.iter_child_nodes(node):
            name = owner
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = child.name
            if isinstance(child, ast.Call):
                chain = _dotted(child.func)
                if chain[-1] in _PRODUCT_NAMES and chain[0] not in _HOST_ROOTS:
                    yield name, child
            yield from visit(child, name)
    yield from visit(tree, "<module>")


@pytest.mark.parametrize("module", _MODULES)
def test_no_matmul_operator_remains_in_the_lbm_sources(module):
    """``A @ B`` and ``A @= B`` take the global matmul precision: none may remain."""
    path, tree = _source_tree(module)
    found = [
        f"{path.name}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.BinOp, ast.AugAssign))
        and isinstance(node.op, ast.MatMult)
    ]
    assert not found, (
        f"{module} uses the @ operator at {found}: on an NVIDIA GPU it runs at "
        "jax_default_matmul_precision.  Write jnp.matmul(a, b, "
        "precision=jax.lax.Precision.HIGHEST)."
    )


@pytest.mark.parametrize("module", _MODULES)
def test_every_matrix_product_call_in_the_lbm_sources_pins_highest(module):
    """Each ``jnp.matmul``/``jnp.dot``/``jnp.einsum``-like call passes
    ``precision=jax.lax.Precision.HIGHEST`` (or ``"highest"``)."""
    path, tree = _source_tree(module)
    unpinned = []
    for owner, call in _product_calls(tree):
        given = [kw.value for kw in call.keywords if kw.arg == "precision"]
        if not (len(given) == 1 and _is_highest(given[0])):
            unpinned.append(
                f"{path.name}:{call.lineno} in {owner} "
                f"({'.'.join(_dotted(call.func))})"
            )
    assert not unpinned, (
        f"matrix products without precision=jax.lax.Precision.HIGHEST in "
        f"{module}: {unpinned}"
    )


@pytest.mark.parametrize("module", _MODULES)
def test_the_scan_finds_the_lattice_products_it_guards(module):
    """The scan above must not pass by recognising no call: it sees a product
    in exactly the helpers that form one today."""
    _, tree = _source_tree(module)
    holders = {owner for owner, _ in _product_calls(tree)}
    assert holders == _EXPECTED_PRODUCT_HOLDERS[module]


# ---------------------------------------------------------------------------
# The traced step and its gradient
# ---------------------------------------------------------------------------

def _dot_precisions(jaxpr, out: list) -> list:
    """Every ``dot_general``'s ``precision`` in a jaxpr, sub-jaxprs included."""
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            out.append(eqn.params.get("precision"))
        for value in eqn.params.values():
            _visit_param(value, out)
    return out


def _visit_param(value, out: list) -> None:
    if hasattr(value, "eqns"):
        _dot_precisions(value, out)
    elif hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
        _dot_precisions(value.jaxpr, out)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _visit_param(item, out)


def _wrap_pad(arr, halo: int, ndim: int):
    width = [(halo, halo)] * ndim + [(0, 0)] * (arr.ndim - ndim)
    return jnp.asarray(np.pad(np.asarray(arr), width, mode="wrap"))


def _lbm_step(grid, lattice, padded=False):
    node = LBMNode("fluid", timestep=1.0, grid_shape=grid, viscosity=0.1,
                   lattice=lattice)
    ndim = len(grid)
    state = node.initial_state()
    if padded:
        state = {
            k: (_wrap_pad(v, 1, ndim) if getattr(v, "ndim", 0) >= ndim else v)
            for k, v in state.items()
        }
    force = jnp.full((ndim,), 1e-4, dtype=jnp.float32)  # units: lattice

    def step(s, body_force):
        bi = {"body_force": body_force}
        if padded:
            return node.update_padded(s, bi, 1.0)
        return node.update(s, bi, 1.0)

    return step, (state, force)


def _pipe_step(**kw):
    node = LBMPipeNode("pipe", timestep=0.01, nx=6, ny=5, nz=5,
                       propeller_x=2, **kw)

    def step(s):
        return node.update(s, {}, 0.01)

    return step, (node.initial_state(),)


_STEPS = {
    "LBMNode-D2Q9": lambda: _lbm_step((6, 5), "D2Q9"),
    "LBMNode-D3Q19": lambda: _lbm_step((4, 4, 4), "D3Q19"),
    "LBMNode-D3Q19-padded": lambda: _lbm_step((4, 4, 4), "D3Q19", padded=True),
    "LBMPipeNode-tracer": lambda: _pipe_step(),
    "LBMPipeNode-shan-chen": lambda: _pipe_step(G=-5.0),
}


def _float_sum(tree) -> jax.Array:
    return sum(
        jnp.sum(leaf) for leaf in jax.tree.leaves(tree)
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    )


@pytest.mark.parametrize("case", sorted(_STEPS))
def test_every_dot_in_the_traced_lbm_step_and_its_gradient_is_highest(case):
    """Under a global ``bfloat16`` default, every product of the step and of
    its reverse-mode gradient still carries ``Precision.HIGHEST``."""
    step, args = _STEPS[case]()
    highest = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)

    def loss(*a):
        return _float_sum(step(*a))

    with jax.default_matmul_precision("bfloat16"):
        forward = _dot_precisions(jax.make_jaxpr(step)(*args).jaxpr, [])
        grad = _dot_precisions(
            jax.make_jaxpr(
                jax.grad(loss, argnums=tuple(range(len(args))), allow_int=True)
            )(*args).jaxpr,
            [],
        )
    # The lattice products are there to be checked (a step with no
    # dot_general would pass vacuously).
    assert forward, f"{case}: the traced step holds no dot_general"
    assert len(grad) > len(forward), (
        f"{case}: the gradient holds no transposed product "
        f"({len(grad)} against {len(forward)} in the step)"
    )
    for label, found in (("step", forward), ("gradient", grad)):
        wrong = [p for p in found if p != highest]
        assert not wrong, (
            f"{case}: {len(wrong)} of {len(found)} dot_general in the {label} "
            f"do not pin Precision.HIGHEST: {wrong}"
        )
