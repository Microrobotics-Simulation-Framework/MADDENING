"""Every program the library builds traces its float32 matrix products at the
graph's ``matmul_precision`` (``"highest"`` unless set), and ``None`` inherits
JAX's own setting.

On NVIDIA GPUs, JAX's default ``jax_default_matmul_precision`` runs float32
matrix-matrix products at reduced precision.  Measured on an RTX A2000,
relative error against float64 at the default and at ``"highest"``:
``U.T @ JU`` of the coupling bounds, (1e5, 8), 3.0e-4 against 4.9e-7;
``H @ field`` of an interface mapping on a 3-component field, 2.5e-4 against
6.5e-7; the heat node's ``w @ cells``, (2, 3) @ (3, 1e5), 4.0e-4 against
7.3e-8; ``J.T @ J`` of the Fisher information, (2e4, 12), 9.0e-6 against
1.1e-7 (MADD-ANO-291).

XLA's CPU backend computes every float32 product in full whatever is asked,
so no CPU test can see the error itself.  These tests read the cause in the
programs instead.  Each entry point is called with JAX's global default set
to ``"bfloat16"`` and ``jax_dump_ir_to`` pointed at a scratch directory, so
every module lowered during the call is written out as StableHLO; every
float32 ``stablehlo.dot_general`` in them must then carry ``HIGHEST`` (the
library pinned it: the global setting would have given ``DEFAULT``), and,
with ``matmul_precision=None``, ``DEFAULT`` (inherited from the global
setting).  Reading what was lowered rather than a jaxpr covers the programs a
loop on the host compiles one by one (``run_adaptive``, the fitters), the
operations it runs eagerly, and the derivative rules JAX traces only when the
caller's ``jax.grad`` or ``jax.jvp`` reaches them.

The graphs reach the affected sites: a coupling group joined by interface
mappings applied to 3-component fields, whose implicit-function gradient is
taken, with its diagnostics (the spectral and gradient bounds) on in the slow
test; a heat rod of
4th order behind ``ShardedStencilNode`` (``update_padded``'s ``w @ cells``);
and a float32 fit and Fisher information whose residual forms a product of
its own.
"""

from __future__ import annotations

import ast
import glob
import importlib.util
import os
import re
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening import sysid
from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core import _matmul_precision
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.simulation.profiler import profile_graph
from maddening.fmi.directional_derivatives import (
    DirectionalDerivativeKind,
    get_directional_derivative,
)
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode

HIGHEST = ("HIGHEST", "HIGHEST")
DEFAULT = ("DEFAULT", "DEFAULT")


# ---------------------------------------------------------------------------
# Reading the lowered programs
# ---------------------------------------------------------------------------

_DOT = re.compile(r"stablehlo\.dot_general\b")
_DIMS = re.compile(r"(batching|contracting)_dims\s*=\s*\[([\d,\s]*)\]\s*x\s*\[([\d,\s]*)\]")
_PRECISION = re.compile(r"precision\s*=\s*\[(\w+),\s*(\w+)\]")
_PRECISION_CONFIG = re.compile(
    r"precision_config\s*=\s*\[#stablehlo<precision (\w+)>,\s*#stablehlo<precision (\w+)>\]")
_TENSOR = r"tensor<((?:\d+x)*)([a-z0-9]+(?:<[a-z0-9]+>)?)>"
_OPERANDS = re.compile(r":\s*\(" + _TENSOR + r",\s*" + _TENSOR + r"\)\s*->")
_FLOAT32 = ("f32", "complex<f32>")


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(t) for t in text.replace(" ", "").split(",") if t)


class Dot:
    """One ``stablehlo.dot_general`` of a lowered module."""

    def __init__(self, module: str, line: str):
        self.module, self.line = module, line.strip()
        m = _PRECISION.search(line) or _PRECISION_CONFIG.search(line)
        # A lowering that omits the attribute asked for the default.
        self.precision = m.groups() if m else DEFAULT
        lhs_dims, lt, rhs_dims, rt = _OPERANDS.search(line).groups()
        self.lhs, self.rhs = (tuple(int(d) for d in dims.split("x") if d)
                              for dims in (lhs_dims, rhs_dims))
        self.float32 = lt in _FLOAT32 and rt in _FLOAT32
        dims = {kind: (_ints(a), _ints(b)) for kind, a, b in _DIMS.findall(line)}
        batch = dims.get("batching", ((), ()))
        contract = dims.get("contracting", ((), ()))
        self.free = tuple(
            [size for i, size in enumerate(shape) if i not in batch[k] + contract[k]]
            for k, shape in enumerate((self.lhs, self.rhs)))

    @property
    def matrix_matrix(self) -> bool:
        """Both operands contribute a free dimension longer than one: the class
        a GPU computes at reduced precision (a matrix-vector or vector-vector
        product is computed in full float32 at any setting, measured)."""
        return all(any(size > 1 for size in free) for free in self.free)

    def __repr__(self) -> str:
        return f"{self.precision} {self.lhs}x{self.rhs} in {self.module}"


def lowered_dots(call) -> list[Dot]:
    """Every float32 ``dot_general`` of every module lowered while ``call()``
    runs under a global ``jax.default_matmul_precision("bfloat16")``."""
    with tempfile.TemporaryDirectory() as directory:
        prior = jax.config.values.get("jax_dump_ir_to") or ""
        jax.config.update("jax_dump_ir_to", directory)
        try:
            with jax.default_matmul_precision("bfloat16"):
                call()
        finally:
            jax.config.update("jax_dump_ir_to", prior)
        found = []
        for path in sorted(glob.glob(os.path.join(directory, "*.mlir"))):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if _DOT.search(line) and _OPERANDS.search(line):
                        found.append(Dot(os.path.basename(path), line))
    return [d for d in found if d.float32]


def assert_inherited(dots: list[Dot], what: str) -> None:
    """With ``matmul_precision=None`` the library enters nothing, so the
    products take the global ``bfloat16`` (``DEFAULT``).  Products that pass
    their own precision keep it -- JAX's ``lstsq`` pins its matrix-vector
    products to ``HIGHEST`` -- so the rule is held on the matrix products, and
    at least one product must show the inherited setting."""
    assert any(d.precision == DEFAULT for d in dots), (
        f"{what}: no product inherited the global setting: {dots[:6]}")
    wrong = [d for d in dots if d.matrix_matrix and d.precision != DEFAULT]
    assert not wrong, f"{what}: matrix products not at the inherited setting: {wrong[:6]}"


def assert_all(dots: list[Dot], precision: tuple, what: str) -> None:
    # The programs hold products to check: an entry point whose modules
    # held none would pass vacuously.
    assert dots, f"{what}: no float32 dot_general was lowered"
    wrong = [d for d in dots if d.precision != precision]
    assert not wrong, (
        f"{what}: {len(wrong)} of {len(dots)} float32 products are not traced "
        f"at {precision}: {wrong[:6]}")


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

N, C = 4, 3          # 4 points of a 3-component field
DT = 0.01


class Vec(SimulationNode):
    """``x <- g * u + c`` on an ``(N, C)`` field: memoryless, so the coupled
    pair has a fixed point and an implicit-function gradient."""

    def __init__(self, name, g, c):
        super().__init__(name, DT, g=jnp.asarray(g, jnp.float32),
                         c=jnp.asarray(np.broadcast_to(c, (N, C)), jnp.float32))

    def initial_state(self):
        return {"x": jnp.zeros((N, C), jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(N, C), dtype=jnp.float32,
                                       default=jnp.zeros((N, C), jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["g"] * jnp.asarray(boundary_inputs["u"], jnp.float32) + p["c"]}


def coupled(precision=_matmul_precision.DEFAULT, linear_solver="dense",
            diagnostics=False, acceleration="iqn-ils") -> GraphManager:
    """Two :class:`Vec` members coupled through ``(N, N)`` interface mappings
    applied to their ``(N, C)`` fields (``H @ field``, a matrix product), the
    implicit-function adjoint solved by ``linear_solver``.  IQN-ILS
    accelerates the solve: its least-squares step forms matrix products in
    the forward step and again in the implicit-function rule, which re-solves
    for its primal when JAX traces it (the members' own products reach the
    rule pre-traced, at the step's setting).  ``jnp.linalg.pinv`` in that
    step pins one product of its own to ``HIGHEST`` whatever the setting, so
    the tests of ``None`` turn the acceleration off (:data:`INHERITING`).
    With
    ``diagnostics`` the step also forms the group's Krylov basis
    (``U.T @ JU``), spectral and gradient bounds: five seconds of compile
    here, so only the slow test turns them on."""
    rng = np.random.default_rng(0)
    gm = GraphManager()
    gm.matmul_precision = precision
    gm.add_node(Vec("a", 0.5, np.linspace(0.5, 1.0, N * C).reshape(N, C)))
    gm.add_node(Vec("b", 0.4, 0.2))
    for source, target in (("b", "a"), ("a", "b")):
        H = jnp.asarray(rng.uniform(0.1, 0.3, (N, N)), jnp.float32)
        gm.add_edge(source, target, "x", "u", mapping=matrix_mapping(H))
    gm.add_coupling_group(["a", "b"], max_iterations=30, tolerance=1e-6,
                          acceleration=acceleration, diagnostics=diagnostics,
                          linear_solver=linear_solver)
    return gm


def heat(precision=_matmul_precision.DEFAULT) -> GraphManager:
    """A 4th-order heat rod stepped through ``update_padded`` (one device)."""
    gm = GraphManager()
    gm.matmul_precision = precision
    gm.add_node(ShardedStencilNode(
        HeatNode("rod", DT, n_cells=16, stencil_order=4),
        create_device_mesh(shape=(1,)), axis_map={"devices": 0}))
    return gm


def _float_sum(tree):
    return sum(jnp.sum(leaf) for leaf in jax.tree.leaves(tree)
               if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating))


def _batched(gm, copies=2):
    return {name: jax.tree.map(lambda v: jnp.stack([v] * copies), fields)
            for name, fields in gm._user_state(gm._state).items()}


def _scan_loss(gm):
    def loss(p):
        return _float_sum(gm.run_scan(2, params=p))
    return loss


def _member_step(gm):
    """``x_a -> x_b`` through one compiled step: the FMU sidecar's
    ``unknown_fn`` shape."""
    state = gm._state

    def unknown(x):
        return gm._compiled_step({**state, "a": {"x": x}}, {}, gm.params)["b"]["x"]
    return unknown


def _compiled(gm):
    gm.compile()
    return gm


#: Each graph entry point, as ``(graph builder, call)``.  The call gets a
#: freshly built graph, so every program it lowers is traced afresh.
ENTRY_POINTS = {
    "compile+step": (coupled, lambda gm: (gm.compile(), gm.step())),
    "run_scan": (coupled, lambda gm: _compiled(gm).run_scan(3)),
    "run_scan_with_history": (coupled, lambda gm: _compiled(gm).run_scan_with_history(3)),
    "run_sweep": (coupled, lambda gm: _compiled(gm).run_sweep(2, _batched(gm))),
    "run_adaptive": (coupled, lambda gm: _compiled(gm).run_adaptive(2 * DT, dt_initial=DT)),
    "run_adaptive_scan": (coupled, lambda gm: _compiled(gm).run_adaptive_scan(
        2 * DT, dt_initial=DT, max_steps=4)),
    "grad of run_scan": (coupled, lambda gm: jax.grad(_scan_loss(_compiled(gm)))(gm.params)),
    "jvp of run_scan": (coupled, lambda gm: jax.jvp(
        _scan_loss(_compiled(gm)), (gm.params,), (jax.tree.map(jnp.ones_like, gm.params),))),
    "FMU forward derivative": (coupled, lambda gm: get_directional_derivative(
        _member_step(_compiled(gm)), kind=DirectionalDerivativeKind.FORWARD,
        x=jnp.ones((N, C), jnp.float32), v=jnp.ones((N, C), jnp.float32))),
    "FMU reverse derivative": (coupled, lambda gm: get_directional_derivative(
        _member_step(_compiled(gm)), kind=DirectionalDerivativeKind.REVERSE,
        x=jnp.ones((N, C), jnp.float32), v=jnp.ones((N, C), jnp.float32))),
    "profile_graph": (heat, lambda gm: profile_graph(
        _compiled(gm), n_steps=2, n_warmup=1, measure_coupling=False, counts=False)),
    "heat step": (heat, lambda gm: (gm.compile(), gm.step())),
    "heat run_sweep": (heat, lambda gm: _compiled(gm).run_sweep(2, _batched(gm, 3))),
}


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
def test_every_graph_entry_point_traces_its_products_at_highest(entry):
    build, call = ENTRY_POINTS[entry]
    gm = build()
    assert gm.matmul_precision == "highest"
    assert_all(lowered_dots(lambda: call(gm)), HIGHEST, entry)


#: The graphs of the tests of ``None``: no product of theirs pins its own
#: precision, so every one must inherit the global setting.
INHERITING = {coupled: {"acceleration": "none"}, heat: {}}


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
def test_with_none_every_graph_entry_point_inherits_jax_s_setting(entry):
    build, call = ENTRY_POINTS[entry]
    gm = build(precision=None, **INHERITING[build])
    assert_inherited(lowered_dots(lambda: call(gm)), entry)


# Per push: tests/core/test_library_programs_trace_products_at_their_matmul_precision.py::test_every_graph_entry_point_traces_its_products_at_highest[compile+step]
@pytest.mark.slow
@pytest.mark.parametrize("precision", ["highest", None])
def test_the_coupling_bounds_trace_their_products_at_the_graph_s_setting(precision):
    """With the group's diagnostics on, the step forms the Krylov basis of the
    Jacobian's range (``U.T @ JU``, ``U @ M``), the squarings of the spectral
    estimate and the full resolvent norm: in the step and in its gradient.
    CPU results do not move: the step and the report are bit-identical at
    either setting."""
    gm = coupled(precision, diagnostics=True, acceleration="none")
    dots = lowered_dots(lambda: (gm.compile(), gm.step()))
    stepped = jax.tree.map(np.asarray, gm._state)          # the report included
    dots += lowered_dots(lambda: jax.grad(_scan_loss(gm))(gm.params))
    assert sum(d.matrix_matrix for d in dots) > 20, "the bounds' products are missing"
    if precision is None:
        assert_inherited(dots, "coupling diagnostics")
    else:
        assert_all(dots, HIGHEST, "coupling diagnostics")
    other = _compiled(coupled("highest" if precision is None else None, diagnostics=True,
                              acceleration="none"))
    other.step()
    for x, y in zip(jax.tree.leaves(stepped), jax.tree.leaves(other._state)):
        np.testing.assert_array_equal(x, np.asarray(y))


def test_the_krylov_adjoint_traces_its_matrix_products_at_highest():
    """With the default GMRES adjoint, the gradient's matrix products are
    pinned too.  Its matrix-vector products are held to less: lineax lowers
    its solve as a primitive whose body JAX traces when the program is
    compiled -- after every context of ours has closed -- and lineax 0.0.7's
    GMRES forms ``basis @ proj`` there without a precision.  A matrix-vector
    product is computed in full float32 at any setting (measured), so only
    those may differ."""
    gm = coupled(linear_solver="gmres")
    dots = lowered_dots(lambda: jax.grad(_scan_loss(_compiled(gm)))(gm.params))
    assert [d for d in dots if d.matrix_matrix], "no matrix product in the gradient"
    assert_all([d for d in dots if d.matrix_matrix], HIGHEST, "Krylov adjoint")


# ---------------------------------------------------------------------------
# System identification
# ---------------------------------------------------------------------------

STEPS = 24
_PROJECTION = jnp.asarray(np.random.default_rng(1).uniform(0.5, 1.5, (4, 3)), jnp.float32)


def spring(stiffness=30.0, precision=_matmul_precision.DEFAULT) -> GraphManager:
    gm = GraphManager()
    gm.matmul_precision = precision
    gm.add_node(SpringDamperNode(name="s", timestep=DT, stiffness=stiffness, damping=2.0,
                                 mass=1.0, rest_length=1.0, initial_position=0.2,
                                 initial_velocity=0.0))
    gm.compile()
    return gm


def _project(x):
    """A product the residual forms itself, outside any graph: ``(6, 4) @ (4, 3)``."""
    return (x.reshape(-1, 4) @ _PROJECTION).ravel()


_TRUTH = spring(40.0).run_scan_with_history(STEPS)[1]["s"]["position"]


def _residual(p):
    # A fresh graph per evaluation: ``run_scan*`` stores its final state.
    return _project(spring().run_scan_with_history(STEPS, params=p)[1]["s"]["position"] - _TRUTH)


def _mask(gm):
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    mask["nodes"]["s"]["stiffness"] = True
    mask["nodes"]["s"]["damping"] = True
    return mask


def _observations():
    gm = spring(40.0)
    start = gm._user_state(gm._state)
    history = gm.run_scan_with_history(STEPS)[1]
    return sysid.observations_from_history({"s": start["s"]}, {"s": history["s"]})


def _obs_fn(state):
    return _project(jnp.tile(state["s"]["position"], 8))


def _windowed(gm, **kw):
    obs = _observations()
    return jax.jit(lambda p: sysid.windowed_loss(gm, p, obs, obs_fn=_obs_fn, window=6, **kw))


#: Each sysid entry point, as ``call(precision)``.  The fitters and
#: ``windowed_loss`` take the setting from the graph they are handed; ``fim``
#: and ``fim_core`` from their ``matmul_precision`` argument.
SYSID = {
    "fit": lambda prec: (lambda gm: sysid.fit(
        gm, lambda p: jnp.sum(_residual(p) ** 2), mask=_mask(gm), n_iter=3))(spring(precision=prec)),
    "fit_lm": lambda prec: (lambda gm: sysid.fit_lm(
        gm, _residual, mask=_mask(gm), n_iter=20))(spring(precision=prec)),
    "fit_multiple_shooting": lambda prec: (lambda gm: sysid.fit_multiple_shooting(
        gm, _observations(), obs_fn=_obs_fn, window=6, mask=_mask(gm), n_iter=3))(
        spring(precision=prec)),
    "windowed_loss": lambda prec: (lambda gm: _windowed(gm)(gm.params))(spring(precision=prec)),
    "grad of windowed_loss, masked": lambda prec: (lambda gm: jax.grad(
        _windowed(gm, mask_unconverged=True))(gm.params))(spring(precision=prec)),
    "fim": lambda prec: sysid.fim(_residual, spring().params, matmul_precision=prec),
    "fim, reused trace": lambda prec: sysid.fim(
        _residual, spring().params, reuse_trace=True, matmul_precision=prec),
    "fim_core under jit": lambda prec: jax.jit(
        lambda p: sysid.fim_core(_residual, p, matmul_precision=prec))(spring().params),
}


@pytest.mark.parametrize("entry", sorted(SYSID))
def test_every_sysid_entry_point_traces_its_products_at_highest(entry):
    assert_all(lowered_dots(lambda: SYSID[entry](_matmul_precision.DEFAULT)), HIGHEST, entry)


@pytest.mark.parametrize("entry", sorted(SYSID))
def test_with_none_every_sysid_entry_point_inherits_jax_s_setting(entry):
    assert_inherited(lowered_dots(lambda: SYSID[entry](None)), entry)


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------

def test_a_new_matmul_precision_recompiles_every_program_of_the_graph():
    """No program traced at the previous setting runs again: the graph is
    marked for recompilation, the next entry point recompiles it (a new
    generation, an empty scan cache) and keeps its state."""
    gm = _compiled(coupled(acceleration="none"))
    gm.run_scan(2)
    generation = gm._compile_generation
    before = jax.tree.map(np.asarray, gm._user_state(gm._state))
    gm.matmul_precision = None
    assert gm._dirty
    # The recompile keeps the state the run had reached.
    gm.compile()
    for x, y in zip(jax.tree.leaves(before), jax.tree.leaves(gm._user_state(gm._state))):
        np.testing.assert_array_equal(x, np.asarray(y))
    assert gm._compile_generation == generation + 1
    assert_inherited(lowered_dots(lambda: gm.run_scan(2)), "run_scan after the change")
    # Setting the value it holds changes nothing.
    gm.matmul_precision = None
    assert not gm._dirty


def test_a_compiled_step_handed_out_before_the_change_is_the_step_of_an_earlier_compile():
    gm = _compiled(coupled())
    gm.step()
    old = gm._compiled_step
    gm.matmul_precision = "float32"
    gm.step()
    assert gm._compiled_step is not old
    assert old._maddening_compile[1] < gm._compiled_step._maddening_compile[1]


@pytest.mark.parametrize("value", ["nonsense", "HIGHEST ", 3, 1.0, jax.lax.Precision.HIGHEST])
def test_a_value_jax_does_not_accept_is_refused_where_it_is_set(value):
    gm = coupled()
    with pytest.raises(ValueError, match="matmul_precision"):
        gm.matmul_precision = value
    assert gm.matmul_precision == "highest"
    with pytest.raises(ValueError, match="matmul_precision"):
        sysid.fim(_residual, spring().params, matmul_precision=value)


@pytest.mark.parametrize("value", [None, "highest", "float32", "tensorfloat32", "bfloat16", "default"])
def test_every_value_jax_accepts_is_taken(value):
    gm = coupled()
    gm.matmul_precision = value
    assert gm.matmul_precision == value


def test_on_cpu_the_pinned_programs_compute_the_same_bits():
    """XLA's CPU backend computes float32 products in full at any setting:
    pinning the precision moves nothing on CPU, in the state or the gradient."""
    for build in (coupled, heat):
        a, b = _compiled(build()), _compiled(build(precision=None))
        for x, y in zip(jax.tree.leaves(a.run_scan(3)), jax.tree.leaves(b.run_scan(3))):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    a, b = _compiled(coupled()), _compiled(coupled(precision=None))
    for x, y in zip(jax.tree.leaves(jax.grad(_scan_loss(a))(a.params)),
                    jax.tree.leaves(jax.grad(_scan_loss(b))(b.params))):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# ---------------------------------------------------------------------------
# Custom derivative rules: traced when the derivative is taken
# ---------------------------------------------------------------------------

#: Every ``jax.custom_jvp`` / ``jax.custom_vjp`` the package defines, by
#: module and name, with how its rule's products are traced at the graph's
#: setting.  A rule is traced when the caller's ``jax.grad`` or ``jax.jvp``
#: reaches it, outside the body that called the function, so a rule that
#: forms a product must go through
#: ``_matmul_precision.custom_jvp_at_call_precision``.  Held equal, so a new
#: rule is classified when it is added.
CUSTOM_DERIVATIVES = {
    ("maddening.core._matmul_precision", "derivative"): "the carrier itself",
    ("maddening.core.coupling._coupled_block", "_interpolate"): "no product",
    ("maddening.core.coupling._reports", "_bit_tie"): "no product",
    ("maddening.core.params", "_clip_into_bounds"): "no product",
    ("maddening.cloud.multigpu.iterative_solver", "as_solved"): "its rule raises",
}


def _custom_derivatives() -> set:
    spec = importlib.util.find_spec("maddening")
    root = Path(spec.origin).parent
    found = set()
    for path in sorted(root.rglob("*.py")):
        if "examples" in path.relative_to(root).parts:
            continue
        module = "maddening." + ".".join(path.relative_to(root).with_suffix("").parts)
        module = module.removesuffix(".__init__")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # ``@jax.custom_jvp`` or ``@functools.partial(jax.custom_jvp, ...)``
                if any(re.search(r"\bcustom_[jv]vp\b", ast.unparse(deco))
                       for deco in node.decorator_list):
                    found.add((module, node.name))
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if re.search(r"\bcustom_[jv]vp$", ast.unparse(node.value.func)):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            found.add((module, target.id))
    return found


def test_every_custom_derivative_rule_of_the_package_is_classified():
    assert _custom_derivatives() == set(CUSTOM_DERIVATIVES)


def test_the_rules_classified_as_forming_no_product_form_none():
    from maddening.core.coupling._coupled_block import _interpolate
    from maddening.core.coupling._reports import _bit_tie
    from maddening.core.params import _clip_into_bounds

    x = jnp.ones((3, 2), jnp.float32)
    cases = {
        "_interpolate": (lambda a, b: _interpolate(a, b, jnp.float32(0.3)), (x, 2 * x)),
        "_bit_tie": (lambda a: _bit_tie(a, jnp.zeros((), jnp.uint32)), (x,)),
        "_clip_into_bounds": (lambda a: _clip_into_bounds(a, 0.0, 2.0), (x,)),
    }
    for name, (f, args) in cases.items():
        jaxpr = jax.make_jaxpr(lambda *a, f=f: jax.jvp(f, a, a))(*args)
        assert "dot_general" not in str(jaxpr), name


def test_the_implicit_function_rule_traces_at_the_precision_of_its_call():
    """The coupling group's implicit-function solve is the one rule that forms
    products (the members' JVPs and the adjoint solve): it is built by the
    carrier, which hands the rule the precision in force at the call."""
    from maddening.core.coupling import _ift

    assert _ift._ift_solve.__wrapped__ is _ift._ift_solve_impl
    assert _ift._ift_solve.__qualname__ == _ift._ift_solve_impl.__qualname__
