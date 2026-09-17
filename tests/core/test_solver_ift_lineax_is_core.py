"""A default coupling group is differentiable on a base install.

``CouplingGroup`` defaults to ``solver="ift"`` with
``linear_solver="gmres"``, whose derivative rule solves the adjoint
system with ``lineax``.  Until v0.4.0 ``lineax`` was an optional
``[ift]`` extra, so the *default* configuration raised ``ImportError``
on a base install the moment anything called ``jax.grad`` through a
coupling group — a default that failed at runtime instead of at
install time.  ``lineax`` is now a base dependency; these tests pin
that, both at the metadata level and by differentiating a default
group in a subprocess where every optional extra is unimportable.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from maddening.core.coupling import CouplingGroup
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


_REPO_ROOT = Path(__file__).resolve().parents[2]

# Top-level import names provided *only* by an optional extra.  A base
# install has none of them, so blocking them is a faithful stand-in for
# one.  ``equinox`` and ``jaxtyping`` are deliberately absent: they come
# in transitively with ``lineax``, which is a base dependency.
_EXTRA_ONLY_MODULES = (
    "matplotlib",    # viz
    "rich",          # terminal
    "zmq",           # network
    "fastapi",       # api
    "uvicorn",       # api
    "websockets",    # api / streaming
    "optax",         # surrogates
    "pyvista",       # viz3d
    "PIL",           # viz3d
    "pygfx",         # gpu-viz
    "rendercanvas",  # gpu-viz
    "glfw",          # gpu-viz
    "skimage",       # gpu-viz
    "pxr",           # usd
    "gi",            # streaming
    "cyclonedx",     # sbom
    "zstandard",     # compression
    "sky",           # cloud providers (skypilot)
    "fsspec",        # ci: cloud checkpoint URLs
    "httpx",         # ci
    "fmpy",          # ci: FMI round trips
    "cupy",          # never declared; some paths probe for it
)


def _coupled(**group_kwargs) -> GraphManager:
    """Two spring-damper nodes in a mutually-coupled group."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.01, initial_position=3.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=10, tolerance=1e-8,
                          **group_kwargs)
    gm.compile()
    return gm


def _grad_k(gm: GraphManager) -> float:
    """d(b.position after one step) / d(a.stiffness)."""
    ext = gm._default_external_inputs()

    def loss(k):
        p = jax.tree.map(lambda x: x, gm.params)
        p["nodes"]["a"]["stiffness"] = k
        return gm._compiled_step(gm._state, ext, p)["b"]["position"]

    return float(jax.grad(loss)(jnp.asarray(100.0, jnp.float32)))


def test_lineax_is_a_base_dependency():
    """lineax is in [project].dependencies, not behind an extra."""
    import tomllib

    py = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    base = py["project"]["dependencies"]
    assert any(d.startswith("lineax") for d in base), base

    # The `ift` extra is kept as an empty compatibility alias so that
    # `pip install maddening[ift]` — valid in v0.3.x — keeps resolving.
    extras = py["project"]["optional-dependencies"]
    assert extras["ift"] == [], extras["ift"]


def test_coupling_group_defaults_to_the_krylov_ift_path():
    """The configuration this file is about is the *default* one."""
    group = CouplingGroup(nodes=frozenset({"a", "b"}))
    assert group.solver == "ift"
    assert group.linear_solver == "gmres"


def test_default_coupling_group_is_differentiable():
    """jax.grad through a default-settings coupling group works."""
    g = _grad_k(_coupled())
    assert jnp.isfinite(g)
    assert g != 0.0 and abs(g) < 1e3


def test_default_coupling_group_differentiable_with_no_extra_installed():
    """The same gradient, with every optional extra unimportable.

    The regression this whole change exists to prevent: a base install
    hitting ``ImportError`` on the default path.  Extras cannot be
    uninstalled from the shared venv, so a child process blocks them
    with a ``sys.meta_path`` finder instead.  The child also asserts
    that ``lineax`` really was imported, which proves the Krylov
    adjoint ran rather than some silent dense fallback.
    """
    src_root = str(_REPO_ROOT / "src")
    child = textwrap.dedent(
        '''
        import sys

        BLOCKED = %(blocked)r


        class _NotInstalled:
            """Reject extra-only modules the way a base install would."""

            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in BLOCKED:
                    raise ImportError(
                        f"no module named {fullname!r} (simulated base install)"
                    )
                return None


        sys.meta_path.insert(0, _NotInstalled())
        sys.path.insert(0, %(src_root)r)

        import jax
        import jax.numpy as jnp

        from maddening.core.graph_manager import GraphManager
        from maddening.nodes.spring import SpringDamperNode

        gm = GraphManager()
        gm.add_node(SpringDamperNode("a", 0.01, initial_position=0.0))
        gm.add_node(SpringDamperNode("b", 0.01, initial_position=3.0))
        gm.add_edge("a", "b", "position", "anchor_position")
        gm.add_edge("b", "a", "position", "anchor_position")
        gm.add_coupling_group(["a", "b"], max_iterations=10, tolerance=1e-8)
        gm.compile()

        ext = gm._default_external_inputs()


        def loss(k):
            p = jax.tree.map(lambda x: x, gm.params)
            p["nodes"]["a"]["stiffness"] = k
            return gm._compiled_step(gm._state, ext, p)["b"]["position"]


        g = float(jax.grad(loss)(jnp.asarray(100.0, jnp.float32)))
        assert "lineax" in sys.modules, "the gmres adjoint never ran"
        assert jnp.isfinite(g) and g != 0.0, g
        print(g)
        '''
    ) % {"blocked": set(_EXTRA_ONLY_MODULES), "src_root": src_root}

    env = dict(os.environ, JAX_PLATFORMS="cpu")
    # PYTHONPATH may point at another checkout; the child prepends this
    # repo's own src to sys.path itself, so drop it to avoid ambiguity.
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True, text=True, timeout=600, env=env,
    )
    assert proc.returncode == 0, (
        f"default coupling group failed on a simulated base install\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    child_grad = float(proc.stdout.strip().splitlines()[-1])
    assert abs(child_grad - _grad_k(_coupled())) < 1e-4, child_grad


def test_dense_adjoint_does_not_import_lineax(monkeypatch):
    """``linear_solver="dense"`` stays a pure-JAX path.

    Not an optionality assertion: dense is the triage escape hatch, so
    it must not drag in lineax (nor its equinox / jaxtyping import
    cost) even though lineax is now always installed.
    """
    monkeypatch.setitem(sys.modules, "lineax", None)
    g = _grad_k(_coupled(linear_solver="dense"))
    assert g != 0.0 and abs(g) < 1e3
