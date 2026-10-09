"""A group's diagnostics take a spectral norm from the singular values alone.

``_spectral_norm`` (``maddening.core.coupling.acceleration``) was
``jnp.linalg.norm(A, ord=2)``: the SVD with ``full_matrices=True``, whose
lowered form declares the full ``U`` and ``V^T`` as outputs although only
the singular values are read.  The bounds call it on ``(probes, k, n)``
tensors (``k <= 8`` Krylov directions against a group of ``n`` entries), so
a step with ``diagnostics=True`` asked for ``probes * n * n`` floats: 14 GB
at ``n = 3e4`` and 160 GB at ``n = 1e5`` in float32, and ``gm.step()``
raised ``RESOURCE_EXHAUSTED`` where the same group without diagnostics ran
in 0.6 GB.

Three things are held here:

* **the numbers did not move**: the norm equals ``jnp.linalg.norm(ord=2)``
  bit for bit on every shape the bounds hand it (and under ``vmap``), so
  no diagnostics slot changed (checked per push on both of CI's jaxlib
  lanes, where the SVD driver may differ);
* **nothing quadratic is declared**: the lowered program of the norm, and
  of a whole step with diagnostics on a pair of 3 and ``n`` entries,
  declares no value with ``n * n`` entries.  This is read from the program
  text, so it needs no memory limit and no large allocation, and it is the
  per-push form of the property;
* **the step runs** (slow): a pair of 3 and 1e5 entries steps with
  diagnostics on in a process whose address space is capped below what the
  old call asked for.
"""

from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import acceleration
from maddening.core.coupling.sparse_mapping import StaticSparseMapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.sparse_mapping_support import x64

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = Path(acceleration.__file__).resolve().parents[3]

M = 3
KEY = "grid+markers"


# ---------------------------------------------------------------------------
# The numbers did not move
# ---------------------------------------------------------------------------

#: Shapes the bounds hand the norm (square ``k x k`` and ``2k x 2k``, a
#: batch of wide ``k x n``, one tall ``n x k``), a nested batch, and a
#: matrix with no rows.  One compiled program per shape and call.
SHAPES = ((2, 2), (8, 8), (16, 16), (300, 8), (20, 8, 1000), (4, 3, 1, 9), (2, 0, 5))


def _matrices(shape, dtype, seed):
    """Four matrices of *shape*: random, graded over forty decades (per
    matrix of the batch), rank-deficient, and zero."""
    rng = np.random.default_rng(seed)
    plain = rng.standard_normal(shape).astype(dtype)
    decades = 20.0 if dtype == np.float64 else 6.0
    graded = (plain * 10.0 ** rng.uniform(-decades, decades, shape[:-2] + (1, 1))).astype(dtype)
    deficient = plain.copy()
    if deficient.shape[-2] > 1:
        deficient[..., 0, :] = deficient[..., 1, :]
    return {"random": plain, "graded": graded, "rank-deficient": deficient,
            "zero": np.zeros(shape, dtype)}


def _bits(value):
    value = np.asarray(value)
    return value.dtype, value.shape, value.tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_the_norm_is_the_two_norm_of_jax_bit_for_bit(dtype):
    """``_spectral_norm`` returns what ``jnp.linalg.norm(ord=2)`` returns, to
    the last bit, each in its own compiled program (and ours untraced):
    the same LAPACK routine on the same matrix, asked for the same
    values.  A float64 matrix needs x64."""
    def reference(A):
        return jnp.linalg.norm(A, ord=2, axis=(-2, -1))

    with x64(dtype == np.float64):
        for seed, shape in enumerate(SHAPES):
            ours, theirs = jax.jit(acceleration._spectral_norm), jax.jit(reference)
            for label, A in _matrices(shape, dtype, seed).items():
                assert _bits(ours(A)) == _bits(theirs(A)), (shape, label)
                assert _bits(acceleration._spectral_norm(A)) == _bits(ours(A)), (shape, label)


def test_the_norm_of_a_batch_is_the_norm_of_each_matrix():
    """Under ``vmap`` (how the bounds reach it for every probe) each matrix
    of a batch gets the value it gets alone, bit for bit."""
    A = _matrices((6, 8, 40), np.float32, 7)["graded"]
    batched = np.asarray(jax.vmap(acceleration._spectral_norm)(A))
    alone = np.stack([np.asarray(acceleration._spectral_norm(a)) for a in A])
    assert batched.tobytes() == alone.tobytes()
    assert np.all(batched > 0)


def test_a_matrix_that_is_not_finite_reads_nan_and_its_neighbours_are_untouched():
    """The guard in front of the SVD (``_lapack_input``) is unchanged: a
    matrix holding an ``inf`` or a NaN reads NaN, and the finite matrices
    beside it in the batch read what they read alone."""
    A = _matrices((4, 8, 40), np.float32, 11)["random"]
    spoiled = A.copy()
    spoiled[1, 2, 3] = np.inf
    spoiled[3, 0, 0] = np.nan
    out = np.asarray(acceleration._spectral_norm(spoiled))
    clean = np.asarray(acceleration._spectral_norm(A))
    assert np.isnan(out[1]) and np.isnan(out[3])
    assert out[[0, 2]].tobytes() == clean[[0, 2]].tobytes()


# ---------------------------------------------------------------------------
# Nothing quadratic is declared
# ---------------------------------------------------------------------------

_TENSOR = re.compile(r"tensor<((?:\d+x)+)[a-z]\w*>")


def largest_declared(text: str) -> tuple:
    """``(entries, type)`` of the largest tensor type a lowered program
    declares (StableHLO text: ``tensor<20x8x30000xf32>``)."""
    best = (0, "")
    for match in _TENSOR.finditer(text):
        entries = int(np.prod([int(d) for d in match.group(1).rstrip("x").split("x")],
                              dtype=np.int64))
        if entries > best[0]:
            best = (entries, match.group(0))
    return best


def test_the_parser_reads_the_types_of_a_lowered_program():
    """The instrument can fail: it finds the full-matrix SVD's ``n x n``
    output in the program of ``jnp.linalg.norm(ord=2)`` itself."""
    x = jnp.ones((20, 8, 3000), jnp.float32)
    text = jax.jit(lambda a: jnp.linalg.norm(a, ord=2, axis=(-2, -1))).lower(x).as_text()
    entries, declared = largest_declared(text)
    assert entries == 20 * 3000 * 3000, declared
    assert largest_declared("tensor<f32> tensor<4xi1>") == (4, "tensor<4xi1>")


def test_the_norm_declares_nothing_larger_than_its_matrix():
    """On a ``(probes, k, n)`` tensor the lowered norm declares no value
    with more entries than the tensor: no ``n x n`` factor."""
    x = jnp.ones((20, 8, 30000), jnp.float32)
    entries, declared = largest_declared(jax.jit(acceleration._spectral_norm).lower(x).as_text())
    assert entries == x.size, declared


class Markers(SimulationNode):
    """``f`` (3 values) ``<- 1 + 0.5 u``."""

    def initial_state(self):
        return {"f": jnp.zeros(M, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(M,), dtype=jnp.float32,
                                       default=jnp.zeros(M, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"f": 1.0 + 0.5 * boundary_inputs["u"]}

    def update_evaluations(self):
        return 1


class Grid(SimulationNode):
    """``u`` (``n`` values) ``<- 1 + F``."""

    def __init__(self, name, timestep, n):
        super().__init__(name, timestep)
        self._n = n

    def initial_state(self):
        return {"u": jnp.ones(self._n, jnp.float32)}

    def boundary_input_spec(self):
        return {"F": BoundaryInputSpec(shape=(self._n,), dtype=jnp.float32,
                                       default=jnp.zeros(self._n, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"u": 1.0 + boundary_inputs["F"]}

    def update_evaluations(self):
        return 1


def pair(n: int, norm: str, *, diagnostics: bool = True) -> GraphManager:
    """Three markers and a grid of *n* cells: the markers' values are
    scattered onto three cells and the grid is sampled back at them."""
    at = (np.arange(M) * (n // M))[:, None]
    weights = jnp.ones((M, 1), jnp.float32)
    gm = GraphManager()
    gm.add_node(Markers("markers", 1.0))
    gm.add_node(Grid("grid", 1.0, n))
    gm.add_edge("markers", "grid", "f", "F", mapping=StaticSparseMapping(
        at, weights, n_source=M, n_target=n, layout="scatter"))
    gm.add_edge("grid", "markers", "u", "u", mapping=StaticSparseMapping(at, weights, n_source=n))
    tolerance = {"rtol": 1e-4} if norm == "interface" else {"tolerance": 1e-4}
    gm.add_coupling_group(["markers", "grid"], convergence_norm=norm, max_iterations=50,
                          solver="ift", diagnostics=diagnostics, **tolerance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


class _Captured(Exception):
    """Raised in place of running a step: carries its program."""


def step_program_text(gm: GraphManager) -> str:
    """The lowered text of what ``gm.step()`` would run (taken where every
    compiled entry point hands its program over, as
    ``scripts/capture_step_programs.py`` does); nothing is compiled."""
    def grab(fn, *args):
        raise _Captured(fn, args)

    gm._call_surfacing_strict = grab            # noqa: SLF001
    try:
        gm.step()
    except _Captured as captured:
        fn, args = captured.args
        return fn.lower(*args).as_text()
    finally:
        del gm._call_surfacing_strict           # noqa: SLF001
    raise AssertionError("GraphManager.step did not reach _call_surfacing_strict")


@pytest.mark.parametrize("norm", ["interface", "l2"])
def test_a_step_with_diagnostics_declares_nothing_quadratic_in_the_group(norm):
    """The whole step program of a pair of 3 and ``n`` entries with
    diagnostics on: its largest declared value is linear in ``n`` (at most
    ``probes * k * (n + 3)`` with at most 8 of each), a hundred times and
    more below ``n * n`` at this size.  The old call declared
    ``8 x (n + 3) x (n + 3)``.  The analysis is in the program measured:
    it holds a basis of several directions over the group, which the
    same step without diagnostics does not."""
    n = 30000
    entries, declared = largest_declared(step_program_text(pair(n, norm)))
    assert entries <= 8 * 8 * (n + M), declared
    assert 100 * entries < n * n
    assert entries >= 4 * (n + M), declared
    plain, _ = largest_declared(step_program_text(pair(n, norm, diagnostics=False)))
    assert plain < 4 * (n + M)


# ---------------------------------------------------------------------------
# The step runs in a capped address space (slow)
# ---------------------------------------------------------------------------

_GIB = 1024 ** 3

#: The cap.  A jax CPU process reserves 3 to 5 GiB of address space before
#: it allocates anything of ours (thread stacks, LLVM, the allocator's
#: arenas: 4.5 GiB measured at the peak of this very step on 4 and on 8
#: cores, jax 0.10.2, 0.11.0 and 0.11.2), so the cap is not the step's
#: memory: it is a ceiling the old call's single request (160 GB) cannot
#: pass on any machine, whatever its overcommit setting.
ADDRESS_SPACE = 8 * _GIB

#: What the step itself may hold (peak resident set of the child).
RESIDENT_LIMIT_GB = 1.5

_STEP_UNDER_A_CAP = '''
    import json, resource, sys, warnings
    from tests.core.test_spectral_norm_takes_singular_values_only import KEY, pair

    n, norm = int(sys.argv[1]), sys.argv[2]
    gm = pair(n, norm)
    try:
        gm.step()
    except Exception as exc:                                    # noqa: BLE001
        out = {"raised": type(exc).__name__ + ": " + str(exc)[:200]}
    else:
        report = gm.coupling_diagnostics()[KEY]
        out = {name: bool(report[name])
               for name in ("converged", "spectral_usable", "gradient_bound_usable")}
        out["iterations"] = int(report["iterations"])
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print("RESULT" + json.dumps(out))
'''


def _capped(script: str, *arguments: str, address_space: int, timeout: int = 600) -> dict:
    """Run *script* in a fresh interpreter whose address space is capped
    (``RLIMIT_AS``, set before jax is imported) and return what it printed
    after ``RESULT``.  The child keeps at most four of this process's
    cores, so the address space its thread pools reserve does not depend
    on the machine."""
    def cap():
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))
        resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))

    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "PYTHONPATH": os.pathsep.join([str(SRC), str(REPO_ROOT),
                                          os.environ.get("PYTHONPATH", "")]),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "2"}
    done = subprocess.run([sys.executable, "-c", textwrap.dedent(script), *arguments],
                          capture_output=True, text=True, timeout=timeout, env=env,
                          preexec_fn=cap)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.rsplit("RESULT", 1)[1])


# Per push: tests/core/test_spectral_norm_takes_singular_values_only.py::test_a_step_with_diagnostics_declares_nothing_quadratic_in_the_group
@pytest.mark.slow
@pytest.mark.parametrize("norm", ["interface", "l2"])
def test_a_step_with_diagnostics_on_a_large_group_runs_in_a_capped_address_space(norm):
    """Three markers and 1e5 cells, one step with diagnostics on, in a
    process capped at 8 GiB of address space: it steps, converges and
    reports usable bounds, holding under 1.5 GB.  The old call asked for
    160 GB here and the step raised."""
    out = _capped(_STEP_UNDER_A_CAP, "100000", norm, address_space=ADDRESS_SPACE)
    assert "raised" not in out, out
    assert out["converged"] and out["spectral_usable"] and out["gradient_bound_usable"], out
    assert out["iterations"] > 1, out
    assert out["peak_rss_gb"] < RESIDENT_LIMIT_GB, out
