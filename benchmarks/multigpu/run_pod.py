#!/usr/bin/env python
"""Pod-side runner for the multi-GPU hardware session.

Runs *on the machine that has the GPUs* (or, with ``--dry-run``, on CPU
virtual devices) and writes one JSON file per goal under ``--out``.  It
never talks to a cloud provider: launching, copying results back and
tearing the pod down are the human's job (see ``README.md`` next to this
file).

Goals (``--goal``).  The first five are the sharding checklist, each a
comparison against a reference computed on the same machine (the
unsharded node, NumPy, or the refusal the library promises); the last
three are the timing goals::

    indivisible  checklist 5: a grid the mesh cannot split is refused by
                 the stencil wrapper (along spatial axis 0 and along axis
                 1, on the 1-D mesh, and on the pencil) and the pointwise
                 wrapper with both numbers named (the stencil refusal
                 saying the unstructured wrapper is not a way out for a
                 stencil node), and the unstructured wrapper takes an
                 uneven split of a node written for it and matches the
                 unsharded node.                    -> indivisible.json
    halo         checklist 2: ``halo_exchange`` on every mesh of
                 ``STENCIL_MESHES`` for every boundary mode and halo
                 widths 1 and 2, and ``exchange_unstructured`` under both
                 transports, forward and adjoint, against a NumPy
                 reference -- bit for bit.          -> halo.json
    coupled      checklist 6: one ``ShardedStencilNode`` member and one
                 replicated member in a single coupling group, coupled
                 through the field's domain integral and a non-uniform
                 grid-shaped input; forward rollout and ``jax.grad`` with
                 respect to trained parameters of both, under the default
                 solver and ``"fori"``, against the same group with the
                 sharded member replaced by the node it wraps and against
                 a float64 model.  Every mesh at the smallest size, the
                 pencil at the others.              -> coupled.json
    stencil      checklist 1 and 3 for ``ShardedStencilNode``: a 2-D field
                 that reads its sharded ``StaticArray`` in the halo, takes
                 a grid-shaped source and carries a domain integral, and a
                 D2Q9 lattice with a grid-shaped body force (its streaming
                 reads the halo corners); forward rollout and the adjoint
                 (initial state and a parameter) against the unsharded
                 node.  At the smallest size, the field under periodic,
                 ``"edge"`` (the wrapper's default) and Dirichlet ends and
                 the lattice, each on every mesh; periodic ends on the
                 pencil at the others.              -> stencil.json
    hybrid       checklist 4: ``HybridNode(ShardedStencilNode(inner))`` in
                 a graph, with a non-local correction and a grid-shaped
                 source, forward (the field and its domain integral) and
                 adjoint against ``HybridNode(inner)``; every mesh at the
                 smallest size, the pencil at the others.
                                                    -> hybrid.json
    exchange     NCCL ranking of the two unstructured halo-exchange
                 transports, ``all_to_all`` vs ``ppermute``, at 1e5-1e6
                 cells -- the measurement that decides whether ``ppermute``
                 becomes the default ``exchange=`` of
                 ``ShardedUnstructuredNode``.       -> exchange.json
    forward      checklist 1 for ``ShardedUnstructuredNode``: 1e6-cell
                 forward run on a real unstructured mesh (``--mesh``) or a
                 synthetic one, both transports, checked against the
                 unsharded node; and its cross-shard sum exactly, on a
                 field of ones (``EXACT_COUNT_CELLS``). -> forward.json
    gradient     checklist 3 for ``ShardedUnstructuredNode``: ``jax.grad``
                 through a sharded rollout (both transports) and through
                 the Jacobi-preconditioned ``sharded_cg`` against the
                 unsharded references, on a system both sides' solves
                 must have converged on.            -> gradient.json
    checklist    the five checklist goals, in the order above.
    all          all eight, in the order above.

The meshes (``STENCIL_MESHES``, as devices along spatial axes 0 and 1 over
``D``): ``"1d"`` (D x 1) and ``"1d-axis1"`` (1 x D), a 1-D mesh sharding
one spatial axis each; ``"2d-flat"`` (1 x D), a 2-D mesh whose two axes
have different sizes; and ``"2d"`` (2 x D/2), the pencil, which exchanges
two axes.  On four devices the pencil is 2 x 2, and on a mesh axis of two
devices a shard's left and right neighbour are one device: the 1-D meshes
give each spatial axis all four.  Every grid is non-square
(``field_shape``: ``nx = ny + D``), and every grid-shaped input differs on
every block of every mesh.

Every goal records ``checks`` (name, measured value, limit, the sense of
the comparison, passed) and ``passed`` in its JSON, prints a failed check
as ``CHECK FAILED``, and the runner exits 1 when any check failed.  A case
the device count cannot express (the 2-D pencil mesh below 4 or on an odd
count; a spatial axis no case splits over three or more devices, where a
halo from the wrong neighbour cannot show) is recorded as a check *not
run* and printed as ``CHECK NOT RUN``: it does not fail its goal when
another check of the goal ran and passed (the run exits 0 for it and the
goal reads ``INCOMPLETE``), but a goal none of whose checks ran is counted
as failed and the run exits 1, as for a goal with no checks; the halo,
stencil, hybrid and coupled goals record
those cases that way on such a count.  Each of the wrapper goals must fail
on a stencil wrapper broken in any of eight ways (a static's halos NaN,
grid-shaped inputs zeroed, domain integrals halved, domain integrals summed
over the first mesh axis only, sharded axes after the first zero-filled at
the global edges, halos taken from the wrong neighbour along spatial axis
1, ``shard_info``'s extent divided by the first mesh axis's size or read
off spatial axis 0):
``tests/cloud/multigpu/test_run_pod_seeded_faults.py`` seeds each into a
scratch copy of the library and requires it.  ``--summarise DIR`` reads
the JSON files back and prints
the checklist verdict, the per-goal tables and the ``ppermute`` vs
``all_to_all`` recommendation.  It re-derives every check's pass/fail from
its value, limit and sense rather than trusting the recorded flag, and
fails a record that disagrees.  It also refuses to take a file's own word
for what it checked (:func:`record_problems`): a file decides nothing
unless it is on the current ``SCHEMA_VERSION``, its ``n_devices`` is no
more than the devices its environment saw, its results hold every case
this runner runs for the file's own config, and its checks are exactly
the ones this runner derives from its results (``GOAL_CHECKS``), limits
from ``LIMITS`` included.  A checklist item reads ``CLOSED`` only when
every goal deciding it passed with every check run and every file valid,
on real GPUs, not a dry run, on at least ``MIN_DECIDING_DEVICES`` (4)
devices, and every deciding file records the same git commit.  Each item's
line names the commit its files record, and a directory whose files come
from more than one commit, or record none, prints a ``MIXED COMMITS``
warning naming each commit's items and files, and exits 4.  It does not
import JAX, so it works on a laptop without a usable jaxlib.

With ``--keep-going`` a goal that raises is recorded -- one failed ``goal
raised`` check, the exception under ``raised`` -- and the run goes on to
the next goal; without it the exception ends the run, with its traceback
and exit status ``EXIT_CRASHED`` (5).  A run this runner refuses (more
devices than are visible, a device count the checklist or the exchange
cannot use, a ``--mesh`` it cannot partition) ends with the reason as its
last line and exit status ``EXIT_REFUSED`` (2, as for an option it does
not take).  So a goal's exit status says which happened: 0 no check
failed, 1 a check failed (a ``CHECK FAILED`` line names it), 2 refused,
5 crashed; ``--summarise``'s statuses (0, 1, 3, 4) are its own.

Every timed callable receives inputs that were placed on the device mesh
once, with the ``NamedSharding`` the compiled executable expects, outside
the timed region (:func:`place_on_mesh`); the runner refuses to time
anything else, so no per-call reshard from device 0 is charged to a
transport.  Compile time is reported separately (``compile_s``) from the
steady-state timings, on both the sharded and the unsharded side.

The recommendation only counts rows from a real accelerator run (not
``--dry-run``) on at least ``MIN_DECIDING_DEVICES`` (4) devices; with
``--allow-fewer-devices`` (recorded in the JSON) a 2- or 3-device run may
decide.  A single device performs no exchange and never decides.  The flag
is about the ranking only: it never lets a checklist item close.

Sizes default to the hardware: on GPUs the cell counts are
``1e5, 3e5, 1e6``; on CPU (or with ``--dry-run``) they are a few hundred
cells so the whole script proves itself in a minute or two.
``--dry-run`` additionally pins JAX to the CPU backend with four virtual
host devices when no accelerator backend was requested, so::

    python benchmarks/multigpu/run_pod.py --goal all --dry-run --out /tmp/mg
    python benchmarks/multigpu/run_pod.py --summarise /tmp/mg

works on any laptop.  On the pod (``README.md`` has the session order,
the time boxes and the stop condition)::

    python benchmarks/multigpu/run_pod.py --goal all --out results/
    python benchmarks/multigpu/run_pod.py --goal forward --mesh mesh.npz --out results/

``--mesh`` accepts an ``.npz`` with an ``edges`` array of shape
``(n_edges, 2)`` (cell adjacency, global ids) and an optional
``partition`` array (``(n_cells,)`` device index, e.g. from PyMetis), or
a ``.npy`` holding just the edges.  Without ``partition`` the cells are
partitioned with PyMetis when importable, else by reverse-Cuthill-McKee
ordering (SciPy) cut into contiguous blocks, else by cell id.  A supplied
partition must use exactly ``--n-devices`` non-empty parts.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json
import math
from collections import Counter
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np


def _pre_import_setup(argv: list[str]) -> None:
    """Environment that must be in place before ``import jax``.

    ``--dry-run`` with no accelerator backend requested (``JAX_PLATFORMS``
    unset or ``cpu``) pins the CPU backend and gives it four virtual
    devices unless ``XLA_FLAGS`` already sets a device count.  It matches
    the literal ``--dry-run`` only, which is why :func:`parse_args` takes
    no abbreviations.
    """
    if "--dry-run" not in argv:
        return
    platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    if platforms not in ("", "cpu"):
        return
    os.environ["JAX_PLATFORMS"] = "cpu"
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_force_host_platform_device_count" not in flags:
        os.environ["XLA_FLAGS"] = (flags + " --xla_force_host_platform_device_count=4").strip()


_pre_import_setup(sys.argv)

SCHEMA_VERSION = 9

#: Exit status of a run this runner refused before a check decided anything
#: (argparse's own status for an option it does not take).
EXIT_REFUSED = 2

#: Exit status of a run that a goal's uncaught exception ended (without
#: ``--keep-going``).  It used to be 1, the status of a failed check.
EXIT_CRASHED = 5
METHODS = ("all_to_all", "ppermute")
MESH_AXIS = "devices"
GPU_CELLS = (100_000, 300_000, 1_000_000)
DRY_RUN_CELLS = (256, 1024)
#: The session's question is "on 4 GPUs".  A transport ranking may decide
#: on fewer with ``--allow-fewer-devices``; a checklist item never closes on
#: fewer, whatever that flag says: on 2 devices a shard's left and right
#: neighbour are the same device, so a halo that swaps directions passes
#: every halo and stencil check, and the 2-D pencil cases do not run.  Four
#: devices are not enough by themselves either: their pencil is 2 x 2, and
#: both of its axes have that two-device property.  So every goal that runs
#: the stencil wrapper or the halo exchange also runs a 1-D mesh along each
#: spatial axis (``STENCIL_MESHES``), where that axis has all four, and
#: records a spatial axis no case split over three or more devices as a
#: check not run (:func:`_neighbour_direction_not_run`).
MIN_DECIDING_DEVICES = 4
MIN_DEVICES_WITH_ESCAPE = 2     # --allow-fewer-devices: still needs a real exchange

#: Global-edge conditions the ``stencil`` goal runs, at its smallest size
#: (the largest runs ``"periodic"`` only, to stay inside its time box).
#: ``"edge"`` is ``ShardedStencilNode``'s default fill; ``"dirichlet"`` holds
#: a boundary input beyond the grid, applied by the node after the exchange.
STENCIL_BOUNDARIES = ("periodic", "edge", "dirichlet")
STENCIL_WALL = 1.5              # the Dirichlet value; unlike any edge cell of the field

#: The sharding checklist's goals, cheapest first, then the timing goals.
CHECKLIST_GOALS = ("indivisible", "halo", "coupled", "stencil", "hybrid")
TIMING_GOALS = ("exchange", "forward", "gradient")
ALL_GOALS = CHECKLIST_GOALS + TIMING_GOALS

#: Checklist item -> (what it claims, the goals whose checks decide it).
CHECKLIST = {
    1: ("sharded == unsharded, stencil and unstructured wrappers", ("stencil", "forward")),
    2: ("halo exchange at the shard and global boundaries", ("halo",)),
    3: ("sharded adjoint == unsharded adjoint", ("stencil", "gradient", "coupled")),
    4: ("HybridNode(ShardedStencilNode(inner))", ("hybrid",)),
    5: ("an indivisible grid is refused", ("indivisible",)),
    6: ("a sharded and a replicated member in one coupling group, with its adjoint",
        ("coupled",)),
}

#: The system the ``gradient`` goal's ``sharded_cg`` part solves, in
#: float32: ``(2 + CG_SHIFT) x[i] - x[i-1] - x[i+1] = b[i]``, the 1-D
#: Dirichlet Laplacian shifted on its diagonal, preconditioned by that
#: diagonal.  The one number the sharded and the unsharded operator, the
#: preconditioner and the host's residual all read.
#:
#: Why a shift: the unshifted operator has condition number
#: ``(2 (n + 1) / pi)**2``, 4e9 at 1e5 unknowns and 4e11 at 1e6, and no
#: float32 solve converges on it -- the goal solved it until schema 7 and
#: its true residual ``|b - A x| / |b|`` was 5e-2 at 1024 unknowns and 6e2
#: to 2e4 at the session's sizes, after all 3000 iterations.  With the
#: shift the condition number is ``(4 + s) / s`` at every size, 134 here.
#:
#: Why this one: the true residual a float32 solve can reach on this
#: operator is about ``eps32 * (4 + s) / s`` for a right-hand side in its
#: low modes, as the adjoint solve's is (measured 0.6 to 1.8 times that for
#: shifts from 1e-3 to 1e-1): 1.6e-5 here, measured 1.5e-5, one sixth of
#: ``CG_RTOL``, so both sides converge whatever a backend's rounding; and
#: the solve is not trivial: 55 to 58 iterations for each of the three
#: solves (``CG_SOLVES``), the same on both sides, at every size from 256
#: to 1e7 unknowns (CPU), so each derivative crosses the exchange about 110
#: times.  Hundreds of iterations a solve are not to be had in float32
#: together with convergence: the count grows as
#: ``sqrt(kappa) ln(2 / rtol) / 2`` while the floor ``eps32 * kappa`` has
#: to stay well under ``rtol`` (at a shift of 0.01 the floor is 4.8e-5, the
#: tolerance has to rise to 4e-4, and the count is 80 to 88).
CG_SHIFT = 0.03

#: The relative tolerance of that solve (``|r| <= CG_RTOL |b|``).  Until
#: schema 7 the goal asked for 1e-6, which float32 cannot reach on this
#: operator at any shift worth solving (a true residual of 2.6e-6 at a
#: shift of 0.1, where the solve takes 28 iterations at 1e5 unknowns).
CG_RTOL = 1e-4

#: The three linear solves behind the goal's two derivatives of
#: ``sum(x**2)``, ``A x = b``: the solve itself, the reverse pass's solve
#: for the gradient (right-hand side ``2 x``) and the forward pass's for the
#: tangent (right-hand side the direction ``v``).
CG_SOLVES = ("solve", "adjoint", "tangent")

#: The limit every parity check is held to, as a relative difference
#: ``max|sharded - reference| / max|reference|`` (componentwise for
#: gradient vectors), and why; and the limit on a Krylov solve's true
#: residual.  A dry run on CPU virtual devices is held to the same limits,
#: and its parity checks land one to three orders of magnitude inside
#: them.  Measured for every goal at the dry run's 256 and 1024 cells (the
#: record under tests/cloud/multigpu/run_pod_record/): the closest is the
#: D2Q9 lattice's forward velocity at 0.11 of the forward limit, then the
#: coupled and hybrid gradients at 0.03 of theirs, the rollouts at 0.01,
#: and ``sharded_cg`` at 0.001.  And for the ``gradient`` goal at the
#: session's 1e5, 3e5 and 1e6 cells: rollouts 1e-7 against 1e-5,
#: ``sharded_cg`` 4e-7 to 6e-7 against 4e-4 (8e-7 at 1e7 unknowns).  The
#: residual limit is of another kind: a converged solve reads just under
#: its tolerance, half the limit.  The few-ulp CPU numbers are pinned by
#: the unit tests under tests/cloud/multigpu/.
LIMITS = {
    # Pure data movement: the halo exchange and its adjoint on
    # integer-valued data, where every sum is exact.  And the ``forward``
    # goal's count of its cells: the total of a field of ones, which the
    # node leaves a field of ones (``EXACT_COUNT_CELLS``).
    "exact": 0.0,
    # float32 round-off across a rollout: the order in which XLA fuses the
    # sharded and the unsharded program may differ, and a cross-device
    # mean or loss reduces in another order.  Unchanged from schema 2.
    # The ``forward`` goal's ``total`` was held to it until schema 8 and is
    # not any more: that field's sum cancels (``sum|x| / |sum x|`` is 22
    # to 69,000 at the session's sizes and 1e7 cells), so eight float32
    # orders of addition of one and the same field differ by up to 3.5e-4
    # of the total (7.9e-4 at 1e7 cells) while every ghost row counted
    # into it moves it by 3.6e-4 at 1,000,000 cells: no limit on that
    # number tells the one from the other.  The field ``x`` is compared
    # entry by entry, and the reduction is checked exactly on a field of
    # ones (``EXACT_COUNT_CELLS``).
    "forward": 1e-5,
    # A rollout adjoint without a Krylov solve: the same round-off, once
    # more through the reverse pass.  Unchanged from schema 2.
    "gradient": 1e-5,
    # The coupling group's implicit-function-theorem adjoint is a GMRES
    # solve that stops at rtol = 100 ulp = 1.2e-5 of the right-hand side;
    # two correct solves (sharded and unsharded) may stop on different
    # iterations and land that far apart.  ~8x that tolerance.
    "coupled_gradient_ift": 1e-4,
    # "fori" differentiates straight through the iterates, no Krylov solve.
    "coupled_gradient_fori": 1e-5,
    # sharded_cg gradient and jvp.  Each is the result of two CG solves
    # that stop at rtol = CG_RTOL = 1e-4 and have converged (the next
    # limit holds them to it).  Against a float64 direct solve each side's
    # derivative is then within 1.2e-4 of exact at every size from 256 to
    # 1e7 unknowns, so two correct solves are within 2.4e-4 of each other
    # however each got there.  Measured, the sharded and the unsharded
    # side stop on the same iteration and differ by float32 round-off
    # (2e-7 to 8e-7 on CPU), and a stop one iteration apart moves a
    # derivative by 2.6e-5 at most.  4x the tolerance.  (1e-3 until
    # schema 7, with no reason given, on solves that had not converged:
    # the check read 5.3e-4, 5.7e-4 and 9.2e-4 at the session's sizes.)
    "krylov": 4 * CG_RTOL,
    # The true residual |rhs - A x| / |rhs| of each of those solves,
    # computed on the host in float64 from the result.  The loop stops
    # when its recursively updated residual is under rtol; the true one
    # differs from that by the float32 floor of the operator (measured
    # 4e-6 for the solve and its tangent, 1.5e-5 for the adjoint solve),
    # so a converged solve reads at most rtol plus that: measured 8.3e-5
    # to 1.0e-4.  2x the tolerance.  The unshifted operator reads 0.25 to
    # 23 at 1024 unknowns and up after the dry run's 300 iterations.
    "krylov_residual": 2 * CG_RTOL,
    # The coupled group's gradient against central differences of the
    # float64 model: the IFT adjoint is exact at the fixed point up to its
    # GMRES tolerance (1.2e-5) and "fori" differentiates the returned
    # iterate, which the group's tolerance (1e-7) keeps next to it.  The
    # float64 differences themselves are good to ~1e-9.
    "model_gradient": 1e-4,
}

#: The ``forward`` goal checks its cross-shard reduction on a field of
#: ones, at every size under this many cells.  ``NeighbourMeanNode`` leaves
#: a field of ones a field of ones, bit for bit (the mean of ones is 1,
#: ``(1 - w) + w`` is 1 at ``w = 0.5``, and a padded neighbour slot points
#: at the cell itself), so the step's ``total`` is a float32 sum of ones:
#: exact in ANY order of addition while the count is under 2**24, since
#: every partial sum is an integer float32 holds, and equal to the number
#: of cells.  One public ``update`` of the sharded node from such a field,
#: placed as the wrapper places its own initial state (the compiled step
#: is the one already timed: same shape, dtype and sharding), must
#: therefore return ``total == cells`` and a field still all ones, with a
#: limit of zero; and so must the unsharded node.  A shard left out of the
#: reduction or counted twice reads off by that shard's cells (25,122 to
#: 250,000 at the session's sizes), ghost rows counted off by their number
#: (1,532, 2,646 and 4,830 there; 15,276 at 10,004,569 cells), and pad rows
#: counted -- the node's ``own`` mask ignored -- off by half a cell each (a
#: pad row starts at zero and takes half of its shard's first cell in the
#: step): 1.5 cells at 100,489 and at 10,004,569 cells, which leave one
#: pad row on each of three shards.  300,304 and 1,000,000 cells divide by
#: four and have no pad row: the mask masks nothing there, and no check of
#: a total can see it ignored.  At this many cells or more (the stress
#: tail's 3e7 and 1e8) the count is recorded as a check not run.
EXACT_COUNT_CELLS = 2 ** 24

#: The parameters the ``hybrid`` and ``coupled`` goals differentiate with
#: respect to, as ``(node, params key)``; the check names spell them out.
HYBRID_WRITES = (("field", "diffusivity"), ("field", "exchange"))
COUPLED_WRITES = (("field", "diffusivity"), ("far", "conductance"), ("field", "exchange"))

#: Coupling group of the ``coupled`` goal.
COUPLED_MAX_ITERATIONS = 40
COUPLED_TOLERANCE = 1e-7
COUPLED_SOLVERS = ("ift", "fori")   # "ift" is CouplingGroup's default: passed as nothing

# JAX and the sharding helpers are imported on first use by
# ``_load_backend()`` (goal runners and ``environment()``), never at
# module scope: ``--summarise`` reads JSON only and ``recommend()`` is
# unit-tested without JAX.  The names are declared here so the rest of
# the module refers to ordinary module globals.
jax = jnp = lax = shard_map = P = NamedSharding = None
create_device_mesh = build_unstructured_partition = exchange_traffic = None
exchange_unstructured = gather_value = partition_value = None
jacobi_preconditioner = sharded_cg = ShardedUnstructuredNode = None
SimulationNode = StaticArray = NeighbourMeanNode = None
ShardedStencilNode = ShardedPointwiseNode = halo_exchange = None
GraphManager = HybridNode = Field2D = FarField = Pointwise2D = LBMNode = None


def _load_backend() -> None:
    """Import JAX and the MADDENING sharding helpers (idempotent)."""
    global jax, jnp, lax, shard_map, P, NamedSharding
    global create_device_mesh, build_unstructured_partition, exchange_traffic
    global exchange_unstructured, gather_value, partition_value
    global jacobi_preconditioner, sharded_cg, ShardedUnstructuredNode
    global SimulationNode, StaticArray, NeighbourMeanNode
    global ShardedStencilNode, ShardedPointwiseNode, halo_exchange
    global GraphManager, HybridNode, Field2D, FarField, Pointwise2D, LBMNode
    if jax is not None:
        return
    # Benchmarks share the GPU with nothing else; not preallocating keeps
    # the unsharded reference (device 0) and the sharded run from fighting
    # over memory at 1e6 cells.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax as _jax
    import jax.numpy as _jnp
    from jax import lax as _lax
    from jax import shard_map as _shard_map
    from jax.sharding import NamedSharding as _NamedSharding
    from jax.sharding import PartitionSpec as _P

    from maddening.cloud.multigpu import device_mesh as _dm
    from maddening.cloud.multigpu import halo as _halo
    from maddening.cloud.multigpu import halo_unstructured as _hu
    from maddening.cloud.multigpu import iterative_solver as _its
    from maddening.cloud.multigpu import sharded_node as _sn
    from maddening.cloud.multigpu import sharded_unstructured as _su
    from maddening.core import graph_manager as _gm
    from maddening.core import node as _node
    from maddening.core import params as _params
    from maddening.core import static_data as _sd
    from maddening.core.simulation import hybrid_node as _hn
    from maddening.nodes import lbm as _lbm

    jax, jnp, lax, shard_map, P, NamedSharding = _jax, _jnp, _lax, _shard_map, _P, _NamedSharding
    create_device_mesh = _dm.create_device_mesh
    build_unstructured_partition = _hu.build_unstructured_partition
    exchange_traffic = _hu.exchange_traffic
    exchange_unstructured = _hu.exchange_unstructured
    gather_value = _hu.gather_value
    partition_value = _hu.partition_value
    jacobi_preconditioner = _its.jacobi_preconditioner
    sharded_cg = _its.sharded_cg
    ShardedUnstructuredNode = _su.ShardedUnstructuredNode
    ShardedStencilNode = _sn.ShardedStencilNode
    ShardedPointwiseNode = _sn.ShardedPointwiseNode
    halo_exchange = _halo.halo_exchange
    GraphManager = _gm.GraphManager
    HybridNode = _hn.HybridNode
    LBMNode = _lbm.LBMNode
    SimulationNode = _node.SimulationNode
    StaticArray = _sd.StaticArray
    NeighbourMeanNode = _make_node_class(SimulationNode, StaticArray, jnp)
    Field2D, FarField, Pointwise2D = _make_checklist_classes(
        SimulationNode, StaticArray, _node.BoundaryInputSpec, _params.ParamSpec, jnp)


# ---------------------------------------------------------------------------
# Environment record
# ---------------------------------------------------------------------------


#: A commit as git names it: a full SHA-1, or a full SHA-256 in a
#: repository of that object format.
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


def _git_commit() -> str | None:
    """The commit ``git rev-parse HEAD`` names, or ``None`` when git fails
    or prints anything but a full SHA.  Git's exit status used to be
    ignored: in a repository with no commit yet (``git init`` in a tree
    synced without its ``.git``) git prints ``HEAD`` and exits 128, and
    every file recorded the commit ``"HEAD"``, which the commit gate took
    as one session's commit."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and _COMMIT_SHA.fullmatch(sha) else None


def _nvidia_smi() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def environment(*, dry_run: bool = False) -> dict:
    """What the numbers were measured on.

    A dry run never shells out to ``nvidia-smi``: its numbers come from
    CPU virtual devices and a GPU listed next to ``platform: cpu`` would
    only mislead the reader.
    """
    _load_backend()
    devices = jax.devices()
    import jaxlib  # noqa: PLC0415

    return {
        "hostname": socket.gethostname(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "platform": devices[0].platform,
        "devices": [str(d) for d in devices],
        "device_kinds": sorted({d.device_kind for d in devices}),
        "n_devices_visible": len(devices),
        "nvidia_smi": "skipped (dry run)" if dry_run else _nvidia_smi(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
        "git_commit": _git_commit(),
    }


def on_gpu() -> bool:
    _load_backend()
    return jax.devices()[0].platform == "gpu"


# ---------------------------------------------------------------------------
# Meshes, neighbour tables, partitions (NumPy only)
# ---------------------------------------------------------------------------


class MeshPartitionError(ValueError):
    """A ``--mesh`` partition that cannot be laid out on ``--n-devices``."""


def ring_edges(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.int32)
    return np.stack([i, (i + 1) % n], axis=1)


def _grid_side(n: int) -> int:
    """The side of the smallest square grid holding at least ``n`` cells (and
    at least 2 x 2).  It used to round ``sqrt(n)`` to the nearest integer,
    which put the session's "1e5-cell" exchange row at 316**2 = 99 856
    cells, below the 1e5 cells :func:`recommend` needs, so that row could
    never decide the transport."""
    side = math.isqrt(max(int(n), 0))
    if side * side < n:
        side += 1
    return max(side, 2)


def synthetic_cells(kind: str, n: int) -> int:
    """The cell count :func:`synthetic_mesh` builds for ``n``, without
    building it: ``n`` for a ring, and for a grid the smallest square
    holding at least ``n`` cells (``ceil(sqrt(n))**2``), so a synthetic mesh
    never has fewer cells than were asked for."""
    if kind == "ring":
        return n
    if kind == "grid":
        return _grid_side(n) ** 2
    raise ValueError(f"unknown synthetic mesh {kind!r} (ring|grid)")


def grid_edges(n: int) -> tuple[int, np.ndarray]:
    """4-neighbour lattice of at least ``n`` cells (the smallest square
    holding them, :func:`synthetic_cells`)."""
    side = _grid_side(n)
    ids = np.arange(side * side, dtype=np.int32).reshape(side, side)
    horiz = np.stack([ids[:, :-1].ravel(), ids[:, 1:].ravel()], axis=1)
    vert = np.stack([ids[:-1, :].ravel(), ids[1:, :].ravel()], axis=1)
    return side * side, np.concatenate([horiz, vert]).astype(np.int32)


def synthetic_mesh(kind: str, n: int) -> tuple[int, np.ndarray]:
    if kind == "ring":
        return n, ring_edges(n)
    if kind == "grid":
        return grid_edges(n)
    raise ValueError(f"unknown synthetic mesh {kind!r} (ring|grid)")


def load_mesh(path: str) -> tuple[int, np.ndarray, np.ndarray | None]:
    """``(n_cells, edges, partition-or-None)`` from ``.npz`` / ``.npy``."""
    p = Path(path)
    partition = None
    if p.suffix == ".npz":
        with np.load(p) as z:
            edges = np.asarray(z["edges"])
            if "partition" in z.files:
                partition = np.asarray(z["partition"], dtype=np.int32)
    else:
        edges = np.load(p)
    edges = np.asarray(edges, dtype=np.int32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"{path}: edges must be (n_edges, 2), got {edges.shape}")
    n = int(edges.max()) + 1 if partition is None else int(partition.size)
    return n, edges, partition


def check_file_partition(pa: np.ndarray, n_devices: int) -> np.ndarray:
    """A supplied partition must fill every one of ``n_devices`` shards.

    Fewer non-empty parts than devices would run silently with empty
    shards (and time nothing on them); more parts cannot be placed.
    """
    pa = np.asarray(pa, dtype=np.int32)
    if pa.size and (int(pa.min()) < 0 or int(pa.max()) >= n_devices):
        raise MeshPartitionError(
            f"--mesh partition uses part ids {int(pa.min())}..{int(pa.max())} but "
            f"--n-devices is {n_devices} (valid ids are 0..{n_devices - 1})")
    counts = np.bincount(pa, minlength=n_devices)
    empty = np.flatnonzero(counts == 0).tolist()
    if empty:
        non_empty = int((counts > 0).sum())
        raise MeshPartitionError(
            f"--mesh partition has {non_empty} non-empty part(s) but --n-devices is "
            f"{n_devices}: device(s) {empty} would own no cells; re-partition the mesh "
            f"for {n_devices} devices or pass --n-devices {non_empty}")
    return pa


def neighbour_table(n: int, edges: np.ndarray) -> np.ndarray:
    """``(n, max_degree)`` neighbour ids, rows padded with the cell itself."""
    u = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    v = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
    keep = u != v
    u, v = u[keep], v[keep]
    order = np.argsort(u, kind="stable")
    u, v = u[order], v[order]
    deg = np.bincount(u, minlength=n)
    max_deg = int(deg.max()) if deg.size else 1
    start = np.zeros(n, dtype=np.int64)
    start[1:] = np.cumsum(deg)[:-1]
    pos = np.arange(u.size) - start[u]
    tbl = np.repeat(np.arange(n, dtype=np.int32)[:, None], max(max_deg, 1), axis=1)
    tbl[u, pos] = v
    return tbl


def partition_cells(n: int, edges: np.ndarray, n_devices: int, how: str) -> tuple[np.ndarray, str]:
    """Global cell -> device index.  Returns ``(assignment, method_used)``."""
    if n_devices == 1:
        return np.zeros(n, dtype=np.int32), "single"
    if how in ("auto", "metis"):
        try:
            import pymetis  # noqa: PLC0415

            tbl = neighbour_table(n, edges)
            adjacency = [row[row != i] for i, row in enumerate(tbl)]
            _, parts = pymetis.part_graph(n_devices, adjacency=adjacency)
            return np.asarray(parts, dtype=np.int32), "metis"
        except ImportError:
            if how == "metis":
                raise
    if how in ("auto", "rcm"):
        try:
            from scipy.sparse import coo_matrix  # noqa: PLC0415
            from scipy.sparse.csgraph import reverse_cuthill_mckee  # noqa: PLC0415

            u, v = edges[:, 0], edges[:, 1]
            a = coo_matrix((np.ones(2 * u.size, np.float32),
                            (np.concatenate([u, v]), np.concatenate([v, u]))), shape=(n, n)).tocsr()
            order = reverse_cuthill_mckee(a, symmetric_mode=True)
            pa = np.empty(n, dtype=np.int32)
            pa[order] = (np.arange(n) * n_devices // n).astype(np.int32)
            return pa, "rcm"
        except ImportError:
            if how == "rcm":
                raise
    return (np.arange(n) * n_devices // n).astype(np.int32), "contiguous"


def slab_table(layout, tbl_global: np.ndarray) -> np.ndarray:
    """Per-cell neighbour table in *slab* indices of the owning shard.

    Row ``g`` holds, for each neighbour of global cell ``g``, its position
    in the ``(n_local_max + n_ghost_max)`` slab that ``pa[g]`` sees after
    the ghost exchange.  Vectorised: a ``(n_devices, n_cells)`` lookup is
    16 MB at 1e6 cells on 4 shards.
    """
    D = layout.n_devices
    n = tbl_global.shape[0]
    slab = np.full((D, n), -1, dtype=np.int32)
    for d in range(D):
        slab[d, layout.local_global_ids[d]] = np.arange(layout.n_local[d], dtype=np.int32)
        slab[d, layout.ghost_global_ids[d]] = (
            layout.n_local_max + np.arange(layout.n_ghost[d], dtype=np.int32))
    out = slab[layout.partition_assignment[:, None], tbl_global]
    if (out < 0).any():
        raise RuntimeError("neighbour not owned or ghosted -- layout/edges mismatch")
    return out


# The file mesh is loaded, and every (mesh, n_devices, method) partition
# computed, once per process: ``--goal all --mesh big.npz`` used to load
# and PyMetis-partition the same 1e6-cell mesh once per goal *and* per
# ``--cells`` entry.
_MESH_CACHE: dict = {}


def _mesh_source(args, n_hint: int):
    """``(n, edges, partition-or-None, source_label)`` for one size."""
    if args.mesh:
        key = ("file", str(args.mesh))
        if key not in _MESH_CACHE:
            _MESH_CACHE[key] = load_mesh(args.mesh)
        n, edges, pa = _MESH_CACHE[key]
        return n, edges, pa, f"file:{args.mesh}"
    n, edges = synthetic_mesh(args.synthetic, n_hint)
    return n, edges, None, args.synthetic


def _sizes(args) -> list[int]:
    """A file mesh has one size; the ``--cells`` list only sizes synthetic meshes."""
    return [args.cells[-1]] if args.mesh else list(args.cells)


def _partition_for(args, source: str, n: int, edges: np.ndarray,
                   pa: np.ndarray | None, n_devices: int) -> tuple[np.ndarray, str]:
    """Validated file partition, or a cached computed one."""
    if pa is not None:
        return check_file_partition(pa, n_devices), "file"
    key = ("partition", source, n, n_devices, args.partition)
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = partition_cells(n, edges, n_devices, args.partition)
    return _MESH_CACHE[key]


# ---------------------------------------------------------------------------
# The node under test
# ---------------------------------------------------------------------------


def _make_node_class(SimulationNode, StaticArray, jnp):  # noqa: N803 -- class factory
    class NeighbourMeanNode(SimulationNode):
        """``x <- (1 - w) x + w * mean_j x[nbr_j]`` over an arbitrary graph.

        The unsharded ``update`` gathers from a global neighbour table; the
        sharded ``update_padded`` gathers from the same table rewritten in
        slab indices (``slab_table``).  Padded neighbour slots point at the
        cell itself, so both paths compute exactly the same arithmetic and
        the sharded result is bit-comparable to the unsharded one.  ``total``
        (sum of ``x`` over owned cells) exercises the ``psum`` path.
        """

        def __init__(self, name: str, n: int, tbl: np.ndarray, *,
                     partition_assignment: np.ndarray | None = None,
                     weight: float = 0.5, timestep: float = 1.0) -> None:
            super().__init__(name=name, timestep=timestep)
            self._n = int(n)
            self._tbl = np.asarray(tbl, dtype=np.int32)
            self._pa = partition_assignment
            self._w = float(weight)

        def state_fields(self) -> list[str]:
            return ["x"]

        def domain_integral_fields(self) -> set[str]:
            return {"total"}

        @property
        def static_data(self) -> dict:
            if self._pa is None:
                return {}
            return {
                "nbr": StaticArray(self._tbl, replication="partition",
                                   partition_assignment=self._pa),
                "own": StaticArray(np.ones(self._n, np.float32), replication="partition",
                                   partition_assignment=self._pa),
            }

        def initial_state(self) -> dict:
            i = np.arange(self._n, dtype=np.float32)
            return {"x": jnp.asarray(np.sin(0.01 * i) + 0.1 * np.cos(0.37 * i))}

        def _relax(self, x, tbl):
            g = jnp.take(x, tbl, axis=0).mean(axis=-1)
            return (1.0 - self._w) * x[: tbl.shape[0]] + self._w * g

        def update(self, state, boundary_inputs, dt):
            new = self._relax(state["x"], jnp.asarray(self._tbl))
            return {"x": new, "total": jnp.sum(new)}

        def update_padded(self, state_padded, boundary_inputs, dt, *, static_padded=None,
                          shard_info=None):
            n_local = shard_info[0][1]
            new = self._relax(state_padded["x"], static_padded["nbr"][:n_local])
            own = static_padded["own"][:n_local]
            return {"x": new, "total": jnp.sum(new * own)}

    return NeighbourMeanNode


def build_pair(n: int, edges: np.ndarray, mesh, pa: np.ndarray, exchange: str):
    """Unsharded reference node, its sharded twin, and the layout."""
    _load_backend()
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges,
                                          n_devices=int(mesh.shape[MESH_AXIS]))
    tbl = neighbour_table(n, edges)
    ref = NeighbourMeanNode("mesh", n, tbl)
    inner = NeighbourMeanNode("mesh", n, slab_table(layout, tbl), partition_assignment=pa)
    return ref, ShardedUnstructuredNode(inner, mesh, layout, exchange=exchange), layout


def _make_checklist_classes(SimulationNode, StaticArray, BoundaryInputSpec,  # noqa: N803
                            ParamSpec, jnp):
    """The nodes of the checklist goals (class factory: JAX is imported late)."""

    class Field2D(SimulationNode):
        """A 2-D field solver: variable-coefficient diffusion relaxing towards
        an ambient field, with a source.

        ``f <- f + dt * (diffusivity * div(k grad f) + exchange * (ambient - f)
        + smooth(source))``, in flux form on the 5-point stencil.  Each part
        of it is there to make one path of ``ShardedStencilNode`` carry a
        value into the answer, so that a fault on that path moves the
        result (the ``stencil``, ``hybrid`` and ``coupled`` goals, and the
        seeded-fault test that runs them against a broken wrapper):

        * the conductance ``k`` of a face is the mean of the ``mask`` of
          the two cells it joins, so every step reads ``mask`` one cell
          into its halo.  ``mask`` is a ``StaticArray(replication="shard")``
          along ``mask_axis`` (0, or 1 where the wrapper shards axis 1
          only): the wrapper halo-exchanges it along that axis
          (periodically under ``"periodic"``, the edge cell repeated
          otherwise), and the node fills the other axis itself by the same
          rule, taking its own slice out of the whole static by
          ``shard_info`` when that axis is sharded too (a 2-D mesh) --
          so a wrong offset or extent in ``shard_info`` moves the answer;
        * ``source``, a grid-shaped boundary input, is read through a
          5-point smoothing, so its halo counts as well as its interior;
        * ``ambient`` is a scalar or a grid-shaped boundary input;
        * ``averages``, ``[mean f, mean f**2]`` of the new field, is a
          declared domain integral carried in the state (the wrapper sums
          the shards' partial sums across the mesh).  Its name sorts
          before ``f``, as the wrapper's state dict is iterated in sorted
          order inside ``shard_map``: the local extent in ``shard_info``
          must come from the grid field, not from the integral.

        ``update`` pads the whole field itself, ``update_padded`` receives
        the wrapper's halos, and both then run the same arithmetic:
        sharded, the node is the unsharded node up to where XLA places the
        operations.

        ``boundary`` is what lies beyond the grid, on both axes
        (``STENCIL_BOUNDARIES``): ``"periodic"`` wraps and ``"edge"``
        repeats the edge cell -- the wrapper's fill of the same name does
        it, sharded -- and ``"dirichlet"`` holds the boundary input
        ``wall`` there: the wrapper fills ``"edge"`` and ``update_padded``
        overwrites the halos at the *global* edges, finding them on each
        sharded axis from ``shard_info``.
        """

        def __init__(self, name: str, ny: int, nx: int, *, diffusivity: float = 0.2,
                     exchange: float = 0.0, timestep: float = 0.1,
                     boundary: str = "periodic", mask_axis: int = 0) -> None:
            super().__init__(name=name, timestep=timestep, diffusivity=diffusivity,
                             exchange=exchange)
            if boundary not in STENCIL_BOUNDARIES:
                raise ValueError(f"boundary {boundary!r} is not one of {STENCIL_BOUNDARIES}")
            if mask_axis not in (0, 1):
                raise ValueError(f"mask_axis {mask_axis!r} is not 0 or 1")
            self.boundary = boundary
            self._mask_axis = int(mask_axis)
            self._shape = (int(ny), int(nx))
            j = np.arange(ny, dtype=np.float32)[:, None]
            i = np.arange(nx, dtype=np.float32)[None, :]
            mask = (0.75 + 0.25 * np.sin(0.37 * j) * np.cos(0.11 * i)).astype(np.float32)
            # Built once: the wrapper snapshots static_data at construction.
            self._static = {"mask": StaticArray(value=mask, replication="shard",
                                                shard_axis=self._mask_axis)}

        def halo_width(self) -> dict:
            return {0: 1, 1: 1}

        def state_fields(self) -> list:
            return ["f"]

        def domain_integral_fields(self) -> set:
            return {"averages"}

        @property
        def static_data(self) -> dict:
            return self._static

        def _averages(self, f):
            """``[sum f, sum f**2] / n_cells`` of ``f`` -- this block's share of the
            global means when ``f`` is one block."""
            n = self._shape[0] * self._shape[1]
            return jnp.stack([jnp.sum(f), jnp.sum(f * f)]) / n

        def initial_state(self) -> dict:
            ny, nx = self._shape
            y = np.linspace(0.0, 1.0, ny, endpoint=False, dtype=np.float32)[:, None]
            x = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float32)[None, :]
            f = jnp.asarray((1.0 + np.sin(2 * np.pi * y) * np.cos(4 * np.pi * x)
                             + 0.25 * np.cos(6 * np.pi * (x + y))).astype(np.float32))
            return {"f": f, "averages": self._averages(f)}

        def boundary_input_spec(self) -> dict:
            spec = {"ambient": BoundaryInputSpec(shape=(), description="far-field value, "
                                                 "a scalar or one per cell"),
                    "source": BoundaryInputSpec(shape=self._shape,
                                                description="per-cell source")}
            if self.boundary == "dirichlet":
                spec["wall"] = BoundaryInputSpec(shape=(), description="value beyond the grid")
            return spec

        def param_specs(self) -> dict:
            return {**super().param_specs(),
                    "diffusivity": ParamSpec(bounds=(0.0, None), transform="log"),
                    "exchange": ParamSpec(bounds=(0.0, None), transform="log")}

        @property
        def _pad_mode(self) -> str:
            """How ``update`` fills the halo of the field and of a grid-shaped
            input: as the wrapper's fill does (``"edge"`` under Dirichlet)."""
            return {"periodic": "wrap", "edge": "edge", "dirichlet": "edge"}[self.boundary]

        @property
        def _static_mode(self) -> str:
            """How a sharded static's halo is filled: the wrapper's rule."""
            return "wrap" if self.boundary == "periodic" else "edge"

        @staticmethod
        def _new(f_pad, mask_pad, ambient, source_pad, p, dt):
            f = f_pad[1:-1, 1:-1]
            m = mask_pad[1:-1, 1:-1]
            div = sum(0.5 * (m + mask_pad[sl]) * (f_pad[sl] - f)
                      for sl in ((slice(None, -2), slice(1, -1)), (slice(2, None), slice(1, -1)),
                                 (slice(1, -1), slice(None, -2)), (slice(1, -1), slice(2, None))))
            out = f + dt * (p["diffusivity"] * div + p["exchange"] * (ambient - f))
            if source_pad is not None:
                s = (0.5 * source_pad[1:-1, 1:-1]
                     + 0.125 * (source_pad[:-2, 1:-1] + source_pad[2:, 1:-1]
                                + source_pad[1:-1, :-2] + source_pad[1:-1, 2:]))
                out = out + dt * s
            return out

        def _hold_walls(self, f_pad, wall, shard_info):
            """Dirichlet halos: the rows and columns beyond a *global* edge.
            A block holds an edge of a sharded axis when ``shard_info`` puts
            it there; an axis not in ``shard_info`` is whole on the block."""
            info = shard_info or {}
            held = []
            for axis in (0, 1):
                local = f_pad.shape[axis] - 2
                offset, extent = info.get(axis, (0, local))
                held.append((offset == 0, offset + extent == self._shape[axis]))
            (top, bottom), (left, right) = held
            f_pad = f_pad.at[0, :].set(jnp.where(top, wall, f_pad[0, :]))
            f_pad = f_pad.at[-1, :].set(jnp.where(bottom, wall, f_pad[-1, :]))
            f_pad = f_pad.at[:, 0].set(jnp.where(left, wall, f_pad[:, 0]))
            return f_pad.at[:, -1].set(jnp.where(right, wall, f_pad[:, -1]))

        def update(self, state, boundary_inputs, dt, *, params=None):
            p = self.params if params is None else {**self.params, **params}
            ambient = jnp.asarray(boundary_inputs.get("ambient", 0.0), jnp.float32)
            source = boundary_inputs.get("source")
            source_pad = None if source is None else jnp.pad(
                jnp.asarray(source, jnp.float32), 1, mode=self._pad_mode)
            f_pad = jnp.pad(state["f"], 1, mode=self._pad_mode)
            if self.boundary == "dirichlet":
                wall = jnp.asarray(boundary_inputs.get("wall", 0.0), jnp.float32)
                f_pad = self._hold_walls(f_pad, wall, None)
            mask_pad = jnp.pad(jnp.asarray(self._static["mask"].value), 1,
                               mode=self._static_mode)
            new = self._new(f_pad, mask_pad, ambient, source_pad, p, dt)
            return {"f": new, "averages": self._averages(new)}

        def update_padded(self, state_padded, boundary_inputs, dt, *, static_padded=None,
                          shard_info=None, params=None):
            p = self.params if params is None else {**self.params, **params}
            ambient = jnp.asarray(boundary_inputs.get("ambient", 0.0), jnp.float32)
            if ambient.ndim == 2:                  # grid-shaped: arrives halo-padded
                ambient = ambient[1:-1, 1:-1]
            source_pad = boundary_inputs.get("source")
            f_pad = state_padded["f"]
            if self.boundary == "dirichlet":
                wall = jnp.asarray(boundary_inputs.get("wall", 0.0), jnp.float32)
                f_pad = self._hold_walls(f_pad, wall, shard_info)
            # The wrapper halo-exchanges the static along mask_axis only; it
            # is whole along the other axis on every device, so fill that
            # axis here and take this block's slice of it (all of it unless
            # that axis is sharded too).
            other = 1 - self._mask_axis
            pad = [(0, 0), (0, 0)]
            pad[other] = (1, 1)
            mask_pad = jnp.pad(static_padded["mask"], pad, mode=self._static_mode)
            width = f_pad.shape[other]
            offset = (shard_info or {}).get(other, (0, width - 2))[0]
            mask_pad = lax.dynamic_slice_in_dim(mask_pad, offset, width, axis=other)
            new = self._new(f_pad, mask_pad, ambient, source_pad, p, dt)
            return {"f": f_pad.at[1:-1, 1:-1].set(new), "averages": self._averages(new)}

    class FarField(SimulationNode):
        """A replicated scalar solver: ``u <- u + dt * conductance * (field_mean - u)``."""

        def __init__(self, name: str = "far", *, conductance: float = 2.0,
                     timestep: float = 0.1) -> None:
            super().__init__(name=name, timestep=timestep, conductance=conductance)

        def state_fields(self) -> list:
            return ["u"]

        def initial_state(self) -> dict:
            return {"u": jnp.asarray(0.2, jnp.float32)}

        def boundary_input_spec(self) -> dict:
            return {"field_mean": BoundaryInputSpec(shape=(), description="mean of the field")}

        def param_specs(self) -> dict:
            return {**super().param_specs(),
                    "conductance": ParamSpec(bounds=(0.0, None), transform="log")}

        def update(self, state, boundary_inputs, dt, *, params=None):
            p = self.params if params is None else {**self.params, **params}
            mean = jnp.asarray(boundary_inputs.get("field_mean", 0.0), jnp.float32)
            return {"u": state["u"] + dt * p["conductance"] * (mean - state["u"])}

    class Pointwise2D(SimulationNode):
        """A pointwise ``(rows, cols)`` decay, for the pointwise wrapper's refusal."""

        def __init__(self, rows: int, cols: int) -> None:
            super().__init__(name="pointwise", timestep=0.1)
            self._shape = (int(rows), int(cols))

        def halo_width(self) -> dict:
            return {}

        def state_fields(self) -> list:
            return ["x"]

        def initial_state(self) -> dict:
            return {"x": jnp.ones(self._shape, jnp.float32)}

        def update(self, state, boundary_inputs, dt):
            return {"x": state["x"] * (1.0 - dt)}

    return Field2D, FarField, Pointwise2D


# ---------------------------------------------------------------------------
# Placement and timing helpers
# ---------------------------------------------------------------------------


def mesh_sharding(mesh):
    """The ``NamedSharding`` every ``shard_map`` here compiles its inputs to."""
    return NamedSharding(mesh, P(MESH_AXIS))


def layout_slab(value, layout) -> np.ndarray:
    """Global-order ``value`` -> ``(D * n_local_max, ...)`` partition-layout host array."""
    per = partition_value(value=np.asarray(value), layout=layout)
    return per.reshape((layout.n_devices * layout.n_local_max,) + per.shape[2:])


def place_on_mesh(value, mesh):
    """One ``device_put`` with the mesh sharding -- done outside every timed region.

    An uncommitted array (``jnp.asarray`` of host data) lives on device 0
    and would be scattered to the mesh on *every* call of a compiled
    ``shard_map``, inside the timed window; that scatter is not exchange.
    """
    return jax.device_put(jnp.asarray(value), mesh_sharding(mesh))


def require_presharded(arrays, mesh, what: str, compiled=None) -> bool:
    """Refuse to time inputs that would be resharded on each call.

    Every array must already carry the mesh sharding; when the
    ``compiled`` executable is given (positional, non-pytree arguments)
    its input shardings must agree too.  Returns ``True`` for the JSON.
    """
    expected = mesh_sharding(mesh)
    for i, a in enumerate(arrays):
        if not a.sharding.is_equivalent_to(expected, a.ndim):
            raise RuntimeError(
                f"{what}: timed input {i} is placed as {a.sharding}, not {expected}; "
                "a timed call would reshard it from device 0 -- place it once with "
                "place_on_mesh() outside the timed region")
    if compiled is not None:
        for i, (a, s) in enumerate(zip(arrays, compiled.input_shardings[0])):
            if not s.is_equivalent_to(a.sharding, a.ndim):
                raise RuntimeError(
                    f"{what}: compiled input sharding {s} differs from the placed "
                    f"input {i} ({a.sharding})")
    return True


def timed(fn, *, warmup: int, repeats: int) -> dict:
    """``fn()`` must return something ``block_until_ready`` can wait on."""
    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - t0) * 1e3)
    return {
        "warmup": warmup,
        "repeats": repeats,
        "ms": samples,
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
    }


def compile_seconds(jitted, *args) -> tuple[float, object]:
    """Ahead-of-time compile of ``jitted`` for ``args``; primes the jit cache.

    Returns ``(seconds, compiled)``; the following ``jitted(*args)`` calls
    hit the cache, so compile time never leaks into the steady-state
    timings and never includes an execution.
    """
    t0 = time.perf_counter()
    compiled = jitted.lower(*args).compile()
    return time.perf_counter() - t0, compiled


def _diff(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
    return {
        "max_abs": max_abs,
        "max_rel": (max_abs / scale) if scale > 0 else max_abs,
        "reference_scale": scale,
        "finite": bool(np.all(np.isfinite(a))),
    }


def _rel_each(a, b) -> float:
    """Largest componentwise ``|a - b| / |b|`` (for gradients of unlike size)."""
    a = np.atleast_1d(np.asarray(a, np.float64))
    b = np.atleast_1d(np.asarray(b, np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(b != 0, np.abs(a - b) / np.abs(b), np.abs(a - b))
    return float(np.max(rel)) if rel.size else 0.0


#: How a check's ``value`` is compared with its ``limit``.  Recorded with
#: every check (schema 4) so that ``--summarise`` can re-derive pass/fail
#: instead of trusting the recorded flag.
SENSES = ("<=", "==")


def expected_pass(c: dict) -> bool | None:
    """Pass/fail of a check, re-derived from its ``value``, ``limit`` and
    ``sense``; ``None`` when the record cannot be judged (an unknown sense,
    or a value of the wrong type for its sense).

    ``"<="``: a finite number no greater than the limit.  ``"=="``: a
    yes/no value equal to the limit (``true``).  A schema-3 record has no
    ``sense``; it is read from the limit's type, which is how schema 3
    wrote them.  The recording side (:func:`check`, :func:`check_that`)
    uses this same rule, so a record that disagrees with it was not
    written by this runner as it stands.
    """
    value, limit = c.get("value"), c.get("limit")
    sense = c.get("sense", "==" if isinstance(limit, bool) else "<=")
    if sense == "==":
        if not (isinstance(value, bool) and isinstance(limit, bool)):
            return None
        return value == limit
    if sense == "<=":
        if (isinstance(value, bool) or isinstance(limit, bool)
                or not isinstance(value, (int, float))
                or not isinstance(limit, (int, float))):
            return None
        return math.isfinite(value) and value <= limit
    return None


def check(name: str, value, limit: float) -> dict:
    """A measured ``value`` that must not exceed ``limit``; non-finite fails."""
    c = {"name": name, "value": float(value), "limit": float(limit), "sense": "<="}
    c["passed"] = bool(expected_pass(c))
    return c


def check_that(name: str, ok, detail: str = "") -> dict:
    """A yes/no check (a refusal raised, an array partitioned, ...)."""
    c = {"name": name, "value": bool(ok), "limit": True, "sense": "==", "detail": detail}
    c["passed"] = bool(expected_pass(c))
    return c


def check_not_run(name: str, why: str) -> dict:
    """A check this run could not make (too few devices for the case, say).

    Recorded rather than silently skipped, so that the goal cannot read
    as complete: ``passed`` is ``False`` for any reader that looks only at
    that, and ``--summarise`` keeps the checklist items the goal decides
    open.  The runner reports it as ``NOT RUN`` rather than as a failure --
    it exits 0 for it -- when at least one other check of the goal ran and
    passed; a goal none of whose checks ran is counted as failed and the
    run exits 1, as for a goal with no checks.
    """
    return {"name": name, "value": None, "limit": None, "sense": None, "passed": False,
            "not_run": True, "detail": why}


def finish_checks(out: dict, checks: list) -> dict:
    """Store ``checks`` and ``passed``: every check ran and passed, and there
    was at least one (no checks is not a pass, and neither is a check not run)."""
    out["checks"] = checks
    out["passed"] = bool(checks) and all(c["passed"] for c in checks)
    return out


#: The name of the one check a goal that raised records (``--keep-going``).
GOAL_RAISED_CHECK = "goal raised"


def goal_raised_check(raised: dict) -> dict:
    """The failed check a goal that raised records, derived from its
    ``raised`` entry (the exception's type and message)."""
    return check_that(GOAL_RAISED_CHECK, False,
                      detail=f"{raised.get('type')}: {raised.get('message')}")


def record_goal_raised(doc: dict, exc: BaseException) -> dict:
    """Make *doc* the record of a goal that raised *exc*: no results, one
    failed ``goal raised`` check, and the exception under ``raised``.

    For ``--keep-going``, which used to stop at the first goal that raised
    (only a goal whose *checks* failed was kept going past), leaving every
    later goal not run -- a seeded wrapper fault that makes ``shard_map``
    refuse the step recorded nothing for three of the five checklist goals.
    The record is valid (:func:`record_problems` derives the same single
    check from ``raised``), so ``--summarise`` reads the goal ``FAIL`` and
    names the exception, rather than ``not run``.
    """
    doc["raised"] = {
        "type": type(exc).__name__,
        "message": str(exc)[:4000],
        "traceback": "".join(traceback.format_exception(exc))[-8000:],
    }
    doc["results"] = []
    return finish_checks(doc, [goal_raised_check(doc["raised"])])


def check_status(c: dict) -> str:
    """``"passed"``, ``"failed"``, ``"not run"``, or ``"inconsistent"`` -- the
    recorded ``passed`` disagrees with what the value, limit and sense say,
    or the record cannot be judged.  ``"inconsistent"`` counts as a failure."""
    if c.get("not_run"):
        return "not run" if c.get("passed") is False else "inconsistent"
    want = expected_pass(c)
    recorded = c.get("passed")
    if want is None or not isinstance(recorded, bool) or recorded != want:
        return "inconsistent"
    return "passed" if want else "failed"


def _parity_checks(prefix: str, diff: dict, limit: float) -> list:
    return [check(f"{prefix} max_rel", diff["max_rel"], limit),
            check_that(f"{prefix} finite", diff["finite"])]


def _mesh_for(n_devices: int):
    _load_backend()
    return create_device_mesh(shape=(n_devices,))


def _is_partitioned(arr, n_devices: int) -> bool:
    """``arr`` is split across ``n_devices`` devices, not replicated on them."""
    sharding = arr.sharding
    return len(sharding.device_set) == n_devices and not sharding.is_fully_replicated


def field_shape(cells: int, n_devices: int) -> tuple:
    """``(ny, nx)`` of about ``cells`` cells, both a multiple of the devices,
    so that one shape splits on every mesh of :data:`STENCIL_MESHES`.

    Never square: ``nx = ny + D``.  On a square grid every per-axis
    quantity of the wrapper is the same number for both spatial axes, so a
    wrapper that reads one axis's extent for the other's (a ``shard_info``
    taken off spatial axis 0 for every axis, say) passes every goal.  Until
    schema 6 the grids were square, and that fault passed even on 8
    devices.
    """
    side = max(int(round(math.sqrt(cells))), 2 * n_devices)
    ny = -(-side // n_devices) * n_devices
    return ny, ny + n_devices


#: The meshes the stencil wrapper runs on, each named for what it shards.
#: Over ``D`` devices, as (devices along spatial axis 0, along axis 1):
#:
#: * ``"1d"``, ``(D, 1)``: a 1-D mesh sharding spatial axis 0; axis 1 is
#:   whole on each device and filled by the wrapper as unsharded;
#: * ``"1d-axis1"``, ``(1, D)``: the same 1-D mesh sharding spatial axis 1
#:   (``axis_map={MESH_AXIS: 1}``); axis 0 is the unsharded one;
#: * ``"2d-flat"``, ``(1, D)``: a 2-D mesh whose first axis has one device,
#:   both spatial axes in its ``axis_map`` -- two mesh axes of different
#:   sizes (the wrapper's one-device exchange on axis 0, all ``D`` on axis
#:   1), which the pencil on four devices cannot give;
#: * ``"2d"``, ``(2, D/2)``: the pencil mesh, sharding both, so the wrapper
#:   exchanges two axes and a node reading a corner reads one that crossed
#:   two.
#:
#: On four devices the pencil is 2 x 2: both of its axes have two devices,
#: where a shard's left and right neighbour are the same device and a halo
#: taken from the wrong neighbour cannot show.  The two 1-D meshes give
#: each spatial axis a four-device axis of its own (see
#: :func:`_neighbour_direction_not_run`).
STENCIL_MESHES = ("1d", "1d-axis1", "2d-flat", "2d")


def stencil_mesh(label: str, n_devices: int) -> tuple:
    """``(mesh, axis_map, mesh_shape)`` of mesh ``label`` over ``n_devices``."""
    _load_backend()
    shape = _mesh_shape_of(label, n_devices)
    if label == "1d":
        return create_device_mesh(shape=(n_devices,)), {MESH_AXIS: 0}, shape
    if label == "1d-axis1":
        return create_device_mesh(shape=(n_devices,)), {MESH_AXIS: 1}, shape
    if label == "2d" and not _pencil_mesh_fits(n_devices):
        raise ValueError(_pencil_mesh_needs(n_devices))
    return create_device_mesh(shape=shape), {"spatial_y": 0, "spatial_z": 1}, shape


def meshes_that_fit(n_devices: int) -> tuple:
    """The ``STENCIL_MESHES`` this device count can build: every one but the
    pencil, which needs an even count of at least 4."""
    return STENCIL_MESHES if _pencil_mesh_fits(n_devices) else STENCIL_MESHES[:-1]


def graph_mesh(n_devices: int) -> str:
    """The one mesh the larger sizes of the ``stencil``, ``hybrid`` and
    ``coupled`` goals run on: the pencil where it fits -- it exchanges
    two axes -- and the 1-D mesh otherwise, with the pencil recorded as not
    run.  The smallest size runs every mesh (:func:`graph_cases`)."""
    return "2d" if _pencil_mesh_fits(n_devices) else "1d"


def graph_cases(cells, n_devices: int) -> list:
    """``(cells, mesh)`` cases of the ``hybrid`` and ``coupled`` goals: every
    mesh this device count can build at the smallest size, so that each
    spatial axis is sharded on all ``D`` devices somewhere, and the
    :func:`graph_mesh` at the others."""
    smallest = min(cells)
    return [(n, mesh) for n in cells
            for mesh in (meshes_that_fit(n_devices) if n == smallest
                         else (graph_mesh(n_devices),))]


def _mesh_shape_of(label: str, n_devices: int) -> tuple:
    """Devices along (spatial axis 0, spatial axis 1) on mesh ``label``."""
    shapes = {"1d": (n_devices, 1), "1d-axis1": (1, n_devices), "2d-flat": (1, n_devices),
              "2d": (2, n_devices // 2)}
    if label not in shapes:
        raise ValueError(f"unknown mesh {label!r} (one of {STENCIL_MESHES})")
    return shapes[label]


def _neighbour_direction_not_run(mesh_shapes, claim: str) -> list:
    """A check not run for each spatial axis that no case split over at
    least three devices.

    On a mesh axis of one or two devices a shard's left and right neighbour
    are the same device (the two ring permutations are the same pair), so
    a wrapper or exchange that takes a halo from the wrong neighbour along
    that spatial axis passes every comparison.  ``mesh_shapes`` are the
    cases' ``(devices along axis 0, devices along axis 1)``.
    """
    shapes = [tuple(s) for s in mesh_shapes]
    out = []
    for axis in (0, 1):
        most = max((s[axis] for s in shapes), default=0)
        if most < 3:
            out.append(check_not_run(
                f"spatial axis {axis} split over >= 3 devices: {claim} with left and right "
                "neighbours on different devices, so a halo taken from the wrong one shows",
                f"needs a mesh axis of >= 3 devices on spatial axis {axis}; this run's cases "
                f"split it over at most {most}"))
    return out


def _host(tree):
    return jax.tree.map(lambda a: np.asarray(jax.device_get(a)), tree)


def _placed_statics(sharded, layout, mesh) -> dict:
    """The wrapper's partitioned static arrays, placed on the mesh once."""
    return {k: place_on_mesh(layout_slab(np.asarray(sa.value), layout), mesh)
            for k, sa in sharded._sharded_static.items()}


# ---------------------------------------------------------------------------
# Goal a: exchange ranking
# ---------------------------------------------------------------------------


def exchange_input(values: np.ndarray, layout, mesh):
    """The exchange benchmark's timed input: partition layout, placed on the mesh."""
    return place_on_mesh(layout_slab(values, layout), mesh)


def run_exchange(args, out: dict) -> dict:
    _load_backend()
    mesh = _mesh_for(args.n_devices)
    D = args.n_devices
    trailing = (args.fields,) if args.fields > 1 else ()
    results = []
    for requested in args.cells:
        n, edges = synthetic_mesh(args.synthetic, requested)
        pa, how = _partition_for(args, args.synthetic, n, edges, None, D)
        t0 = time.perf_counter()
        layout = build_unstructured_partition(partition_assignment=pa, edges=edges, n_devices=D)
        layout_s = time.perf_counter() - t0
        traffic = exchange_traffic(layout)
        itemsize = 4 * args.fields
        rng = np.random.default_rng(0)
        values = rng.standard_normal((n,) + trailing).astype(np.float32)
        slab = exchange_input(values, layout, mesh)
        entry = {
            "cells": n, "requested_cells": int(requested), "mesh": args.synthetic,
            "partition": how,
            "n_devices": D, "fields_per_cell": args.fields,
            "n_local_max": layout.n_local_max, "n_ghost_max": layout.n_ghost_max,
            "layout_build_s": layout_s, "traffic_cells_per_shard": traffic,
            "input_sharding": str(slab.sharding), "input_presharded": False,
            "methods": {},
        }
        outputs = {}
        for method in METHODS:
            def local(x, _m=method):
                return exchange_unstructured(x, layout=layout, mesh_axis=MESH_AXIS, method=_m)

            fn = jax.jit(shard_map(local, mesh=mesh, in_specs=P(MESH_AXIS), out_specs=P(MESH_AXIS)))
            compile_s, compiled = compile_seconds(fn, slab)
            entry["input_presharded"] = require_presharded([slab], mesh, f"exchange/{method}",
                                                           compiled=compiled)
            stats = timed(lambda: fn(slab), warmup=args.warmup, repeats=args.repeats)
            outputs[method] = np.asarray(jax.device_get(fn(slab)))
            per_shard_bytes = int(traffic[method]) * itemsize
            entry["methods"][method] = {
                **stats,
                "compile_s": compile_s,
                "bytes_per_shard": per_shard_bytes,
                "bytes_total": per_shard_bytes * D,
                "messages": traffic["ppermute_messages"] if method == "ppermute" else 1,
                "bandwidth_GBps": (per_shard_bytes * D / 1e9) / (stats["min_ms"] / 1e3)
                if stats["min_ms"] > 0 else None,
            }
        entry["bit_identical"] = bool(np.array_equal(outputs["all_to_all"], outputs["ppermute"]))
        a, p = entry["methods"]["all_to_all"], entry["methods"]["ppermute"]
        entry["ppermute_speedup_median"] = (a["median_ms"] / p["median_ms"]) if p["median_ms"] else None
        entry["ppermute_speedup_min"] = (a["min_ms"] / p["min_ms"]) if p["min_ms"] else None
        results.append(entry)
        speedup = entry["ppermute_speedup_median"]
        speedup_txt = f"x{speedup:.2f}" if speedup is not None else "x n/a (zero median)"
        print(f"[exchange] cells={n:>8} {how:<10} a2a {a['median_ms']:8.3f} ms  "
              f"ppermute {p['median_ms']:8.3f} ms  {speedup_txt}  "
              f"identical={entry['bit_identical']}")
    out["results"] = results
    return finish_checks(out, exchange_checks(results, D))


def exchange_checks(results: list, n_devices: int) -> list:
    """The ``exchange`` goal's checks, from its ``results`` alone.

    Every goal's checks are a pure function of what it records, so that
    ``--summarise`` can derive them again from a file's ``results`` with
    this same code and refuse a file whose ``checks`` say anything else
    (:func:`record_problems`).
    """
    checks = []
    for r in results:
        checks.append(check_that(f"{r['cells']} cells: all_to_all == ppermute bit for bit",
                                 r["bit_identical"]))
        checks.append(check_that(f"{r['cells']} cells: timed input pre-placed on the mesh",
                                 r["input_presharded"]))
    return checks


# ---------------------------------------------------------------------------
# Goal b: forward run on a real (or synthetic) mesh
# ---------------------------------------------------------------------------


def run_forward(args, out: dict) -> dict:
    _load_backend()
    mesh = _mesh_for(args.n_devices)
    D = args.n_devices
    results = []
    for n_hint in _sizes(args):
        n, edges, pa, source = _mesh_source(args, n_hint)
        pa, how = _partition_for(args, source, n, edges, pa, D)
        entry = {"cells": n, "mesh": source, "partition": how, "n_devices": D,
                 "steps": args.steps, "methods": {}}
        ref_node = None
        ref_state = None
        for method in METHODS:
            ref_node, sharded, layout = build_pair(n, edges, mesh, pa, method)
            entry["n_local_max"] = layout.n_local_max
            entry["n_ghost_max"] = layout.n_ghost_max
            entry["traffic_cells_per_shard"] = exchange_traffic(layout)
            state = sharded.initial_state()       # placed on the mesh by the wrapper
            dt = 1.0
            dt_j = jnp.asarray(dt)

            # The compiled step alone, with pre-placed statics: this is the
            # number that compares across transports.
            fn = sharded._get_sharded_fn(state, {}, None)
            statics = _placed_statics(sharded, layout, mesh)
            presharded = require_presharded([state["x"], *statics.values()], mesh,
                                            f"forward/{method}")
            compile_s, _ = compile_seconds(fn, state, {}, dt_j, statics, {})

            def device_block(state=state, fn=fn, statics=statics, dt_j=dt_j):
                for _ in range(args.steps):
                    state = {"x": fn(state, {}, dt_j, statics, {})["x"]}
                return state["x"]

            device = timed(device_block, warmup=args.warmup, repeats=args.repeats)

            # The public ``update()`` re-partitions the static arrays on the
            # host every call; its first call is host work only (the
            # executable is already compiled).
            t0 = time.perf_counter()
            jax.block_until_ready(sharded.update(state, {}, dt)["x"])
            wrapper_first_call_s = time.perf_counter() - t0

            def step_block(state=state, sharded=sharded, dt=dt):
                for _ in range(args.steps):
                    state = {"x": sharded.update(state, {}, dt)["x"]}
                return state["x"]

            wrapper = timed(step_block, warmup=args.warmup, repeats=args.repeats)
            # parity against the unsharded node after ``steps`` steps
            final = state
            for _ in range(args.steps):
                final = sharded.update({"x": final["x"]}, {}, dt)
            got = sharded.gather_global(final)
            if ref_state is None:
                ref_state = ref_node.initial_state()
                for _ in range(args.steps):
                    ref_state = ref_node.update({"x": ref_state["x"]}, {}, dt)
            m = {
                "compile_s": compile_s,
                "wrapper_first_call_s": wrapper_first_call_s,
                "input_presharded": presharded,
                "wrapper_step": {**wrapper, "ms_per_step": wrapper["median_ms"] / args.steps},
                "device_step": {**device, "ms_per_step": device["median_ms"] / args.steps},
                "parity_x": _diff(got["x"], np.asarray(jax.device_get(ref_state["x"]))),
                "parity_total": _diff(np.asarray(got["total"]).reshape(-1),
                                      np.asarray(jax.device_get(ref_state["total"])).reshape(-1)),
            }
            if n < EXACT_COUNT_CELLS:
                # One more public step, from a field of ones placed as the
                # wrapper places its own initial state: the program is the
                # one compiled above (require_presharded: same sharding).
                ones = {"x": place_on_mesh(layout_slab(np.ones(n, np.float32), layout), mesh)}
                require_presharded([ones["x"]], mesh, f"forward/{method} ones")
                m["ones"] = _ones_record(sharded.gather_global(sharded.update(ones, {}, dt)))
            entry["methods"][method] = m
            print(f"[forward] cells={n:>8} {method:<10} wrapper {m['wrapper_step']['ms_per_step']:8.3f}"
                  f" ms/step  device {m['device_step']['ms_per_step']:8.3f} ms/step"
                  f"  compile {compile_s:6.2f} s  max|dx|={m['parity_x']['max_abs']:.2e}")
        if n < EXACT_COUNT_CELLS:
            entry["ones_unsharded"] = _ones_record(
                ref_node.update({"x": jnp.ones(n, jnp.float32)}, {}, 1.0))
        results.append(entry)
    out["results"] = results
    return finish_checks(out, forward_checks(results, D))


def _ones_record(stepped: dict) -> dict:
    """What one step from a field of ones returned, in global cell order:
    its ``total`` (the number of cells when the reduction is right), the
    largest ``|x - 1|`` of the stepped field and how many entries are not
    exactly 1 (zero and none when the step leaves ones ones)."""
    x = np.asarray(stepped["x"], np.float64)
    return {"total": float(np.asarray(stepped["total"]).reshape(-1)[0]),
            "max_abs_from_one": float(np.max(np.abs(x - 1.0))) if x.size else 0.0,
            "entries_not_one": int(np.count_nonzero(x != 1.0))}


def _ones_checks(prefix: str, cells: int, record: dict) -> list:
    """The two exact checks of one side's step from a field of ones: the
    count, and its precondition as a check of its own, so that a failed
    count on a field that did not stay ones reads as that."""
    return [check(f"{prefix} ones: |total - cells|", abs(record["total"] - cells),
                  LIMITS["exact"]),
            check(f"{prefix} ones: max |x - 1| after the step", record["max_abs_from_one"],
                  LIMITS["exact"])]


def forward_checks(results: list, n_devices: int) -> list:
    """The ``forward`` goal's checks, from its ``results`` alone (see
    :func:`exchange_checks`): the field against the unsharded node's, entry
    by entry; the total of the cancelling field finite; and the reduction
    exactly, on a field of ones (``EXACT_COUNT_CELLS``)."""
    checks = []
    for r in results:
        for method in METHODS:
            m = r["methods"][method]
            prefix = f"{r['cells']} cells {method}"
            checks += _parity_checks(f"{prefix} x vs unsharded", m["parity_x"], LIMITS["forward"])
            # ``parity_total`` is recorded as information: its max_rel is
            # decided by the order of addition (see LIMITS["forward"]).
            checks.append(check_that(f"{prefix} total vs unsharded finite",
                                     m["parity_total"]["finite"]))
            if r["cells"] < EXACT_COUNT_CELLS:
                checks += _ones_checks(prefix, r["cells"], m["ones"])
        if r["cells"] < EXACT_COUNT_CELLS:
            checks += _ones_checks(f"{r['cells']} cells unsharded", r["cells"],
                                   r["ones_unsharded"])
        else:
            checks.append(check_not_run(
                f"{r['cells']} cells ones: total == cells exactly, sharded and unsharded",
                f"a float32 sum of ones is exact only under 2**24 = {EXACT_COUNT_CELLS} "
                "cells"))
    return checks


# ---------------------------------------------------------------------------
# Goal c: gradient parity
# ---------------------------------------------------------------------------


def _laplacian_matvec_unsharded():
    """The shifted 1-D Dirichlet Laplacian of the ``sharded_cg`` part
    (``CG_SHIFT`` says why it is shifted), on one device."""
    def matvec(x):
        left = jnp.concatenate([jnp.zeros((1,), dtype=x.dtype), x[:-1]])
        right = jnp.concatenate([x[1:], jnp.zeros((1,), dtype=x.dtype)])
        return (2 + CG_SHIFT) * x - left - right
    return matvec


def _laplacian_matvec_sharded(mesh):
    """The same operator over the mesh: each shard takes its two ghosts
    from its neighbours by ``ppermute`` and the global ends take zero."""
    D = int(mesh.shape[MESH_AXIS])

    def shard_matvec(x):
        left_ghost = lax.ppermute(x[-1], MESH_AXIS, [(i, (i + 1) % D) for i in range(D)])
        right_ghost = lax.ppermute(x[0], MESH_AXIS, [(i, (i - 1) % D) for i in range(D)])
        idx = lax.axis_index(MESH_AXIS)
        left_ghost = jnp.where(idx == 0, 0.0, left_ghost)
        right_ghost = jnp.where(idx == D - 1, 0.0, right_ghost)
        left = jnp.concatenate([jnp.asarray([left_ghost], dtype=x.dtype), x[:-1]])
        right = jnp.concatenate([x[1:], jnp.asarray([right_ghost], dtype=x.dtype)])
        return (2 + CG_SHIFT) * x - left - right

    def matvec(x):
        return shard_map(shard_matvec, mesh=mesh, in_specs=(P(MESH_AXIS),),
                         out_specs=P(MESH_AXIS))(x)
    return matvec


def _cg_true_residual(x, rhs) -> float:
    """``|rhs - A x| / |rhs|`` of the ``sharded_cg`` part's operator, in
    float64 on the host: the residual of the system itself, measured
    outside the solve and by neither of the two JAX operators.

    ``A`` has the diagonal the float32 solves were given (``2 + CG_SHIFT``
    rounded to float32).  A solve of a wrong operator -- a ghost from the
    wrong neighbour, say -- has a large residual here, whatever its own
    solver reported.
    """
    x, rhs = np.asarray(x, np.float64), np.asarray(rhs, np.float64)
    ax = float(np.float32(2 + CG_SHIFT)) * x
    ax[1:] -= x[:-1]
    ax[:-1] -= x[1:]
    return float(np.linalg.norm(rhs - ax) / np.linalg.norm(rhs))


def run_gradient_cg(args, mesh, n_cg: int, rng) -> dict:
    """The ``sharded_cg`` part of the ``gradient`` goal at ``n_cg`` unknowns:
    reverse and forward mode through the Jacobi-preconditioned CG, sharded
    against unsharded, and whether the solves behind them converged.

    The right-hand side and the tangent direction are white noise from
    ``rng``.  A smooth right-hand side (until schema 7, a sine) is nearly
    an eigenvector of the shifted operator: its residual lives in the two
    boundary layers, the solver's test -- relative to ``|b|``, which grows
    as ``sqrt(n)`` -- is met sooner the larger the system (21 iterations at
    1e6 unknowns, 14 at 1e7), and the solution it stops on at 1e6 is 40 to
    60 times ``CG_RTOL`` from a float64 direct solve.  On white noise the
    iteration count and that distance (about ``CG_RTOL``) are the same at
    every size.

    ``sharded_cg(differentiable=True)``, which the derivatives are taken
    through, cannot report an iteration count (``iters`` is -1: the solve
    is behind ``lax.custom_linear_solve``), and its ``converged`` is the
    float32 true residual against ``rtol`` with no slack, which reads
    ``False`` on a converged solve whenever the loop stopped within the
    float32 floor of the tolerance (measured at 1024 unknowns with a
    shift of 0.01 and a smooth right-hand side: stopped after 75 of 3000
    iterations, true residual 1.04e-4).  So the
    flag and the count recorded here are the loop's own (the same solve
    without ``differentiable``: it stopped on its tolerance, after so many
    iterations), and convergence is decided on the true residual of each
    of the three solves (``CG_SOLVES``), measured on the host against
    ``LIMITS["krylov_residual"]``.
    """
    matvec_ref = _laplacian_matvec_unsharded()
    matvec_sh = _laplacian_matvec_sharded(mesh)
    pc = jacobi_preconditioner(jnp.full((n_cg,), 2 + CG_SHIFT, jnp.float32))
    kw = dict(max_iters=args.cg_max_iters, rtol=CG_RTOL, atol=1e-8, backend="loop",
              preconditioner=pc, differentiable=True)
    on_mesh = dict(mesh=mesh, in_specs=P(MESH_AXIS))
    b_host = rng.standard_normal(n_cg).astype(np.float32)
    v_host = rng.standard_normal(n_cg).astype(np.float32)
    b, v = jnp.asarray(b_host), jnp.asarray(v_host)
    b_sh = place_on_mesh(b_host, mesh)

    def cg_loss_sh(bb):
        return jnp.sum(sharded_cg(matvec_sh, bb, **on_mesh, **kw).value ** 2)

    def cg_loss_ref(bb):
        return jnp.sum(sharded_cg(matvec_ref, bb, **kw).value ** 2)

    g_sh_fn, g_ref_fn = jax.jit(jax.grad(cg_loss_sh)), jax.jit(jax.grad(cg_loss_ref))
    cg_presharded = require_presharded([b_sh], mesh, "gradient/sharded_cg")
    sh_compile_s, _ = compile_seconds(g_sh_fn, b_sh)
    ref_compile_s, _ = compile_seconds(g_ref_fn, b)
    g_sh = np.asarray(jax.device_get(g_sh_fn(b_sh)))
    g_ref = np.asarray(jax.device_get(g_ref_fn(b)))
    # Jitted: an eager jvp through the solver dispatched op by op and
    # took 13 of the dry run's 21 s at 256 dof; compiled, it is the same
    # derivative.
    jvp_sh = jax.jit(lambda bb, vv: jax.jvp(
        lambda x: sharded_cg(matvec_sh, x, **on_mesh, **kw).value, (bb,), (vv,))[1])
    jvp_ref = jax.jit(lambda bb, vv: jax.jvp(
        lambda x: sharded_cg(matvec_ref, x, **kw).value, (bb,), (vv,))[1])
    t_sh = np.asarray(jax.device_get(jvp_sh(b, v)))
    t_ref = np.asarray(jax.device_get(jvp_ref(b, v)))

    def report(matvec, where):
        # The solution of the route the derivatives are taken through, and
        # the flag and count of the loop itself.
        def solve(bb):
            through = sharded_cg(matvec, bb, **where, **kw)
            loop = sharded_cg(matvec, bb, **where, **dict(kw, differentiable=False))
            return through.value, loop.converged, loop.iters
        return jax.jit(solve)

    solves = {}
    for side, fn, rhs, g, t in (("sharded", report(matvec_sh, on_mesh), b_sh, g_sh, t_sh),
                                ("unsharded", report(matvec_ref, {}), b, g_ref, t_ref)):
        x, stopped, iterations = jax.device_get(fn(rhs))
        x = np.asarray(x, np.float64)
        solves[side] = {
            "converged": bool(stopped), "iterations": int(iterations),
            "true_residual": {"solve": _cg_true_residual(x, b_host),
                              "adjoint": _cg_true_residual(g, 2 * x),
                              "tangent": _cg_true_residual(t, v_host)},
        }
    cg = {
        "dof": n_cg,
        "shift": CG_SHIFT, "rtol": CG_RTOL, "max_iters": args.cg_max_iters,
        "input_presharded": cg_presharded,
        "grad_sharded": timed(lambda: g_sh_fn(b_sh), warmup=args.warmup, repeats=args.repeats),
        "grad_unsharded": timed(lambda: g_ref_fn(b), warmup=args.warmup, repeats=args.repeats),
        "compile_s": {"sharded": sh_compile_s, "unsharded": ref_compile_s},
        "grad_parity": _diff(g_sh, g_ref),
        "jvp_parity": _diff(t_sh, t_ref),
        "solve": solves,
    }
    worst = max(r for s in solves.values() for r in s["true_residual"].values())
    print(f"[gradient] dof={n_cg:>8} sharded_cg grad {cg['grad_sharded']['median_ms']:8.2f} ms "
          f"(unsharded {cg['grad_unsharded']['median_ms']:8.2f})  "
          f"rel grad={cg['grad_parity']['max_rel']:.2e} jvp={cg['jvp_parity']['max_rel']:.2e}  "
          f"iterations {solves['sharded']['iterations']}/{solves['unsharded']['iterations']} "
          f"of {args.cg_max_iters}  true residual <= {worst:.2e}")
    return cg


def run_gradient(args, out: dict) -> dict:
    _load_backend()
    mesh = _mesh_for(args.n_devices)
    D = args.n_devices
    results = []
    for n_hint in _sizes(args):
        n, edges, pa, source = _mesh_source(args, n_hint)
        pa, how = _partition_for(args, source, n, edges, pa, D)
        rng = np.random.default_rng(1)
        w_global = rng.standard_normal(n).astype(np.float32)
        entry = {"cells": n, "mesh": source, "partition": how, "n_devices": D,
                 "grad_steps": args.grad_steps, "rollout": {}, "sharded_cg": None}

        # (i) reverse mode through a rollout: jitted grad on both sides,
        # compile time apart, statics placed once (like-for-like).
        ref_node, _, _ = build_pair(n, edges, mesh, pa, "all_to_all")
        w_j = jnp.asarray(w_global)

        def loss_ref(x0, ref_node=ref_node, w_j=w_j):
            st = {"x": x0}
            for _ in range(args.grad_steps):
                st = {"x": ref_node.update(st, {}, 1.0)["x"]}
            return jnp.sum(st["x"] * w_j)

        x0_global = ref_node.initial_state()["x"]
        g_ref_fn = jax.jit(jax.grad(loss_ref))
        ref_compile_s, _ = compile_seconds(g_ref_fn, x0_global)
        g_ref = np.asarray(jax.device_get(g_ref_fn(x0_global)))
        ref_t = timed(lambda: g_ref_fn(x0_global), warmup=args.warmup, repeats=args.repeats)
        entry["rollout"]["unsharded"] = {"grad": ref_t, "compile_s": ref_compile_s}
        for method in METHODS:
            _, sharded, layout = build_pair(n, edges, mesh, pa, method)
            state0 = sharded.initial_state()
            x0 = state0["x"]
            w_layout = place_on_mesh(layout_slab(w_global, layout), mesh)
            step_fn = sharded._get_sharded_fn(state0, {}, None)
            statics = _placed_statics(sharded, layout, mesh)
            dt_j = jnp.asarray(1.0)
            presharded = require_presharded([x0, w_layout, *statics.values()], mesh,
                                            f"gradient/rollout/{method}")

            def loss_sh(x0, step_fn=step_fn, statics=statics, dt_j=dt_j, w_layout=w_layout):
                st = {"x": x0}
                for _ in range(args.grad_steps):
                    st = {"x": step_fn(st, {}, dt_j, statics, {})["x"]}
                return jnp.sum(st["x"] * w_layout)

            g_fn = jax.jit(jax.grad(loss_sh))
            compile_s, _ = compile_seconds(g_fn, x0)
            g_layout = np.asarray(jax.device_get(g_fn(x0)))
            g_sh = gather_value(per_shard=g_layout.reshape(D, layout.n_local_max), layout=layout)
            stats = timed(lambda: g_fn(x0), warmup=args.warmup, repeats=args.repeats)
            entry["rollout"][method] = {"grad": stats, "compile_s": compile_s,
                                        "input_presharded": presharded,
                                        "parity": _diff(g_sh, g_ref)}
            print(f"[gradient] cells={n:>8} rollout {method:<10} grad {stats['median_ms']:8.2f} ms"
                  f" (unsharded {ref_t['median_ms']:8.2f})  compile {compile_s:6.2f} s"
                  f"  max|dg|={entry['rollout'][method]['parity']['max_abs']:.2e}"
                  f"  rel={entry['rollout'][method]['parity']['max_rel']:.2e}")

        # (ii) reverse + forward mode through the Jacobi-preconditioned sharded CG
        entry["sharded_cg"] = run_gradient_cg(args, mesh, (n // D) * D, rng)
        results.append(entry)
    out["results"] = results
    return finish_checks(out, gradient_checks(results, D))


def gradient_checks(results: list, n_devices: int) -> list:
    """The ``gradient`` goal's checks, from its ``results`` alone (see
    :func:`exchange_checks`)."""
    checks = []
    for r in results:
        for method in METHODS:
            checks += _parity_checks(f"{r['cells']} cells rollout {method} grad vs unsharded",
                                     r["rollout"][method]["parity"], LIMITS["gradient"])
        checks += gradient_cg_checks(r["sharded_cg"])
    return checks


def gradient_cg_checks(cg: dict) -> list:
    """The checks of one size's ``sharded_cg`` entry: the two derivatives
    against the unsharded ones, and that each side's solves converged.

    Parity alone passes two solves that failed alike -- it did, until
    schema 7, at every size of the session.  So each side must have stopped
    on its tolerance and not on the iteration cap (the loop's own flag),
    and the true residual of each of its three solves must be within
    ``LIMITS["krylov_residual"]``.
    """
    checks = _parity_checks(f"{cg['dof']} dof sharded_cg grad vs unsharded",
                            cg["grad_parity"], LIMITS["krylov"])
    checks += _parity_checks(f"{cg['dof']} dof sharded_cg jvp vs unsharded",
                             cg["jvp_parity"], LIMITS["krylov"])
    for side in ("sharded", "unsharded"):
        solve = cg["solve"][side]
        prefix = f"{cg['dof']} dof sharded_cg {side}"
        checks.append(check_that(
            f"{prefix} solve stopped on its tolerance, not on the iteration cap",
            solve["converged"] is True,
            detail=f"{solve['iterations']} iterations, cap {cg['max_iters']}"))
        checks += [check(f"{prefix} {which} true residual", solve["true_residual"][which],
                         LIMITS["krylov_residual"]) for which in CG_SOLVES]
    return checks


# ---------------------------------------------------------------------------
# Checklist 5: an indivisible grid is refused
# ---------------------------------------------------------------------------


def _refusal(build) -> tuple:
    """``(exception type name or None, message)`` of calling ``build()``."""
    try:
        build()
    except Exception as e:  # noqa: BLE001 - the type is what gets recorded
        return type(e).__name__, str(e)
    return None, ""


def _names_both(message: str, cells: int, devices: int) -> bool:
    return f"{cells} cells" in message and f"{devices} devices" in message


def _pencil_mesh_fits(n_devices: int) -> bool:
    """The ``2 x (D / 2)`` pencil mesh the 2-D cases use exists for ``D``."""
    return n_devices >= 4 and n_devices % 2 == 0


def _pencil_mesh_needs(n_devices: int) -> str:
    return (f"needs an even device count >= 4 for a 2 x (D/2) pencil mesh; this run has "
            f"{n_devices}")


def run_indivisible(args, out: dict) -> dict:
    """The stencil and pointwise wrappers refuse a grid the mesh cannot split,
    at construction and in the caller's terms; the unstructured wrapper
    takes such a count and still matches the unsharded node."""
    _load_backend()
    D = args.n_devices
    mesh = _mesh_for(D)
    ny_ok, nx = field_shape(min(args.cells), D)
    ny_bad = ny_ok + 1                              # D >= 2: never a multiple of D
    entry: dict = {"n_devices": D}

    def stencil(ny, nx_, axis=0):
        return ShardedStencilNode(Field2D("field", ny, nx_, mask_axis=axis), mesh,
                                  axis_map={MESH_AXIS: axis}, boundary="periodic")

    kind, msg = _refusal(lambda: stencil(ny_bad, nx))
    kind_ok, msg_ok = _refusal(lambda: stencil(ny_ok, nx))
    entry["stencil"] = {"shape": [ny_bad, nx], "raised": kind, "message": msg,
                        "divisible_shape": [ny_ok, nx], "divisible_raised": kind_ok,
                        "divisible_message": msg_ok}
    # The same 1-D mesh sharding spatial axis 1 (axis_map={MESH_AXIS: 1}),
    # as the stencil goals now run it: an axis-1 extent it cannot split.
    nx_bad = nx + 1
    kind, msg = _refusal(lambda: stencil(ny_ok, nx_bad, axis=1))
    kind_ok, msg_ok = _refusal(lambda: stencil(ny_ok, nx, axis=1))
    entry["stencil_axis1"] = {"shape": [ny_ok, nx_bad], "raised": kind, "message": msg,
                              "divisible_shape": [ny_ok, nx], "divisible_raised": kind_ok,
                              "divisible_message": msg_ok}

    kind, msg = _refusal(lambda: ShardedPointwiseNode(Pointwise2D(ny_bad, nx), mesh,
                                                      shard_axes=(0,)))
    entry["pointwise"] = {"shape": [ny_bad, nx], "raised": kind, "message": msg}

    if _pencil_mesh_fits(D):
        # A pencil mesh: the refusal names the axis that does not divide.
        nz = D // 2
        rows, cols = 16, 8 * nz + 1
        pencil = create_device_mesh(shape=(2, nz))
        kind, msg = _refusal(lambda: ShardedStencilNode(
            Field2D("field", rows, cols), pencil,
            axis_map={"spatial_y": 0, "spatial_z": 1}, boundary="periodic"))
        entry["pencil"] = {"mesh": [2, nz], "shape": [rows, cols], "raised": kind,
                           "message": msg}

    # The rule is the Cartesian wrappers': the unstructured wrapper carries
    # a padded layout and takes an uneven split -- for a node written for
    # its partition layout (a ring here), not for a stencil node.
    n = ny_bad * nx + (1 if (ny_bad * nx) % D == 0 else 0)
    edges = ring_edges(n)
    pa, how = partition_cells(n, edges, D, "contiguous")
    ref, sharded, _layout = build_pair(n, edges, mesh, pa, "all_to_all")
    counts = np.bincount(pa, minlength=D)
    state, ref_state = sharded.initial_state(), ref.initial_state()
    for _ in range(args.steps):
        state = sharded.update({"x": state["x"]}, {}, 1.0)
        ref_state = ref.update({"x": ref_state["x"]}, {}, 1.0)
    got = sharded.gather_global(state)
    parity = _diff(got["x"], np.asarray(jax.device_get(ref_state["x"])))
    entry["unstructured"] = {"cells": n, "partition": how, "cells_per_device": counts.tolist(),
                             "steps": args.steps, "parity_x": parity}
    print(f"[indivisible] stencil {ny_bad}x{nx}: {entry['stencil']['raised']}  stencil axis 1 "
          f"{ny_ok}x{nx_bad}: {entry['stencil_axis1']['raised']}  pointwise: "
          f"{entry['pointwise']['raised']}  unstructured {n} cells {counts.tolist()}: "
          f"max|dx|={parity['max_abs']:.2e}")
    out["results"] = [entry]
    return finish_checks(out, indivisible_checks(out["results"], D))


def indivisible_checks(results: list, n_devices: int) -> list:
    """The ``indivisible`` goal's checks, from its ``results`` alone (see
    :func:`exchange_checks`)."""
    [entry] = results
    D = n_devices
    st = entry["stencil"]
    ny_bad, nx = st["shape"]
    ny_ok = st["divisible_shape"][0]
    kind, msg = st["raised"], st["message"]
    checks = [
        check_that(f"stencil {ny_bad}x{nx} on {D} devices: ValueError at construction",
                   kind == "ValueError", msg[:300]),
        # Until 0.4.0 the refusal recommended ShardedUnstructuredNode, and this
        # check asked for that name; the wrapper now refuses a stencil node
        # (it hands update_padded the partition layout), and the refusal says
        # so.  The name alone would pass either message.
        check_that("stencil refusal names the cell count, the device count and that "
                   "the unstructured wrapper is not a way out for a stencil node",
                   _names_both(msg, ny_bad, D)
                   and "ShardedUnstructuredNode is not a way out for a stencil node" in msg
                   and "or use ShardedUnstructuredNode" not in msg),
        check_that(f"stencil {ny_ok}x{nx} (divisible) is accepted",
                   st["divisible_raised"] is None, st.get("divisible_message", "")[:300]),
    ]
    ax1 = entry["stencil_axis1"]
    rows, nx_bad = ax1["shape"]
    nx_ok = ax1["divisible_shape"][1]
    kind, msg = ax1["raised"], ax1["message"]
    checks += [
        check_that(f"stencil {rows}x{nx_bad} sharded along spatial axis 1 on {D} devices: "
                   "ValueError at construction", kind == "ValueError", msg[:300]),
        check_that("axis-1 stencil refusal names spatial axis 1, the cell count and the "
                   "device count", "spatial axis 1" in msg and _names_both(msg, nx_bad, D)),
        check_that(f"stencil {rows}x{nx_ok} sharded along spatial axis 1 (divisible) is "
                   "accepted", ax1["divisible_raised"] is None,
                   ax1.get("divisible_message", "")[:300]),
    ]
    pw = entry["pointwise"]
    kind, msg = pw["raised"], pw["message"]
    checks += [
        check_that(f"pointwise {ny_bad}x{nx} on {D} devices: ValueError at construction",
                   kind == "ValueError", msg[:300]),
        check_that("pointwise refusal names the cell count and the device count",
                   _names_both(msg, ny_bad, D)),
    ]
    if _pencil_mesh_fits(D):
        pencil = entry["pencil"]
        nz = pencil["mesh"][1]
        rows, cols = pencil["shape"]
        kind, msg = pencil["raised"], pencil["message"]
        checks += [
            check_that(f"pencil {rows}x{cols} on a 2x{nz} mesh: ValueError at construction",
                       kind == "ValueError", msg[:300]),
            check_that("pencil refusal names spatial axis 1 and both numbers",
                       "spatial axis 1" in msg and _names_both(msg, cols, nz)),
        ]
    else:
        checks.append(check_not_run("pencil (2-D mesh) refusal naming the axis that does "
                                    "not divide", _pencil_mesh_needs(D)))
    un = entry["unstructured"]
    n, counts = un["cells"], list(un["cells_per_device"])
    checks.append(check_that(f"unstructured {n} cells on {D} devices is an uneven split",
                             len(set(counts)) > 1, str(counts)))
    checks += _parity_checks(f"unstructured {n} cells x vs unsharded", un["parity_x"],
                             LIMITS["forward"])
    return checks


# ---------------------------------------------------------------------------
# Checklist 2: halo exchange at the shard and global boundaries
# ---------------------------------------------------------------------------

HALO_BOUNDARIES = ("periodic", "edge", "zero")
HALO_WIDTHS = (1, 2)


def halo_index_map(n_global: int, n_shards: int, halo: int, boundary: str) -> np.ndarray:
    """``(n_shards, n_global // n_shards + 2 * halo)`` global indices, ``-1`` = zero.

    Slot ``k`` of shard ``d``'s padded block holds global cell
    ``map[d, k]`` after ``halo_exchange``: interior halos are the
    neighbouring shard's cells; at the global edges ``periodic`` wraps,
    ``edge`` repeats the outermost cell across the whole halo
    (``numpy.pad``'s ``mode="edge"``) and ``zero`` fills zeros.  (Until
    0.4.0 ``edge`` repeated the shard's ``halo`` outermost cells in order,
    ``r0, r1`` before ``r0`` at width 2, and this reference pinned that;
    MADD-ANO-029.)
    """
    per = n_global // n_shards
    rows = []
    for d in range(n_shards):
        own = np.arange(d * per, (d + 1) * per)
        left = np.arange(d * per - halo, d * per) % n_global
        right = np.arange((d + 1) * per, (d + 1) * per + halo) % n_global
        if d == 0 and boundary != "periodic":
            left = np.full(halo, own[0] if boundary == "edge" else -1)
        if d == n_shards - 1 and boundary != "periodic":
            right = np.full(halo, own[-1] if boundary == "edge" else -1)
        rows.append(np.concatenate([left, own, right]))
    return np.stack(rows)


def halo_reference(a: np.ndarray, row_map: np.ndarray, col_map: np.ndarray,
                   cotangent: np.ndarray) -> tuple:
    """NumPy forward and adjoint of an exchange whose shards hold ``row_map x col_map``.

    Returns ``(padded, grad)``: the assembled padded blocks, and the
    cotangent of those blocks summed back onto the cells they came from.
    """
    a_pad = np.pad(a, ((0, 1), (0, 1)))              # index -1 -> the zero row/column
    grad = np.zeros_like(a_pad)
    by, bz = row_map.shape[1], col_map.shape[1]
    blocks = []
    for i, rows in enumerate(row_map):
        line = []
        for j, cols in enumerate(col_map):
            line.append(a_pad[np.ix_(rows, cols)])
            np.add.at(grad, (rows[:, None], cols[None, :]),
                      cotangent[i * by:(i + 1) * by, j * bz:(j + 1) * bz])
        blocks.append(line)
    return np.block(blocks), grad[:-1, :-1]


def _integer_field(shape, modulus: int) -> np.ndarray:
    """float32 integers: every sum the adjoint forms is exact."""
    return (np.arange(int(np.prod(shape))) % modulus + 1).reshape(shape).astype(np.float32)


def _max_abs(got: np.ndarray, want: np.ndarray) -> float:
    if got.shape != want.shape:
        return math.inf
    return float(np.max(np.abs(got - want), initial=0.0))


def run_halo(args, out: dict) -> dict:
    """Every halo slot, forward and adjoint, against NumPy, bit for bit."""
    _load_backend()
    D = args.n_devices
    ny, nx = field_shape(max(args.cells), D)
    meshes = meshes_that_fit(D)
    cases = []
    for label in meshes:
        mesh, axis_map, (py, pz) = stencil_mesh(label, D)
        # An axis in axis_map is exchanged (on one device too, "2d-flat"'s
        # axis 0: its halos are the global-edge fill); one not in it has
        # no halo slots at all.
        mesh_axis_of = {sax: ma for ma, sax in axis_map.items()}
        spec = P(*(mesh_axis_of.get(sax) for sax in (0, 1)))
        a = _integer_field((ny, nx), 9973)
        placed = NamedSharding(mesh, spec)
        fns, cotangents, refs = [], [], []
        for boundary in HALO_BOUNDARIES:
            for h in HALO_WIDTHS:
                axes = [(ma, sax, h) for ma, sax in axis_map.items()]
                row_map, col_map = (
                    halo_index_map(n, k, h, boundary) if sax in mesh_axis_of
                    else np.arange(n)[None, :]
                    for sax, n, k in ((0, ny, py), (1, nx, pz)))

                def local(x, _axes=axes, _b=boundary, _mesh=mesh):
                    return halo_exchange(x, mesh=_mesh, axes=_axes, boundary=_b)

                fns.append(shard_map(local, mesh=mesh, in_specs=spec, out_specs=spec))
                ct = _integer_field((row_map.size, col_map.size), 13)
                cotangents.append(jax.device_put(jnp.asarray(ct), placed))
                refs.append((boundary, h, *halo_reference(a, row_map, col_map, ct)))

        # Every mode and width of this mesh in one program: one compile.
        def every_case(x, cts, _fns=tuple(fns)):
            return [(fn(x), jax.vjp(fn, x)[1](ct)[0]) for fn, ct in zip(_fns, cts)]

        outs = jax.jit(every_case)(jax.device_put(jnp.asarray(a), placed), cotangents)
        for (boundary, h, want, want_grad), (got, got_grad) in zip(refs, outs):
            fwd_err = _max_abs(np.asarray(jax.device_get(got)), want)
            adj_err = _max_abs(np.asarray(jax.device_get(got_grad)), want_grad)
            cases.append({"mesh": label, "mesh_shape": [py, pz], "shape": [ny, nx],
                          "halo": h, "boundary": boundary,
                          "forward_max_abs": fwd_err, "adjoint_max_abs": adj_err})

    # The unstructured exchange: every owned slot and every ghost slot.
    mesh = _mesh_for(D)
    n, edges = synthetic_mesh(args.synthetic, max(args.cells))
    pa, how = _partition_for(args, args.synthetic, n, edges, None, D)
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges, n_devices=D)
    values = _integer_field((n,), 9973)
    slab = place_on_mesh(layout_slab(values, layout), mesh)
    width = layout.n_local_max + layout.n_ghost_max
    ct = np.zeros((D, width), np.float32)
    want_out = np.zeros((D, width), np.float32)
    grad_global = np.zeros(n, np.float32)
    valid = []
    for d in range(D):
        nl, ng = layout.n_local[d], layout.n_ghost[d]
        ghost_slots = slice(layout.n_local_max, layout.n_local_max + ng)
        ct[d, :nl] = np.arange(nl) % 13 + 1
        ct[d, ghost_slots] = np.arange(ng) % 11 + 1
        want_out[d, :nl] = values[layout.local_global_ids[d]]
        want_out[d, ghost_slots] = values[layout.ghost_global_ids[d]]
        np.add.at(grad_global, layout.local_global_ids[d], ct[d, :nl])
        np.add.at(grad_global, layout.ghost_global_ids[d], ct[d, ghost_slots])
        valid.append(np.r_[0:nl, layout.n_local_max:layout.n_local_max + ng])
    want_grad = partition_value(value=grad_global, layout=layout)
    ct_placed = place_on_mesh(ct.reshape(-1), mesh)
    unstructured = {"cells": n, "partition": how, "n_local_max": layout.n_local_max,
                    "n_ghost_max": layout.n_ghost_max, "methods": {}}
    def both_methods(x, c):
        out = {}
        for method in METHODS:
            def ulocal(y, _m=method):
                return exchange_unstructured(y, layout=layout, mesh_axis=MESH_AXIS, method=_m)

            fn = shard_map(ulocal, mesh=mesh, in_specs=P(MESH_AXIS), out_specs=P(MESH_AXIS))
            out[method] = (fn(x), jax.vjp(fn, x)[1](c)[0])
        return out

    results_by_method = jax.jit(both_methods)(slab, ct_placed)
    for method in METHODS:
        got, got_grad = results_by_method[method]
        got = np.asarray(jax.device_get(got)).reshape(D, width)
        got_grad = np.asarray(jax.device_get(got_grad)).reshape(D, layout.n_local_max)
        fwd_err = max(_max_abs(got[d, valid[d]], want_out[d, valid[d]]) for d in range(D))
        adj_err = max(_max_abs(got_grad[d, :layout.n_local[d]], want_grad[d, :layout.n_local[d]])
                      for d in range(D))
        unstructured["methods"][method] = {"forward_max_abs": fwd_err, "adjoint_max_abs": adj_err}
    worst = max([c["forward_max_abs"] for c in cases] + [c["adjoint_max_abs"] for c in cases])
    print(f"[halo] {len(cases)} stencil cases on {'+'.join(meshes)} meshes, "
          f"worst |diff| {worst:.1e}; unstructured {n} cells: "
          + "  ".join(f"{m} fwd {v['forward_max_abs']:.1e} adj {v['adjoint_max_abs']:.1e}"
                      for m, v in unstructured["methods"].items()))
    out["results"] = [{"n_devices": D, "stencil_cases": cases, "unstructured": unstructured}]
    return finish_checks(out, halo_checks(out["results"], D))


def halo_checks(results: list, n_devices: int) -> list:
    """The ``halo`` goal's checks, from its ``results`` alone (see
    :func:`exchange_checks`)."""
    [entry] = results
    D = n_devices
    checks = []
    if not _pencil_mesh_fits(D):
        checks.append(check_not_run("2d pencil mesh: every boundary mode and width, forward "
                                    "and adjoint vs NumPy", _pencil_mesh_needs(D)))
    # The forward and backward ring permutations are the same pair on a
    # mesh axis of two devices, so no case there can tell a correct
    # exchange from one that takes each halo from the wrong neighbour --
    # on four devices that is both axes of the pencil.  Until schema 6 this
    # asked only for D >= 3, and an exchange wrong along spatial axis 1
    # passed on four devices: only the pencil sharded that axis.
    checks += _neighbour_direction_not_run(
        (c["mesh_shape"] for c in entry["stencil_cases"]), "halo_exchange")
    for c in entry["stencil_cases"]:
        (py, pz), (ny, nx_m) = c["mesh_shape"], c["shape"]
        name = f"{c['mesh']} mesh {py}x{pz}, {ny}x{nx_m}, halo {c['halo']}, {c['boundary']}"
        checks.append(check(f"{name}: forward vs NumPy max_abs", c["forward_max_abs"],
                            LIMITS["exact"]))
        checks.append(check(f"{name}: adjoint vs NumPy max_abs", c["adjoint_max_abs"],
                            LIMITS["exact"]))
    un = entry["unstructured"]
    n = un["cells"]
    for method in METHODS:
        m = un["methods"][method]
        checks.append(check(f"unstructured {n} cells {method}: owned and ghost slots vs NumPy "
                            "max_abs", m["forward_max_abs"], LIMITS["exact"]))
        checks.append(check(f"unstructured {n} cells {method}: adjoint vs NumPy max_abs",
                            m["adjoint_max_abs"], LIMITS["exact"]))
    return checks


# ---------------------------------------------------------------------------
# Checklist 1 and 3 for ShardedStencilNode
# ---------------------------------------------------------------------------


def _no_block_symmetry(ny: int, nx: int, phase: float) -> np.ndarray:
    """A smooth periodic ``(ny, nx)`` pattern no block translation leaves alone.

    It carries a frequency-one mode along each axis, so shifting it by any
    fraction of the grid short of the whole -- one block of any mesh --
    changes it: every shard's block differs from every other's.  A
    grid-shaped input with a block period (until schema 6 the source was
    periodic with period 1/2 in x, exactly one block on the 2 x 2 pencil)
    lets a wrapper that hands a shard another shard's block pass.
    ``phase`` makes two patterns unlike each other.
    """
    y = np.linspace(0.0, 1.0, ny, endpoint=False, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float64)[None, :]
    two_pi = 2 * np.pi
    return (np.cos(two_pi * x + phase) + 0.6 * np.sin(two_pi * y + 1.7 * phase + 0.5)
            + 0.4 * np.cos(two_pi * (x - 2 * y) + 2.3 * phase + 1.0)).astype(np.float32)


def _loss_weight(ny: int, nx: int) -> np.ndarray:
    return (1.0 + 0.3 * _no_block_symmetry(ny, nx, 0.9)).astype(np.float32)


def _source_field(ny: int, nx: int) -> np.ndarray:
    """The ``source`` the stencil goals feed ``Field2D``: smooth, of the
    field's own size, unlike the field (so it cannot hide in it), and
    different on every block of every mesh (:func:`_no_block_symmetry`)."""
    return (0.5 * _no_block_symmetry(ny, nx, 0.4)).astype(np.float32)


def coupled_profile(ny: int, nx: int) -> np.ndarray:
    """How the ``coupled`` goal spreads the far field over the grid: the
    field's ``ambient`` is ``u * coupled_profile``, positive and different
    on every block of every mesh.  Until schema 6 it was ``u`` broadcast to
    every cell, so a wrapper that handed each shard another shard's block
    of a grid-shaped input passed the goal."""
    return (1.0 + 0.4 * _no_block_symmetry(ny, nx, 2.1)).astype(np.float32)


#: The lattice case of the ``stencil`` goal: ``LBMNode`` on D2Q9, whose
#: streaming reads the diagonal neighbours -- the halo corners, which on
#: the pencil mesh hold cells that crossed both mesh axes.  Periodic, the
#: one fill the node takes (``LBMNode.halo_boundary``).
LBM_VISCOSITY = 0.1
LBM_FORCE = 1e-3                # body force amplitude, lattice units


def _lbm_start(ny: int, nx: int, node) -> dict:
    """``LBMNode``'s equilibrium state with a smooth perturbation of ``f``, so
    the populations stream something across the shard boundaries.

    Large enough (velocities of a few 1e-2) that the velocity, a small
    difference of populations of about 0.1, is well above float32
    rounding of the populations: at 1e-3 the sharded and unsharded
    velocities already differed by 4e-6 relative after three steps, from
    a single-ulp difference in ``f``.
    """
    state = dict(node.initial_state())
    # One pattern per population, none with a block period (until schema 6
    # the perturbation had period 1/2 along y, one block on the pencil).
    shape = np.asarray(state["f"]).shape
    pattern = np.stack([_no_block_symmetry(ny, nx, 0.7 * q) for q in range(shape[-1])],
                       axis=-1)
    pert = 0.1 * pattern * np.asarray(state["f"])
    state["f"] = state["f"] + jnp.asarray(pert.astype(np.float32))
    return state


def _lbm_force(ny: int, nx: int) -> np.ndarray:
    """A grid-shaped ``body_force``, ``(ny, nx, 2)``, different on every block
    of every mesh, in both components (until schema 6 its x component was
    uniform along x and its y component along y)."""
    fx, fy = _no_block_symmetry(ny, nx, 1.3), _no_block_symmetry(ny, nx, 3.7)
    return (0.5 * LBM_FORCE * np.stack([fx, fy], axis=-1)).astype(np.float32)


#: What each node kind of the ``stencil`` goal compares forward (besides the
#: rollout's loss and gradients), the parameter it differentiates with
#: respect to, and the field its loss weighs.  ``averages`` is ``Field2D``'s
#: domain integral.  The lattice's loss is a kinetic energy: a loss on
#: ``f**2`` is nearly blind to the viscosity (the populations of a cell
#: sum to its density whatever the viscosity), and its gradient came out
#: of cancellation, 5e-5 apart between two correct programs.
STENCIL_NODES = {
    "field": {"fields": ("f", "averages"), "parameter": "diffusivity", "loss_of": "f"},
    "lbm": {"fields": ("f", "velocity"), "parameter": "viscosity", "loss_of": "velocity"},
}


def _stencil_fns(node, steps: int, grad_steps: int, dt: float, weight, boundary_inputs,
                 parameter: str, loss_of: str):
    """``(rollout, value_and_grad)``, both jitted, both through the public
    ``update(..., params=)`` -- the call a graph traces into its step.

    ``rollout(state, p)`` returns the whole state after ``steps`` steps;
    ``value_and_grad(f0, rest, p)`` differentiates the loss of a
    ``grad_steps`` rollout from ``{**rest, "f": f0}`` with respect to the
    initial ``f`` and to the parameter: the weighted mean square of field
    ``loss_of``, plus the first of the ``averages`` where there are any.
    """
    def advance(state, p, n):
        def body(_, s):
            return node.update(s, boundary_inputs, dt, params={parameter: p})
        return lax.fori_loop(0, n, body, state)

    def loss(f0, rest, p):
        final = advance({**rest, "f": f0}, p, grad_steps)
        x = final[loss_of]
        w = weight if x.ndim == weight.ndim else weight[..., None]
        value = jnp.sum(w * x * x) / x.size
        if "averages" in final:
            value = value + final["averages"][0]
        return value

    return (jax.jit(lambda s, p: advance(s, p, steps)),
            jax.jit(jax.value_and_grad(loss, argnums=(0, 2))))


def stencil_cases(cells, n_devices: int) -> list:
    """``(cells, node, boundary, mesh)`` cases the ``stencil`` goal runs.

    At the smallest size: ``Field2D`` under every ``STENCIL_BOUNDARIES``
    entry and the D2Q9 lattice, on every mesh this device count can build
    (:func:`meshes_that_fit`).  At the other sizes: ``Field2D`` with
    periodic ends on the goal's own mesh (:func:`graph_mesh`, the pencil
    where it fits).
    """
    smallest = min(cells)
    cases = []
    for n in cells:
        if n == smallest:
            for mesh in meshes_that_fit(n_devices):
                cases += [(n, "field", b, mesh) for b in STENCIL_BOUNDARIES]
                cases.append((n, "lbm", "periodic", mesh))
        else:
            cases.append((n, "field", "periodic", graph_mesh(n_devices)))
    return cases


def _stencil_prefix(entry: dict) -> str:
    (ny, nx), (py, pz) = entry["shape"], entry["mesh_shape"]
    return f"{entry['node']} {entry['mesh']} {py}x{pz} {ny}x{nx} {entry['boundary']}"


def mask_axis_for(axis_map: dict) -> int:
    """The spatial axis ``Field2D`` shards its static ``mask`` along under
    ``axis_map``: one the wrapper shards (it refuses any other), axis 0
    where it can."""
    return min(axis_map.values())


def _stencil_pair(kind: str, ny: int, nx: int, boundary: str, mesh, axis_map):
    """``(unsharded node, sharded node, boundary inputs, parameter value)``."""
    if kind == "lbm":
        def make():
            return LBMNode("lbm", 1.0, grid_shape=(ny, nx), viscosity=LBM_VISCOSITY,
                           lattice="D2Q9")
        return (make(), ShardedStencilNode(make(), mesh, axis_map=axis_map, boundary="periodic"),
                {"body_force": jnp.asarray(_lbm_force(ny, nx))}, jnp.float32(LBM_VISCOSITY))

    def make():
        return Field2D("field", ny, nx, exchange=0.3, boundary=boundary,
                       mask_axis=mask_axis_for(axis_map))
    # "dirichlet" is the node's own condition over the wrapper's default
    # "edge" fill; the other two are the wrapper's fill of that name.
    sharded = ShardedStencilNode(make(), mesh, axis_map=axis_map,
                                 boundary="edge" if boundary == "dirichlet" else boundary)
    bi = {"source": jnp.asarray(_source_field(ny, nx))}
    if boundary == "dirichlet":
        bi["wall"] = jnp.float32(STENCIL_WALL)
    return make(), sharded, bi, jnp.float32(0.2)


def _placed_like(sharded, state: dict) -> dict:
    """``state`` placed as the wrapper places a state it built: each grid
    field partitioned over the mesh, each domain integral replicated."""
    integrals = set(sharded.domain_integral_fields())
    return {k: (sharded._place_integral(k, v) if k in integrals        # noqa: SLF001
                else jax.device_put(v, sharded._sharding_for_field(v)))  # noqa: SLF001
            for k, v in state.items()}


def _stencil_side(node, s0, p, *, dt, weight, bi, spec, args) -> tuple:
    """One side (sharded or not) of a stencil case: ``(got, forward record,
    gradient record)``, ``got`` = (final state, loss, d loss / d initial f,
    d loss / d parameter)."""
    rollout, vg = _stencil_fns(node, args.steps, args.grad_steps, dt, weight, bi,
                               spec["parameter"], spec["loss_of"])
    f0, rest = s0["f"], {k: v for k, v in s0.items() if k != "f"}
    fwd_compile_s, _ = compile_seconds(rollout, s0, p)
    final = _host(rollout(s0, p))
    t_fwd = timed(lambda: rollout(s0, p), warmup=args.warmup, repeats=args.repeats)
    grad_compile_s, _ = compile_seconds(vg, f0, rest, p)
    value, (g_f0, g_p) = vg(f0, rest, p)
    t_grad = timed(lambda: vg(f0, rest, p), warmup=args.warmup, repeats=args.repeats)
    got = (final, float(value), np.asarray(jax.device_get(g_f0)), float(g_p))
    return (got,
            {"compile_s": fwd_compile_s,
             "rollout": {**t_fwd, "ms_per_step": t_fwd["median_ms"] / args.steps}},
            {"compile_s": grad_compile_s, "grad": t_grad, "loss": float(value),
             "grad_parameter": float(g_p)})


def run_stencil_case(case, args, unsharded_cache: dict | None = None) -> dict:
    """One :func:`stencil_cases` case: forward rollout and adjoint of the
    sharded node against the unsharded one, timed on both sides.

    The unsharded side does not depend on the mesh: with
    ``unsharded_cache`` (a dict, one per goal) it is run once per node,
    ends and grid, and every mesh's case is compared with that one run.
    """
    _load_backend()
    D = args.n_devices
    n_hint, kind, boundary, mesh_label = case
    mesh, axis_map, (py, pz) = stencil_mesh(mesh_label, D)
    ny, nx = field_shape(n_hint, D)
    spec = STENCIL_NODES[kind]
    ref, sharded, bi, p = _stencil_pair(kind, ny, nx, boundary, mesh, axis_map)
    start = _lbm_start(ny, nx, ref) if kind == "lbm" else dict(ref.initial_state())
    sharded_start = _placed_like(sharded, start)
    dt = ref.delta_t
    weight = jnp.asarray(_loss_weight(ny, nx))
    entry = {"node": kind, "mesh": mesh_label, "mesh_shape": [py, pz], "cells": ny * nx,
             "shape": [ny, nx], "boundary": boundary, "n_devices": D, "steps": args.steps,
             "grad_steps": args.grad_steps, "parameter": spec["parameter"],
             "input_partitioned": _is_partitioned(sharded_start["f"], D),
             "forward": {}, "gradient": {}}
    side = functools.partial(_stencil_side, dt=dt, weight=weight, bi=bi, spec=spec, args=args)
    key = (kind, boundary, ny, nx)
    if unsharded_cache is None or key not in unsharded_cache:
        unsharded = side(ref, start, p)
        if unsharded_cache is not None:
            unsharded_cache[key] = unsharded
    else:
        unsharded = unsharded_cache[key]
    got = {"unsharded": unsharded[0]}
    entry["forward"]["unsharded"], entry["gradient"]["unsharded"] = unsharded[1:]
    got["sharded"], entry["forward"]["sharded"], entry["gradient"]["sharded"] = side(
        sharded, sharded_start, p)
    (s_s, l_s, gf_s, gp_s), (s_u, l_u, gf_u, gp_u) = got["sharded"], got["unsharded"]
    entry["forward"]["parity"] = {field: _diff(s_s[field], s_u[field])
                                  for field in spec["fields"]}
    entry["gradient"]["parity_loss"] = _rel_each(l_s, l_u)
    entry["gradient"]["parity_grad_initial_field"] = _diff(gf_s, gf_u)
    entry["gradient"]["parity_grad_parameter"] = _rel_each(gp_s, gp_u)
    fwd = entry["forward"]
    print(f"[stencil] {_stencil_prefix(entry):>34} fwd "
          f"{fwd['sharded']['rollout']['ms_per_step']:8.3f} ms/step (unsharded "
          f"{fwd['unsharded']['rollout']['ms_per_step']:8.3f})  max_rel "
          + " ".join(f"{k} {v['max_rel']:.1e}" for k, v in fwd["parity"].items())
          + f"  grad f0 {entry['gradient']['parity_grad_initial_field']['max_rel']:.1e}"
          f"  grad {spec['parameter']} {entry['gradient']['parity_grad_parameter']:.1e}")
    return entry


def run_stencil(args, out: dict) -> dict:
    """``Field2D`` and a D2Q9 lattice sharded on the 1-D and the pencil mesh
    against the unsharded node, in every case of :func:`stencil_cases`."""
    _load_backend()
    unsharded: dict = {}
    results = [run_stencil_case(case, args, unsharded)
               for case in stencil_cases(args.cells, args.n_devices)]
    out["results"] = results
    return finish_checks(out, stencil_checks(results, args.n_devices))


def stencil_checks(results: list, n_devices: int) -> list:
    """The ``stencil`` goal's checks, from its ``results`` alone (see
    :func:`exchange_checks`)."""
    checks = []
    if not _pencil_mesh_fits(n_devices):
        checks.append(check_not_run(
            "2d pencil mesh: Field2D under every boundary and the D2Q9 lattice, forward "
            "and adjoint vs unsharded", _pencil_mesh_needs(n_devices)))
    checks += _neighbour_direction_not_run((r["mesh_shape"] for r in results),
                                           "ShardedStencilNode")
    for entry in results:
        fwd, grad = entry["forward"], entry["gradient"]
        prefix, param = _stencil_prefix(entry), entry["parameter"]
        gp_u = grad["unsharded"]["grad_parameter"]
        checks.append(check_that(f"{prefix}: sharded state is partitioned over {n_devices} "
                                 "devices", entry["input_partitioned"]))
        for field in STENCIL_NODES[entry["node"]]["fields"]:
            checks += _parity_checks(f"{prefix} forward {field} vs unsharded",
                                     fwd["parity"][field], LIMITS["forward"])
        checks.append(check(f"{prefix} loss vs unsharded rel", grad["parity_loss"],
                            LIMITS["gradient"]))
        checks += _parity_checks(f"{prefix} d loss / d initial f vs unsharded",
                                 grad["parity_grad_initial_field"], LIMITS["gradient"])
        checks.append(check(f"{prefix} d loss / d {param} vs unsharded rel",
                            grad["parity_grad_parameter"], LIMITS["gradient"]))
        checks.append(check_that(f"{prefix} d loss / d {param} is not zero", gp_u != 0.0,
                                 f"{gp_u:.6e}"))
    return checks


# ---------------------------------------------------------------------------
# Graph-level goals: hybrid (checklist 4) and coupled (checklist 6)
# ---------------------------------------------------------------------------


def graph_value_and_grad(gm, steps: int, writes, loss_of_state, external_inputs=None):
    """``jit(value_and_grad)`` of ``theta -> loss`` over ``steps`` graph steps.

    ``theta[k]`` is written into a copy of ``gm.params`` at
    ``writes[k] = (node, key)``; the step is the graph's own pure step
    function, scanned, as ``maddening.sysid`` does, with
    ``external_inputs`` held at every step (zeros where not given, as
    ``run_scan`` does).  The aux output is the final state (``_meta``
    included).  ``gm``'s own state is not advanced.
    """
    step_fn = gm._build_step_fn()  # noqa: SLF001
    ext = gm._resolve_external_inputs(external_inputs)  # noqa: SLF001
    state0 = gm._state  # noqa: SLF001

    def loss(theta):
        params = jax.tree.map(lambda x: x, gm.params)
        for k, (node, key) in enumerate(writes):
            params["nodes"][node][key] = theta[k]

        def body(state, _):
            return step_fn(state, ext, params), None

        final, _ = lax.scan(body, state0, None, length=steps)
        return loss_of_state(final), final

    return jax.jit(jax.value_and_grad(loss, has_aux=True))


def make_hybrid_correction(shift: tuple):
    """An additive correction on the *global* field, outside any ``shard_map``:
    a nonlinear term and a shift by ``shift = (rows, columns)`` along both
    spatial axes, which on a field partitioned along either is a
    cross-device permutation (until schema 6 it shifted rows only, local on
    a mesh that shards axis 1 alone)."""
    def correction(state, boundary_inputs, dt):
        f = state["f"]
        moved = jnp.roll(f, tuple(shift), axis=(0, 1))
        return {"f": dt * (0.05 * jnp.tanh(f - 1.0) + 0.1 * (moved - f))}
    return correction


def _pencil_not_run(goal_claim: str, n_devices: int) -> list:
    """A graph goal's check not run on a device count with no pencil mesh."""
    if _pencil_mesh_fits(n_devices):
        return []
    return [check_not_run(f"2d pencil mesh: {goal_claim}", _pencil_mesh_needs(n_devices))]


def _graph_prefix(entry: dict) -> str:
    (ny, nx), (py, pz) = entry["shape"], entry["mesh_shape"]
    return f"{entry['mesh']} {py}x{pz} {ny}x{nx}"


def _cached(cache: dict, key, compute):
    """``cache[key]``, computed once."""
    if key not in cache:
        cache[key] = compute()
    return cache[key]


def run_hybrid(args, out: dict) -> dict:
    """``HybridNode(ShardedStencilNode(inner))`` against ``HybridNode(inner)``,
    on every mesh at the smallest size and the pencil at the others
    (:func:`graph_cases`), with a grid-shaped ``source`` held through an
    external input and ``Field2D``'s domain integral compared."""
    _load_backend()
    D = args.n_devices
    results = []
    writes = list(HYBRID_WRITES)
    theta = jnp.asarray([0.2, 0.3], jnp.float32)
    unsharded: dict = {}
    for n_hint, mesh_label in graph_cases(args.cells, D):
        mesh, axis_map, (py, pz) = stencil_mesh(mesh_label, D)
        ny, nx = field_shape(n_hint, D)
        shift = (max(1, ny // (2 * D)), max(1, nx // (2 * D)))
        correction = make_hybrid_correction(shift)
        ext = {"field": {"source": jnp.asarray(_source_field(ny, nx))}}

        def graph(sharded: bool, _ny=ny, _nx=nx, _correction=correction, _mesh=mesh,
                  _axis_map=axis_map):
            inner = Field2D("field", _ny, _nx, exchange=0.3, mask_axis=mask_axis_for(_axis_map))
            physics = (ShardedStencilNode(inner, _mesh, axis_map=_axis_map,
                                          boundary="periodic") if sharded else inner)
            gm = GraphManager()
            gm.add_node(HybridNode(physics, _correction))
            gm.add_external_input("field", "source", shape=(_ny, _nx))
            gm.compile()
            return gm

        def side(sharded: bool, _graph=graph, _ext=ext):
            gm = _graph(sharded)
            t0 = time.perf_counter()
            final_state = gm.run_scan(args.steps, external_inputs=_ext)["field"]
            f, averages = final_state["f"], final_state["averages"]
            jax.block_until_ready(f)
            scan_s = time.perf_counter() - t0
            # run_scan advanced that graph's state: the adjoint gets a fresh graph.
            vg = graph_value_and_grad(_graph(sharded), args.grad_steps, writes,
                                      lambda st: jnp.mean(st["field"]["f"] ** 2),
                                      external_inputs=_ext)
            compile_s, _ = compile_seconds(vg, theta)
            (value, _final), grad = vg(theta)
            t_grad = timed(lambda: vg(theta), warmup=args.warmup, repeats=args.repeats)
            got = (np.asarray(jax.device_get(f)), np.asarray(jax.device_get(averages)),
                   float(value), np.asarray(grad))
            record = {"run_scan_first_call_s": scan_s, "partitioned": _is_partitioned(f, D),
                      "compile_s": compile_s, "value_and_grad": t_grad,
                      "loss": float(value), "grad": np.asarray(grad).tolist()}
            return got, record

        entry = {"mesh": mesh_label, "mesh_shape": [py, pz], "cells": ny * nx,
                 "shape": [ny, nx], "n_devices": D, "steps": args.steps,
                 "grad_steps": args.grad_steps, "correction_shift": list(shift)}
        # The unsharded graph does not depend on the mesh: once per grid.
        got_u, entry["unsharded"] = _cached(unsharded, (ny, nx), lambda _s=side: _s(False))
        got_s, entry["sharded"] = side(True)
        (f_s, a_s, l_s, g_s), (f_u, a_u, l_u, g_u) = got_s, got_u
        corr = np.asarray(correction({"f": jnp.asarray(f_u)}, {}, 0.1)["f"])
        entry["correction_rel"] = float(np.max(np.abs(corr)) / np.max(np.abs(f_u)))
        entry["parity_f"] = _diff(f_s, f_u)
        entry["parity_averages"] = _diff(a_s, a_u)
        entry["parity_loss"] = _rel_each(l_s, l_u)
        entry["parity_grad"] = _rel_each(g_s, g_u)
        results.append(entry)
        print(f"[hybrid] {_graph_prefix(entry):>22} max_rel f {entry['parity_f']['max_rel']:.1e}"
              f"  averages {entry['parity_averages']['max_rel']:.1e}  grad "
              f"{entry['parity_grad']:.1e}  value+grad "
              f"{entry['sharded']['value_and_grad']['median_ms']:8.2f} ms (unsharded "
              f"{entry['unsharded']['value_and_grad']['median_ms']:8.2f})")
    out["results"] = results
    return finish_checks(out, hybrid_checks(results, D))


def hybrid_checks(results: list, n_devices: int) -> list:
    """The ``hybrid`` goal's checks, from its ``results`` alone (see
    :func:`exchange_checks`)."""
    params = ", ".join(key for _node, key in HYBRID_WRITES)
    checks = _pencil_not_run("HybridNode(ShardedStencilNode(inner)) forward and adjoint vs "
                             "HybridNode(inner)", n_devices)
    checks += _neighbour_direction_not_run((r["mesh_shape"] for r in results),
                                           "HybridNode(ShardedStencilNode(inner))")
    for entry in results:
        g_u = entry["unsharded"]["grad"]
        prefix = _graph_prefix(entry)
        checks += [
            check_that(f"{prefix}: the field inside the hybrid is partitioned over {n_devices} "
                       "devices", entry["sharded"]["partitioned"]),
            check_that(f"{prefix}: the correction is not negligible (> 100x the forward limit)",
                       entry["correction_rel"] > 100 * LIMITS["forward"],
                       f"{entry['correction_rel']:.3e}"),
            *_parity_checks(f"{prefix} run_scan f vs HybridNode(inner)", entry["parity_f"],
                            LIMITS["forward"]),
            *_parity_checks(f"{prefix} run_scan domain integral averages vs HybridNode(inner)",
                            entry["parity_averages"], LIMITS["forward"]),
            check(f"{prefix} loss vs HybridNode(inner) rel", entry["parity_loss"],
                  LIMITS["gradient"]),
            check(f"{prefix} d loss / d ({params}) vs HybridNode(inner) rel",
                  entry["parity_grad"], LIMITS["gradient"]),
            check_that(f"{prefix} both gradient components are non-zero",
                       bool(np.all(np.asarray(g_u, np.float64) != 0.0)), str(list(g_u))),
        ]
    return checks


def _first(averages):
    """The mean out of ``Field2D``'s ``averages`` (an edge transform)."""
    return averages[0]


def _spread(u, *, profile):
    """The far field's value spread over the grid (an edge transform)."""
    return u * profile


def coupled_graph(ny: int, nx: int, mesh, solver: str, axis_map=None):
    """A ``Field2D`` member (sharded when ``mesh`` is given) and a replicated
    ``FarField`` member in one coupling group.

    The field reaches the far field through its domain integral (the mean
    in ``averages``), and the far field reaches the field as a grid-shaped
    ``ambient``, ``u * coupled_profile`` -- different on every block of
    every mesh -- so both of the wrapper's paths for them are inside the
    group's fixed point and its adjoint.
    """
    axis_map = axis_map or {MESH_AXIS: 0}
    field = Field2D("field", ny, nx, exchange=0.8, mask_axis=mask_axis_for(axis_map))
    gm = GraphManager()
    gm.add_node(field if mesh is None else ShardedStencilNode(
        field, mesh, axis_map=axis_map, boundary="periodic"))
    gm.add_node(FarField("far"))
    gm.add_edge("field", "far", "averages", "field_mean", transform=_first)
    gm.add_edge("far", "field", "u", "ambient",
                transform=functools.partial(_spread,
                                            profile=jnp.asarray(coupled_profile(ny, nx))))
    kwargs = {} if solver == "ift" else {"solver": solver}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)      # solver="fori"
        group = gm.add_coupling_group(["field", "far"], max_iterations=COUPLED_MAX_ITERATIONS,
                                      tolerance=COUPLED_TOLERANCE, **kwargs)
    if group.solver != solver:
        raise RuntimeError(f"coupling group solver is {group.solver!r}, expected {solver!r}: "
                           "the default moved; update COUPLED_SOLVERS and the limits")
    gm.compile()
    return gm


def _device0_pins(lowered) -> int:
    """Ops the lowered program pins to one device (``maximal`` sharding)."""
    return sum(1 for line in lowered.as_text().splitlines()
               if "maximal" in line and "sdy.mesh" not in line)


def coupled_model(field, theta, steps: int, u0: float) -> tuple:
    """``(f, u, loss)`` of the coupled group's rollout in float64 NumPy.

    Independent of the graph machinery: at convergence each step solves
    both members' updates at once, each reading the other's *new* value --
    ``f' = f + dt (D div(k grad f) + h (u' p - f))`` and
    ``u' = u + dt c (mean f' - u)``, ``k`` the face-averaged mask of
    ``Field2D`` on a periodic grid and ``p`` the :func:`coupled_profile`
    the far field is spread by.  ``mean f'`` and ``u'`` satisfy a 2x2
    linear system (``mean(div(k grad f))`` is known from ``f``, and
    ``mean(u' p) = u' mean(p)``), and ``f'`` follows.  ``theta = (D, c,
    h)``; the field's own float32 initial state, mask and profile are
    widened exactly.
    """
    d, c, h = (float(t) for t in theta)
    dt = float(np.float32(field.delta_t))
    f = np.asarray(field.initial_state()["f"], np.float64)
    mask = np.asarray(field.static_data["mask"].value, np.float64)
    profile = np.asarray(coupled_profile(*f.shape), np.float64)
    p_mean = profile.mean()
    u = float(np.float32(u0))
    # (shift, axis) of each neighbour and the conductance of the face to it
    faces = [((s, a), 0.5 * (mask + np.roll(mask, s, a))) for a in (0, 1) for s in (1, -1)]
    for _ in range(steps):
        div = sum(k * (np.roll(f, s, a) - f) for (s, a), k in faces)
        m, mean_div = f.mean(), div.mean()
        m_new, u_new = np.linalg.solve(
            np.array([[1.0, -dt * h * p_mean], [-dt * c, 1.0]]),
            np.array([m + dt * d * mean_div - dt * h * m, u - dt * c * u]))
        f = f + dt * (d * div + h * (u_new * profile - f))
        u = float(u_new)
    return f, u, float(np.mean(f ** 2) + u ** 2)


def coupled_model_gradient(field, theta, steps: int, u0: float) -> np.ndarray:
    """Central differences of :func:`coupled_model`'s loss (relative step 1e-6)."""
    theta = np.asarray(theta, np.float64)
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = 1e-6 * theta[i]
        grad[i] = (coupled_model(field, theta + step, steps, u0)[2]
                   - coupled_model(field, theta - step, steps, u0)[2]) / (2 * step[i])
    return grad


def run_coupled(args, out: dict) -> dict:
    """Checklist 6: forward and adjoint of the group, sharded vs unsharded,
    and both against the float64 model of the coupled step, on every mesh
    at the smallest size and the pencil at the others (:func:`graph_cases`)."""
    _load_backend()
    D = args.n_devices
    writes = list(COUPLED_WRITES)
    theta = jnp.asarray([0.2, 2.0, 0.8], jnp.float32)
    iterations_key = "coupling_far+field_iterations"
    results = []
    models: dict = {}
    unsharded: dict = {}

    def loss_of(state):
        return jnp.mean(state["field"]["f"] ** 2) + state["far"]["u"] ** 2

    def model(ny, nx):
        t0 = time.perf_counter()
        model_field = Field2D("field", ny, nx, exchange=0.8)
        u0 = float(FarField("far").initial_state()["u"])
        theta64 = np.asarray(theta, np.float64)             # the float32 values, widened
        f_model, u_model, loss_model = coupled_model(model_field, theta64, args.grad_steps, u0)
        averages_model = np.array([np.mean(f_model), np.mean(f_model ** 2)])
        grad_model = coupled_model_gradient(model_field, theta64, args.grad_steps, u0)
        return (f_model, u_model, loss_model, averages_model, grad_model,
                {"loss": loss_model, "grad": grad_model.tolist(),
                 "wall_s": time.perf_counter() - t0})

    def side(ny, nx, mesh, solver, axis_map):
        vg = graph_value_and_grad(coupled_graph(ny, nx, mesh, solver, axis_map),
                                  args.grad_steps, writes, loss_of)
        t0 = time.perf_counter()
        lowered = vg.lower(theta)
        lowered.compile()
        compile_s = time.perf_counter() - t0
        (value, final), grad = vg(theta)
        t = timed(lambda: vg(theta), warmup=args.warmup, repeats=args.repeats)
        f = final["field"]["f"]
        meta = final.get("_meta", {})
        iterations = int(meta[iterations_key]) if iterations_key in meta else None
        got = (np.asarray(jax.device_get(f)), float(final["far"]["u"]),
               float(value), np.asarray(grad),
               np.asarray(jax.device_get(final["field"]["averages"])))
        record = {"compile_s": compile_s,
                  "value_and_grad": {**t, "ms_per_step": t["median_ms"] / args.grad_steps},
                  "loss": float(value), "grad": np.asarray(grad).tolist(),
                  "last_step_iterations": iterations,
                  "partitioned": _is_partitioned(f, D),
                  "device0_pinned_ops": _device0_pins(lowered)}
        return got, record

    for n_hint, mesh_label in graph_cases(args.cells, D):
        mesh, axis_map, (py, pz) = stencil_mesh(mesh_label, D)
        ny, nx = field_shape(n_hint, D)
        # The field, its two averages (a domain integral in its state) and u.
        entry = {"mesh": mesh_label, "mesh_shape": [py, pz], "cells": ny * nx,
                 "shape": [ny, nx], "n_devices": D, "steps": args.grad_steps,
                 "coupled_dof": ny * nx + 2 + 1, "max_iterations": COUPLED_MAX_ITERATIONS,
                 "tolerance": COUPLED_TOLERANCE, "parameters": [f"{n}.{k}" for n, k in writes],
                 "solvers": {}}
        # Neither the model nor the unsharded group depends on the mesh.
        (f_model, u_model, loss_model, averages_model, grad_model,
         entry["model"]) = _cached(models, (ny, nx), lambda _ny=ny, _nx=nx: model(_ny, _nx))
        for solver in COUPLED_SOLVERS:
            got, sol = {}, {}
            got["unsharded"], sol["unsharded"] = _cached(
                unsharded, (ny, nx, solver),
                lambda _ny=ny, _nx=nx, _s=solver: side(_ny, _nx, None, _s, None))
            got["sharded"], sol["sharded"] = side(ny, nx, mesh, solver, axis_map)
            (f_s, u_s, l_s, g_s, a_s), (f_u, u_u, l_u, g_u, a_u) = (got["sharded"],
                                                                   got["unsharded"])
            sol["parity_f"] = _diff(f_s, f_u)
            sol["parity_averages"] = _diff(a_s, a_u)
            sol["parity_u"] = _rel_each(u_s, u_u)
            sol["parity_loss"] = _rel_each(l_s, l_u)
            sol["parity_grad"] = _rel_each(g_s, g_u)
            sol["model"] = {
                label: {"f": _diff(got[label][0], f_model),
                        "averages": _diff(got[label][4], averages_model),
                        "u": _rel_each(got[label][1], u_model),
                        "loss": _rel_each(got[label][2], loss_model),
                        "grad": _rel_each(got[label][3], grad_model)}
                for label in ("sharded", "unsharded")}
            entry["solvers"][solver] = sol
            prefix = f"{_graph_prefix(entry)} {solver}"
            print(f"[coupled] {prefix:>27} max_rel f {sol['parity_f']['max_rel']:.1e}  u "
                  f"{sol['parity_u']:.1e}  grad {sol['parity_grad']:.1e}  vs model grad "
                  f"{max(v['grad'] for v in sol['model'].values()):.1e}  value+grad "
                  f"{sol['sharded']['value_and_grad']['median_ms']:8.2f} ms (unsharded "
                  f"{sol['unsharded']['value_and_grad']['median_ms']:8.2f})  compile "
                  f"{sol['sharded']['compile_s']:5.1f} s  pinned ops "
                  f"{sol['sharded']['device0_pinned_ops']}")
        results.append(entry)
    out["results"] = results
    return finish_checks(out, coupled_checks(results, D))


def coupled_checks(results: list, n_devices: int) -> list:
    """The ``coupled`` goal's checks, from its ``results`` alone (see
    :func:`exchange_checks`)."""
    params = ", ".join(f"{node}.{key}" for node, key in COUPLED_WRITES)
    checks = _pencil_not_run("a sharded and a replicated member in one coupling group, "
                             "forward and adjoint", n_devices)
    checks += _neighbour_direction_not_run((r["mesh_shape"] for r in results),
                                           "the coupled group's sharded member")
    for entry in results:
        for solver in COUPLED_SOLVERS:
            sol = entry["solvers"][solver]
            prefix = f"{_graph_prefix(entry)} {solver}"
            g_u = sol["unsharded"]["grad"]
            for label in ("sharded", "unsharded"):
                vs = sol["model"][label]
                checks += [
                    check(f"{prefix} {label} field vs float64 model max_rel", vs["f"]["max_rel"],
                          LIMITS["forward"]),
                    check(f"{prefix} {label} domain integral averages vs float64 model max_rel",
                          vs["averages"]["max_rel"], LIMITS["forward"]),
                    check(f"{prefix} {label} far field and loss vs float64 model rel",
                          max(vs["u"], vs["loss"]), LIMITS["forward"]),
                    check(f"{prefix} {label} gradient vs float64 model differences rel",
                          vs["grad"], LIMITS["model_gradient"]),
                ]
            checks += [
                check_that(f"{prefix}: the field member is partitioned over {n_devices} devices",
                           sol["sharded"]["partitioned"]),
                *_parity_checks(f"{prefix} field vs unsharded group", sol["parity_f"],
                                LIMITS["forward"]),
                *_parity_checks(f"{prefix} domain integral averages vs unsharded group",
                                sol["parity_averages"], LIMITS["forward"]),
                check(f"{prefix} far field vs unsharded group rel", sol["parity_u"],
                      LIMITS["forward"]),
                check(f"{prefix} loss vs unsharded group rel", sol["parity_loss"],
                      LIMITS["forward"]),
                check(f"{prefix} d loss / d ({params}) vs unsharded group rel",
                      sol["parity_grad"], LIMITS[f"coupled_gradient_{solver}"]),
                check_that(f"{prefix} every gradient component is non-zero",
                           bool(np.all(np.asarray(g_u, np.float64) != 0.0)), str(list(g_u))),
            ]
            if solver == "ift":
                its = (sol["unsharded"]["last_step_iterations"],
                       sol["sharded"]["last_step_iterations"])
                checks.append(check_that(
                    f"{prefix}: the last step iterated (2 <= passes < {COUPLED_MAX_ITERATIONS}) "
                    "on both paths",
                    all(isinstance(i, int) and not isinstance(i, bool)
                        and 2 <= i < COUPLED_MAX_ITERATIONS for i in its), str(its)))
    return checks


def check_checklist_device_count(n_devices: int) -> None:
    """A sharding check on one device compares a program with itself.

    2 or 3 devices run (to prove the script, or to localise a failure),
    but their results never close a checklist item: see
    ``MIN_DECIDING_DEVICES`` and :func:`closes_the_gap`."""
    if n_devices < MIN_DEVICES_WITH_ESCAPE:
        raise SystemExit(f"the checklist goals need >= {MIN_DEVICES_WITH_ESCAPE} devices "
                         f"(got --n-devices {n_devices}): on one device nothing is sharded")


# ---------------------------------------------------------------------------
# Summary / recommendation (no JAX)
# ---------------------------------------------------------------------------


def _load_results(directory: Path, goal: str) -> list[dict]:
    """Every ``<goal>*.json`` under *directory*.

    A file that cannot be read as a JSON object -- one ``timeout`` killed the
    runner in the middle of writing, an empty one, one holding a list -- is
    kept as a stand-in that records *goal* and why it cannot be read
    (``_unreadable``): :func:`record_problems` reads it ``INVALID``, so the
    summary says which file and exits 3.  It used to end the summary on the
    ``JSONDecodeError`` (or ``AttributeError``) with exit status 1, the one
    the runbook gives for "no goal JSON".  A file named for *goal* that
    records another goal, or none, is kept the same way (``_misfiled``).
    """
    docs = []
    for path in sorted(directory.glob(f"{goal}*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            why = None if isinstance(doc, dict) else \
                f"it holds a JSON {type(doc).__name__}, not an object"
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            why = f"{type(exc).__name__}: {exc}"
        if why is not None:
            # Read as a file with no results and no checks, recorded as not
            # passed, so every verdict below lands on INVALID for it.
            doc = {"goal": goal, "_unreadable": why, "results": [], "checks": [],
                   "passed": False}
        elif doc.get("goal") != goal:
            # A file named for this goal that records another goal, or none
            # (an empty object, a re-run saved under a goal's name): kept the
            # same way, so it reads INVALID and the summary exits 3.  It used
            # to be dropped without a word, and the summary exited 0.
            doc = {"goal": goal, "_misfiled": (
                       f"its name says goal {goal!r}, but it records goal "
                       f"{doc.get('goal')!r}: a file this runner would not have written"),
                   "results": [], "checks": [], "passed": False}
        else:
            # A JSON object of another shape -- a field of another type --
            # read INVALID too.  It used to end the summary on a traceback
            # (exit 1, "no goal JSON"): ``environment: null``, a check that is
            # a string, a value past the float range, ``n_devices: [4]``.
            why = _shape_problem(doc) or _verdict_problem(goal, doc)
            if why is not None:
                doc = {"goal": goal, "_unreadable": why, "results": [], "checks": [],
                       "passed": False}
        doc["_file"] = path.name              # for the reader; never part of a record
        docs.append(doc)
    return docs


def _kind(value) -> str:
    return "null" if value is None else f"a JSON {type(value).__name__}"


def _shape_problem(doc: dict) -> str | None:
    """Why *doc* -- a goal file read as a JSON object -- has a field of a
    type the summary cannot read, or ``None``.  Absent fields are left to
    :func:`record_problems`, which says what a record lacks."""
    env = doc.get("environment", {})
    if not isinstance(env, dict):
        return f"its environment is {_kind(env)}, not an object"
    for key in ("platform", "jax", "jaxlib"):
        if key in env and not isinstance(env[key], str):
            return f"its environment's {key} is {_kind(env[key])}, not a string"
    kinds = env.get("device_kinds", [])
    if not (isinstance(kinds, list) and all(isinstance(k, str) for k in kinds)):
        return "its environment's device_kinds is not a list of strings"
    cfg = doc.get("config", {})
    if not isinstance(cfg, dict):
        return f"its config is {_kind(cfg)}, not an object"
    for where, value in (("n_devices", doc.get("n_devices")),
                         ("config's n_devices", cfg.get("n_devices"))):
        if value is not None and not _is_count(value):
            return f"its {where} is {value!r}, not a device count"
    for key in ("checks", "results"):
        if key in doc and not isinstance(doc[key], list):
            return f"its {key} is {_kind(doc[key])}, not a list"
    for i, c in enumerate(doc.get("checks") or []):
        if not isinstance(c, dict):
            return f"its check {i} is {_kind(c)}, not a record"
        for key in ("value", "limit"):
            v = c.get(key)
            if _is_count(v):
                try:
                    float(v)
                except OverflowError:
                    return f"its check {c.get('name')!r} has a {key} past the float range"
    for i, r in enumerate(doc.get("results") or []):
        if not isinstance(r, dict):
            return f"its result {i} is {_kind(r)}, not a record"
    return None


def _verdict_problem(goal: str, doc: dict) -> str | None:
    """Why the summary's own reading of *doc* -- its verdict, its line in
    the run table, whether it closes the CPU gap -- raises, or ``None``:
    the net under :func:`_shape_problem` for a shape it does not name, so
    that a file of any shape reads INVALID rather than ending the summary
    on a traceback."""
    try:
        record_problems(doc)
        goal_verdict([doc])
        _run_line(goal, doc)
        closes_the_gap(doc)
        _commit_of(doc)
    except _UNREADABLE as exc:
        return f"the summary cannot read it ({type(exc).__name__}: {exc})"
    return None


# --- record integrity -------------------------------------------------------
#
# A file's checks are only evidence if this runner, as it stands, would
# have written them.  ``--summarise`` used to take four things on each
# file's word: the limit a check was held to, the set of checks, the
# schema, and where and on what the file was measured.  Every one of them
# is recorded, so each is now checked against the runner itself:
#
# * the file is on the current ``SCHEMA_VERSION``;
# * its ``n_devices`` is no more than the devices its environment saw, and
#   agrees with its config and with every result entry;
# * its results hold exactly the cases the runner runs for the file's own
#   config (``cells``, ``n_devices``, ``synthetic``, ``mesh``) -- the
#   minimum check set, derived from the runner's own case lists
#   (``stencil_cases``, ``HALO_BOUNDARIES``, ``COUPLED_SOLVERS``, ...);
# * its checks are exactly the ones ``GOAL_CHECKS[goal]`` derives from its
#   results: same names, values, limits (so every limit is ``LIMITS``'),
#   senses and flags -- results that disagree with the checks, a limit
#   loosened in the record, a check deleted or one added all show here;
#
# and across the files that decide one checklist item, a single recorded
# ``git_commit``.  A file that fails any of this reads ``INVALID`` and
# cannot close an item; the item says which file and why.

GOAL_CHECKS = {
    "indivisible": indivisible_checks, "halo": halo_checks, "coupled": coupled_checks,
    "stencil": stencil_checks, "hybrid": hybrid_checks, "exchange": exchange_checks,
    "forward": forward_checks, "gradient": gradient_checks,
}
"""Goal -> the function its runner builds its checks with, from its results."""


def _file_of(doc: dict) -> str:
    return doc.get("_file") or f"{doc.get('goal')}.json"


def _is_count(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _shape(x) -> tuple:
    return tuple(x)


def case_keys(goal: str, doc: dict) -> tuple[Counter, Counter]:
    """``(expected, recorded)``: the cases the runner runs for this file's
    own config, and the cases its results hold, as comparable keys.

    Derived from the same case lists and size rules the runners use, so
    the two cannot drift apart.  Raises ``KeyError`` / ``TypeError`` /
    ``ValueError`` on results or a config it cannot read.
    """
    cfg, results, n_dev = doc["config"], doc["results"], doc["n_devices"]
    cells = [int(n) for n in cfg["cells"]]
    methods = tuple(sorted(METHODS))
    want: Counter = Counter()
    got: Counter = Counter()
    if goal == "exchange":
        # The requested size too: recommend() decides on it (and on the
        # measured count reaching it), so a row must record the one it was
        # measured for.
        want.update((synthetic_cells(cfg["synthetic"], n), n, methods) for n in cells)
        got.update((r["cells"], r["requested_cells"], tuple(sorted(r["methods"])))
                   for r in results)
    elif goal in ("forward", "gradient"):
        file_mesh = bool(cfg.get("mesh"))
        sizes = [cells[-1]] if file_mesh else cells
        if goal == "forward":
            per, extra = methods, ()
        else:
            per, extra = tuple(sorted((*METHODS, "unsharded"))), ("sharded_cg",)
        want.update((None if file_mesh else synthetic_cells(cfg["synthetic"], n), per, extra)
                    for n in sizes)
        got.update((None if file_mesh else r["cells"],
                    tuple(sorted(r["methods" if goal == "forward" else "rollout"])),
                    tuple(k for k in extra if r.get(k) is not None)) for r in results)
    elif goal == "indivisible":
        ny_ok, nx = field_shape(min(cells), n_dev)
        ny_bad = ny_ok + 1
        n_un = ny_bad * nx + (1 if (ny_bad * nx) % n_dev == 0 else 0)
        pencil = None
        if _pencil_mesh_fits(n_dev):
            nz = n_dev // 2
            pencil = ((2, nz), (16, 8 * nz + 1))
        want[((ny_bad, nx), (ny_ok, nx + 1), pencil, n_un)] += 1
        for r in results:
            p = r.get("pencil")
            got[(_shape(r["stencil"]["shape"]), _shape(r["stencil_axis1"]["shape"]),
                 None if p is None else (_shape(p["mesh"]), _shape(p["shape"])),
                 r["unstructured"]["cells"])] += 1
            if r.get("pointwise") is None:
                got[("pointwise", "missing")] += 1
    elif goal == "halo":
        ny, nx = field_shape(max(cells), n_dev)
        for label in meshes_that_fit(n_dev):
            want.update((label, _mesh_shape_of(label, n_dev), (ny, nx), h, b)
                        for b in HALO_BOUNDARIES for h in HALO_WIDTHS)
        want[("unstructured", synthetic_cells(cfg["synthetic"], max(cells)), methods)] += 1
        for r in results:
            got.update((c["mesh"], _shape(c["mesh_shape"]), _shape(c["shape"]), c["halo"],
                        c["boundary"]) for c in r["stencil_cases"])
            un = r["unstructured"]
            got[("unstructured", un["cells"], tuple(sorted(un["methods"])))] += 1
    elif goal == "stencil":
        want.update((kind, mesh, _mesh_shape_of(mesh, n_dev), field_shape(n, n_dev), b)
                    for n, kind, b, mesh in stencil_cases(cells, n_dev))
        got.update((r["node"], r["mesh"], _shape(r["mesh_shape"]), _shape(r["shape"]),
                    r["boundary"]) for r in results)
    elif goal == "hybrid":
        want.update((mesh, _mesh_shape_of(mesh, n_dev), field_shape(n, n_dev))
                    for n, mesh in graph_cases(cells, n_dev))
        got.update((r["mesh"], _shape(r["mesh_shape"]), _shape(r["shape"])) for r in results)
    elif goal == "coupled":
        solvers = tuple(sorted(COUPLED_SOLVERS))
        want.update((mesh, _mesh_shape_of(mesh, n_dev), field_shape(n, n_dev), solvers)
                    for n, mesh in graph_cases(cells, n_dev))
        got.update((r["mesh"], _shape(r["mesh_shape"]), _shape(r["shape"]),
                    tuple(sorted(r["solvers"]))) for r in results)
    else:
        raise ValueError(f"unknown goal {goal!r}")
    return want, got


def _fmt_case(key) -> str:
    if isinstance(key, tuple):
        if len(key) == 2 and all(_is_count(k) for k in key):
            return f"{key[0]}x{key[1]}"
        return " ".join(_fmt_case(k) for k in key if k not in (None, ()))
    return str(key)


def _fmt_few(items, n: int = 3) -> str:
    items = list(items)
    text = ", ".join(items[:n])
    return text + (f" (+{len(items) - n} more)" if len(items) > n else "")


def _canonical(c: dict) -> tuple:
    """What a check claims, comparably (NaN equal to NaN, ``detail`` aside)."""
    return (c.get("name"), json.dumps(c.get("value"), default=repr),
            json.dumps(c.get("limit"), default=repr), c.get("sense"), c.get("passed"),
            bool(c.get("not_run", False)))


def _check_differences(recorded: list, derived: list) -> list[str]:
    """How a file's recorded checks differ from the ones derived from its
    results, one line per kind of difference."""
    if Counter(map(_canonical, recorded)) == Counter(map(_canonical, derived)):
        return []
    rec: dict = {}
    for c in recorded:
        rec.setdefault(c.get("name"), []).append(c)
    der: dict = {}
    for c in derived:
        der.setdefault(c["name"], []).append(c)
    out = []
    lacking = [n for n in der if n not in rec]
    if lacking:
        out.append(f"lacks {len(lacking)} check(s) the runner derives from its results: "
                   + _fmt_few(repr(n) for n in lacking))
    unknown = [n for n in rec if n not in der]
    if unknown:
        out.append(f"has {len(unknown)} check(s) the runner does not emit for its results: "
                   + _fmt_few(repr(n) for n in unknown))
    limits, values, other = [], [], []
    for name in (n for n in rec if n in der):
        r, d = rec[name], der[name]
        if Counter(map(_canonical, r)) == Counter(map(_canonical, d)):
            continue
        if len(r) != len(d):
            other.append(f"{name!r} recorded {len(r)} time(s), derived {len(d)}")
            continue
        for rc, dc in zip(r, d):
            if _canonical(rc)[2] != _canonical(dc)[2]:
                limits.append(f"{name!r} records limit {rc.get('limit')!r}, the runner "
                              f"holds it to {dc.get('limit')!r}")
            elif _canonical(rc)[1] != _canonical(dc)[1]:
                values.append(f"{name!r} records {rc.get('value')!r}, its results give "
                              f"{dc.get('value')!r}")
            elif _canonical(rc) != _canonical(dc):
                other.append(f"{name!r} records sense/passed/not_run "
                             f"{(rc.get('sense'), rc.get('passed'), rc.get('not_run', False))!r}"
                             f", derived {(dc.get('sense'), dc.get('passed'), dc.get('not_run', False))!r}")
    if limits:
        out.append(f"{len(limits)} limit(s) differ from LIMITS: " + _fmt_few(limits, 2))
    if values:
        out.append(f"{len(values)} check value(s) disagree with its results: "
                   + _fmt_few(values, 2))
    if other:
        out.append(f"{len(other)} check(s) differ: " + _fmt_few(other, 2))
    return out


def record_problems(doc: dict) -> list[str]:
    """Why this file's checks are not evidence this runner would have
    written, one line each; empty when they are.  See the block comment
    above for what is checked and why."""
    problems: list[str] = []
    if "_unreadable" in doc:
        return [f"the file cannot be read as a JSON object ({doc['_unreadable']}): "
                "a run killed while writing it leaves one like this"]
    if "_misfiled" in doc:
        return [doc["_misfiled"]]
    goal = doc.get("goal")
    if goal not in GOAL_CHECKS:
        return [f"unknown goal {goal!r}"]
    version = doc.get("schema_version")
    if not (_is_count(version) and version == SCHEMA_VERSION):
        problems.append(f"schema_version {version!r}, not {SCHEMA_VERSION}: written by "
                        "another version of this runner")
    env, cfg, n_dev = doc.get("environment"), doc.get("config"), doc.get("n_devices")
    if not isinstance(env, dict) or not isinstance(cfg, dict):
        return problems + ["no environment or config recorded"]
    if not _is_count(n_dev) or n_dev < 1:
        return problems + [f"n_devices {n_dev!r} is not a device count"]
    visible, devices = env.get("n_devices_visible"), env.get("devices")
    if not _is_count(visible) or not isinstance(devices, list) or len(devices) != visible:
        n_listed = len(devices) if isinstance(devices, list) else None
        problems.append(f"its environment lists {n_listed} device(s) but records "
                        f"n_devices_visible {visible!r}")
    elif n_dev > visible:
        problems.append(f"n_devices {n_dev}, but its environment saw {visible} device(s)")
    if cfg.get("n_devices") != n_dev:
        problems.append(f"n_devices {n_dev}, but its config says {cfg.get('n_devices')!r}")
    if ("allow_fewer_devices" in doc
            and bool(cfg.get("allow_fewer_devices", False)) != bool(doc["allow_fewer_devices"])):
        problems.append("allow_fewer_devices differs between the file and its config")
    results, checks = doc.get("results"), doc.get("checks")
    if not isinstance(results, list) or not isinstance(checks, list):
        return problems + ["no results or no checks recorded"]
    if "raised" in doc:
        # A goal that raised (--keep-going): no cases, and exactly the one
        # check its exception gives.
        raised = doc["raised"]
        if not (isinstance(raised, dict) and isinstance(raised.get("type"), str)
                and isinstance(raised.get("message"), str)):
            return problems + ["its 'raised' entry does not record the exception's "
                               "type and message"]
        if results:
            problems.append("records results, though its goal raised")
        if not all(isinstance(c, dict) for c in checks):
            return problems + ["a recorded check is not a record"]
        return problems + _check_differences(checks, [goal_raised_check(raised)])
    # One rule for every goal: a goal that did not raise ran at least one
    # size and recorded at least one check.  ``--cells`` needs a value, so
    # the runner writes no file without either; one edited to hold none
    # (cells [], results [], checks []) used to be a complete record for
    # the goals whose cases are a plain product over the cells (forward,
    # gradient, exchange: no sizes, no cases, no checks derived, nothing
    # to differ) and so read "no checks", over which --summarise exited
    # 0 -- while the same edit to stencil, hybrid or coupled read INVALID
    # only because their case lists take min() of the cells.
    if isinstance(cfg.get("cells"), list) and not cfg["cells"]:
        return problems + ["its config names no cells: the runner runs at least one "
                           "size, and a run of none measures nothing"]
    if not checks:
        return problems + ["records no checks, and its goal did not raise: no checks "
                           "is not a pass"]
    measured_on = sorted({repr(r.get("n_devices")) for r in results
                          if not isinstance(r, dict) or r.get("n_devices") != n_dev})
    if measured_on:
        problems.append(f"n_devices {n_dev}, but its results were measured on "
                        f"{', '.join(measured_on)}")
    try:
        want, got = case_keys(goal, doc)
    except _UNREADABLE as exc:
        return problems + [f"its results or config cannot be read ({type(exc).__name__}: {exc})"]
    if want != got:
        missing, extra = want - got, got - want
        if missing:
            problems.append(f"lacks case(s) the runner runs for cells {cfg.get('cells')} on "
                            f"{n_dev} devices: " + _fmt_few(_fmt_case(k) for k in missing))
        if extra:
            problems.append("holds case(s) the runner does not run for its config: "
                            + _fmt_few(_fmt_case(k) for k in extra))
    try:
        derived = GOAL_CHECKS[goal](results, n_dev)
    except _UNREADABLE as exc:
        return problems + [f"its results cannot be read ({type(exc).__name__}: {exc})"]
    if not all(isinstance(c, dict) for c in checks):
        return problems + ["a recorded check is not a record"]
    return problems + _check_differences(checks, derived)


def _commit_of(doc: dict) -> str | None:
    """The git commit a file records, or ``None`` when it records none --
    or records something that is not a full SHA (``"HEAD"``, which a runner
    before 0.4.0 wrote in a repository with no commit)."""
    env = doc.get("environment")
    commit = env.get("git_commit") if isinstance(env, dict) else None
    return commit if isinstance(commit, str) and _COMMIT_SHA.fullmatch(commit) else None


def _short_commit(commit: str | None) -> str:
    return commit[:12] if commit else "(none)"


def files_by_commit(docs: list) -> dict:
    """``{commit or None: [file, ...]}`` over ``docs``, in their order."""
    by_commit: dict = {}
    for d in docs:
        by_commit.setdefault(_commit_of(d), []).append(_file_of(d))
    return by_commit


def item_problems(docs: list) -> list[str]:
    """Why the files deciding one checklist item cannot decide it together:
    each must record a commit, and it must be the same one."""
    by_commit = files_by_commit(docs)
    problems = []
    if None in by_commit:
        problems.append(f"no git commit recorded in {', '.join(by_commit.pop(None))}")
    if len(by_commit) > 1:
        problems.append(f"its files come from {len(by_commit)} commits ("
                        + "; ".join(f"{c[:12]}: {', '.join(files)}"
                                    for c, files in sorted(by_commit.items())) + ")")
    return problems


def _excluded_because(row: dict, *, min_cells: int, min_devices: int) -> str | None:
    """Why a row does not decide the transport, or ``None`` if it does."""
    if not row["hardware"]:
        return "dry-run / CPU row (does not rank NCCL transports)"
    if row["record"] != "PASS":
        return (f"its file reads {row['record']}, not PASS (a failed check, or a record "
                "--summarise lists under 'Records that cannot decide')")
    requested = row["requested_cells"]
    if not _is_count(requested):
        return (f"records no requested cell count ({requested!r}): written by another "
                "version of this runner")
    if requested < min_cells:
        return f"{requested} cells requested < {min_cells}"
    if row["cells"] < requested:
        return f"measured {row['cells']} cells, fewer than the {requested} requested"
    if row["n_devices"] < MIN_DEVICES_WITH_ESCAPE:
        return f"n_devices={row['n_devices']}: no exchange happens on one device"
    if row["n_devices"] < min_devices and not row["allow_fewer_devices"]:
        return (f"n_devices={row['n_devices']} < {min_devices} "
                "(re-run with --allow-fewer-devices to let it decide)")
    if row["speedup_median"] is None:
        if row["ppermute_median_ms"] == 0:
            return "ppermute median is 0 ms (below timer resolution); speedup undefined"
        return (f"speedup undefined: medians all_to_all {row['a2a_median_ms']!r} ms, ppermute "
                f"{row['ppermute_median_ms']!r} ms are not finite with ppermute's above 0")
    return None


def recommend(exchange_docs: list[dict], *, min_cells: int = 100_000,
              margin: float = 1.05, min_devices: int = MIN_DECIDING_DEVICES) -> dict:
    """Which transport should be the default.

    A row *decides* only when it comes from a real accelerator run (not
    ``--dry-run``) whose file reads ``PASS`` (:func:`goal_verdict`: every
    check passed, and the record is one this runner would have written),
    was requested at ``requested_cells >= min_cells`` and measured at no
    fewer cells than requested (a synthetic grid is the smallest square
    holding the requested count, :func:`synthetic_cells`), ran on
    ``n_devices >= min_devices`` (or ``>= 2`` when that run recorded
    ``allow_fewer_devices``), and has a finite speedup, re-derived from
    the two medians (:func:`_speedup_of`), not read from the file.
    ``ppermute`` wins when its median exchange time beats ``all_to_all``
    by at least ``margin`` at every deciding row; ``all_to_all`` keeps the
    default when it is faster at any such row; anything in between is a
    tie.  Without a deciding row the result is ``undecided`` and
    ``reason`` says what each row lacked; with one, a row that does not
    decide is counted in ``reason`` and listed by ``--summarise``.
    """
    rows = []
    for doc in exchange_docs:
        env = doc["environment"]
        hw = env["platform"] == "gpu" and not doc.get("dry_run", False)
        record = goal_verdict([doc])
        allow_fewer = bool(doc.get("allow_fewer_devices",
                                   doc.get("config", {}).get("allow_fewer_devices", False)))
        for r in doc["results"]:
            a, p = r["methods"]["all_to_all"], r["methods"]["ppermute"]
            row = {
                "cells": r["cells"], "requested_cells": r.get("requested_cells"),
                "hardware": hw, "n_devices": r["n_devices"],
                "allow_fewer_devices": allow_fewer,
                "device_kinds": env["device_kinds"],
                "a2a_median_ms": a["median_ms"], "ppermute_median_ms": p["median_ms"],
                "a2a_min_ms": a["min_ms"], "ppermute_min_ms": p["min_ms"],
                "a2a_bytes_total": a["bytes_total"], "ppermute_bytes_total": p["bytes_total"],
                "speedup_median": _speedup_of(a["median_ms"], p["median_ms"]),
                "recorded_speedup_median": r.get("ppermute_speedup_median"),
                "bit_identical": r["bit_identical"],
                "record": record,
            }
            row["excluded_because"] = _excluded_because(row, min_cells=min_cells,
                                                        min_devices=min_devices)
            row["deciding"] = row["excluded_because"] is None
            rows.append(row)
    deciding = [r for r in rows if r["deciding"]]
    if not deciding:
        reasons = sorted({r["excluded_because"] for r in rows})
        reason = (f"no deciding measurement: needs a real-GPU run (not --dry-run) at "
                  f">= {min_cells} cells on >= {min_devices} devices "
                  f"(>= {MIN_DEVICES_WITH_ESCAPE} with --allow-fewer-devices)")
        if reasons:
            reason += "; rows excluded because: " + "; ".join(reasons)
        return {"rows": rows, "decision": "undecided", "reason": reason}
    speedups = [r["speedup_median"] for r in deciding]
    if all(s >= margin for s in speedups):
        decision = "ppermute"
        reason = (f"ppermute is >= {margin:.2f}x faster (median) at every deciding point "
                  f"(>= {min_cells} cells, real GPUs); make it the default exchange")
    elif any(s < 1.0 for s in speedups):
        decision = "all_to_all"
        reason = "all_to_all is faster at at least one deciding point; keep it the default"
    else:
        decision = "tie"
        reason = (f"ppermute is faster but by less than {margin:.2f}x somewhere; keep "
                  "all_to_all the default (fewer collectives) unless the byte savings matter")
    excluded = len(rows) - len(deciding)
    if excluded:
        # A decision on fewer rows than were measured says so in its own
        # line: the 1e5-cell row once dropped out silently.
        reason += (f" -- decided on {len(deciding)} of {len(rows)} row(s); {excluded} "
                   "excluded (listed under 'Rows that do not decide')")
    return {"rows": rows, "decision": decision, "reason": reason,
            "min_speedup": min(speedups), "max_speedup": max(speedups),
            "deciding_rows": len(deciding)}


def _speedup_of(a2a_median_ms, ppermute_median_ms) -> float | None:
    """``all_to_all``'s median over ``ppermute``'s, or ``None`` unless both
    are finite non-negative numbers, ppermute's is above zero, and the
    ratio is finite.  :func:`recommend` derives the speedup from the
    medians rather than trusting the file's ``ppermute_speedup_median``:
    a hand-edited speedup, or a NaN or infinite one, used to decide the
    transport (``NaN >= 1.05`` and ``NaN < 1.0`` are both false, so a NaN
    row read as a tie)."""
    vals = (a2a_median_ms, ppermute_median_ms)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return None
    a2a, ppm = (float(v) for v in vals)
    if not (math.isfinite(a2a) and math.isfinite(ppm)) or a2a < 0 or ppm <= 0:
        return None
    ratio = a2a / ppm
    return ratio if math.isfinite(ratio) else None


def _fmt_speedup(s) -> str:
    return f"{s:8.2f}" if s is not None else f"{'n/a':>8}"


def closes_the_gap(doc: dict) -> bool:
    """Can this run close the CPU-only gap?  Real GPUs, not a dry run, on
    ``>= MIN_DECIDING_DEVICES`` (4) devices -- whatever
    ``--allow-fewer-devices`` says.  That flag lets a *transport ranking*
    decide on 2-3 devices; it never closes a checklist item, because on
    two devices a halo taken from the wrong neighbour passes every halo
    and stencil check, and the 2-D pencil cases do not run."""
    env = doc.get("environment", {})
    n = int(doc.get("n_devices") or doc.get("config", {}).get("n_devices") or 0)
    return (env.get("platform") == "gpu" and not doc.get("dry_run", False)
            and n >= MIN_DECIDING_DEVICES)


def _file_flag_consistent(doc: dict) -> bool:
    """The file-level ``passed`` says what its checks say: every check ran
    and passed, and there was at least one."""
    statuses = [check_status(c) for c in doc["checks"]]
    want = bool(statuses) and all(s == "passed" for s in statuses)
    return isinstance(doc.get("passed"), bool) and doc["passed"] == want


def goal_verdict(docs: list) -> str:
    """``PASS`` / ``FAIL`` / ``INVALID`` / ``INCOMPLETE`` / ``not run`` /
    ``no checks`` over every file of a goal.

    Pass/fail is re-derived from each check's value, limit and sense
    (:func:`check_status`), not read from the recorded flag: a flag that
    disagrees with them, at the check or the file level, is a ``FAIL``.
    ``INVALID``: nothing failed, but a file is not evidence this runner
    would have written (:func:`record_problems`: another schema, more
    devices than its environment saw, a case missing, or checks that are
    not the ones its results give under ``LIMITS``).  ``INCOMPLETE``:
    nothing failed and every file is valid, but a check was recorded as
    not run.
    """
    if not docs:
        return "not run"
    if any("checks" not in d for d in docs):
        # A file with no checks (what a schema-2 runner wrote) is no evidence
        # this runner would have written: INVALID, not "no checks", so the
        # summary counts it and exits 3.  It used to exit 0.
        return "INVALID" if any(record_problems(d) for d in docs) else "no checks"
    statuses = [check_status(c) for d in docs for c in d["checks"]]
    if ("failed" in statuses or "inconsistent" in statuses
            or not all(_file_flag_consistent(d) for d in docs)):
        return "FAIL"
    if any(record_problems(d) for d in docs):
        return "INVALID"
    if "not run" in statuses:
        return "INCOMPLETE"
    return "PASS" if "passed" in statuses else "no checks"


def _why_still_open(docs: list) -> str:
    if any(d.get("dry_run", False) or d.get("environment", {}).get("platform") != "gpu"
           for d in docs):
        return "open: passed on CPU / dry run only"
    return f"open: passed on fewer than {MIN_DECIDING_DEVICES} devices"


def checklist_status(docs_by_goal: dict) -> dict:
    """Checklist item -> ``(status, detail)`` from the goals that decide it.

    ``CLOSED`` needs every deciding goal to ``PASS`` (which includes every
    file being valid, :func:`record_problems`), every deciding file from a
    real GPU run, not a dry run, on at least ``MIN_DECIDING_DEVICES``
    devices, and all of them from one recorded commit
    (:func:`item_problems`).  An item held open by a file or by the
    commits says which, and why, in its status.
    """
    status = {}
    for item, (_claim, goals) in CHECKLIST.items():
        verdicts = {g: goal_verdict(docs_by_goal.get(g, [])) for g in goals}
        detail = ", ".join(f"{g} {v}" for g, v in verdicts.items())
        docs = [d for g in goals for d in docs_by_goal.get(g, [])]
        if "FAIL" in verdicts.values():
            status[item] = ("FAILED", detail)
        elif "INVALID" in verdicts.values():
            file, why = next((_file_of(d), p) for d in docs for p in record_problems(d))
            status[item] = (f"open: {file} cannot decide it ({why})", detail)
        elif any(v != "PASS" for v in verdicts.values()):
            status[item] = ("open", detail)
        elif not all(closes_the_gap(d) for d in docs):
            status[item] = (_why_still_open(docs), detail)
        elif problems := item_problems(docs):
            status[item] = (f"open: {problems[0]}", detail)
        else:
            status[item] = ("CLOSED", detail)
    return status


def _fmt_value(c: dict) -> str:
    v = c.get("value")
    if v is None:
        return "not run"
    return str(v) if isinstance(v, bool) or not isinstance(v, (int, float)) else f"{v:.3e}"


def _why_failed(c: dict) -> str:
    """One line for a check that failed or whose record is inconsistent."""
    line = f"{c.get('name')}: {_fmt_value(c)} (limit {c.get('limit')})"
    if check_status(c) == "inconsistent":
        if c.get("not_run"):
            why = "a check that was not run cannot pass"
        else:
            want = expected_pass(c)
            why = (f"value, limit and sense {c.get('sense')!r} cannot be judged" if want is None
                   else f"the value {'passes' if want else 'fails'} its limit")
        line += f" -- recorded passed={c.get('passed')!r}, but {why}"
    elif c.get("detail"):
        line += f" -- {c['detail']}"
    return line


def _item_commits(docs: list) -> str:
    """The commit(s) an item's files record, for its checklist line: one
    commit, or each commit with the files that record it."""
    by_commit = files_by_commit(docs)
    if not by_commit:
        return ""
    if len(by_commit) == 1:
        return f"commit {_short_commit(next(iter(by_commit)))}"
    return "commits " + "; ".join(f"{_short_commit(c)} ({', '.join(files)})"
                                  for c, files in by_commit.items())


def _print_mixed_commits(docs_by_goal: dict) -> bool:
    """A prominent warning when the directory's files record more than one
    commit, or any of them records none, naming each commit's items and
    files.  Returns whether it printed one.

    An item is decided only by files of one commit (:func:`item_problems`),
    but the rule is per item: items decided by different goals can each
    close on a different commit, and the checklist then reads as one
    session when it was not.  So the summary says so, and exits 4.
    """
    docs = [d for goal in ALL_GOALS for d in docs_by_goal.get(goal, [])]
    by_commit = files_by_commit(docs)
    if len(by_commit) < 2 and None not in by_commit:
        return False
    if list(by_commit) == [None]:
        # No file records a commit (a tree synced without .git): nothing
        # ties the files to one session's code.  This used to exit 0.
        print("\nWARNING: MIXED COMMITS -- no file in this directory records a git "
              "commit, so nothing says its checklist is one session's verdict (exit 4)")
    else:
        print(f"\nWARNING: MIXED COMMITS -- this directory's files come from "
              f"{len(by_commit)} commits (counting none recorded as one), so its "
              "checklist is not one session's verdict (exit 4)")
    for commit, files in by_commit.items():
        items = [str(item) for item, (_claim, goals) in CHECKLIST.items()
                 if any(_commit_of(d) == commit for g in goals for d in docs_by_goal.get(g, []))]
        print(f"  commit {_short_commit(commit)}: items {', '.join(items) or '-'}; "
              f"{', '.join(files)}")
    print("  Re-run every goal on one commit, or summarise each commit's files apart.")
    return True


def _run_line(goal: str, doc: dict) -> str:
    """One file's line in the run table."""
    env = doc.get("environment", {})
    checks = doc.get("checks")
    statuses = [check_status(c) for c in checks or []]
    n_ok = "-" if checks is None else f"{statuses.count('passed')}/{len(checks)}"
    n_dev = doc.get("n_devices") or doc.get("config", {}).get("n_devices", "?")
    kinds = ",".join(env.get("device_kinds", [])) or "?"
    return (f"{goal:<12} {env.get('platform', '?'):<8} {n_dev:>3}  {kinds[:26]:<26} "
            f"{env.get('jax', '?') + ' / ' + env.get('jaxlib', '?'):<17} "
            f"{'yes' if doc.get('dry_run') else 'no':<7} {n_ok:>7}  "
            f"{_short_commit(_commit_of(doc)):<12}  {goal_verdict([doc])}")


def _print_runs_and_checklist(docs_by_goal: dict) -> int:
    """The per-file run table, the checklist verdict, every record that
    cannot decide, every failed check and every check not run.  Returns the
    number of failures: failed checks, inconsistent records, files whose
    ``passed`` disagrees with their checks, and files that are not
    evidence this runner would have written (:func:`record_problems`)."""
    print("Runs (one line per JSON file)")
    print(f"{'goal':<12} {'platform':<8} {'dev':>3}  {'device kind':<26} {'jax / jaxlib':<17} "
          f"{'dry run':<7} {'checks':>7}  {'commit':<12}  verdict")
    failed, not_run, bad_files = [], [], []
    for goal in ALL_GOALS:
        for doc in docs_by_goal.get(goal, []):
            checks = doc.get("checks")
            statuses = [check_status(c) for c in checks or []]
            print(_run_line(goal, doc))
            for c, s in zip(checks or [], statuses):
                if s in ("failed", "inconsistent"):
                    failed.append((goal, c))
                elif s == "not run":
                    not_run.append((goal, c))
            if checks is not None and not _file_flag_consistent(doc):
                bad_files.append((goal, doc.get("passed")))
    print(f"\nChecklist (CLOSED needs a PASS from real GPUs, not a dry run, on >= "
          f"{MIN_DECIDING_DEVICES} devices, with every check run, every file valid and "
          "one commit)")
    for item, (status, detail) in checklist_status(docs_by_goal).items():
        deciding = [d for g in CHECKLIST[item][1] for d in docs_by_goal.get(g, [])]
        commits = _item_commits(deciding)
        print(f"{item}  {CHECKLIST[item][0]:<74} {status}  [{detail}]"
              + (f"  {commits}" if commits else ""))
    # Every file, a file with no checks included: it used to be left out
    # here, so it was neither listed nor counted and the summary exited 0.
    invalid = [(goal, doc, record_problems(doc)) for goal in ALL_GOALS
               for doc in docs_by_goal.get(goal, [])]
    invalid = [(goal, doc, problems) for goal, doc, problems in invalid if problems]
    mixed = [(item, item_problems([d for g in goals for d in docs_by_goal.get(g, [])]))
             for item, (_claim, goals) in CHECKLIST.items()]
    mixed = [(item, problems) for item, problems in mixed if problems]
    if invalid or mixed:
        print("\nRecords that cannot decide (not evidence this runner, as it stands, "
              "would have written)")
        for goal, doc, problems in invalid:
            for problem in problems:
                print(f"  [{goal}] {_file_of(doc)}: {problem}")
        for item, problems in mixed:
            for problem in problems:
                print(f"  [item {item}] {problem}")
    if failed or bad_files:
        print("\nFailed checks")
        for goal, c in failed:
            print(f"  [{goal}] {_why_failed(c)}")
        for goal, flag in bad_files:
            print(f"  [{goal}] file records passed={flag!r}, which its checks do not bear out")
    if not_run:
        print("\nChecks not run (each keeps its checklist items open)")
        for goal, c in not_run:
            print(f"  [{goal}] {c.get('name')} -- {c.get('detail', '')}")
    return len(failed) + len(bad_files) + len(invalid)


def _print_checklist_goal_tables(docs_by_goal: dict) -> None:
    for doc in docs_by_goal.get("indivisible", []):
        for r in doc["results"]:
            un = r["unstructured"]
            pencil = r.get("pencil") or {}
            print(f"\nIndivisible grid: stencil {r['stencil']['shape']} -> "
                  f"{r['stencil']['raised']}, along spatial axis 1 "
                  f"{r['stencil_axis1']['shape']} -> {r['stencil_axis1']['raised']}, "
                  f"pointwise -> {r['pointwise']['raised']}, pencil "
                  f"{pencil.get('shape', '-')} -> {pencil.get('raised', '-')}; unstructured "
                  f"{un['cells']} cells {un['cells_per_device']} max_rel "
                  f"{un['parity_x']['max_rel']:.1e}")
    for doc in docs_by_goal.get("halo", []):
        for r in doc["results"]:
            cases = r["stencil_cases"]
            worst_f = max(c["forward_max_abs"] for c in cases)
            worst_a = max(c["adjoint_max_abs"] for c in cases)
            meshes = sorted({f"{c['mesh']} {c['mesh_shape'][0]}x{c['mesh_shape'][1]}"
                             for c in cases})
            un = r["unstructured"]
            print(f"\nHalo exchange vs NumPy (max |diff|; must be 0): {len(cases)} stencil cases "
                  f"on {', '.join(meshes)} meshes, forward {worst_f:.1e}, adjoint {worst_a:.1e}; "
                  f"unstructured {un['cells']} cells "
                  + ", ".join(f"{m} {v['forward_max_abs']:.1e}/{v['adjoint_max_abs']:.1e}"
                              for m, v in un["methods"].items()))
    if docs_by_goal.get("coupled"):
        print("\nCoupled group, sharded vs unsharded (max-rel; model = worst gradient against "
              "the float64 model; value+grad = one differentiated rollout)")
        print(f"{'mesh, shape':>22} {'solver':<6} {'field':>8} {'avgs':>8} {'far':>8} "
              f"{'grad':>8} {'limit':>8} {'model':>8} {'sh ms':>9} {'unsh ms':>9} "
              f"{'compile':>8} {'passes':>7} {'pinned':>6}")
        for doc in docs_by_goal["coupled"]:
            for r in doc["results"]:
                shape = _graph_prefix(r)
                for solver, sol in r["solvers"].items():
                    its = "/".join("-" if sol[k]["last_step_iterations"] is None
                                   else str(sol[k]["last_step_iterations"])
                                   for k in ("unsharded", "sharded"))
                    print(f"{shape:>22} {solver:<6} {sol['parity_f']['max_rel']:8.1e} "
                          f"{sol['parity_averages']['max_rel']:8.1e} "
                          f"{sol['parity_u']:8.1e} {sol['parity_grad']:8.1e} "
                          f"{LIMITS['coupled_gradient_' + solver]:8.0e} "
                          f"{max(v['grad'] for v in sol['model'].values()):8.1e} "
                          f"{sol['sharded']['value_and_grad']['median_ms']:9.2f} "
                          f"{sol['unsharded']['value_and_grad']['median_ms']:9.2f} "
                          f"{sol['sharded']['compile_s']:7.1f}s {its:>7} "
                          f"{sol['sharded']['device0_pinned_ops']:>6}")
    if docs_by_goal.get("stencil"):
        print("\nStencil wrapper, sharded vs unsharded (node, mesh, shape, ends; max-rel; "
              "ms per step)")
        for doc in docs_by_goal["stencil"]:
            for r in doc["results"]:
                fw, gr = r["forward"], r["gradient"]
                print(f"{_stencil_prefix(r):<34} "
                      + "  ".join(f"{k} {v['max_rel']:.1e}" for k, v in fw["parity"].items())
                      + f"  loss {gr['parity_loss']:.1e}  grad f0 "
                      f"{gr['parity_grad_initial_field']['max_rel']:.1e}  grad "
                      f"{r['parameter']} {gr['parity_grad_parameter']:.1e}  fwd "
                      f"{fw['sharded']['rollout']['ms_per_step']:.3f} vs "
                      f"{fw['unsharded']['rollout']['ms_per_step']:.3f} ms/step  grad "
                      f"{gr['sharded']['grad']['median_ms']:.2f} vs "
                      f"{gr['unsharded']['grad']['median_ms']:.2f} ms")
    if docs_by_goal.get("hybrid"):
        print("\nHybridNode(ShardedStencilNode(inner)) vs HybridNode(inner) (max-rel)")
        for doc in docs_by_goal["hybrid"]:
            for r in doc["results"]:
                print(f"{_graph_prefix(r):<22} f {r['parity_f']['max_rel']:.1e}  averages "
                      f"{r['parity_averages']['max_rel']:.1e}  "
                      f"loss {r['parity_loss']:.1e}  grad {r['parity_grad']:.1e}  correction "
                      f"{r['correction_rel']:.1e} of |f|  value+grad "
                      f"{r['sharded']['value_and_grad']['median_ms']:.2f} vs "
                      f"{r['unsharded']['value_and_grad']['median_ms']:.2f} ms")


def _print_timing_tables(docs_by_goal: dict) -> None:
    """The exchange ranking and the forward and gradient tables."""
    exchange_docs = docs_by_goal["exchange"]
    forward_docs = docs_by_goal["forward"]
    gradient_docs = docs_by_goal["gradient"]
    rec = recommend(exchange_docs)
    if exchange_docs:
        print("\nExchange ranking (all_to_all vs ppermute), median / min ms per exchange")
        print(f"{'cells':>9} {'asked':>9} {'dev':>4} {'hw':>3} {'a2a med':>9} {'ppm med':>9} "
              f"{'a2a min':>9} {'ppm min':>9} {'speedup':>8} {'a2a MB':>8} {'ppm MB':>8} "
              f"{'same':>5} {'decides':>7}")
        for r in rec["rows"]:
            asked = r["requested_cells"] if _is_count(r["requested_cells"]) else "?"
            print(f"{r['cells']:>9} {asked:>9} {r['n_devices']:>4} "
                  f"{'gpu' if r['hardware'] else 'cpu':>3} "
                  f"{r['a2a_median_ms']:9.3f} {r['ppermute_median_ms']:9.3f} "
                  f"{r['a2a_min_ms']:9.3f} {r['ppermute_min_ms']:9.3f} "
                  f"{_fmt_speedup(r['speedup_median'])} {r['a2a_bytes_total'] / 1e6:8.2f} "
                  f"{r['ppermute_bytes_total'] / 1e6:8.2f} {'yes' if r['bit_identical'] else 'NO':>5} "
                  f"{'yes' if r['deciding'] else 'no':>7}")
        excluded = [r for r in rec["rows"] if not r["deciding"]]
        if excluded:
            # Every row that does not decide, and why, on a line of its own:
            # the decision rule is "at every point >= 1e5 cells", and a row
            # dropped from it in silence once let two of three points decide.
            print("Rows that do not decide (the recommendation below ignores them)")
            for r in excluded:
                asked = r["requested_cells"] if _is_count(r["requested_cells"]) else "?"
                print(f"  {'WARNING: ' if r['hardware'] else ''}{r['cells']} cells "
                      f"(requested {asked}) on {r['n_devices']} "
                      f"{'GPU' if r['hardware'] else 'CPU / dry-run'} device(s): "
                      f"{r['excluded_because']}")
        print(f"Recommendation: {rec['decision']} -- {rec['reason']}")
    if forward_docs:
        print("\nForward run (ms per step; wrapper = public update(), device = compiled step)")
        for doc in forward_docs:
            for r in doc["results"]:
                for method, m in r["methods"].items():
                    print(f"{r['cells']:>9} cells {r['mesh']:<24} {method:<10} "
                          f"wrapper {m['wrapper_step']['ms_per_step']:9.3f}  "
                          f"device {m['device_step']['ms_per_step']:9.3f}  "
                          f"compile {m['compile_s']:6.2f} s  "
                          f"max|dx| {m['parity_x']['max_abs']:.2e}  "
                          f"finite={m['parity_x']['finite']}")
    if gradient_docs:
        print("\nGradient parity (sharded vs unsharded, max-abs / max-rel; jitted grads, compile apart)")
        for doc in gradient_docs:
            for r in doc["results"]:
                ref = r["rollout"]["unsharded"]
                print(f"{r['cells']:>9} cells rollout   {'unsharded':<10} "
                      f"{'-':>8}   {'-':>8}  grad {ref['grad']['median_ms']:8.2f} ms  "
                      f"compile {ref['compile_s']:6.2f} s")
                for method in METHODS:
                    m = r["rollout"][method]
                    p = m["parity"]
                    print(f"{r['cells']:>9} cells rollout   {method:<10} "
                          f"{p['max_abs']:.2e} / {p['max_rel']:.2e}  "
                          f"grad {m['grad']['median_ms']:8.2f} ms  compile {m['compile_s']:6.2f} s")
                cg = r["sharded_cg"]
                print(f"{cg['dof']:>9} dof   sharded_cg grad     {cg['grad_parity']['max_abs']:.2e} / "
                      f"{cg['grad_parity']['max_rel']:.2e}  jvp {cg['jvp_parity']['max_abs']:.2e} / "
                      f"{cg['jvp_parity']['max_rel']:.2e}")


#: What reading a record of another runner's shape raises.
_UNREADABLE = (KeyError, TypeError, ValueError, IndexError, AttributeError,
               ZeroDivisionError, OverflowError)


def _readable_for_tables(docs_by_goal: dict) -> tuple[dict, list[str]]:
    """``(docs, left out)``: per goal, the files whose tables this runner
    can print, and a line for each it cannot (a file of an older runner
    lacks keys the tables read)."""
    readable: dict = {}
    left_out: list[str] = []
    for goal, docs in docs_by_goal.items():
        readable[goal] = []
        for doc in docs:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    _print_checklist_goal_tables({goal: [doc]})
                    _print_timing_tables({g: ([doc] if g == goal else [])
                                          for g in ("exchange", "forward", "gradient")})
            except _UNREADABLE as exc:
                left_out.append(f"{_file_of(doc)}: {type(exc).__name__}: {exc}")
            else:
                readable[goal].append(doc)
    return readable, left_out


def summarise(directory: Path) -> int:
    """Print the verdicts and tables; 0 = no check failed and every file
    records one commit (checks recorded as not run are listed and keep
    their items open, but are not failures), 1 = no goal JSON under
    ``directory``, 3 = at least one check failed, a recorded pass/fail
    disagrees with its value and limit, or a file is not evidence this
    runner would have written (:func:`record_problems`), 4 = nothing
    failed, but the files come from more than one commit, or a file
    records none (every file recording none included -- that used to exit
    0): the items they decide may each have closed on a different commit,
    and the summary prints a ``MIXED COMMITS`` warning naming each
    commit's items and files."""
    docs_by_goal = {goal: _load_results(directory, goal) for goal in ALL_GOALS}
    if not any(docs_by_goal.values()):
        print(f"no goal JSON ({'/'.join(ALL_GOALS)}) under {directory}")
        return 1
    n_failed = _print_runs_and_checklist(docs_by_goal)
    mixed = _print_mixed_commits(docs_by_goal)
    readable, left_out = _readable_for_tables(docs_by_goal)
    _print_checklist_goal_tables(readable)
    _print_timing_tables(readable)
    if left_out:
        # A file whose results the tables cannot read mostly reads INVALID
        # above as well (its checks are derived from those results), and
        # the summary exits 3; one whose unreadable field is a timing no
        # check reads keeps its verdict.  Their tables used to stop the
        # summary with a traceback instead.
        print(f"\nTables leave out {len(left_out)} file(s) whose results this runner "
              "cannot read:")
        for line in left_out:
            print(f"  {line}")
    if mixed and not n_failed:
        # Repeated last, so that it is the line a reader sees.
        print("\nWARNING: MIXED COMMITS -- see above; exit 4")
    return 3 if n_failed else 4 if mixed else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    """The CLI.  Kept apart from :func:`parse_args` so a test can list its
    options."""
    # No abbreviations: _pre_import_setup() runs before argparse and pins the
    # CPU backend only for the literal ``--dry-run``, so an accepted
    # ``--dry`` used to start a "dry run" on whatever accelerator was there.
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal", choices=ALL_GOALS + ("checklist", "all"))
    ap.add_argument("--out", type=Path, help="directory for the JSON results")
    ap.add_argument("--summarise", type=Path, metavar="DIR",
                    help="print the checklist verdict, the tables and the transport "
                         "recommendation from DIR and exit (no JAX needed)")
    ap.add_argument("--keep-going", action="store_true",
                    help="with --goal checklist/all, run the remaining goals after one fails "
                         "or raises (a goal that raises is recorded as one failed 'goal "
                         "raised' check); default: stop after the goal whose checks failed, "
                         "and let an exception end the run")
    ap.add_argument("--dry-run", action="store_true",
                    help="small sizes on CPU virtual devices; proves the script, ranks nothing")
    ap.add_argument("--cells", type=int, nargs="+",
                    help=f"cell counts (default {GPU_CELLS} on GPU, {DRY_RUN_CELLS} otherwise)")
    ap.add_argument("--n-devices", type=int, help="mesh size (default: min(4, visible devices))")
    ap.add_argument("--allow-fewer-devices", action="store_true",
                    help=f"let a real-GPU exchange run on 2..{MIN_DECIDING_DEVICES - 1} devices "
                         f"decide the transport (default: only >= {MIN_DECIDING_DEVICES} decide); "
                         "recorded in the JSON.  It never closes a checklist item, which "
                         f"needs >= {MIN_DECIDING_DEVICES} devices")
    ap.add_argument("--warmup", type=int, help="untimed calls before timing (default 5; dry-run 1)")
    ap.add_argument("--repeats", type=int, help="timed calls (default 20; dry-run 3)")
    ap.add_argument("--steps", type=int, help="steps per timed block for --goal forward (default 20; dry-run 3)")
    ap.add_argument("--grad-steps", type=int, default=5, help="rollout length differentiated through")
    ap.add_argument("--cg-max-iters", type=int, help="sharded_cg iteration cap (default 3000; dry-run 300)")
    ap.add_argument("--fields", type=int, default=1, help="float32 fields per cell in the exchange payload")
    ap.add_argument("--synthetic", choices=("ring", "grid"), default="grid",
                    help="synthetic mesh when no --mesh is given")
    ap.add_argument("--mesh", help=".npz (edges[, partition]) or .npy edges of a real mesh")
    ap.add_argument("--partition", choices=("auto", "metis", "rcm", "contiguous"), default="auto")
    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = _parser()
    args = ap.parse_args(argv)
    if args.summarise is None and (args.goal is None or args.out is None):
        ap.error("--goal and --out are required unless --summarise is given")
    return args


def check_exchange_device_count(n_devices: int, *, dry_run: bool,
                                allow_fewer: bool) -> None:
    """Refuse an ``exchange`` run whose result could never decide.

    One device performs no exchange at all; 2-3 devices decide only with
    ``--allow-fewer-devices``.  A dry run may use any count (it never
    decides anyway).
    """
    if dry_run:
        return
    if n_devices < MIN_DEVICES_WITH_ESCAPE:
        raise SystemExit(f"--goal exchange on {n_devices} device performs no exchange and "
                         "cannot rank transports; use --dry-run to prove the script")
    if n_devices < MIN_DECIDING_DEVICES and not allow_fewer:
        raise SystemExit(f"--goal exchange on {n_devices} devices will not decide the "
                         f"transport (the session's question is on {MIN_DECIDING_DEVICES}); "
                         "pass --allow-fewer-devices to record it as deciding anyway")


#: The least value each counting option takes: a run with fewer has
#: nothing to time, to step or to solve.
OPTION_MINIMUMS = {"cells": 1, "n_devices": 1, "warmup": 0, "repeats": 1, "steps": 1,
                   "grad_steps": 1, "cg_max_iters": 1, "fields": 1}


def check_option_values(args: argparse.Namespace) -> None:
    """Refuse (``SystemExit`` with the reason: ``EXIT_REFUSED``) an option
    value no goal can use, before the backend is loaded or anything runs.

    ``--cells -5``, ``--repeats 0``, ``--steps 0`` and a ``--mesh`` file
    that does not exist used to raise inside the first goal that read them
    and exit as a crashed goal (5) with a traceback, where the runbook says
    a refusal (2); ``--cells 0`` and ``--warmup -1`` ran and wrote a
    passing record.  Every option of :func:`_parser` is asked: a count
    against :data:`OPTION_MINIMUMS`, ``--mesh`` by loading it, ``--out`` as
    a directory that exists or can be made; the options with ``choices``
    and the flags are argparse's.
    """
    for dest, least in OPTION_MINIMUMS.items():
        value = getattr(args, dest)
        if value is None:
            continue
        option = "--" + dest.replace("_", "-")
        for item in value if isinstance(value, list) else [value]:
            if item < least:
                raise SystemExit(f"{option} {item}: must be an integer >= {least}")
    if args.out.exists() and not args.out.is_dir():
        raise SystemExit(f"--out {args.out}: exists and is not a directory")
    if args.mesh:
        if not Path(args.mesh).is_file():
            raise SystemExit(f"--mesh {args.mesh}: no such file")
        try:
            _MESH_CACHE[("file", str(args.mesh))] = load_mesh(args.mesh)
        except Exception as exc:  # noqa: BLE001 - whatever makes it unreadable
            raise SystemExit(f"--mesh {args.mesh}: cannot be read as a mesh "
                             f"({type(exc).__name__}: {exc})") from None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarise is not None:
        return summarise(args.summarise)
    check_option_values(args)

    _load_backend()
    gpu = on_gpu()
    visible = len(jax.devices())
    if args.n_devices is None:
        args.n_devices = min(MIN_DECIDING_DEVICES, visible)
    if args.n_devices > visible:
        raise SystemExit(f"--n-devices {args.n_devices} but only {visible} device(s) visible")
    small = args.dry_run or not gpu
    if args.cells is None:
        args.cells = list(DRY_RUN_CELLS if small else GPU_CELLS)
    args.warmup = args.warmup if args.warmup is not None else (1 if small else 5)
    args.repeats = args.repeats if args.repeats is not None else (3 if small else 20)
    args.steps = args.steps if args.steps is not None else (3 if small else 20)
    args.cg_max_iters = args.cg_max_iters if args.cg_max_iters is not None else (300 if small else 3000)
    goals = {"all": ALL_GOALS, "checklist": CHECKLIST_GOALS}.get(args.goal, (args.goal,))
    if "exchange" in goals:
        check_exchange_device_count(args.n_devices, dry_run=args.dry_run,
                                    allow_fewer=args.allow_fewer_devices)
    if any(g in CHECKLIST_GOALS for g in goals):
        check_checklist_device_count(args.n_devices)

    env = environment(dry_run=args.dry_run)
    print(f"devices: {env['devices']}  jax {env['jax']}  platform {env['platform']}"
          f"{'  [DRY RUN]' if args.dry_run else ''}")
    if not gpu and not args.dry_run:
        print("WARNING: no GPU backend -- these numbers do not rank NCCL transports")

    args.out.mkdir(parents=True, exist_ok=True)
    runners = {"exchange": run_exchange, "forward": run_forward, "gradient": run_gradient,
               "indivisible": run_indivisible, "halo": run_halo, "coupled": run_coupled,
               "stencil": run_stencil, "hybrid": run_hybrid}
    failed_goals, incomplete_goals = [], []
    for goal in goals:
        doc = {
            "schema_version": SCHEMA_VERSION,
            "goal": goal,
            "dry_run": bool(args.dry_run),
            "allow_fewer_devices": bool(args.allow_fewer_devices),
            "n_devices": args.n_devices,
            "environment": env,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        }
        t0 = time.perf_counter()
        try:
            runners[goal](args, doc)
        except MeshPartitionError as e:
            raise SystemExit(str(e)) from e
        except Exception as e:  # noqa: BLE001 - recorded with --keep-going, re-raised without
            if not args.keep_going:
                raise
            record_goal_raised(doc, e)
            print(f"GOAL RAISED [{goal}] {type(e).__name__}: {str(e)[:500]} "
                  "(--keep-going: recorded as a failed check, running the rest)")
        doc["wall_s"] = time.perf_counter() - t0
        path = args.out / f"{goal}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        not_run = [c for c in doc["checks"] if c.get("not_run")]
        ran = [c for c in doc["checks"] if not c.get("not_run")]
        failed = [c for c in ran if not c["passed"]]
        print(f"wrote {path} ({doc['wall_s']:.1f} s)  checks "
              f"{len(ran) - len(failed)}/{len(ran)} passed"
              + (f", {len(not_run)} not run" if not_run else ""))
        for c in failed:
            print(f"CHECK FAILED [{goal}] {c['name']}: {_fmt_value(c)} (limit {c['limit']})"
                  + (f" -- {c['detail']}" if c.get("detail") else ""))
        for c in not_run:
            print(f"CHECK NOT RUN [{goal}] {c['name']} -- {c['detail']}")
        # A check not run is not a failure while another check of the goal
        # ran (the exit status stays 0 for it), but it keeps the goal from
        # reading complete in --summarise; a goal none of whose checks ran
        # is a failure, like one with no checks.
        if failed or not ran:
            failed_goals.append(goal)
            if len(goals) > 1 and not args.keep_going and goal != goals[-1]:
                print(f"STOPPED after {goal}: its checks failed (--keep-going runs the rest)")
                break
        elif not_run:
            incomplete_goals.append(goal)
    if incomplete_goals:
        print(f"INCOMPLETE (checks not run; the checklist items stay open): "
              f"{', '.join(incomplete_goals)}")
    if failed_goals:
        print(f"FAILED: {', '.join(failed_goals)}")
        return 1
    return 0


def exit_status(argv: list[str] | None = None) -> int:
    """:func:`main`'s status, with a refusal and a crash told apart from a
    failed check.  A refusal -- ``SystemExit`` with a message -- prints the
    message and is ``EXIT_REFUSED``; any other exception prints its
    traceback and is ``EXIT_CRASHED``.  Both used to exit 1, which the
    runbook reads as "a check failed"."""
    try:
        return main(argv)
    except SystemExit as exc:
        if exc.code is None or isinstance(exc.code, int):
            raise                       # argparse's 2, an explicit status
        print(exc.code, file=sys.stderr)
        return EXIT_REFUSED
    except Exception:  # noqa: BLE001 - every crash gets the one status
        traceback.print_exc()
        print(f"CRASHED: the run raised (exit {EXIT_CRASHED}); nothing after it ran",
              file=sys.stderr)
        return EXIT_CRASHED


if __name__ == "__main__":
    sys.exit(exit_status())
