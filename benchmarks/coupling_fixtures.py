#!/usr/bin/env python
"""Graph fixtures that span the properties deciding a coupling algorithm.

MADDENING's :class:`~maddening.core.coupling.group.CouplingGroup` offers
two iteration modes and five accelerations.  Which one wins is decided by
four properties of the graph, and each fixture here isolates one of them:

=====================  ===========================================
fixture                property it isolates
=====================  ===========================================
``chain-N``            sequential information flow (depth)
``star-N``             width with no leaf-to-leaf path
``ring-N``             a cycle with no natural first node
``stiff-pair-G``       contraction factor, weak to divergent
``expensive-pair``     cost per iteration (compute-bound)
``heterogeneous``      one expensive node among cheap ones
``mixed-modes``        two groups with different schedules
``slow-drift``         a fixed point that barely moves per step
=====================  ===========================================

Every fixture is built from stock nodes (``SpringDamperNode``,
``HeatNode``) so nothing here depends on a downstream package.

The spring fixtures are parameterised by a dimensionless **coupling
gain** ``g``, not by a stiffness.  A ``SpringDamperNode`` integrates with
semi-implicit Euler using its own old position and its anchor's new one,
so the derivative of its new position with respect to its anchor is
``dt**2 * k / m``; putting the coupling strength in the *edge weight*
``w`` and fixing that derivative at 1 makes ``g = w`` exactly the gain
the fixed-point iteration sees, with the remaining ``1 - w`` of the
node's stiffness acting as a spring to ground that pins the equilibrium.
The contraction factor is then a knob rather than an accident of the
stiffness and the timestep, and the shapes are comparable:

* two mutually coupled nodes contract as ``g`` under Jacobi and
  ``g**2`` under Gauss-Seidel,
* the line and ring shapes add the usual ``cos(pi/(N+1))``-type factor
  from the graph adjacency,
* ``g > 1`` is past the convergence limit: every *fixed-point*
  acceleration must report itself unconverged rather than truncate at
  the cap.  IQN is a quasi-Newton root solver rather than a contraction
  and does converge there, which is the point of keeping a divergent
  fixture in the registry.

See the "Spring helpers" comment below for why the damping is a function
of ``g`` and why every spring fixture carries a driver node.

The heat fixtures are parameterised by the Fourier number
``Fo = alpha*dt/dx**2``, which plays the same role.  ``HeatNode``
imposes its Dirichlet datum at the rod end through the ghost cell
``T_ghost = 2*T_b - T[0]`` (MADD-ANO-007), so the gain from a
neighbour's interface temperature to this node's interface cell is
``2*Fo``: a slab pair contracts as ``2*Fo`` under Jacobi and
``(2*Fo)**2`` under Gauss-Seidel.  Until 0.4.0 the datum was written
into the end cell and the gain was ``Fo``; every heat fixture's Fourier
number was halved when that changed, so each still contracts at the
rate it was designed and first recorded at.

Exchanging the two slabs' end-cell temperatures as each other's
Dirichlet data also sets a *time-step* limit tighter than one node's:
the exactly coupled pair has an alternating interface mode whose
amplification leaves the unit circle at ``Fo = 3/8`` (-1.5 per step at
``Fo = 0.4``, -4 at ``0.45``), where ``HeatNode`` alone is stable to
``1/2``.  ``_HEAT_PAIR_FOURIER_LIMIT`` records it and the slab fixtures
stay below it.

Usage::

    from coupling_fixtures import FIXTURES, CouplingConfig
    built = FIXTURES["chain-5"].build(CouplingConfig(acceleration="aitken"))
    built.gm.step()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

__all__ = [
    "ScopeNotApplicable",
    "CouplingConfig",
    "BuiltGraph",
    "FixtureSpec",
    "FIXTURES",
    "fixture_names",
    "sweep_configs",
]


class ScopeNotApplicable(ValueError):
    """An ``accel_scope`` that selects no node of the fixture it was given.

    ``"cheap"`` on a fixture whose nodes are all expensive is not a
    failure to report, it is a combination that does not exist; the
    driver skips such rows rather than recording them as errors.
    """


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CouplingConfig:
    """One point of the coupling option sweep.

    Parameters
    ----------
    iteration_mode : {"gauss-seidel", "jacobi"}
        Forwarded to :class:`CouplingGroup`.
    acceleration : {"none", "aitken", "fixed", "iqn-ils", "iqn-imvj"}
        Forwarded to :class:`CouplingGroup`.
    relaxation : float
        Omega for ``acceleration="fixed"``.
    convergence_norm : {"l2", "mixed", "interface"}
        Forwarded to :class:`CouplingGroup`.
    accel_scope : {"auto", "all", "interface", "expensive", "cheap"}
        Which state fields enter the quasi-Newton problem.  ``"auto"``
        leaves ``accelerated_fields=None`` (the group auto-detects the
        interface fields); ``"all"`` accelerates every state field of
        every node in the group; ``"interface"`` states the
        auto-detected set explicitly; ``"expensive"`` / ``"cheap"``
        restrict it to the fixture's expensive / cheap nodes.  Only
        ``iqn-ils`` and ``iqn-imvj`` read ``accelerated_fields``.
    jacobian_reuse : int
        Columns retained across timesteps by ``iqn-imvj``.
    max_iterations, tolerance, atol, rtol : float or None
        ``None`` takes the fixture's default.
    """

    iteration_mode: str = "gauss-seidel"
    acceleration: str = "none"
    relaxation: float = 1.0
    convergence_norm: str = "l2"
    accel_scope: str = "auto"
    jacobian_reuse: int = 0
    max_iterations: Optional[int] = None
    tolerance: Optional[float] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None

    @property
    def label(self) -> str:
        """Compact, filename-safe identifier for this configuration."""
        mode = "gs" if self.iteration_mode == "gauss-seidel" else "jac"
        accel = self.acceleration
        if accel == "fixed":
            accel = f"fixed{self.relaxation:g}"
        elif accel == "iqn-imvj" and self.jacobian_reuse:
            accel = f"iqn-imvj{self.jacobian_reuse}"
        parts = [mode, accel, self.convergence_norm]
        if self.accel_scope != "auto":
            parts.append(f"fields-{self.accel_scope}")
        return "/".join(parts)

    def group_kwargs(
        self,
        *,
        max_iterations: int,
        tolerance: float,
        atol: float,
        rtol: float,
        accelerated_fields: Optional[dict] = None,
    ) -> dict:
        """Build the ``add_coupling_group`` keyword arguments.

        Only the knobs this configuration actually reads are emitted.
        The tolerances: ``"l2"`` tests against ``tolerance`` and never sees
        ``atol`` / ``rtol``, while ``"mixed"`` and ``"interface"`` scale
        the residual by ``atol`` / ``rtol`` and never see ``tolerance``.
        Passing all three used to put a dead number in every sweep row,
        which reads — in the results table and in the group itself — as
        if it had been part of the configuration under test.
        ``CouplingGroup`` now warns about exactly that.
        """
        kwargs = {
            "max_iterations": (
                self.max_iterations if self.max_iterations is not None
                else max_iterations
            ),
            "convergence_norm": self.convergence_norm,
            "acceleration": self.acceleration,
            "iteration_mode": self.iteration_mode,
            "accelerated_fields": accelerated_fields,
        }
        # Same rule for the acceleration knobs: only ``"fixed"`` reads
        # ``relaxation`` and only ``"iqn-imvj"`` reads
        # ``jacobian_reuse``, and a sweep row that carries the other's
        # number reads as if it had been part of the configuration
        # under test.  ``CouplingGroup`` warns about exactly that.
        if self.acceleration == "fixed":
            kwargs["relaxation"] = self.relaxation
        if self.acceleration == "iqn-imvj":
            kwargs["jacobian_reuse"] = self.jacobian_reuse
        if self.convergence_norm == "l2":
            kwargs["tolerance"] = (
                self.tolerance if self.tolerance is not None else tolerance
            )
        else:
            kwargs["atol"] = self.atol if self.atol is not None else atol
            kwargs["rtol"] = self.rtol if self.rtol is not None else rtol
        return kwargs


def _resolve_accel_fields(
    config: CouplingConfig,
    *,
    interface: dict[str, tuple[str, ...]],
    all_fields: dict[str, tuple[str, ...]],
    expensive: frozenset[str],
) -> Optional[dict]:
    """Turn ``accel_scope`` into an ``accelerated_fields`` mapping.

    ``None`` means "let the group auto-detect", which is not the same
    object as the explicit ``"interface"`` mapping even though the two
    normally agree — stating it explicitly is what the AR4 experiment
    did, so the sweep can confirm the two agree.
    """
    if config.acceleration not in ("iqn-ils", "iqn-imvj"):
        return None
    scope = config.accel_scope
    if scope == "auto":
        return None
    if scope == "all":
        return dict(all_fields)
    if scope == "interface":
        return dict(interface)
    if scope in ("expensive", "cheap"):
        want_expensive = scope == "expensive"
        picked = {k: v for k, v in interface.items()
                  if (k in expensive) is want_expensive}
        if not picked:
            raise ScopeNotApplicable(
                f"accel_scope={scope!r} selects no node of this fixture "
                f"(expensive nodes: {sorted(expensive) or 'none'})"
            )
        return picked
    raise ValueError(f"unknown accel_scope {scope!r}")


# ---------------------------------------------------------------------------
# Fixture plumbing
# ---------------------------------------------------------------------------


@dataclass
class BuiltGraph:
    """A compiled fixture graph plus everything a driver needs to run it."""

    gm: object
    external_inputs: Optional[dict] = None
    #: ``"+"``-joined sorted node names, one per coupling group.
    group_keys: tuple[str, ...] = ()
    #: Nodes whose cost dominates a coupling iteration (empty when none).
    expensive_nodes: frozenset[str] = frozenset()
    #: Analytic Jacobi/Gauss-Seidel spectral radii where they are
    #: known, in the canonical schema built by :func:`_rho`: always the
    #: keys ``"jacobi"``, ``"gauss-seidel"`` and ``"groups"``, with
    #: ``None`` where no closed form is known, so a consumer can index
    #: it without a per-fixture special case.
    predicted_rho: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FixtureSpec:
    """A named graph shape plus its sweep defaults."""

    name: str
    family: str
    summary: str
    build: Callable[[CouplingConfig], BuiltGraph]
    #: Timed steps for the benchmark driver.
    steps: int = 60
    #: Warmup steps; ``slow-drift`` needs enough to leave the transient.
    warmup: int = 5
    #: Opt-in only (``--include-slow``): minutes rather than seconds.
    slow: bool = False
    #: Expected regime.  ``bench_coupling_sweep`` compares it against
    #: the measured ``regime`` of every row and records the answer as
    #: ``regime_matches_expect``; it was recorded and never compared to
    #: anything until that was added.
    expect: str = "launch-bound"
    #: True when the fixture pins ``iteration_mode`` itself, so sweeping
    #: that axis would produce duplicate rows.
    mode_fixed: bool = False


# ---------------------------------------------------------------------------
# Spring helpers
# ---------------------------------------------------------------------------
#
# Every spring fixture is built so that the *coupling gain* ``g`` is the
# only thing that varies between shapes.  Two facts drive the
# parameterisation, and both were found the hard way:
#
# 1. ``SpringDamperNode`` integrates with semi-implicit Euler using its
#    own *old* position and the anchor's *new* one, so the derivative of
#    a node's new position with respect to its anchor is exactly
#    ``dt**2 k / m``.  Writing that as ``kappa`` and putting the coupling
#    strength in the *edge weight* ``w`` instead of the stiffness makes
#    ``g = kappa * w`` a knob, and leaves ``kappa * (1 - w)`` as a
#    spring to ground that pins the equilibrium at the origin.  Without
#    that ground term a mutually anchored pair has a zero-stiffness
#    rigid-translation mode and drifts.
#
# 2. Solving the coupled system amplifies by ``1/(1 - mu)`` for each
#    eigenvalue ``mu`` of the coupling operator, so a fixture with
#    ``g`` close to 1 is time-step-unstable unless the node's own
#    dynamics contract by the same factor.  Choosing
#    ``kappa = 1`` and ``xi = dt*c/m = 1 - (1 - g) * LIVELINESS`` makes
#    the per-mode amplification matrix have trace = determinant =
#    ``(1 - xi)/(1 - mu)``, whose eigenvalues have modulus
#    ``sqrt((1-xi)/(1-mu)) <= sqrt(LIVELINESS)`` — stable for every
#    ``g < 1`` and every shape here, with the decay rate set by one
#    dimensionless constant instead of per-fixture guesswork.
#
# The price is that a strongly coupled spring fixture is necessarily
# well damped, so its own transient dies within a few tens of steps.  A
# driver node outside every coupling group (see ``_add_driver``) keeps
# the residual excited for the whole run.

#: Total self-stiffness of every coupled spring, as ``dt**2 k / m``.
_SELF_KAPPA = 1.0
#: ``|lambda|**2`` of the slowest mode: 0.8 means the amplitude decays
#: ~11% per step, slow enough to stay interesting and fast enough that
#: the fixture does not ring forever.
_LIVELINESS = 0.8
#: Fraction of a coupled node's anchor supplied by the driver.
_DRIVE_W = 0.15


def _rho(jacobi=None, gauss_seidel=None, groups=None) -> dict:
    """The canonical ``predicted_rho`` schema.

    Every fixture records the same three keys whether or not it knows a
    value for them.  Before this was uniform, ``ring-*`` and
    ``heterogeneous`` omitted ``"gauss-seidel"`` and ``mixed-modes``
    used two entirely different keys, so a consumer reading
    ``predicted_rho["jacobi"]`` got a silent miss on one fixture and a
    ``KeyError`` on another.

    Parameters
    ----------
    jacobi, gauss_seidel : float or None
        Analytic spectral radii of the fixture's coupling operator, or
        ``None`` where no closed form is known: a ring's per-node gains
        are unequal, and a mixed heat/spring group has no single
        consistently-ordered operator whose Gauss-Seidel radius is the
        square of its Jacobi one.
    groups : dict or None
        For a multi-group fixture, ``{group_key: {"jacobi": ...,
        "gauss-seidel": ...}}``; the top-level pair then stays ``None``
        because the graph has no single radius.
    """
    return {
        "jacobi": jacobi,
        "gauss-seidel": gauss_seidel,
        "groups": groups or {},
    }


def _xi_for(gain: float) -> float:
    """Dimensionless damping ``dt*c/m`` that keeps gain *gain* stable."""
    return 1.0 - (1.0 - gain) * _LIVELINESS


def _mode_radius(gain: float, mu: float) -> float:
    """Per-timestep amplification of a coupling mode with eigenvalue *mu*."""
    s = (1.0 - _xi_for(gain)) / (1.0 - mu)
    disc = s * s - 4.0 * s
    if disc < 0.0:
        return math.sqrt(abs(s))
    return max(abs((s + math.sqrt(disc)) / 2.0), abs((s - math.sqrt(disc)) / 2.0))


def _spring(name, dt, gain, mass=1.0, x0=0.0, v0=0.0, damped=True):
    """A coupled spring node parameterised by its coupling gain.

    *gain* is ``g``: the derivative of this node's new position with
    respect to the sum of its coupling edges' contributions.  The node's
    total stiffness is fixed at ``kappa = 1`` and the damping follows
    :func:`_xi_for`, so *gain* alone decides both the contraction factor
    the iteration sees and the damping needed to keep the coupled
    time-step stable.  ``damped=False`` builds the undamped driver.
    """
    from maddening.nodes.spring import SpringDamperNode

    if damped and gain < 1.0:
        # Cheap guard against the failure that cost the most time while
        # these fixtures were being built: a time-stepping blow-up that
        # looks exactly like a coupling failure in the diagnostics.  The
        # worst mode of a symmetric coupling operator has |mu| <= gain,
        # and the parameterisation is chosen so this always passes — but
        # it stops passing the moment someone changes _LIVELINESS or
        # hands in a gain the damping was not derived for.  (For the
        # ring, where the gain varies per node, this is approximate: it
        # checks each node against its own gain rather than against the
        # ring's largest.)
        radius = _mode_radius(gain, gain)
        if radius >= 1.0:
            raise ValueError(
                f"spring {name!r}: gain={gain:g} with LIVELINESS="
                f"{_LIVELINESS:g} gives a per-step amplification of "
                f"{radius:.3f} >= 1.  The fixture would diverge for "
                f"time-stepping reasons that have nothing to do with the "
                f"coupling algorithm."
            )
    stiffness = _SELF_KAPPA * mass / (dt * dt)
    xi = _xi_for(gain) if damped else 0.0
    damping = xi * mass / dt
    return SpringDamperNode(
        name, dt, stiffness=stiffness, damping=damping, mass=mass,
        rest_length=0.0, initial_position=x0, initial_velocity=v0,
    )


def _add_driver(gm, dt, targets, *, name="driver", period_steps=44.0):
    """Add an undamped oscillator outside the group and wire it to *targets*.

    A strongly coupled spring fixture has to be well damped to stay
    time-step stable (see the module comment), which means its own
    transient is gone within a few tens of steps and every configuration
    then converges in one iteration on a dead graph.  The driver is an
    undamped spring anchored to ground — semi-implicit Euler is
    symplectic for that, so it oscillates at constant amplitude forever
    — feeding a fixed fraction of each coupled node's anchor.  Nothing
    feeds back into it, so it stays outside the coupling group's
    strongly connected component and does not change the group's
    contraction factor; it only stops the residual decaying to the
    float32 floor halfway through the run.
    """
    from maddening.core.transforms import scale

    from maddening.nodes.spring import SpringDamperNode

    omega_dt = 2.0 * math.pi / period_steps
    gm.add_node(SpringDamperNode(
        name, dt, stiffness=(omega_dt / dt) ** 2, damping=0.0, mass=1.0,
        rest_length=0.0, initial_position=1.0, initial_velocity=0.0,
    ))
    w = scale(_DRIVE_W)
    for target in targets:
        gm.add_edge(name, target, "position", "anchor_position",
                    transform=w, additive=True)
    return name


def _first_scaled(factor: float):
    """``extract_first`` composed with a scale, as one registered transform."""
    from maddening.core.transforms import register_transform

    name = f"coupling_fixture_first_scaled_{factor:g}"

    def _fn(arr, _f=factor):
        return _f * arr[0]

    _fn.__qualname__ = name
    try:
        register_transform(name, f"First element times {factor:g}.")(_fn)
    except Exception:  # already registered by an earlier build
        from maddening.core.transforms import resolve_transform
        return resolve_transform(name)
    return _fn


def _spring_fields(names: Sequence[str]) -> tuple[dict, dict]:
    """(interface fields, all state fields) for a group of spring nodes."""
    interface = {n: ("position",) for n in names}
    all_fields = {n: ("position", "velocity") for n in names}
    return interface, all_fields


_SPRING_DT = 0.05
#: Absolute L2 tolerance for the spring fixtures.  The coupled state
#: carries velocities of order ``g * x / dt``, i.e. ~10 for O(1)
#: displacements, so the float32 residual floor is ~1e-6 and an
#: "obvious" 1e-8 would never be reachable.  1e-4 is ~1e-5 relative.
_SPRING_TOL = 1e-4
#: High enough that the ``g = 0.95`` fixture without acceleration is
#: inside the cap for Gauss-Seidel and outside it for Jacobi — which is
#: the row that shows what an accelerator is for.
_SPRING_MAXIT = 60
#: The gain at which the ``stiff-pair`` sweep's tolerance is exactly
#: ``_SPRING_TOL``; the other gains are scaled around it.
_STIFF_REF_GAIN = 0.8


def _stiff_pair_tolerance(gain: float) -> float:
    """Absolute tolerance that keeps the *relative* one flat across the sweep.

    Solving the coupled system amplifies the driver's forcing by
    ``1/(1 - g)``, so the state amplitude of ``stiff-pair-G`` grows
    about fifteen-fold from ``g = 0.25`` to ``g = 0.95``.  Held against
    a fixed *absolute* L2 tolerance that is a fifteen-fold swing in the
    effective relative tolerance, and the iteration count the sweep
    records — modelled well by ``ln(tol/||x||)/ln(rho)`` — then moves
    with the amplitude as well as with the contraction factor the
    fixture exists to vary.  Scaling the tolerance by the same
    ``1/(1 - g)`` leaves the gain as the only thing that changes.

    Anchored at :data:`_STIFF_REF_GAIN` so the fixture the rest of the
    suite and the iteration-cap sweep quote keeps exactly the tolerance
    the other spring shapes use.  ``g >= 1`` is divergent by
    construction and never reaches any tolerance, so it keeps the
    unscaled value rather than a negative one.
    """
    if gain >= 1.0:
        return _SPRING_TOL
    return _SPRING_TOL * (1.0 - _STIFF_REF_GAIN) / (1.0 - gain)


def _finish(gm, names, config, *, expensive=frozenset(), predicted=None,
            max_iterations=_SPRING_MAXIT, tolerance=_SPRING_TOL,
            atol=1e-8, rtol=1e-4, groups=None):
    """Attach coupling group(s), compile, and wrap in a :class:`BuiltGraph`."""
    interface, all_fields = _spring_fields(names)
    if groups is None:
        groups = [(list(names), interface, all_fields)]
    keys = []
    for group_nodes, iface, allf in groups:
        accel = _resolve_accel_fields(
            config, interface=iface, all_fields=allf, expensive=expensive,
        )
        gm.add_coupling_group(
            group_nodes,
            **config.group_kwargs(
                max_iterations=max_iterations, tolerance=tolerance,
                atol=atol, rtol=rtol, accelerated_fields=accel,
            ),
        )
        keys.append("+".join(sorted(group_nodes)))
    gm.compile()
    return BuiltGraph(
        gm=gm, group_keys=tuple(keys), expensive_nodes=expensive,
        predicted_rho=predicted or {},
    )


# ---------------------------------------------------------------------------
# 1. chain-N — sequential information flow
# ---------------------------------------------------------------------------


def build_chain(n: int, config: CouplingConfig, gain: float = 0.8,
                dt: float = _SPRING_DT) -> BuiltGraph:
    """*n* spring-damper nodes in a line, each anchored to its neighbours.

    Each interior node's ``anchor_position`` is ``gain/2`` times the sum
    of its two neighbours' positions (two additive edges), so the
    coupling operator is ``gain/2`` times the path-graph adjacency and
    the rest of the node's stiffness pins it to the origin.  Information
    has to travel the length of the line, which is the property
    Gauss-Seidel is supposed to exploit and Jacobi is not.

    Predicted Jacobi spectral radius ``gain * cos(pi/(n+1))``; a
    tridiagonal operator is consistently ordered, so the theory says
    Gauss-Seidel is exactly its square.
    """
    from maddening.core.graph_manager import GraphManager
    from maddening.core.transforms import scale

    gm = GraphManager()
    names = [f"link{i}" for i in range(n)]
    for i, nm in enumerate(names):
        gm.add_node(_spring(nm, dt, gain, x0=float(i) - 0.5 * (n - 1)))
    half = scale(0.5 * gain)
    for a, b in zip(names[:-1], names[1:]):
        gm.add_edge(a, b, "position", "anchor_position",
                    transform=half, additive=True)
        gm.add_edge(b, a, "position", "anchor_position",
                    transform=half, additive=True)
    _add_driver(gm, dt, names)
    rho_j = gain * math.cos(math.pi / (n + 1))
    return _finish(gm, names, config,
                   predicted=_rho(jacobi=rho_j, gauss_seidel=rho_j ** 2))


# ---------------------------------------------------------------------------
# 2. star-N — width without leaf-to-leaf paths
# ---------------------------------------------------------------------------


def build_star(n_leaves: int, config: CouplingConfig, gain: float = 0.8,
               dt: float = _SPRING_DT) -> BuiltGraph:
    """A hub node with *n_leaves* independent leaves.

    Every leaf is anchored to the hub at weight *gain*; the hub is
    anchored to the mean of the leaves, also at weight *gain*.  No leaf
    can see another, so a Gauss-Seidel sweep gains nothing from the
    ordering *among* the leaves — the only sequential dependency is
    hub-to-leaf.  The ``1/n_leaves`` on the hub's edges cancels the
    ``n_leaves`` terms, so the predicted Jacobi radius is *gain*
    independent of width.
    """
    from maddening.core.graph_manager import GraphManager
    from maddening.core.transforms import scale

    gm = GraphManager()
    gm.add_node(_spring("hub", dt, gain, x0=0.0))
    leaves = [f"leaf{i}" for i in range(n_leaves)]
    full = scale(gain)
    for i, nm in enumerate(leaves):
        gm.add_node(_spring(nm, dt, gain, x0=1.0 + 0.1 * i))
        gm.add_edge("hub", nm, "position", "anchor_position",
                    transform=full, additive=True)
    w = scale(gain / n_leaves)
    for nm in leaves:
        gm.add_edge(nm, "hub", "position", "anchor_position",
                    transform=w, additive=True)
    names = ["hub"] + leaves
    _add_driver(gm, dt, names)
    return _finish(gm, names, config,
                   predicted=_rho(jacobi=gain, gauss_seidel=gain ** 2))


# ---------------------------------------------------------------------------
# 3. ring-N — a cycle with no natural first node
# ---------------------------------------------------------------------------


def build_ring(n: int, config: CouplingConfig, gain: float = 0.8,
               dt: float = _SPRING_DT, rotation: int = 0,
               reverse: bool = False) -> BuiltGraph:
    """A closed, *bidirectional* cycle of *n* spring-damper nodes.

    Each node is anchored to the mean of its two ring neighbours, so
    there is no preferred direction and no natural first node.  Which
    node Gauss-Seidel updates first is decided purely by the order the
    nodes were added (``topological_sort`` returns the nodes of a cycle
    in insertion order), so *rotation* and *reverse* permute that
    insertion order while leaving the physical ring identical.  That is
    the handle the order-dependence test pulls: Gauss-Seidel iterates
    differ between rotations, Jacobi's do not.

    A *directed* ring would not show this — a one-way cycle gives
    Gauss-Seidel a radius of ``gain**n`` from any cut point.

    The per-node gains are deliberately unequal.  The coupling operator
    is ``gain_i`` times the ring adjacency and ``kappa = dt**2 k/m`` is
    invariant to the mass, so varying the *mass* would leave the
    iteration matrix perfectly symmetric and the sweep direction
    irrelevant; varying the gain is what makes the order matter.
    """
    from maddening.core.graph_manager import GraphManager
    from maddening.core.transforms import scale

    gm = GraphManager()
    names = [f"seg{i}" for i in range(n)]
    gains = {nm: gain * (0.6 + 0.4 * i / max(n - 1, 1))
             for i, nm in enumerate(names)}
    # The uniform (rigid-translation) mode is the slowest-converging one
    # — eigenvalue ``gain`` — so the initial state must excite it.  A pure
    # ``sin`` profile is orthogonal to it and converges in one iteration
    # on an even ring, which is a property of the initial condition
    # rather than of the algorithm.
    x0 = {nm: 1.0 + math.sin(2 * math.pi * i / n)
          for i, nm in enumerate(names)}

    order = [names[(i + rotation) % n] for i in range(n)]
    if reverse:
        order = list(reversed(order))
    for nm in order:
        gm.add_node(_spring(nm, dt, gains[nm], x0=x0[nm]))
    for i, nm in enumerate(names):
        nxt = names[(i + 1) % n]
        gm.add_edge(nm, nxt, "position", "anchor_position",
                    transform=scale(0.5 * gains[nxt]), additive=True)
        gm.add_edge(nxt, nm, "position", "anchor_position",
                    transform=scale(0.5 * gains[nm]), additive=True)
    _add_driver(gm, dt, names)
    # A ring's per-node gains are deliberately unequal, so it is not a
    # consistently-ordered operator and the Gauss-Seidel radius is not
    # the square of the Jacobi one; recorded as unknown rather than
    # guessed.
    return _finish(gm, names, config, predicted=_rho(jacobi=gain))


# ---------------------------------------------------------------------------
# 4. stiff-pair(gain) — the contraction axis
# ---------------------------------------------------------------------------


def build_stiff_pair(config: CouplingConfig, gain: float = 0.5,
                     dt: float = _SPRING_DT,
                     max_iterations: int = _SPRING_MAXIT) -> BuiltGraph:
    """Two mutually anchored spring-damper nodes with coupling gain *gain*.

    Jacobi contracts as ``gain`` per iteration and Gauss-Seidel as
    ``gain**2``.  ``gain >= 1`` is past the convergence limit: the group
    cannot converge and must say so rather than truncate at the cap.

    The tolerance follows :func:`_stiff_pair_tolerance` rather than the
    shared ``_SPRING_TOL``, so that the gain is the only property that
    varies along this sweep.
    """
    from maddening.core.graph_manager import GraphManager
    from maddening.core.transforms import scale

    gm = GraphManager()
    gm.add_node(_spring("a", dt, gain, x0=-1.0))
    gm.add_node(_spring("b", dt, gain, x0=1.0))
    w = scale(gain)
    gm.add_edge("a", "b", "position", "anchor_position",
                transform=w, additive=True)
    gm.add_edge("b", "a", "position", "anchor_position",
                transform=w, additive=True)
    _add_driver(gm, dt, ["a", "b"])
    return _finish(gm, ["a", "b"], config, max_iterations=max_iterations,
                   tolerance=_stiff_pair_tolerance(gain),
                   predicted=_rho(jacobi=gain, gauss_seidel=gain ** 2))


# ---------------------------------------------------------------------------
# 5. expensive-pair — cost per iteration
# ---------------------------------------------------------------------------


#: Largest Fourier number at which two ``HeatNode`` slabs that exchange
#: their end-cell temperatures as Dirichlet data are stable in *time*
#: once the exchange is solved to convergence.  The pair's interface
#: mode has amplification ``-1`` exactly at ``3/8`` (derived from the
#: coupled operator, and measured: a 200-cell pair stays bounded over
#: 60 steps at 0.375 and grows to 4e5 at 0.4 and 1e33 at 0.45 under
#: ``gs/none``).  ``HeatNode``'s own limit, 1/2, is for fixed data.
_HEAT_PAIR_FOURIER_LIMIT = 3.0 / 8.0


def _check_heat_pair_fourier(fourier):
    """Refuse a slab-pair Fourier number the coupled pair cannot step."""
    if not 0.0 < fourier < _HEAT_PAIR_FOURIER_LIMIT:
        raise ValueError(
            f"fourier={fourier} is outside (0, {_HEAT_PAIR_FOURIER_LIMIT}): "
            "two slabs exchanging end-cell temperatures are unstable in "
            "time there once the exchange converges"
        )


def _heat_pair(config, n_cells, fourier, *, names=("slabA", "slabB"),
               max_iterations=15, tolerance=1e-5):
    """Two 1-D heat grids exchanging interface temperatures.

    Each slab's end-cell temperature is the other's Dirichlet datum, and
    ``HeatNode`` imposes that datum through the ghost cell
    ``2*T_b - T[0]``, so the gain from one slab's interface cell to the
    other's is ``2*Fo`` with ``Fo = alpha*dt/dx**2`` — Jacobi contracts
    as ``2*Fo``, Gauss-Seidel as ``(2*Fo)**2``.
    """
    _check_heat_pair_fourier(fourier)
    from maddening.core.graph_manager import GraphManager
    from maddening.core.transforms import extract_first, extract_last
    from maddening.nodes.heat import HeatNode

    a, b = names
    length = 1.0
    alpha = 0.1
    dx = length / n_cells
    dt = fourier * dx * dx / alpha

    gm = GraphManager()
    gm.add_node(HeatNode(a, dt, n_cells=n_cells, length=length,
                         thermal_diffusivity=alpha, initial_temperature=300.0))
    gm.add_node(HeatNode(b, dt, n_cells=n_cells, length=length,
                         thermal_diffusivity=alpha, initial_temperature=360.0))
    gm.add_edge(a, b, "temperature", "left_temperature", transform=extract_last)
    gm.add_edge(b, a, "temperature", "right_temperature", transform=extract_first)

    interface = {a: ("temperature",), b: ("temperature",)}
    accel = _resolve_accel_fields(
        config, interface=interface, all_fields=interface,
        expensive=frozenset(names),
    )
    gm.add_coupling_group(
        [a, b],
        **config.group_kwargs(max_iterations=max_iterations,
                              tolerance=tolerance, atol=1e-6, rtol=1e-4,
                              accelerated_fields=accel),
    )
    gm.compile()
    return BuiltGraph(
        gm=gm, group_keys=("+".join(sorted(names)),),
        expensive_nodes=frozenset(names),
        predicted_rho=_rho(jacobi=2.0 * fourier,
                           gauss_seidel=(2.0 * fourier) ** 2),
    )


def build_expensive_pair(config: CouplingConfig, n_cells: int = 100_000,
                         fourier: float = 0.2) -> BuiltGraph:
    """Two ~1e5-cell heat grids coupled at one interface.

    The compute-bound counterpart to ``chain-N``: the iteration *count*
    is small and flat, and what an accelerator has to beat is the cost
    of a whole grid sweep.  Because the auto-detected interface field is
    the entire ``temperature`` array, the quasi-Newton least-squares
    runs on 2*n_cells rows — which is exactly the trade IQN is meant to
    win and exactly where it can lose.

    ``fourier=0.2`` puts the interface gain at ``2*Fo = 0.4``, the rate
    this fixture was designed and first recorded at (it was ``Fo = 0.4``
    while ``HeatNode`` wrote its datum into the end cell), and keeps the
    pair under ``_HEAT_PAIR_FOURIER_LIMIT``: at 0.4 the converged pair
    is unstable in time.
    """
    return _heat_pair(config, n_cells, fourier)


# ---------------------------------------------------------------------------
# 6. heterogeneous — one expensive node among cheap ones
# ---------------------------------------------------------------------------


def build_heterogeneous(config: CouplingConfig, n_cells: int = 60_000,
                        n_cheap: int = 4, fourier: float = 0.2,
                        gain: float = 0.6,
                        dt: float = _SPRING_DT) -> BuiltGraph:
    """One expensive heat grid coupled to *n_cheap* scalar spring nodes.

    A synthetic thermo-structural loop: the grid's left interface
    temperature is driven by the mean of the scalar nodes' positions, and
    each scalar node is anchored to the grid's left interface cell.  The
    grid carries ``n_cells`` accelerated degrees of freedom and the
    scalars carry one each, so this is the fixture where restricting
    ``accelerated_fields`` changes the size of the quasi-Newton problem
    by four orders of magnitude.

    The timestep is the *spring's*, and the diffusivity is solved for to
    hit the requested Fourier number, which is the reverse of the
    pure-heat fixtures.  Doing it the other way round — fixing a
    physical ``alpha`` and letting ``dt = Fo*dx**2/alpha`` — gives
    ``dt ~ 1e-9`` at this grid resolution, and a spring at that timestep
    carries velocities of order ``position/dt``, i.e. ten million times
    its positions.  The group's global L2 residual is then entirely
    velocity round-off: every Jacobi L2 row exhausts the cap, the
    surviving state sits at the float32 floor, and the fixture measures
    nothing but its own conditioning.  The resulting ``alpha`` is not a
    physical material property, and does not need to be: the interface
    gain this fixture exists to control is the Fourier number, which is
    the same either way.

    ``fourier=0.2`` makes the grid's interface gain ``2*Fo = 0.4`` and
    the loop's Jacobi rate ``sqrt(0.4 * gain) = 0.49``, the rate the
    fixture was designed and first recorded at (``Fo = 0.4`` while
    ``HeatNode`` wrote its datum into the end cell).  Unlike the slab
    pairs this loop is stable in time either way; at 0.4 it contracts
    at 0.69, and plain Jacobi then converges on 5% of steps within the
    cap (a 2 000-cell grid, 40 steps).
    """
    from maddening.core.graph_manager import GraphManager
    from maddening.core.transforms import scale
    from maddening.nodes.heat import HeatNode

    import numpy as np

    length = 1.0
    dx = length / n_cells
    alpha = fourier * dx * dx / dt
    xs = np.linspace(0.0, 1.0, n_cells, dtype=np.float32)

    gm = GraphManager()
    gm.add_node(HeatNode("grid", dt, n_cells=n_cells, length=length,
                         thermal_diffusivity=alpha,
                         initial_temperature=np.cos(np.pi * xs)))
    cheap = [f"probe{i}" for i in range(n_cheap)]
    for i, nm in enumerate(cheap):
        gm.add_node(_spring(nm, dt, gain, x0=1.0 + 0.2 * i))
        # ``additive=True`` matters: ``_add_driver`` wires a second
        # edge into the same ``anchor_position``, and a non-additive
        # edge *overwrites* whichever contribution was resolved first.
        # With both additive the boundary value is the sum whatever
        # order the edges were inserted in; with one of them
        # non-additive the fixture is correct only by insertion luck,
        # and the losing case silently drops the driver and lets the
        # graph go dead -- the exact failure the driver exists to stop.
        gm.add_edge("grid", nm, "temperature", "anchor_position",
                    transform=_first_scaled(gain), additive=True)
    w = scale(1.0 / n_cheap)
    for nm in cheap:
        gm.add_edge(nm, "grid", "position", "left_temperature",
                    transform=w, additive=True)
    # Same reason as the pure-spring fixtures: without a driver the
    # scalars settle onto the grid interface within a few steps and every
    # configuration then converges in one iteration on a dead graph.
    _add_driver(gm, dt, cheap)

    names = ["grid"] + cheap
    interface = {"grid": ("temperature",), **{n: ("position",) for n in cheap}}
    all_fields = {"grid": ("temperature",),
                  **{n: ("position", "velocity") for n in cheap}}
    accel = _resolve_accel_fields(
        config, interface=interface, all_fields=all_fields,
        expensive=frozenset({"grid"}),
    )
    gm.add_coupling_group(
        names,
        **config.group_kwargs(max_iterations=20, tolerance=_SPRING_TOL,
                              atol=1e-6, rtol=1e-4, accelerated_fields=accel),
    )
    gm.compile()
    return BuiltGraph(gm=gm, group_keys=("+".join(sorted(names)),),
                      expensive_nodes=frozenset({"grid"}),
                      # The grid's interface gain is ``2*Fo`` (the
                      # ghost-cell datum, see ``_heat_pair``) and the
                      # probes' is ``gain``; the loop through both is
                      # their product.  Measured: 0.694 under Jacobi at
                      # Fo = 0.4, 0.490 at 0.2.
                      predicted_rho=_rho(
                          jacobi=math.sqrt(2.0 * fourier * gain)))


# ---------------------------------------------------------------------------
# 7. mixed-modes — two groups, two schedules
# ---------------------------------------------------------------------------


def build_mixed_modes(config: CouplingConfig, n_chain: int = 5,
                      n_leaves: int = 8, gain: float = 0.8,
                      dt: float = _SPRING_DT, *,
                      chain_mode: str = "gauss-seidel",
                      star_mode: str = "jacobi") -> BuiltGraph:
    """One graph holding a stiff chain and a wide star as separate groups.

    The chain group always runs ``gauss-seidel`` and the star group
    always runs ``jacobi``, whatever *config* says about
    ``iteration_mode`` — the point of the fixture is that two groups in
    one graph keep their own schedules.  *chain_mode* and *star_mode*
    override that, and exist for the invariant test: the only way to
    show that a group *honoured* its declared mode, rather than that
    the dataclass remembered it, is to build the counterfactual where
    that one group runs the other schedule and watch the iterate
    change.  The other options
    (acceleration, norm, accelerated fields) are applied to both, so the
    sweep still means something.  A single one-way edge from the chain's
    tail to the star's hub keeps the two cycles in one graph without
    merging them into one strongly connected component.
    """
    from maddening.core.graph_manager import GraphManager
    from maddening.core.transforms import scale

    gm = GraphManager()
    chain = [f"chain{i}" for i in range(n_chain)]
    for i, nm in enumerate(chain):
        gm.add_node(_spring(nm, dt, gain, x0=float(i) - 0.5 * (n_chain - 1)))
    half = scale(0.5 * gain)
    for a, b in zip(chain[:-1], chain[1:]):
        gm.add_edge(a, b, "position", "anchor_position",
                    transform=half, additive=True)
        gm.add_edge(b, a, "position", "anchor_position",
                    transform=half, additive=True)

    star_gain = 0.5
    star_hub = "star_hub"
    leaves = [f"star_leaf{i}" for i in range(n_leaves)]
    gm.add_node(_spring(star_hub, dt, star_gain, x0=0.0))
    full = scale(star_gain)
    for i, nm in enumerate(leaves):
        gm.add_node(_spring(nm, dt, star_gain, x0=1.0 + 0.1 * i))
        gm.add_edge(star_hub, nm, "position", "anchor_position",
                    transform=full, additive=True)
    w = scale(star_gain / (n_leaves + 1))
    for nm in leaves:
        gm.add_edge(nm, star_hub, "position", "anchor_position",
                    transform=w, additive=True)
    # One-way link: the chain's tail feeds the star's hub and nothing
    # comes back, so the two SCCs stay separate.
    gm.add_edge(chain[-1], star_hub, "position", "anchor_position",
                transform=w, additive=True)

    star = [star_hub] + leaves
    _add_driver(gm, dt, chain + star)

    chain_cfg = replace(config, iteration_mode=chain_mode)
    star_cfg = replace(config, iteration_mode=star_mode)
    for nodes, cfg in ((chain, chain_cfg), (star, star_cfg)):
        iface, allf = _spring_fields(nodes)
        accel = _resolve_accel_fields(
            cfg, interface=iface, all_fields=allf, expensive=frozenset(),
        )
        gm.add_coupling_group(
            nodes,
            **cfg.group_kwargs(max_iterations=_SPRING_MAXIT,
                               tolerance=_SPRING_TOL, atol=1e-8, rtol=1e-4,
                               accelerated_fields=accel),
        )
    gm.compile()
    return BuiltGraph(
        gm=gm,
        group_keys=("+".join(sorted(chain)), "+".join(sorted(star))),
        # Two groups, so there is no single radius for the graph; the
        # pair per group goes in ``groups`` under the same keys the
        # diagnostics use.
        predicted_rho=_rho(groups={
            "+".join(sorted(chain)): {
                "jacobi": gain * math.cos(math.pi / (n_chain + 1)),
                "gauss-seidel": (gain * math.cos(math.pi / (n_chain + 1))) ** 2,
            },
            "+".join(sorted(star)): {
                "jacobi": star_gain, "gauss-seidel": star_gain ** 2,
            },
        }),
    )


# ---------------------------------------------------------------------------
# 8. slow-drift — a fixed point that barely moves
# ---------------------------------------------------------------------------


def build_slow_drift(config: CouplingConfig, n_cells: int = 2_000,
                     fourier: float = 0.225) -> BuiltGraph:
    """The ``expensive-pair`` shape, run where the fixed point creeps.

    Two heat slabs started at uniform ``+1`` and ``-1``, so the interface
    carries a step discontinuity that diffusion smooths out.  After the
    first few steps the interface value moves a little, smoothly and
    monotonically, every step — the coupling still has real work each
    step (the two slabs disagree about the interface temperature) but
    this step's interface Jacobian is almost exactly last step's.  That
    is the only regime in which ``jacobian_reuse`` has anything to
    reuse.

    Why not the spring pair: solving a coupled system amplifies by
    ``1/(1 - g)``, so a spring pair with a coupling gain high enough to
    need several iterations has to be damped hard enough to stay
    time-step stable, and then settles within a few tens of steps
    instead of drifting.  Diffusion separates the two scales — the
    interface gain is ``2*Fo`` while the profile relaxes over
    ``O(n_cells**2)`` steps.

    ``fourier=0.225`` is an interface gain of 0.45 under Jacobi and
    0.2025 under Gauss-Seidel, the rates the fixture was designed and
    first recorded at, when ``HeatNode`` wrote its datum into the end
    cell and the gain was ``Fo = 0.45``.  Left at 0.45 after the
    ghost-cell fix the gain was 0.9 and the pair was past
    ``_HEAT_PAIR_FOURIER_LIMIT``: ``gs/none/l2`` exhausted its cap of 25
    on every step and the state left float range within 90 steps.
    """
    _check_heat_pair_fourier(fourier)
    import numpy as np

    from maddening.core.graph_manager import GraphManager
    from maddening.core.transforms import extract_first, extract_last
    from maddening.nodes.heat import HeatNode

    length = 1.0
    alpha = 0.1
    dx = length / n_cells
    dt = fourier * dx * dx / alpha

    gm = GraphManager()
    gm.add_node(HeatNode("slabA", dt, n_cells=n_cells, length=length,
                         thermal_diffusivity=alpha,
                         initial_temperature=np.float32(1.0)))
    gm.add_node(HeatNode("slabB", dt, n_cells=n_cells, length=length,
                         thermal_diffusivity=alpha,
                         initial_temperature=np.float32(-1.0)))
    gm.add_edge("slabA", "slabB", "temperature", "left_temperature",
                transform=extract_last)
    gm.add_edge("slabB", "slabA", "temperature", "right_temperature",
                transform=extract_first)

    interface = {"slabA": ("temperature",), "slabB": ("temperature",)}
    accel = _resolve_accel_fields(
        config, interface=interface, all_fields=interface,
        expensive=frozenset({"slabA", "slabB"}),
    )
    gm.add_coupling_group(
        ["slabA", "slabB"],
        **config.group_kwargs(max_iterations=25, tolerance=1e-5,
                              atol=1e-7, rtol=1e-4, accelerated_fields=accel),
    )
    gm.compile()
    return BuiltGraph(gm=gm, group_keys=("slabA+slabB",),
                      predicted_rho=_rho(jacobi=2.0 * fourier,
                                         gauss_seidel=(2.0 * fourier) ** 2))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _reg(spec: FixtureSpec, into: dict) -> None:
    into[spec.name] = spec


def _build_registry() -> dict[str, FixtureSpec]:
    reg: dict[str, FixtureSpec] = {}

    for n in (2, 5, 20, 50):
        _reg(FixtureSpec(
            name=f"chain-{n}", family="chain",
            summary=f"{n} spring-damper nodes in a line (sequential depth)",
            build=(lambda cfg, _n=n: build_chain(_n, cfg)),
            steps=60 if n <= 20 else 30,
        ), reg)

    for n in (2, 4, 8, 16):
        _reg(FixtureSpec(
            name=f"star-{n}", family="star",
            summary=f"one hub, {n} independent leaves (width, no leaf-leaf path)",
            build=(lambda cfg, _n=n: build_star(_n, cfg)),
            steps=60 if n <= 8 else 30,
        ), reg)

    for n in (4, 8, 16):
        _reg(FixtureSpec(
            name=f"ring-{n}", family="ring",
            summary=f"closed bidirectional cycle of {n} nodes (no first node)",
            build=(lambda cfg, _n=n: build_ring(_n, cfg)),
            steps=60 if n <= 8 else 30,
        ), reg)

    for gain in (0.25, 0.5, 0.8, 0.95, 1.2):
        _reg(FixtureSpec(
            name=f"stiff-pair-{gain:g}", family="stiff-pair",
            summary=f"two nodes with coupling gain {gain:g}"
                    + (" (past the convergence limit)" if gain >= 1 else ""),
            build=(lambda cfg, _k=gain: build_stiff_pair(
                cfg, gain=_k, max_iterations=12 if _k >= 1 else _SPRING_MAXIT)),
            # A divergent group grows the state by ~rho**max_iterations
            # every step, so the run is kept short and the cap low: the
            # row exists to show the group reporting itself unconverged,
            # not to produce a meaningful step time.
            steps=10 if gain >= 1 else 60,
        ), reg)

    _reg(FixtureSpec(
        name="expensive-pair", family="expensive-pair",
        summary="two 1e5-cell heat grids coupled at one interface",
        build=build_expensive_pair, steps=20, slow=True,
        expect="compute-bound",
    ), reg)

    _reg(FixtureSpec(
        name="heterogeneous", family="heterogeneous",
        summary="one 6e4-cell grid plus four scalar nodes in one group",
        build=build_heterogeneous, steps=20, slow=True,
        expect="compute-bound",
    ), reg)

    _reg(FixtureSpec(
        name="mixed-modes", family="mixed-modes",
        summary="one graph, two groups: Gauss-Seidel chain + Jacobi star",
        build=build_mixed_modes, steps=40, mode_fixed=True,
    ), reg)

    _reg(FixtureSpec(
        name="slow-drift", family="slow-drift",
        summary="heat pair whose fixed point creeps (jacobian_reuse's home)",
        build=build_slow_drift, steps=60, warmup=40,
        # 2 000 cells x two slabs is past this CPU's dispatch floor:
        # every recorded row measured compute-bound or mixed, none
        # launch-bound.  Declared to match what it does.
        expect="compute-bound",
    ), reg)

    return reg


FIXTURES: dict[str, FixtureSpec] = _build_registry()


def fixture_names(include_slow: bool = False) -> list[str]:
    """Registry names, slow fixtures only when *include_slow*."""
    return [n for n, f in FIXTURES.items() if include_slow or not f.slow]


# ---------------------------------------------------------------------------
# The sweep grid
# ---------------------------------------------------------------------------


def sweep_configs(norms: Sequence[str] = ("l2", "interface"),
                  *, extra_fields: bool = False) -> list[CouplingConfig]:
    """The product the brief asks for.

    Two iteration modes x five accelerations (``fixed`` at two relaxation
    values) x the requested convergence norms.  With *extra_fields* the
    IQN rows are repeated with ``accel_scope`` set to ``"all"``,
    ``"expensive"`` and ``"cheap"`` so the ``accelerated_fields``
    question can be answered on the heterogeneous fixture.
    """
    accels: list[tuple[str, dict]] = [
        ("none", {}),
        ("aitken", {}),
        ("fixed", {"relaxation": 0.5}),
        ("fixed", {"relaxation": 0.8}),
        ("iqn-ils", {}),
        ("iqn-imvj", {"jacobian_reuse": 5}),
    ]
    out: list[CouplingConfig] = []
    for mode in ("gauss-seidel", "jacobi"):
        for norm in norms:
            for accel, kw in accels:
                out.append(CouplingConfig(
                    iteration_mode=mode, acceleration=accel,
                    convergence_norm=norm, **kw,
                ))
                if extra_fields and accel.startswith("iqn"):
                    for scope in ("all", "expensive", "cheap"):
                        out.append(CouplingConfig(
                            iteration_mode=mode, acceleration=accel,
                            convergence_norm=norm, accel_scope=scope, **kw,
                        ))
    return out
