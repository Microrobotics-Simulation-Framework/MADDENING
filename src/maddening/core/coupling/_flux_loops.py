"""Boundary fluxes that read each other inside a coupling group: what a
coupling pass computes for them, and the groups that are refused because
the pass would not compute the fluxes of its own iterate.

Private.  Read by ``GraphManager.compile()``, ``GraphManager.validate()``
and the readers of a graph that was never compiled
(``GraphManager._boundary_inputs_from``).

What a pass computes
--------------------
A coupling group iterates the **states** of its members.  A boundary flux
is a value ``compute_boundary_fluxes(state, boundary_inputs, dt)`` returns;
it is not a state field, it is not part of the iterate, the convergence
norm does not measure it, and nothing carries it from one pass to the
next.  Every pass rebuilds the fluxes from the iterate before it updates
anything (``one_pass_gs`` and ``one_pass_jacobi`` in
``_coupled_block._run_coupled_block_impl``)::

    flux_s = {}                                  # nothing is known
    for sweep in (1, 2):                         # SEED_SWEEPS
        for member in sweep_order:               # the flux producers
            inputs = the member's boundary inputs, read from the iterate
                     and from flux_s; in sweep 1 an input whose flux is
                     not in flux_s yet is left out
            flux_s[member] = member.compute_boundary_fluxes(state, inputs)

Write ``u`` for the iterate.  *The fluxes of* ``u`` are the values ``q``
that satisfy every member's flux equation at once,
``q_i = F_i(u_i, r_i(u, q))``, where ``r_i`` are the inputs the edges
deliver.  The coupled equations are the members' updates with their inputs
taken from ``u`` and from the fluxes of ``u``.  If the seed is the fluxes
of ``u`` for every ``u``, then a fixed point of the pass solves the coupled
equations and the reverse: at a fixed point the in-pass state is ``u``
throughout, the seed is the fluxes of ``u``, the flux a Gauss-Seidel pass
recomputes after each update is computed from the same values and so is
the same flux, and every update therefore read ``u`` and its fluxes.  So
the question is only whether the seed is the fluxes of ``u``.  Call a flux
*right after sweep k* when the value ``flux_s`` holds for it after sweep
``k`` is its value in the fluxes of ``u``.

(a) **A flux computed from the state alone**, and from inputs that edges
    deliver from state fields.  Its equation holds no other flux, so its
    one evaluation in sweep 1 is right, whatever else the group holds.
    This is every flux of the built-in nodes.

(b) **A flux that reads an input another member's flux delivers, with no
    loop.**  Say flux ``g`` of member ``i`` *reads* flux ``f`` of member
    ``j`` when an edge ``j.f -> i.x`` inside the group delivers ``f`` to an
    input ``x`` that ``g`` depends on.  With no loop among the reads, the
    flux equations are solved by substitution along the reads, and the
    solution is unique.  Sweep ``k`` computes ``i``'s fluxes from the
    values ``flux_s`` holds when ``i`` is visited: this sweep's for a
    member visited earlier, the previous sweep's for ``i`` itself and for
    a member visited later.  So ``g`` is right after sweep
    ``S(g) = max(1, max over the fluxes f it reads of S(f) + late(f, g))``,
    where ``late`` is 0 when ``f``'s member is swept before ``g``'s and 1
    otherwise, by induction on the reads; and it stays right in every
    later sweep, which recomputes it from right values.  Two sweeps make
    the seed right exactly when ``S <= 2`` for every flux an edge of the
    group reads: a chain along the sweep order of any length (``S = 1``),
    and one step against it.  (In sweep 1 the member is handed its inputs
    without the one a later member's flux delivers, so its hook must
    return its fluxes without that key, ``boundary_inputs.get(...)``; the
    value it returns then is replaced in sweep 2.)

(c) **Fluxes that read each other in a loop.**  The flux equations are
    then an implicit system in the fluxes.  Two sweeps from nothing are
    two Gauss-Seidel sweeps on that system, started from whatever the
    hooks return without the missing inputs, and not its solution; and
    because the seed does not depend on any earlier flux, no number of
    passes improves it.  The pass has a fixed point all the same, the
    iteration reaches it and reports ``converged=True`` with a residual at
    the float floor, and that state solves the equations with the flux
    loop cut after two sweeps.  On two affine members coupled by
    ``add_robin_coupling`` it differs from the coupled solution by 4e-4 to
    0.8 of itself for loop gains of 0.06 to 0.56.

The criterion
-------------
:func:`_refuse_unsolved_fluxes` refuses a group in case (c), and in case
(b) with ``S > 2`` (a chain that runs against the sweep order for more
than one step), with one ``ValueError``.  Dependence is **structural**: a
flux depends on an input when the traced program of the hook has a path
from the input to the flux (:func:`_flux_reads`), whatever its numerical
size, so a flux that multiplies its input by zero reads it; and where the
hook cannot be traced, every flux of the member is taken to read every
input.  Both err towards refusing.  An edge's transform or mapping is not
looked at: an edge that delivers a flux times zero still delivers it.

Hooks are traced only for a group that would be refused if every flux of
a member read every input a flux edge delivers to it.  A group that
passes on its edges alone (no flux edge between two flux producers, or
only chains the two sweeps reach) is not traced at all, and nothing about
its step changes.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax

from maddening.core.coupling._interface_plan import FLUX

#: How many sweeps a pass seeds its members' fluxes in: ``one_pass_gs``,
#: ``one_pass_jacobi`` wherever a flux producer reads a flux, and
#: ``_read_gain`` (``_coupled_block._run_coupled_block_impl``).
SEED_SWEEPS = 2


def _flux_reads(spec, state, node_params, inputs) -> Optional[dict]:
    """Which boundary inputs each flux of one node depends on.

    ``{flux field: set of input names}``, from **one abstract trace** of
    the node's ``compute_boundary_fluxes``, called as the graph calls it
    (``_graph_specs._node_fluxes``) on the node's state, a zero for every
    input ``boundary_input_spec()`` declares, and its parameters.
    ``jax.make_jaxpr`` records the program without running or compiling
    it; ``_param_probes._live_jaxpr_inputs`` then walks it backwards from
    one flux at a time and marks the inputs that can reach it.  That walk
    is conservative in one direction only (an equation it does not know
    keeps every input): an input it leaves out has no path to the flux, an
    input it reports may still be unused.  Only ``make_jaxpr`` and the
    equations' ``invars`` / ``outvars`` / ``params`` are read, which are
    the same on the jax versions the project supports.

    ``None`` when dependence cannot be established this way: the hook
    raises under the trace, an input in *inputs* is not declared (there is
    no value to trace it with), or the hook does not return a dict.  The
    caller then takes every flux to read every input.
    """
    from maddening.core._graph_specs import _node_fluxes  # noqa: PLC0415
    from maddening.core._param_probes import (  # noqa: PLC0415
        _declared_boundary_zeros,
        _live_jaxpr_inputs,
    )

    try:
        boundary = _declared_boundary_zeros(spec.node)
        if any(name not in boundary for name in inputs):
            return None
        args = (state, boundary, node_params)

        def hook(st, bi, p):
            return _node_fluxes(spec, st, bi, spec.timestep, p)

        closed, returned = jax.make_jaxpr(hook, return_shape=True)(*args)
        if not isinstance(returned, dict):
            return None
        in_paths = [path for path, _ in jax.tree_util.tree_flatten_with_path(args)[0]]
        out_paths = [path for path, _ in jax.tree_util.tree_flatten_with_path(returned)[0]]
        jaxpr = closed.jaxpr
        if len(in_paths) != len(jaxpr.invars) or len(out_paths) != len(jaxpr.outvars):
            return None
        reads: dict = {}
        for flux in returned:
            wanted = [getattr(path[0], "key", None) == flux for path in out_paths]
            live = _live_jaxpr_inputs(jaxpr, wanted)
            reads[flux] = {
                path[1].key for path, keep in zip(in_paths, live)
                if keep and len(path) >= 2 and getattr(path[0], "idx", None) == 1
            }
        return reads
    except Exception:  # noqa: BLE001 - not traceable this way: the caller assumes dependence
        return None


def _flux_records(plan) -> list:
    """The edges inside the group whose source is a boundary flux, as declared.

    ``plan.flux_edges()`` without the edges whose source is neither a
    state field nor a field of a node that has a flux hook (a field that
    does not exist: ``validate()`` has that error).
    """
    return [r for r in plan.flux_edges() if r.source_kind == FLUX]


def _flux_arcs(plan, depends: Callable[[str, str, str], bool],
               is_flux: Callable[[str, str], bool] = lambda member, field: True) -> dict:
    """The reads among the fluxes the group's edges carry.

    ``{flux: [(flux it reads, input, edge key), ...]}`` over every flux
    ``(member, field)`` that is the source of an edge inside the group,
    in the order the edges were declared.  Flux ``(i, g)`` reads
    ``(j, f)`` through an edge ``j.f -> i.x`` inside the group when
    ``depends(i, g, x)``.  A flux no edge of the group reads from is not a
    key: no pass reads it, so it is neither part of a loop nor of a chain.

    The plan calls every source that is not a state field of a node with
    a flux hook a flux.  *is_flux* takes out the ones a traced hook does
    not return: such an edge reads a field that does not exist, and
    ``validate()`` has that error for it.
    """
    flux_edges = [r for r in _flux_records(plan) if is_flux(*r.source)]
    fluxes = list(dict.fromkeys(r.source for r in flux_edges))
    arcs: dict = {flux: [] for flux in fluxes}
    for member, field in fluxes:
        for r in flux_edges:
            if r.target[0] == member and depends(member, field, r.target[1]):
                arcs[(member, field)].append((r.source, r.target[1], r.key))
    return arcs


def _a_loop(arcs: dict) -> Optional[list]:
    """A loop among the reads, or ``None``.

    ``[(flux, flux it reads, input, edge key), ...]``, each entry's second
    flux being the next entry's first and the last entry's the first
    entry's.  A flux that reads itself is a loop of one.  The first loop a
    depth-first walk in the order of *arcs* closes.
    """
    done: set = set()
    for start in arcs:
        if start in done:
            continue
        path: list = []
        on_path: dict = {start: 0}
        stack = [(start, iter(arcs[start]))]
        while stack:
            flux, reads = stack[-1]
            step = next(reads, None)
            if step is None:
                stack.pop()
                done.add(flux)
                del on_path[flux]
                if path:
                    path.pop()
                continue
            read, name, key = step
            if read in on_path:
                return path[on_path[read]:] + [(flux, read, name, key)]
            if read in done:
                continue
            path.append((flux, read, name, key))
            on_path[read] = len(path)
            stack.append((read, iter(arcs[read])))
    return None


def _sweeps_needed(arcs: dict, order) -> tuple[int, list]:
    """How many seeding sweeps make every flux of *arcs* right, and the
    chain that needs them.

    *arcs* holds no loop.  ``S`` of the module docstring: a flux is right
    after ``max(1, S(f) + late)`` sweeps over the fluxes ``f`` it reads,
    ``late`` being 1 unless ``f``'s member is swept before its own in
    *order*.  Returns the largest ``S`` and that flux's chain,
    ``[(flux, flux it reads, input, edge key), ...]`` from the flux that
    is right last down to one that reads no flux (empty when no flux
    reads another).
    """
    place = {member: i for i, member in enumerate(order)}
    needs: dict = {}
    through: dict = {}

    def visit(flux):
        if flux in needs:
            return needs[flux]
        count, via = 1, None
        for read, name, key in arcs[flux]:
            late = 0 if place[read[0]] < place[flux[0]] else 1
            if visit(read) + late > count:
                count, via = needs[read] + late, (flux, read, name, key)
        needs[flux], through[flux] = count, via
        return count

    worst, most = None, 0
    for flux in arcs:
        if visit(flux) > most:
            worst, most = flux, needs[flux]
    chain = []
    while worst is not None and through[worst] is not None:
        chain.append(through[worst])
        worst = through[worst][1]
    return most, chain


def _describe(steps: list) -> str:
    """The reads of a loop or a chain, one clause each."""
    return "; ".join(
        f"flux {flux[0]}.{flux[1]} reads input {name!r}, which edge {key!r} "
        f"delivers from flux {read[0]}.{read[1]}"
        for flux, read, name, key in steps)


def _refuse_unsolved_fluxes(group, plan, nodes, state,
                            params_of: Callable[[str], Any],
                            sweep_order: Callable[[], Any]) -> None:
    """Raise ``ValueError`` if a pass of *group* would not compute the
    fluxes of its own iterate (cases (c) and (b) with ``S > 2`` of the
    module docstring).

    Parameters
    ----------
    group : CouplingGroup
    plan : InterfacePlan
        The group's edges, described in *state*.
    nodes : dict
        ``{name: node spec}``.
    state : dict
        ``{name: {field: value}}``: what the hooks are traced on, where
        they are traced at all.
    params_of : callable
        ``params_of(name)``: the parameters the step hands that node's
        hooks.  Called only for a member that is traced.
    sweep_order : callable
        Returns the members in the order a pass sweeps them.  Called only
        when a flux edge of the group delivers to a flux producer.

    The edges are asked first, with every flux of a member taken to read
    every input a flux edge delivers to it.  A group that passes that way
    returns here, untraced.  Otherwise the members such edges deliver to
    are traced, once each (:func:`_flux_reads`), and the group is asked
    again with what their fluxes read.
    """
    names = sorted(group.nodes)
    if not all(n in nodes and n in state for n in names):
        return                      # validate() has the error for it
    arcs = _flux_arcs(plan, lambda member, flux, name: True)
    if not any(arcs.values()):
        return
    order = list(sweep_order())
    if _a_loop(arcs) is None and _sweeps_needed(arcs, order)[0] <= SEED_SWEEPS:
        return

    fed: dict = {}
    for r in _flux_records(plan):
        if r.target[0] in plan.flux_members:
            fed.setdefault(r.target[0], set()).add(r.target[1])
    traced = {
        member: _flux_reads(nodes[member], state[member], params_of(member),
                            sorted(inputs))
        for member, inputs in fed.items()
    }

    def depends(member, flux, name):
        reads = traced.get(member)
        # A hook that could not be traced: every flux is taken to read it.
        return reads is None or name in reads.get(flux, ())

    def is_flux(member, field):
        reads = traced.get(member)
        return reads is None or field in reads

    arcs = _flux_arcs(plan, depends, is_flux)
    loop = _a_loop(arcs)
    if loop is None:
        sweeps, chain = _sweeps_needed(arcs, order)
        if sweeps <= SEED_SWEEPS:
            return
        found = (
            f"its boundary fluxes read each other along a chain that runs against "
            f"the order the group sweeps its members in ({order}) for more than one "
            f"step: {_describe(chain)}. Each pass rebuilds the fluxes from nothing in "
            f"{SEED_SWEEPS} sweeps over the members in that order, and this chain needs "
            f"{sweeps}, so flux {chain[0][0][0]}.{chain[0][0][1]} would be computed from "
            f"a flux that had not yet been computed from the current states")
        ways = (
            "Or add the nodes to the graph so that each flux's producer is added "
            "before the node whose flux reads it: a group sweeps the members of a "
            "loop in add_node order, and a chain along that order is computed in "
            "one sweep. ")
        steps = chain
    else:
        found = (
            f"its boundary fluxes depend on each other in a loop: {_describe(loop)}. "
            f"Each pass rebuilds the fluxes from nothing in {SEED_SWEEPS} sweeps over "
            f"the members, which is exact for a flux computed from the state alone "
            f"and is not a solution of fluxes that need each other")
        ways = (
            "Taking the nodes out of the group does not help: outside a group the "
            "flux edge that closes a loop cannot be read at all (MADD-ANO-157). ")
        steps = loop
    assumed = sorted({
        f"{member}.{field}" for (member, field), _read, _name, _key in steps
        if traced.get(member) is None})
    if assumed:
        found += (
            f" (what {assumed} read could not be established by tracing "
            f"compute_boundary_fluxes on the inputs boundary_input_spec() declares, "
            f"so each is taken to read every input: that dependence is assumed)")
    producer = steps[0][0]
    raise ValueError(
        f"coupling group {names} is not compiled: {found}. A boundary flux is "
        "returned by compute_boundary_fluxes and is not a state field, so it is not "
        "part of what the group iterates and the convergence norm does not see it. "
        "With such fluxes the iteration can stop with converged=True, and a residual "
        "at the float floor, at a state that does not solve the coupled equations; "
        "how far off depends on how strongly the fluxes act on each other, and "
        "nothing in the diagnostics shows it. One way to get here: add_robin_coupling "
        "between two nodes whose flux depends on the Robin datum they are handed. "
        "To couple such nodes, make the quantity part of the iterate: have each node "
        "compute it in update() and write it to a state field (for example "
        f"{producer[0]}'s state[{producer[1]!r}]), and name that state field as the "
        "edge's source_field in place of the flux; the group then iterates it and "
        f"measures its convergence. {ways}"
        "A flux that is computed from the node's state alone is never refused. "
        "Before 0.4.0 such a group compiled, and returned that state."
    )


def _unseeded_input_error(group, plan, member, handed, exc) -> Optional[ValueError]:
    """What a seeding sweep raises in place of *exc*, the ``KeyError`` of
    *member*'s ``compute_boundary_fluxes``, or ``None`` when *exc* is not
    about an input the sweep left out.

    The first seeding sweep hands a member its inputs (*handed*) without
    the ones a flux not yet computed delivers (module docstring, (b)); the
    second hands them all.  A hook that indexes such an input cannot be
    called in the first, whether or not its fluxes use it.
    """
    name = exc.args[0] if exc.args else None
    if not isinstance(name, str) or name in handed:
        return None
    late = [r for r in _flux_records(plan) if r.target == (member, name)]
    if not late:
        return None
    edge = late[0]
    return ValueError(
        f"coupling group {sorted(group.nodes)}: compute_boundary_fluxes of {member!r} "
        f"indexes boundary_inputs[{name!r}], which edge {edge.key!r} delivers from the "
        f"boundary flux {edge.source[0]}.{edge.source[1]}. Each pass rebuilds the "
        f"group's fluxes from nothing in {SEED_SWEEPS} sweeps over its members; the "
        f"first reaches {member!r} before that flux has been computed and hands it "
        f"its inputs without {name!r}, and the second hands it every input and "
        f"computes its fluxes again. Read the input with boundary_inputs.get({name!r}, "
        "<a value of its shape and dtype>) in compute_boundary_fluxes: what the hook "
        "returns from the default is replaced in the second sweep and reaches no "
        f"result. Or carry the quantity in a state field of {edge.source[0]!r} and "
        "name that field as the edge's source_field."
    )
