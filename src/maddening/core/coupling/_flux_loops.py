"""Boundary fluxes that read each other inside a coupling group: what a
coupling pass computes for them, and the groups that are refused because
a pass would read a flux that is not the flux of its own iterate.

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
anything, the *seed* (``one_pass_gs`` and ``one_pass_jacobi`` in
``_coupled_block._run_coupled_block_impl``)::

    flux_s = {}                                  # nothing is known
    for sweep in (1, 2):                         # SEED_SWEEPS
        for member in sweep_order:               # the flux producers
            inputs = the member's boundary inputs, read from the iterate
                     and from flux_s; in sweep 1 an input whose flux is
                     not in flux_s yet is left out
            flux_s[member] = member.compute_boundary_fluxes(state, inputs)

A Jacobi pass then updates every member from the iterate and the seed.  A
Gauss-Seidel pass updates the members in sweep order, each from the states
and fluxes as they stand, and recomputes a member's fluxes right after its
update, so a member swept later reads that value and not the seed.

Write ``u`` for the iterate.  *The fluxes of* ``u`` are the values ``q``
that satisfy every member's flux equation at once,
``q_i = F_i(u_i, r_i(u, q))``, where ``r_i`` are the inputs the edges
deliver.  The coupled equations are the members' updates with their inputs
taken from ``u`` and from the fluxes of ``u``.  Call a flux value *right*
when it is that flux in the fluxes of ``u``.  If every flux value a pass
reads at a fixed point ``u`` is right, the fixed point solves the coupled
equations, and a solution of them is a fixed point: at a fixed point the
in-pass state is ``u`` throughout, so every update read ``u`` and its
fluxes.  So the question is which values a pass reads, and whether each
is right.

Say flux ``g`` of member ``i`` *reads* flux ``f`` of member ``j`` when an
edge ``j.f -> i.x`` inside the group delivers ``f`` to an input ``x``
that ``g`` depends on.

(a) **A flux that reads no flux**: computed from the state, and from
    inputs that edges deliver from state fields.  Its equation holds no
    other flux, so its one evaluation in sweep 1 is right, whatever else
    the group holds, and so is every later evaluation.  (A built-in
    node's flux is of this kind as the coupling helpers wire it:
    ``HeatNode``'s end fluxes read its end temperatures and
    ``SpringDamperNode``'s force its anchor position, inputs that a
    state field delivers.  Fed by another member's flux, such an input
    makes the flux one of (b) or (c).)

(b) **Fluxes that read each other, with no loop among the reads.**  The
    flux equations are then solved by substitution along the reads, and
    the solution is unique.  Seeding sweep ``k`` computes ``i``'s fluxes
    from the values ``flux_s`` holds when ``i`` is visited: this sweep's
    for a member visited earlier, the previous sweep's for ``i`` itself
    and for a member visited later.  So, by induction on the reads, the
    seed of ``g`` is right after sweep
    ``S(g) = max(1, max over the fluxes f it reads of S(f) + late(f, g))``,
    with ``late`` 0 when ``f``'s member is swept before ``g``'s and 1
    otherwise, and it stays right in every later sweep, which recomputes
    it from right values.  **The seed of a flux is right exactly when**
    ``S <= 2``: a chain along the sweep order of any length (``S = 1``),
    and one step against it.

    Which values a pass reads:

    * *Jacobi*: only seeds.  Every flux an edge of the group carries must
      have ``S <= 2``.
    * *Gauss-Seidel*: an edge whose source is swept **before** its target
      delivers the flux the pass recomputed after the source's update;
      every other edge (source swept after its target, or the same
      member) delivers the seed.  A recomputed flux is right when the
      fluxes it reads are, and it reads them through edges under the
      same rule, so by induction along the sweep it is enough that every
      flux an edge carries **against or within** the sweep order
      (source swept at or after its target) has ``S <= 2``.

(c) **Fluxes that read each other in a loop.**  The flux equations are
    then an implicit system in the fluxes.  Two sweeps from nothing are
    two Gauss-Seidel sweeps on that system, started from whatever the
    hooks return without the missing inputs, and not its solution
    (``S`` has no value); and because the seed does not depend on any
    earlier flux, no number of passes improves it.  Going round the loop,
    at least one edge has its source swept at or after its target, so
    both schedules read such a seed.  The pass has a fixed point all the
    same, the iteration reaches it and reports ``converged=True`` with a
    residual at the float floor, and that state solves the equations with
    the flux loop cut after two sweeps.  On two affine members coupled by
    ``add_robin_coupling`` its states differ from their coupled values by
    4e-4 to 0.8 of those values for loop gains of 0.06 to 0.56.

The criterion
-------------
:func:`_refuse_unsolved_fluxes` refuses, with one ``ValueError``, a group
in case (c), and in case (b) a group whose pass reads the seed of a flux
with ``S > 2`` (a chain that runs against the sweep order for more than
one step).  Sufficient for a right fixed point under both schedules; it
errs towards refusing in the ways stated here.

*What a hook is traced on.*  The inputs the pass hands it: one for each
edge into the node and each external input, **declared in**
``boundary_input_spec()`` **or not**, each with the shape and dtype it
will have (:func:`_handed`): what the edge's own rule delivers, evaluated
abstractly on the state field it reads or on the abstract flux of the
producer it reads (:func:`_abstract_fluxes`), and the declared shape only
for an input no edge's value could be derived for.  Not every input the
node declares: a hook may branch on what it is handed.

*Dependence is structural.*  A flux depends on an input when the traced
program of the hook has a path from the input to the flux
(:func:`_flux_reads`), whatever its numerical size: a flux that
multiplies its input by zero reads it.  An edge's transform or mapping is
not looked at (an edge that delivers a flux times zero still delivers
it), and every edge is taken to be read by its target.

*A hook may branch on which inputs it is handed*
(``if "x" in boundary_inputs``), and sweep 1 hands it fewer than sweep 2.
So the hook is traced again as sweep 1 calls it, and a flux whose two
programs are not shown to be the same is taken to read every input sweep
1 left out.

*Dependence is assumed, and the message says so with the reason,* only
where it cannot be established: the hook raises under the trace on the
inputs the pass hands it (a hook that computes on values, such as
``float(state["u"])``; that is not a JAX-traceable function, which the
node contract asks for), it does not return a dict, or the shape of an
input cannot be derived at all (every edge into it carries the flux of a
hook that cannot be evaluated, and the node does not declare it).  Every
flux of that member is then taken to read every input a flux edge
delivers to it.

Hooks are traced only for a group that would be refused if every flux of
a member read every input a flux edge delivers to it.  A group that
passes on its edges alone (no flux edge between two flux producers, or
only chains the seed reaches) is not traced at all, and nothing about any
group's step changes: the passes are what they were.

What the criterion cannot see
-----------------------------
It is asked when the graph is compiled or validated, of its own trace of
each hook with the hook's arguments as the only inputs.  Two things are
outside that, both outside the node contract (a hook is a pure function
of its arguments, and a structural switch is fixed at compile), and both
wrong in silence:

* a dependence that does not pass through the hook's arguments: an
  ``update`` that stashes its boundary input on the node object for
  ``compute_boundary_fluxes`` to read;
* a hook that changes after the question was asked: a Python switch on
  ``node.params`` (or any attribute) flipped after ``compile()``.
  ``step()`` keeps the program it traced, and an entry point traced
  later (``run_scan``) runs the new hook, with a loop that was never
  asked about.

(A seeding sweep hands a member its inputs without the ones a flux not
yet computed delivers, so a hook must return its fluxes without such a
key, ``boundary_inputs.get(...)``; what it returns then is replaced in
sweep 2.  ``validate()`` asks every flux hook for its flux names with no
inputs at all, and refuses by name one that indexes an input.)
"""

from __future__ import annotations

import enum
import operator
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.coupling._interface_plan import FLUX, INBOUND

#: How many sweeps a pass seeds its members' fluxes in: ``one_pass_gs``,
#: ``one_pass_jacobi`` wherever a flux producer reads a flux, and
#: ``_read_gain`` (``_coupled_block._run_coupled_block_impl``).
SEED_SWEEPS = 2


def _plain(value) -> Any:
    """A hashable stand-in for an equation's parameter or a constant:
    equal for equal values, and for a value of a kind not listed here
    (a sub-program, a function) equal to nothing, itself included."""
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return (type(value).__name__, value)
    if isinstance(value, enum.Enum):
        return ("enum", type(value).__name__, value.name)
    if isinstance(value, (np.dtype, type)):
        return ("type", str(value))
    if isinstance(value, (np.ndarray, np.generic, jax.Array)):
        array = np.asarray(value)
        return ("array", str(array.dtype), array.shape, array.tobytes())
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_plain(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted(((str(k), _plain(v)) for k, v in value.items()),
                                     key=lambda item: item[0])))
    return ("unknown", object())


def _programs(closed, in_paths, out_paths, table: dict) -> dict:
    """``{flux: its program}`` for one trace of a flux hook.

    A flux's program is a number for each of its leaves that stands for
    the whole expression the leaf is computed by: the primitive of each
    equation with its parameters, applied to the programs of its
    operands, down to the hook's arguments (by their place in the
    arguments), constants and literals (by value).  Numbers are drawn
    from *table*, shared between the traces compared, so two leaves have
    the same number exactly when they are the same expression of the
    same arguments.  An equation whose parameters hold something
    :func:`_plain` does not know (control flow, a jitted call) is the
    same as no other, so its flux is never shown to be the same.
    """
    from jax.extend.core import Literal  # noqa: PLC0415

    def number(key):
        return table.setdefault(key, len(table))

    jaxpr = closed.jaxpr
    ids = {v: number(("argument", jax.tree_util.keystr(path)))
           for v, path in zip(jaxpr.invars, in_paths)}
    for v, const in zip(jaxpr.constvars, closed.consts):
        ids[v] = number(("constant", _plain(const)))

    def of(v):
        return number(("literal", _plain(v.val))) if isinstance(v, Literal) else ids[v]

    for eqn in jaxpr.eqns:
        key = (eqn.primitive.name, _plain(dict(eqn.params)), tuple(of(v) for v in eqn.invars))
        for k, v in enumerate(eqn.outvars):
            ids[v] = number(("equation", key, k))
    programs: dict = {}
    for v, path in zip(jaxpr.outvars, out_paths):
        programs.setdefault(path[0].key, []).append((jax.tree_util.keystr(path[1:]), of(v)))
    return {flux: tuple(leaves) for flux, leaves in programs.items()}


def _zeros(value) -> Any:
    """A zero of each leaf's shape and dtype: a value to trace on."""
    return jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), value)


def _handed(incoming: dict, given: dict, state, fluxes: dict, declared: dict,
            mappings) -> tuple[dict, set]:
    """The shapes and dtypes of the inputs a pass hands one member's hook.

    ``({input: abstract value}, the inputs no value could be derived
    for)``.  *incoming* is ``{input: the plan's records of the edges into
    it}``, in the order the edges were declared, and *given* the
    external inputs, ``{input: abstract value}``.  Each edge's value is
    what its own rule delivers (``InterfaceEdge.delivered_abstract``:
    the mapping, then the transform, under :func:`jax.eval_shape`) for
    the state field it reads, or for its producer's flux in *fluxes*
    (``{member: {flux: abstract value}}``); the edges into one input are
    combined as the pass combines them (an additive edge adds, any other
    replaces).  An edge whose source is not known yet is left out, as the
    first seeding sweep leaves it out.  An input no edge gave a value is
    handed at the shape and dtype ``boundary_input_spec()`` declares
    (*declared*), where it declares one.
    """
    values: dict = dict(given)
    unknown: set = set()
    for name, records in incoming.items():
        if name in given:
            continue
        value = None
        for r in records:
            held = fluxes if r.source_kind == FLUX else state
            source = (held.get(r.source[0]) or {}).get(r.source[1])
            delivered = None if source is None else r.delivered_abstract(
                source, state, mappings)
            if delivered is None:
                continue
            value = (jax.eval_shape(operator.add, value, delivered)
                     if value is not None and r.additive else delivered)
        if value is None and name in declared:
            value = jax.eval_shape(lambda v: v, declared[name])
        if value is None:
            unknown.add(name)
        else:
            values[name] = value
    return values, unknown


def _abstract_fluxes(members, handed_to: Callable, nodes, state,
                     params_of: Callable[[str], Any]) -> dict:
    """``{member: {flux: shapes and dtypes}}`` of the members' fluxes.

    Each hook is evaluated abstractly (:func:`jax.eval_shape`: nothing is
    run), called as the graph calls it on the inputs
    ``handed_to(member, fluxes so far)`` gives.  A member whose hook
    cannot be evaluated yet (an input it needs is another member's flux)
    is tried again once more fluxes are known, until a round adds none;
    one that never can is left out.
    """
    from maddening.core._graph_specs import _node_fluxes  # noqa: PLC0415

    fluxes: dict = {}
    waiting = list(members)
    while waiting:
        still = []
        for member in waiting:
            spec = nodes[member]

            def hook(st, bi, p, spec=spec):
                return _node_fluxes(spec, st, bi, spec.timestep, p)

            try:
                out = jax.eval_shape(hook, state[member],
                                     _zeros(handed_to(member, fluxes)[0]), params_of(member))
                if not isinstance(out, dict):
                    raise TypeError("compute_boundary_fluxes did not return a dict")
                fluxes[member] = out
            except Exception:  # noqa: BLE001 - not yet, or never: tried again below
                still.append(member)
        if len(still) == len(waiting):
            break
        waiting = still
    return fluxes


def _flux_reads(spec, state, node_params, handed: dict, absent=(),
                why: Optional[list] = None) -> Optional[dict]:
    """Which boundary inputs each flux of one node depends on.

    ``{flux field: set of input names}``, from **abstract traces** of the
    node's ``compute_boundary_fluxes``, called as the graph calls it
    (``_graph_specs._node_fluxes``) on the node's state, its parameters,
    and a zero of the shape and dtype of each input in *handed*
    (``{input: abstract value}``, :func:`_handed`): the inputs a pass
    hands the hook.  ``jax.make_jaxpr`` records the program without
    running or compiling it.

    * ``_param_probes._live_jaxpr_inputs`` walks that program backwards
      from one flux at a time and marks the inputs that can reach it.
      The walk is conservative in one direction only (an equation it does
      not know keeps every input): an input it leaves out has no path to
      the flux, an input it reports may still be unused.
    * *absent* are the inputs the first seeding sweep leaves out.  Where
      there are any, the hook is traced again without them, as sweep 1
      calls it, and once more for each flux that reads some of them by
      value, without those alone.  A flux whose program
      (:func:`_programs`) is the same in those two traces does in sweep 1
      what it would do with only the inputs it reads by value left out:
      it reads no other.  One whose program differs, or whose hook
      cannot be traced without them (it indexes one), is reported to
      read every absent input.

    Only ``make_jaxpr`` and the equations' ``primitive`` / ``invars`` /
    ``outvars`` / ``params`` are read, which are the same on the jax
    versions the project supports.

    ``None`` when dependence cannot be established this way: the hook
    raises under the trace on the inputs it is handed, or does not return
    a dict.  The reason is appended to *why*, and the caller takes every
    flux to read every input.
    """
    from maddening.core._graph_specs import _node_fluxes  # noqa: PLC0415
    from maddening.core._param_probes import _live_jaxpr_inputs  # noqa: PLC0415

    def hook(st, bi, p):
        return _node_fluxes(spec, st, bi, spec.timestep, p)

    def trace(boundary):
        args = (state, boundary, node_params)
        closed, returned = jax.make_jaxpr(hook, return_shape=True)(*args)
        if not isinstance(returned, dict):
            raise TypeError("compute_boundary_fluxes did not return a dict")
        in_paths = [path for path, _ in jax.tree_util.tree_flatten_with_path(args)[0]]
        out_paths = [path for path, _ in jax.tree_util.tree_flatten_with_path(returned)[0]]
        if (len(in_paths) != len(closed.jaxpr.invars)
                or len(out_paths) != len(closed.jaxpr.outvars)):
            raise ValueError("the traced program does not line up with its arguments")
        return closed, returned, in_paths, out_paths

    try:
        zeros = {name: _zeros(value) for name, value in handed.items()}
        closed, returned, in_paths, out_paths = trace(zeros)
        reads: dict = {}
        for flux in returned:
            wanted = [getattr(path[0], "key", None) == flux for path in out_paths]
            live = _live_jaxpr_inputs(closed.jaxpr, wanted)
            reads[flux] = {
                path[1].key for path, keep in zip(in_paths, live)
                if keep and len(path) >= 2 and getattr(path[0], "idx", None) == 1
            }
    except Exception as exc:  # noqa: BLE001 - not traceable: the caller assumes dependence
        if why is not None:
            why.append(f"{type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}")
        return None
    if absent:
        table: dict = {}
        traces = {frozenset(): _programs(closed, in_paths, out_paths, table)}

        def programs_without(missing):
            key = frozenset(missing)
            if key not in traces:
                other = trace({n: zero for n, zero in zeros.items() if n not in key})
                if set(other[1]) != set(returned):
                    raise ValueError("other fluxes without these inputs")
                traces[key] = _programs(other[0], other[2], other[3], table)
            return traces[key]

        for flux in reads:
            by_value = reads[flux] & set(absent)
            try:
                same = (programs_without(by_value).get(flux)
                        == programs_without(absent).get(flux))
            except Exception:  # noqa: BLE001 - a hook that indexes an absent input
                same = False
            if not same:
                reads[flux] |= set(absent)
    return reads


def _flux_records(plan, is_flux: Callable[[str, str], bool] = lambda member, field: True) -> list:
    """The edges inside the group whose source is a boundary flux, as declared.

    ``plan.flux_edges()`` without the edges whose source is neither a
    state field nor a field of a node that has a flux hook, and without
    those *is_flux* takes out (a field a traced hook does not return).
    Both read a field that does not exist; ``validate()`` has that error.
    """
    return [r for r in plan.flux_edges() if r.source_kind == FLUX and is_flux(*r.source)]


def _flux_arcs(records: list, depends: Callable[[str, str, str], bool]) -> dict:
    """The reads among the fluxes the group's edges carry.

    ``{flux: [(flux it reads, input, edge key), ...]}`` over every flux
    ``(member, field)`` that is the source of one of *records*, in their
    order.  Flux ``(i, g)`` reads ``(j, f)`` through an edge
    ``j.f -> i.x`` when ``depends(i, g, x)``.  A flux no edge of the
    group reads from is not a key: no pass reads it, so it is neither
    part of a loop nor of a chain.
    """
    fluxes = list(dict.fromkeys(r.source for r in records))
    arcs: dict = {flux: [] for flux in fluxes}
    for member, field in fluxes:
        for r in records:
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


def _read_from_the_seed(records: list, order, jacobi: bool) -> list:
    """The fluxes whose **seed** a pass reads, in the order of *records*.

    Under Jacobi every flux an edge carries.  Under Gauss-Seidel the
    fluxes an edge carries to a member swept at or before the flux's own
    (an edge along the sweep delivers the flux the pass recomputed after
    its source's update): (b) of the module docstring.
    """
    place = {member: i for i, member in enumerate(order)}
    return list(dict.fromkeys(
        r.source for r in records
        if jacobi or place[r.source[0]] >= place[r.target[0]]))


def _sweeps_needed(arcs: dict, order, among) -> tuple[int, list]:
    """How many seeding sweeps make the seed of every flux in *among*
    right, and the chain that needs them.

    *arcs* holds no loop.  ``S`` of the module docstring: the seed of a
    flux is right after ``max(1, S(f) + late)`` sweeps over the fluxes
    ``f`` it reads, ``late`` being 1 unless ``f``'s member is swept
    before its own in *order*.  Returns the largest ``S`` among *among*
    and that flux's chain, ``[(flux, flux it reads, input, edge key),
    ...]`` from the flux whose seed is right last down to one that reads
    no flux (empty when it reads none).
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
    for flux in among:
        if visit(flux) > most:
            worst, most = flux, needs[flux]
    chain = []
    while worst is not None and through[worst] is not None:
        chain.append(through[worst])
        worst = through[worst][1]
    return most, chain


def _an_order_the_seed_reaches(arcs: dict, records: list, order, jacobi: bool) -> Optional[list]:
    """An order of the members in which the group would not be refused,
    or ``None`` when this finds none.

    The members sorted so that each flux's producer comes before the
    members whose fluxes read it, otherwise as in *order*; there is no
    such order when the fluxes of two members read each other both ways
    (a chain that bounces between them).  Kept only if the criterion
    passes in it, so an order the message offers is one that compiles.
    """
    after: dict = {member: set() for member in order}
    for flux, reads in arcs.items():
        after[flux[0]].update(read[0] for read, _name, _key in reads if read[0] != flux[0])
    found: list = []
    left = list(order)
    while left:
        ready = next((m for m in left if after[m] <= set(found)), None)
        if ready is None:
            return None
        found.append(ready)
        left.remove(ready)
    reached = _sweeps_needed(arcs, found, _read_from_the_seed(records, found, jacobi))[0]
    return found if reached <= SEED_SWEEPS else None


def _describe(steps: list) -> str:
    """The reads of a loop or a chain, one clause each."""
    return "; ".join(
        f"flux {flux[0]}.{flux[1]} reads input {name!r}, which edge {key!r} "
        f"delivers from flux {read[0]}.{read[1]}"
        for flux, read, name, key in steps)


def _refuse_unsolved_fluxes(group, plan, nodes, state,
                            params_of: Callable[[str], Any],
                            sweep_order: Callable[[], Any],
                            external=(), mappings=None) -> None:
    """Raise ``ValueError`` if a pass of *group* would read a boundary
    flux that is not the flux of its own iterate (the criterion of the
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
    external : iterable of (node, field, abstract value)
        The declared external inputs into the members
        (``InterfacePlan.external_inputs``): inputs a pass hands a hook
        beside what the edges deliver.
    mappings : dict, optional
        The ``"mappings"`` section of the graph's parameters: the weights
        the edges' mappings deliver with.

    The edges are asked first, with every flux of a member taken to read
    every input a flux edge delivers to it.  A group that passes that way
    returns here, untraced.  Otherwise the members such edges deliver to
    are traced (:func:`_flux_reads`) on the inputs the pass hands them
    (:func:`_handed`), and the group is asked again with what their
    fluxes read.
    """
    names = sorted(group.nodes)
    if not all(n in nodes and n in state for n in names):
        return                      # validate() has the error for it
    jacobi = group.iteration_mode == "jacobi"
    records = _flux_records(plan)
    arcs = _flux_arcs(records, lambda member, flux, name: True)
    if not any(arcs.values()):
        return
    order = list(sweep_order())
    if _a_loop(arcs) is None and _sweeps_needed(
            arcs, order, _read_from_the_seed(records, order, jacobi))[0] <= SEED_SWEEPS:
        return

    from maddening.core._param_probes import _declared_boundary_zeros  # noqa: PLC0415

    # The producers a flux edge of the group delivers to (the ones
    # traced), and with them the producers of those fluxes: each with the
    # edges into each of its inputs (one input of the hook per edge target
    # and per external input, as ``_resolve_boundary`` builds them).
    place = {member: i for i, member in enumerate(order)}
    fed = list(dict.fromkeys(
        r.target[0] for r in records if r.target[0] in plan.flux_members))
    involved = [m for m in order
                if m in fed or any(r.source[0] == m for r in records)]
    incoming: dict = {member: {} for member in involved}
    for r in (*plan.internal, *(c for c in plan.crossing if c.role == INBOUND)):
        member, name = r.target
        if member in incoming:
            incoming[member].setdefault(name, []).append(r)
    given: dict = {member: {} for member in involved}
    for member, name, value in external:
        if member in given:
            given[member][name] = value
    declared: dict = {}
    params: dict = {}
    for member in involved:
        params[member] = params_of(member)
        try:
            declared[member] = _declared_boundary_zeros(nodes[member].node)
        except Exception:  # noqa: BLE001 - a spec that gives no shape declares nothing here
            declared[member] = {}

    def handed_to(member, fluxes):
        return _handed(incoming[member], given[member], state, fluxes,
                       declared[member], mappings)

    fluxes = _abstract_fluxes(involved, handed_to, nodes, state, params.__getitem__)
    traced: dict = {}
    reason: dict = {}
    for member in fed:
        values, unknown = handed_to(member, fluxes)
        if unknown:
            traced[member] = None
            reason[member] = (
                f"the shape of its input {sorted(unknown)} could not be derived (every edge "
                f"into it carries the flux of a hook that could not be evaluated, and "
                f"boundary_input_spec() does not declare it)")
            continue
        # Sweep 1 leaves an input out when every edge into it carries a
        # flux of a member swept at or after this one (or of a node
        # outside the group, MADD-ANO-156).
        absent = sorted(
            name for name, into in incoming[member].items()
            if name not in given[member] and all(
                r.source_kind == FLUX
                and place.get(r.source[0], len(place)) >= place[member] for r in into))
        why: list = []
        traced[member] = _flux_reads(
            nodes[member], state[member], params[member], values, absent, why)
        if traced[member] is None:
            reason[member] = (
                "its compute_boundary_fluxes could not be traced on the inputs the pass "
                f"hands it ({why[0] if why else 'no reason recorded'})")

    def depends(member, flux, name):
        reads = traced.get(member)
        # A hook that could not be traced: every flux is taken to read it.
        return reads is None or name in reads.get(flux, ())

    def is_flux(member, field):
        reads = traced.get(member)
        return reads is None or field in reads

    records = _flux_records(plan, is_flux)
    arcs = _flux_arcs(records, depends)
    loop = _a_loop(arcs)
    if loop is None:
        sweeps, chain = _sweeps_needed(
            arcs, order, _read_from_the_seed(records, order, jacobi))
        if sweeps <= SEED_SWEEPS:
            return
        last = f"{chain[0][0][0]}.{chain[0][0][1]}"
        found = (
            f"its boundary fluxes read each other along a chain that runs against "
            f"the order the group sweeps its members in ({order}) for more than one "
            f"step: {_describe(chain)}. Each pass rebuilds the fluxes from nothing in "
            f"{SEED_SWEEPS} sweeps over the members in that order, flux {last} would "
            f"take {sweeps}, and the pass reads it as those {SEED_SWEEPS} sweeps "
            f"leave it")
        better = _an_order_the_seed_reaches(arcs, records, order, jacobi)
        ways = "" if better is None else (
            f"Or add the nodes to the graph in the order of the reads, {better}: a "
            "group sweeps the members of a loop in add_node order, and in that order "
            "the two sweeps reach every flux the pass reads. ")
        steps = chain
    else:
        found = (
            f"its boundary fluxes depend on each other in a loop: {_describe(loop)}. "
            f"Each pass rebuilds the fluxes from nothing in {SEED_SWEEPS} sweeps over "
            f"the members, which is exact for a flux that reads no other flux and is "
            f"not a solution of fluxes that need each other")
        ways = (
            "Taking the nodes out of the group does not help: outside a group the "
            "flux edge that closes a loop cannot be read at all (MADD-ANO-157). ")
        steps = loop
    assumed = list(dict.fromkeys(
        member for (member, _field), _read, _name, _key in steps
        if traced.get(member) is None))
    if assumed:
        found += (
            ". What the fluxes of " + ", ".join(repr(m) for m in assumed) + " read is "
            "assumed, not established: " + "; ".join(
                f"for {m!r}, {reason[m]}" for m in assumed)
            + ". Each of their fluxes is therefore taken to read every input a flux "
            "edge delivers to its node")
        closing = (
            "If these fluxes in fact read no other flux, the group compiles once that "
            "can be established: write compute_boundary_fluxes as JAX operations on "
            "its arguments (the step itself needs that), and declare an input whose "
            "shape could not be derived in boundary_input_spec().")
    else:
        closing = (
            "A group whose fluxes are each computed without reading, or branching on, "
            "an input another flux delivers is never refused.")
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
        f"measures its convergence. {ways}{closing}"
    )
