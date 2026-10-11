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

*What a hook is traced on.*  The inputs the pass hands it, composed as
the pass composes them (:func:`_handed`, after
``_coupled_block._run_coupled_block_impl._resolve_boundary``): every
edge into the node, from a member or from outside the group, in the
order of the ``add_edge`` calls (``InterfacePlan.edges_into``), an
additive edge adding its value to what an earlier edge into the same
input delivered and any other replacing it (whatever the replaced edge
delivered, or failed to), then the external inputs;
each edge's value what its own rule delivers, evaluated abstractly on
the state field it reads or on the abstract flux of the producer it
reads; inputs **declared in** ``boundary_input_spec()`` **or not**.
Not every input the node declares: a hook may branch on what it is
handed.  The declared shape stands in only for an input into which an
edge's value cannot be derived, and the message says so where it
matters.

*The fluxes' shapes are those the sweeps settle at.*  A flux's shape or
dtype may follow an input another flux delivers, so the abstract fluxes
are computed by the pass's own sweeps over **every flux producer of the
group** (:func:`_abstract_fluxes`: the first without the edges whose
flux is not computed yet, so that an input holds what the edges before
such an edge delivered, a value a later edge replaces included; each
later one from the fluxes as they stand when it visits a member),
repeated until a sweep changes no flux's shape or dtype.  The hooks are
traced on what the last sweep hands them.

*Dependence is structural.*  A flux depends on an input when the traced
program of the hook has a path from the input to the flux
(:func:`_flux_reads`), whatever its numerical size: a flux that
multiplies its input by zero reads it.  An edge's transform or mapping is
not looked at (an edge that delivers a flux times zero still delivers
it), and every edge whose value the pass hands on is taken to be read
by its target (:func:`_handed_on`: an edge followed into the same input
by one that is not additive is replaced, and read by nothing from the
second seeding sweep on; what the first sweep hands through it is
counted by the next paragraph).

*A hook may branch on what it is handed*
(``if "x" in boundary_inputs``, or on an input's shape), and a seeding
sweep can hand it less than the settled inputs: sweep 1 leaves out the
inputs whose flux is not computed yet, and where a flux's shape takes
more than one sweep to settle a sweep hands an input at another shape.
So the hook is traced again as each seeding sweep calls it, and a flux
whose programs are not shown to be the same is taken to read the inputs
that sweep hands otherwise whose difference changes its program (each
tried alone, and all of them where only some together change it).  (Its value in that sweep is then not
its settled one, and the reads make its ``S`` count that sweep: a flux
whose sweep-``k`` shape is not its settled one has ``S > k``.)  Two
programs are the same when their equations are, a sub-program (a
``jax.jit`` helper, the branches of a ``cond``, a loop's body) by its
structure and a custom derivative's call by its value, not by its rule;
where they differ in an equation whose parameter is a function or an
object the comparison cannot look into, the read is still counted, and
the message says it is assumed.

*Dependence is assumed, and the message says so with the reason,* only
where it cannot be established: the hook raises under the trace on the
inputs the pass hands it (a hook that computes on values, such as
``float(state["u"])``; that is not a JAX-traceable function, which the
node contract asks for; reported as such even where an input cannot be
derived either), it does not return a dict, the shape of an
input cannot be derived at all (the edge that decides its value, or an
additive one after it, carries the flux of a member whose fluxes cannot
be evaluated, a flux the producer does not return or a flux from outside
the group, or its rule fails, and the node does not declare it: the
message names the edge), or the fluxes' shapes have not settled after
``SEED_SWEEPS`` sweeps and one more per producer (a shape that grows
round a loop never does).  Every flux of that member is then taken to
read every input a flux edge delivers to it.  A member with an input
that cannot be derived is not evaluated in the sweeps either, so its
fluxes are not derived, and the inputs they deliver cannot be: nothing
downstream is traced on a shape the pass would not hand it.

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

Within the contract, the question is asked of one configuration, and
these are its edges:

* *the timestep*: hooks are traced with the node's own ``timestep``.
  ``run_adaptive`` hands the step a ``dt`` chosen at run time; a hook
  whose structure (not only its value) follows ``dt`` is asked about at
  that one value.
* *geometry-anchored mappings*: an edge's value is derived from the
  shape and dtype of its geometry field in the state at compile
  (``InterfaceEdge.delivered_abstract``); a geometry field that is a
  flux, or that the state does not hold, makes the input underivable,
  and the group is refused saying so.  This derivation has not been
  compared, edge by edge, against what an instrumented pass hands, as
  the edges without a geometry have.
* *flux edges from outside the group*: the step does not read them
  (MADD-ANO-156, ``KeyError`` at the first step); the criterion takes
  the input such an edge decides as underivable, and names the edge.
* *dependence is structural and errs one way*: an equation the walk does
  not know (a custom derivative's call among them) keeps every input,
  so a flux can be taken to read an input it does not use, and a group
  refused that would be right; never the reverse.

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

from maddening.core.coupling._interface_plan import FLUX

#: How many sweeps a pass seeds its members' fluxes in: ``one_pass_gs``,
#: ``one_pass_jacobi`` wherever a flux producer reads a flux, and
#: ``_read_gain`` (``_coupled_block._run_coupled_block_impl``).
SEED_SWEEPS = 2


class _Opaque:
    """:func:`_plain` of a value of a kind it cannot compare by value (a
    function, an object of a type outside jax that has no equality of its
    own): equal only to the same object, which it holds (so its identity
    is not reused while the comparison lasts).  A program that holds one
    and differs from another is not shown to differ by what it computes:
    a fresh object in each trace makes two equal programs differ."""

    __slots__ = ("held",)

    def __init__(self, held):
        self.held = held

    def __eq__(self, other):
        return isinstance(other, _Opaque) and other.held is self.held

    def __hash__(self):
        return id(self.held)


def _aval_form(v) -> Any:
    """The shape, dtype and weak type of a variable or literal of a program."""
    aval = getattr(v, "aval", None)
    try:
        return (tuple(aval.shape), str(aval.dtype), bool(getattr(aval, "weak_type", False)))
    except Exception:  # noqa: BLE001 - an abstract value with no shape (a token, a reference)
        return ("aval", str(aval))


#: Primitives whose callable parameters are rules for derivatives: the
#: value is the call program's (``call_jaxpr``), whatever the rule.
_CUSTOM_DERIVATIVES = ("custom_jvp_call", "custom_vjp_call")


def _eqn_params(eqn) -> Any:
    """:func:`_plain` of an equation's parameters, without the derivative
    rules of a custom-derivative call (functions, made again in each
    trace, that do not change the value the call computes)."""
    params = dict(eqn.params)
    if eqn.primitive.name.startswith(_CUSTOM_DERIVATIVES):
        params = {k: v for k, v in params.items()
                  if not (callable(v) or hasattr(v, "call_wrapped"))}
    return _plain(params)


def _jaxpr_form(jaxpr, consts=None) -> tuple:
    """A sub-program (the ``jaxpr`` of a jitted call, the branches of a
    ``cond``, the body of a loop) as a value: its inputs by place, shape
    and dtype, each equation's primitive with its parameters applied to
    its operands, its outputs, and its constants by value.  Two traces of
    the same function give equal forms, whatever objects jax made for
    them."""
    from jax.extend.core import Literal  # noqa: PLC0415

    names: dict = {v: ("input", i, _aval_form(v)) for i, v in enumerate(jaxpr.invars)}
    for i, v in enumerate(jaxpr.constvars):
        names[v] = (("constant", _plain(consts[i])) if consts is not None and i < len(consts)
                    else ("constant input", i, _aval_form(v)))

    def of(v):
        return ("literal", _plain(v.val), _aval_form(v)) if isinstance(v, Literal) else names[v]

    equations = []
    for k, eqn in enumerate(jaxpr.eqns):
        equations.append((eqn.primitive.name, _eqn_params(eqn), tuple(of(v) for v in eqn.invars)))
        for j, v in enumerate(eqn.outvars):
            names[v] = ("equation", k, j)
    return ("program", tuple(names[v] for v in jaxpr.invars), tuple(equations),
            tuple(of(v) for v in jaxpr.outvars))


def _plain(value) -> Any:
    """A hashable stand-in for an equation's parameter or a constant:
    equal for equal values.  A sub-program is taken by its structure
    (:func:`_jaxpr_form`); an object of a jax type with an equality and
    a hash of its own (a sharding, a mesh's axes) by that equality; any
    other object is :class:`_Opaque`, equal only to itself."""
    from jax.extend.core import ClosedJaxpr, Jaxpr  # noqa: PLC0415

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
    if isinstance(value, ClosedJaxpr):
        return _jaxpr_form(value.jaxpr, value.consts)
    if isinstance(value, Jaxpr):
        return _jaxpr_form(value)
    kind = type(value)
    if (kind.__module__.split(".")[0] in ("jax", "jaxlib") and kind.__eq__ is not object.__eq__
            and kind.__hash__ is not None and not callable(value)
            and not hasattr(value, "call_wrapped")):
        try:
            hash(value)
            return ("value", kind.__module__, kind.__qualname__, value)
        except Exception:  # noqa: BLE001 - not hashable after all: compared by identity
            pass
    return _Opaque(value)


def _holds_opaque(key) -> bool:
    """Does a :func:`_plain` form hold an :class:`_Opaque` anywhere?"""
    if isinstance(key, _Opaque):
        return True
    return isinstance(key, tuple) and any(_holds_opaque(k) for k in key)


def _programs(closed, in_paths, out_paths, table: dict, opaque: Optional[set] = None) -> dict:
    """``{flux: its program}`` for one trace of a flux hook.

    A flux's program is a number for each of its leaves that stands for
    the whole expression the leaf is computed by: the primitive of each
    equation with its parameters (:func:`_eqn_params`, a sub-program by
    its structure), applied to the programs of its operands, down to the
    hook's arguments (by their place in the arguments), constants and
    literals (by value).  Numbers are drawn from *table*, shared between
    the traces compared, so two leaves have the same number exactly when
    they are the same expression of the same arguments.  *opaque*
    (shared alike) collects the numbers of the expressions that hold an
    :class:`_Opaque` parameter: two programs that differ only there are
    not shown to compute different things.
    """
    from jax.extend.core import Literal  # noqa: PLC0415

    opaque = set() if opaque is None else opaque

    def number(key):
        return table.setdefault(key, len(table))

    jaxpr = closed.jaxpr
    ids = {v: number(("argument", jax.tree_util.keystr(path)))
           for v, path in zip(jaxpr.invars, in_paths)}
    for v, const in zip(jaxpr.constvars, closed.consts):
        form = _plain(const)
        ids[v] = number(("constant", form))
        if _holds_opaque(form):
            opaque.add(ids[v])

    def of(v):
        return number(("literal", _plain(v.val))) if isinstance(v, Literal) else ids[v]

    for eqn in jaxpr.eqns:
        params = _eqn_params(eqn)
        operands = tuple(of(v) for v in eqn.invars)
        key = (eqn.primitive.name, params, operands)
        hidden = _holds_opaque(params) or any(n in opaque for n in operands)
        for k, v in enumerate(eqn.outvars):
            ids[v] = number(("equation", key, k))
            if hidden:
                opaque.add(ids[v])
    programs: dict = {}
    for v, path in zip(jaxpr.outvars, out_paths):
        programs.setdefault(path[0].key, []).append((jax.tree_util.keystr(path[1:]), of(v)))
    return {flux: tuple(leaves) for flux, leaves in programs.items()}


def _signature(value) -> Any:
    """What a value is on shapes and dtypes, hashable: its tree structure
    and each leaf's shape, dtype and weak type (``None`` for ``None``).
    Two inputs a pass hands are the same to a traced hook exactly when
    their signatures are equal."""
    if value is None:
        return None
    leaves, tree = jax.tree_util.tree_flatten(value)
    return (tree, tuple((tuple(leaf.shape), str(leaf.dtype), bool(getattr(leaf, "weak_type", False)))
                        for leaf in leaves))


def _handed(into, given: dict, state, fluxes: dict, declared: dict, mappings,
            first: bool = False) -> tuple[dict, set, list]:
    """The shapes and dtypes of the inputs a pass hands one member's hook.

    ``({input: abstract value}, the inputs no value could be derived
    for, the inputs taken at their declared shape)``, the dict in the
    order the pass fills it.  Composed as the pass composes it
    (``_coupled_block._run_coupled_block_impl._resolve_boundary`` and
    its ``_resolve_value``), on shapes and dtypes:

    * *into* are the plan's records of the edges into the member in the
      order the pass applies them (``InterfacePlan.edges_into``: every
      edge into it, from a member or from outside the group, in the
      order of the ``add_edge`` calls);
    * an edge's source is the field of its source's *state* where the
      state holds one (the pass looks there first), and otherwise the
      producer's flux in *fluxes* (``{member: {flux: abstract value}}``,
      the fluxes as they stand when the pass visits the member);
    * its value is what its own rule delivers for that source
      (``InterfaceEdge.delivered_abstract``: the mapping, then the
      transform, under :func:`jax.eval_shape`);
    * an additive edge adds its value to what an earlier edge into the
      same input delivered (``+``, with its promotion and broadcasting),
      any other edge replaces it, **whatever it delivered** (a value
      that could not be derived included: the pass hands the later
      edge's value);
    * then the external inputs (*given*, ``{input: abstract value}``)
      replace what the edges delivered;
    * an input keeps the place in the dict its first edge gave it, as a
      key of the pass's dict does when a later edge replaces its value.

    *first*: compose as the first seeding sweep does (``strict=False``),
    which leaves out an edge whose flux is not computed yet (not in
    *fluxes*, or ``None`` there: its hook could not be evaluated in this
    sweep), and the input then holds what the edges before it delivered
    (the value of an edge a later one replaces, where the later one is
    left out).  Otherwise such an edge, like one whose rule cannot be
    evaluated, gives its input no derivable value, and so does an
    additive edge added to such a value: the input is then handed at the
    shape and dtype ``boundary_input_spec()`` declares (*declared*)
    where it declares one, and is unknown otherwise.
    """
    values: dict = {}
    for r in into:
        name = r.target[1]
        source = _source_of(r, state, fluxes, first)
        if source is _LEFT_OUT:
            continue
        delivered = None
        if source is not None:
            try:
                delivered = r.delivered_abstract(source, state, mappings)
            except Exception:  # noqa: BLE001 - the edge's rule fails on these shapes
                delivered = None
        if r.additive and name in values:
            try:
                delivered = (None if delivered is None or values[name] is None
                             else jax.eval_shape(operator.add, values[name], delivered))
            except Exception:  # noqa: BLE001 - the two values do not add
                delivered = None
        values[name] = delivered        # None: not derivable; the key keeps its place
    for name, value in given.items():
        values[name] = value
    fallback: list = []
    unknown: set = set()
    for name in sorted(name for name, value in values.items() if value is None):
        if name in declared:
            values[name] = jax.eval_shape(lambda v: v, declared[name])
            fallback.append(name)
        else:
            del values[name]
            unknown.add(name)
    return values, unknown, fallback


#: :func:`_source_of` of an edge the first seeding sweep leaves out.
_LEFT_OUT = object()


def _source_of(r, state, fluxes: dict, first: bool) -> Any:
    """What edge *r* reads, as ``_coupled_block``'s ``_resolve_value``
    finds it: the field of its source's state where the state holds one,
    else the producer's flux in *fluxes*; where neither holds it,
    :data:`_LEFT_OUT` in the first sweep (``strict=False`` skips the
    edge) and ``None`` in a later one (no value can be derived; the pass
    itself would raise ``KeyError``)."""
    fields = state.get(r.source[0]) or {}
    if r.source[1] in fields:
        return fields[r.source[1]]
    produced = fluxes.get(r.source[0]) or {}
    if r.source[1] in produced:
        return produced[r.source[1]]
    return _LEFT_OUT if first else None


def _underived_causes(into, names, state, fluxes: dict, members, mappings) -> list:
    """Why no value could be derived for the inputs *names* of a member
    (:func:`_handed` composing them on the settled *fluxes*): one clause
    per edge into them whose value could not be derived and is not
    replaced by a later edge that is not additive."""
    causes: list = []
    live: list = []
    for name in sorted(names):
        edges = [r for r in into if r.target[1] == name]
        last = max((i for i, r in enumerate(edges) if not r.additive), default=0)
        live.extend(edges[last:])
    for r in live:
        source = _source_of(r, state, fluxes, False)
        producer, field = r.source
        if source is None:
            if producer not in members:
                causes.append(f"edge {r.key!r} carries a flux from outside the group, which "
                              f"the step does not read (MADD-ANO-156)")
            elif fluxes.get(producer) is None:
                causes.append(f"edge {r.key!r} carries a flux of {producer!r}, whose fluxes "
                              f"could not be evaluated on the inputs the pass hands it")
            else:
                causes.append(f"edge {r.key!r} carries {producer}.{field}, which "
                              f"{producer!r}'s compute_boundary_fluxes does not return")
            continue
        try:
            failed = r.delivered_abstract(source, state, mappings) is None
        except Exception:  # noqa: BLE001 - the edge's rule fails on these shapes
            failed = True
        if failed:
            causes.append(f"the mapping or transform of edge {r.key!r} fails on the shape "
                          f"and dtype it is handed")
    return causes or ["an additive edge's value does not add to what the edges before it "
                      "deliver"]


def _called_as_handed(spec, names) -> Callable:
    """The node's ``compute_boundary_fluxes`` called as the graph calls it
    (``_graph_specs._node_fluxes``), with its inputs in the order *names*:
    a traced function's dict arguments arrive with their keys sorted, and
    the pass hands them in the order it filled them."""
    from maddening.core._graph_specs import _node_fluxes  # noqa: PLC0415

    def hook(st, bi, p):
        return _node_fluxes(spec, st, {name: bi[name] for name in names}, spec.timestep, p)

    return hook


def _abstract_fluxes(members, handed_to: Callable, nodes, state,
                     params_of: Callable[[str], Any]) -> tuple[dict, list, set]:
    """The members' fluxes on shapes and dtypes, as the pass's sweeps
    compute them, swept until they settle.

    ``(fluxes, sweeps, moving)``: *fluxes* ``{member: {flux: abstract
    value}}`` after the last sweep (``None`` for a member whose fluxes
    could not be derived); *sweeps* one ``{member: what handed_to gave
    it}`` per sweep, the first :data:`SEED_SWEEPS` being the pass's
    seeding sweeps and the last the inputs at the settled fluxes;
    *moving* the members whose fluxes still changed in the last sweep
    (empty when the fluxes settled).

    *members* are every flux producer of the group in the sweep order,
    as the pass seeds them (``for nn in group_node_names: if nn in
    flux_producing_nodes``), so that each sweep's fluxes are those the
    pass's own dict holds when it visits a member.  A sweep evaluates
    each hook abstractly (:func:`jax.eval_shape`: nothing is run),
    called as the graph calls it, on ``handed_to(member, fluxes as they
    stand, first sweep)``: the members before it this sweep's fluxes and
    the others the previous sweep's, the first sweep without an input
    whose flux is not computed yet (``one_pass_gs`` and
    ``one_pass_jacobi`` seed so).  A member with an input no value can
    be derived for is not evaluated: its fluxes are not derived either,
    rather than derived at a shape the pass would not give them and
    handed on.  A flux whose shape or dtype follows an input another flux
    delivers can change from one sweep to the next, so the sweeps go on
    until one (after the seeding sweeps) changes no member's fluxes, and
    at most ``SEED_SWEEPS + len(members)``: a dependence of shapes along
    the sweep settles in the sweep it is read in, and each step against
    the sweep takes one sweep more.
    """
    fluxes: dict = {}
    sweeps: list = []
    moving: set = set()
    for sweep in range(SEED_SWEEPS + len(members)):
        moving = set()
        handed: dict = {}
        for member in members:
            handed[member] = handed_to(member, fluxes, sweep == 0)
            values, unknown = handed[member][0], handed[member][1]
            out = None
            if not unknown:
                try:
                    out = jax.eval_shape(_called_as_handed(nodes[member], list(values)),
                                         state[member], values, params_of(member))
                    if not isinstance(out, dict):
                        raise TypeError("compute_boundary_fluxes did not return a dict")
                except Exception:  # noqa: BLE001 - not on these inputs: not derived, asked again next sweep
                    out = None
            if _signature(out) != _signature(fluxes.get(member)):
                moving.add(member)
            fluxes[member] = out
        sweeps.append(handed)
        if sweep + 1 >= SEED_SWEEPS and not moving:
            break
    return fluxes, sweeps, moving


def _flux_reads(spec, state, node_params, handed: dict, absent=(),
                why: Optional[list] = None, sweeps=None,
                unsure: Optional[dict] = None, seen_in: Optional[dict] = None) -> Optional[dict]:
    """Which boundary inputs each flux of one node depends on.

    ``{flux field: set of input names}``, from **abstract traces** of the
    node's ``compute_boundary_fluxes``, called as the graph calls it
    (``_graph_specs._node_fluxes``) on the node's state, its parameters,
    and *handed* (``{input: abstract value}``, :func:`_handed`): the
    inputs a pass hands the hook once the fluxes have settled, in the
    order it hands them.  ``jax.make_jaxpr`` records the program without
    running or compiling it.

    * ``_param_probes._live_jaxpr_inputs`` walks that program backwards
      from one flux at a time and marks the inputs that can reach it.
      The walk is conservative in one direction only (an equation it does
      not know keeps every input): an input it leaves out has no path to
      the flux, an input it reports may still be unused.
    * *sweeps* are the inputs as each seeding sweep hands them
      (``{input: abstract value}``, one per sweep); without it, the first
      sweep hands *handed* without the inputs in *absent*.  Where a sweep
      hands some inputs otherwise than *handed* (it leaves one out, or
      hands it at another shape or dtype, as a sweep does before the
      fluxes have settled), the hook is traced as that sweep calls it,
      and once more for each flux that reads some of those inputs by
      value, with only those handed as the sweep hands them.  A flux
      whose program (:func:`_programs`) is the same in those two traces
      does in that sweep what it would do with only the inputs it reads
      by value changed: it reads no other.  Where its program differs,
      it is reported to read each other input whose sweep's form, added
      alone, changes its program, or every input that sweep hands
      otherwise where those alone do not make up the difference or the
      hook cannot be traced so (it indexes an input the sweep leaves
      out).  Where the programs compared hold an opaque parameter
      (:class:`_Opaque`), the inputs found so are *unsure*: written to
      *unsure* (``{flux: inputs}``, where not also found otherwise), and
      still reported, so the caller can say the read is assumed.  An
      input a flux does not read by value once the fluxes settle, and
      reads as a seeding sweep hands it, is written to *seen_in*
      (``{flux: {input: the first such sweep, from 1}}``).

    Only ``make_jaxpr`` and the equations' ``primitive`` / ``invars`` /
    ``outvars`` / ``params`` are read, which are the same on the jax
    versions the project supports.

    ``None`` when dependence cannot be established this way: the hook
    raises under the trace on the inputs it is handed, or does not return
    a dict.  The reason, then the exception, are appended to *why*, and
    the caller takes every flux to read every input.
    """
    from maddening.core._param_probes import _live_jaxpr_inputs  # noqa: PLC0415

    def trace(boundary):
        args = (state, boundary, node_params)
        closed, returned = jax.make_jaxpr(_called_as_handed(spec, list(boundary)),
                                          return_shape=True)(*args)
        if not isinstance(returned, dict):
            raise TypeError("compute_boundary_fluxes did not return a dict")
        in_paths = [path for path, _ in jax.tree_util.tree_flatten_with_path(args)[0]]
        out_paths = [path for path, _ in jax.tree_util.tree_flatten_with_path(returned)[0]]
        if (len(in_paths) != len(closed.jaxpr.invars)
                or len(out_paths) != len(closed.jaxpr.outvars)):
            raise ValueError("the traced program does not line up with its arguments")
        return closed, returned, in_paths, out_paths

    try:
        closed, returned, in_paths, out_paths = trace(dict(handed))
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
            why.append(exc)
        return None
    if sweeps is None:
        sweeps = [{n: v for n, v in handed.items() if n not in set(absent)}] if absent else []
    by_value = {flux: frozenset(names) for flux, names in reads.items()}
    sure = {flux: set(names) for flux, names in reads.items()}
    doubted: dict = {flux: set() for flux in reads}
    table: dict = {}
    opaque: set = set()
    settled = _programs(closed, in_paths, out_paths, table, opaque)
    for sweep, given in enumerate(sweeps, start=1):
        otherwise = frozenset(
            n for n in (*handed, *given)
            if n not in given or n not in handed or _signature(given[n]) != _signature(handed[n]))
        if not otherwise:
            continue
        traces = {frozenset(): settled}

        def programs_as(changed, given=given, traces=traces):
            # *changed* as the sweep hands them, the other inputs as settled;
            # in the sweep's order where it hands them.
            if changed not in traces:
                order = [*given, *(n for n in handed if n not in given)]
                other = trace({n: (given if n in changed else handed)[n] for n in order
                               if n in (given if n in changed else handed)})
                if set(other[1]) != set(returned):
                    raise ValueError("other fluxes on these inputs")
                traces[changed] = _programs(other[0], other[2], other[3], table, opaque)
            return traces[changed]

        for flux in reads:
            base = by_value[flux] & otherwise
            seen: list = []

            def program(changed, flux=flux, seen=seen):
                seen.append(programs_as(changed).get(flux))
                return seen[-1]

            try:
                whole = program(otherwise)
                if program(base) == whole:
                    continue
                # Which of the others change it, each alone; all of them
                # where only some together do.
                found = frozenset(n for n in otherwise - base
                                  if program(base | {n}) != program(base))
                if program(base | found) != whole:
                    found = otherwise
            except Exception:  # noqa: BLE001 - a hook that indexes an input the sweep leaves out
                found = otherwise
            # A difference in an expression that holds an opaque parameter
            # is not shown to be one of what the flux computes.
            hidden = any(n in opaque for leaves in seen if leaves for _path, n in leaves)
            (doubted if hidden else sure)[flux].update(found)
            if seen_in is not None:
                for name in found - by_value[flux]:
                    seen_in.setdefault(flux, {}).setdefault(name, sweep)
    if unsure is not None:
        for flux in reads:
            if doubted[flux] - sure[flux]:
                unsure[flux] = doubted[flux] - sure[flux]
    return {flux: sure[flux] | doubted[flux] for flux in reads}


def _flux_records(plan, is_flux: Callable[[str, str], bool] = lambda member, field: True) -> list:
    """The edges inside the group whose source is a boundary flux, as declared.

    ``plan.flux_edges()`` without the edges whose source is neither a
    state field nor a field of a node that has a flux hook, and without
    those *is_flux* takes out (a field a traced hook does not return).
    Both read a field that does not exist; ``validate()`` has that error.
    """
    return [r for r in plan.flux_edges() if r.source_kind == FLUX and is_flux(*r.source)]


def _handed_on(plan, records: list, external) -> list:
    """*records* without the edges whose value the pass does not hand on
    once its fluxes are all computed.

    Composed as :func:`_handed` composes: an edge is followed into the
    same input by an edge that is not additive (which replaces what it
    delivered), or delivers to an input that is an external input of its
    target (which replaces it), and from the second seeding sweep on its
    value is read by neither the hook nor the update.

    **The first seeding sweep can hand it**: where the replacing edge
    carries a flux that sweep has not computed yet, the sweep leaves that
    edge out and the input holds the replaced value.  That is not lost:
    :func:`_handed` composes the first sweep so, the hook is traced again
    on it (:func:`_flux_reads`), and a flux that reads that input there
    (by value, or by a program that differs there) is taken to read it
    through the replacing edge, whose source the first
    sweep reaches only after the member: one sweep late, so its seed is
    right no earlier than the second sweep (``S >= 2``), which is what a
    value computed from the replaced one in the first sweep needs.  An
    arc to the replaced edge's own source is not drawn: the flux reads it
    in no sweep after the first, so it is no part of a loop.
    """
    given = {(member, name) for member, name, _value in external}
    replaced: set = set()
    for member in dict.fromkeys(r.target[0] for r in records):
        into = plan.edges_into(member)
        for i, r in enumerate(into):
            if (member, r.target[1]) in given or any(
                    later.target[1] == r.target[1] and not later.additive
                    for later in into[i + 1:]):
                replaced.add(id(r))
    return [r for r in records if id(r) not in replaced]


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


def _describe(steps: list, seen_in: Optional[dict] = None) -> str:
    """The reads of a loop or a chain, one clause each.  A read found only
    as a seeding sweep hands the input (*seen_in*, ``{member: {flux:
    {input: sweep}}}``, :func:`_flux_reads`) says so."""
    seen_in = seen_in or {}

    def clause(flux, read, name, key):
        sweep = seen_in.get(flux[0], {}).get(flux[1], {}).get(name)
        if sweep is None:
            return (f"flux {flux[0]}.{flux[1]} reads input {name!r}, which edge {key!r} "
                    f"delivers from flux {read[0]}.{read[1]}")
        return (f"flux {flux[0]}.{flux[1]} reads input {name!r} in seeding sweep {sweep}, "
                f"which hands it otherwise than once the fluxes settle (edge {key!r} "
                f"delivers it from flux {read[0]}.{read[1]})")

    return "; ".join(clause(*step) for step in steps)


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
        hooks.  Called only for a group its edges alone would refuse,
        once for each flux producer of the group.
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

    The edges are asked first (those whose value the pass hands on,
    :func:`_handed_on`), with every flux of a member taken to read every
    input a flux edge delivers to it.  A group that passes that way
    returns here, untraced.  Otherwise the pass's sweeps are run on
    shapes and dtypes until the fluxes settle (:func:`_abstract_fluxes`),
    the members such edges deliver to are traced (:func:`_flux_reads`) on
    the inputs the last sweep hands them and on those each seeding sweep
    hands them (:func:`_handed`), and the group is asked again with what
    their fluxes read.
    """
    names = sorted(group.nodes)
    if not all(n in nodes and n in state for n in names):
        return                      # validate() has the error for it
    jacobi = group.iteration_mode == "jacobi"
    external = tuple(external)
    records = _handed_on(plan, _flux_records(plan), external)
    arcs = _flux_arcs(records, lambda member, flux, name: True)
    if not any(arcs.values()):
        return
    order = list(sweep_order())
    if _a_loop(arcs) is None and _sweeps_needed(
            arcs, order, _read_from_the_seed(records, order, jacobi))[0] <= SEED_SWEEPS:
        return

    from maddening.core._param_probes import _declared_boundary_zeros  # noqa: PLC0415

    # The members traced: the producers a flux edge of the group delivers
    # to.  The sweeps run over every flux producer of the group in the
    # sweep order, as the pass seeds them, each with the edges into it in
    # the order the pass applies them and its external inputs
    # (``_resolve_boundary``): a producer that no edge the criterion
    # follows reads can still deliver a value the pass hands (an edge a
    # later one replaces, in the first sweep).
    fed = list(dict.fromkeys(
        r.target[0] for r in records if r.target[0] in plan.flux_members))
    involved = [m for m in order if m in plan.flux_members]
    into = {member: plan.edges_into(member) for member in involved}
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

    def handed_to(member, fluxes, first=False):
        return _handed(into[member], given[member], state, fluxes,
                       declared[member], mappings, first)

    fluxes, sweeps, moving = _abstract_fluxes(involved, handed_to, nodes, state,
                                              params.__getitem__)
    # Where the fluxes have not settled, the inputs of the members they
    # reach through the group's flux edges have not either.
    unsettled = set(moving)
    while True:
        reached = {r.target[0] for r in records if r.source[0] in unsettled} - unsettled
        if not reached:
            break
        unsettled |= reached
    traced: dict = {}
    reason: dict = {}
    taken_as_declared: dict = {}
    unsure: dict = {}
    seen_in: dict = {}
    for member in fed:
        values, unknown, fallback = sweeps[-1][member]
        if member in unsettled:
            traced[member] = None
            reason[member] = (
                "unsettled",
                f"the shapes and dtypes of the fluxes delivered to it had not settled after "
                f"{len(sweeps)} sweeps over the members (a flux's shape or dtype follows an "
                f"input another flux delivers, round a loop or along more steps against the "
                f"sweep order than that)")
            continue
        # Traced first, on what can be derived: a hook that cannot be
        # traced at all is reported as such, whatever its inputs.
        why: list = []
        unsure[member], seen_in[member] = {}, {}
        traced[member] = _flux_reads(
            nodes[member], state[member], params[member], values, (), why,
            [sweeps[k][member][0] for k in range(SEED_SWEEPS)], unsure[member],
            seen_in[member])
        taken_as_declared[member] = fallback
        failure = why[1] if len(why) > 1 else None
        if traced[member] is None and not (
                unknown and isinstance(failure, KeyError) and failure.args
                and failure.args[0] in unknown):
            reason[member] = (
                "untraceable",
                "its compute_boundary_fluxes could not be traced on the inputs the pass "
                f"hands it ({why[0] if why else 'no reason recorded'})")
        elif unknown:
            traced[member] = None
            reason[member] = (
                "underived",
                f"no shape could be derived for its input {sorted(unknown)}: "
                + "; ".join(_underived_causes(into[member], unknown, state, fluxes,
                                              plan.members, mappings))
                + "; and boundary_input_spec() does not declare it")

    def depends(member, flux, name):
        reads = traced.get(member)
        # A hook that could not be traced: every flux is taken to read it.
        return reads is None or name in reads.get(flux, ())

    def is_flux(member, field):
        reads = traced.get(member)
        return reads is None or field in reads

    records = _handed_on(plan, _flux_records(plan, is_flux), external)
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
            f"step: {_describe(chain, seen_in)}. Each pass rebuilds the fluxes from nothing in "
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
            f"its boundary fluxes depend on each other in a loop: {_describe(loop, seen_in)}. "
            f"Each pass rebuilds the fluxes from nothing in {SEED_SWEEPS} sweeps over "
            f"the members, which is exact for a flux that reads no other flux and is "
            f"not a solution of fluxes that need each other")
        ways = (
            "Taking the nodes out of the group does not help: outside a group the "
            "flux edge that closes a loop cannot be read at all (MADD-ANO-157). ")
        steps = loop
    readers = list(dict.fromkeys(member for (member, _field), _read, _name, _key in steps))
    assumed = [member for member in readers if traced.get(member) is None]
    as_declared = [member for member in readers
                   if traced.get(member) is not None and taken_as_declared.get(member)]
    if as_declared:
        found += ". " + "; ".join(
            f"{m!r} was traced with its input {taken_as_declared[m]} at the shape and dtype "
            f"its boundary_input_spec() declares, as what the edges deliver there could not "
            f"be derived" for m in as_declared)
    # Reads found only where the programs compared hold a parameter the
    # comparison cannot look into: assumed, and said so.
    doubtful = list(dict.fromkeys(
        (member, field, name) for (member, field), _read, name, _key in steps
        if traced.get(member) is not None and name in unsure.get(member, {}).get(field, ())))
    if assumed:
        found += (
            ". What the fluxes of " + ", ".join(repr(m) for m in assumed) + " read is "
            "assumed, not established: " + "; ".join(
                f"for {m!r}, {reason[m][1]}" for m in assumed)
            + ". Each of their fluxes is therefore taken to read every input a flux "
            "edge delivers to its node")
    if doubtful:
        found += (
            ". " + "; ".join(f"that flux {m}.{f} reads input {n!r}" for m, f, n in doubtful)
            + (" is" if len(doubtful) == 1 else " are") + " assumed, not established: the "
            "programs it was compared by, as a seeding sweep hands that input and as the "
            "settled inputs give it, differ, and hold an operation whose parameters are a "
            "function or an object the comparison cannot look into (a host callback, for "
            "example), so the difference is not shown to be one of what the flux computes")
    if assumed or doubtful:
        remedies = {
            "untraceable": "write compute_boundary_fluxes as JAX operations on its "
                           "arguments (the step itself needs that)",
            "underived": "declare an input whose shape could not be derived in "
                         "boundary_input_spec()",
            "unsettled": "give each flux a shape and dtype that do not follow the shapes "
                         "of the inputs other fluxes deliver",
            "opaque": "compute such a flux without operations that take a function or "
                      "an object as a parameter"}
        kinds = {reason[m][0] for m in assumed} | ({"opaque"} if doubtful else set())
        closing = (
            "If these fluxes in fact read no other flux, the group compiles once that "
            "can be established: " + ", and ".join(
                remedies[kind] for kind in remedies if kind in kinds) + ".")
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
