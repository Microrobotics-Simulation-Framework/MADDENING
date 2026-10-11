"""Integer, Boolean and key fields that a coupling group's pass reads from the iterate.

A coupling group solves ``s = P(s)``, ``P`` one pass over its members (a
Jacobi pass or a Gauss-Seidel sweep).  Its solvers iterate the group's
**floating** fields and read convergence from them: a norm of a change has
no meaning for a flag or a key, and neither has a relaxed step.  This module
says what happens to the other fields, and proves that it is right.

The fields in question
----------------------
Write the group's state as ``s = (x, c, o)``:

* ``x``, every floating field;
* ``c``, the **carried** non-floating fields: the non-floating state fields
  that a pass reads *from the iterate* when it computes any field of any
  member.  A flag one member computes and another reads across an edge of
  the group is one; so is a flag that a member's own boundary flux is
  computed from, where another member reads that flux;
* ``o``, the remaining non-floating fields.  No pass reads them (a member
  integrates from its pre-step state, not from its own iterate), so they
  are outputs only: ``P`` is a function of ``(x, c)``.  A step counter or
  a PRNG key that a node advances from its pre-step state is of this kind
  unless a pass reads it from the iterate -- another member across an
  edge, or a boundary flux computed from it.  Then it is carried, with no
  field feeding it (depth one), and so never on a loop.  Edges and fluxes
  of the node that do not read it change neither.

How "reads" is established
--------------------------
From the program, not from the edges.  The group's Jacobi pass ``J`` is
traced once, abstractly (``jax.make_jaxpr``: nothing is compiled or run),
as a function of its members' iterate, with everything a pass hands a node's
hooks exactly as a step hands it -- every boundary input an edge delivers,
whether or not the node declares it, each flux computed by the producer's
own hook from the producer's state and inputs.  Output field ``g`` *reads*
input field ``f`` when ``f`` can reach ``g`` through the traced equations,
walked back from ``g`` alone (``_param_probes._live_jaxpr_inputs``, the walk
the parameter probes use, with its two keywords for this walk:
``effects=False`` and an ``opaque`` predicate).  "``f`` feeds ``g``" below
means exactly that, for two non-floating fields.

*Effects.*  An equation is followed only through its results.  One whose
results reach no output makes nothing a read, effects or not: a
``jax.debug.print`` or ``jax.debug.callback`` has no result, so printing a
field -- in a node's ``update``, its flux hook, a ``lax.cond`` branch, a
jitted helper or a loop body -- makes it feed nothing.  An equation with an
effect whose result *is* followed (an ``io_callback`` that returns a value,
a read of a mutable array) may return anything the program handed out
before it, so an output reached through one is taken to read every field
of the iterate, and each of those reads is *assumed* (below).
(``jax.pure_callback`` has no effect: it is taken at its word, a function
of its operands -- assumed all the same, since the walk does not look
inside it.)

*Established and assumed reads.*  A read is **established** when ``f``
reaches ``g`` through equations whose operands the walk knows to feed their
results: every primitive with one result (each operand feeds it), every
primitive with one operand, and the structured ones it follows into
(``jit``, ``cond``, ``while``, ``scan``, ...).  A read is **assumed** when
``f`` reaches ``g`` only through an equation the walk does not look inside
and whose results need not each use every operand: a function wrapped in
``jax.custom_jvp`` or ``jax.custom_vjp`` (the derivative rule may read what
the primal does not, so it is not followed), a callback, a primitive with
several operands and several results (``sort`` of several arrays), a
structured primitive whose layout the walk does not recognise, or an
equation with an effect as above.  Those are the only places the walk can
overstate: a field it reports unread is never an operand on the way to the
output.  Each read is classified by walking ``g`` twice, once with such an
equation taken to use all of its operands and once with it taken to use
none; a read the second walk still finds is established.

The trace is taken only for a group that *may* carry such a field, which is
asked of the edges first (:func:`possible_chain`, an upper bound: a member
taken to use everything an edge hands it, a flux its producer's whole
state).  A group whose edges allow none is not traced, and is compiled to
the program it always was.  Where the trace is taken, a read counts only if
the edges allow it too: two upper bounds of what the pass reads, of which
what both allow is a third.  So a loop the trace shows is always a loop of
the edges' bound, and the two places that ask (``compile()``, which traces
a group only when its bound has a loop, and the step, which traces it
wherever its bound has a field) decide alike from the same state.

**When dependence is assumed instead.**  When that one trace cannot be
taken: the pass raises when traced on its own (:func:`carried_nonfloat_chain`
catches the exception and keeps its text).  The bound from the edges then
stands in, the refusal below can be of a loop that is not there, and its
message says that the dependence was assumed, and why.

The rule
--------
**Every pass a solve measures starts from carried fields evaluated at its
floating iterate.**  For an iterate ``x`` the carried fields are set to
``c(x)``, obtained by evaluating the members as the Jacobi pass does, every
member reading the same frozen state, and keeping the carried fields; this
is repeated ``L`` times, ``L`` the length of the longest chain of carried
fields that feed one another.  Both solvers (``"ift"`` and ``"fori"``) and
both sweeps then iterate the one map

    ``G(x) = P_x(x, c(x))``

and measure the residual ``r(x) = ||G(x) - x||`` in the group's norm.  The
state a solve returns holds ``x*`` and the non-floating fields ``P`` computes
at ``(x*, c(x*))``.

Why it is right
---------------
Write ``J_c(x, c)`` for the carried fields the Jacobi pass computes.

*Lemma 1 (the evaluation is a function of the iterate).*  If no chain of
carried fields closes on itself, give each carried field its depth: one
more than the largest depth among the carried fields that feed it, one if
none does.  After ``k`` evaluations ``c <- J_c(x, c)`` every field of depth
at most ``k`` holds a value that depends on ``x`` alone, and keeps it under
further evaluations: a field of depth one is computed from ``x`` and from
nothing an evaluation changes; a field of depth ``k + 1`` is computed from
``x`` and from fields of depth at most ``k``, which the induction has
settled.  So after ``L`` evaluations (``L`` the largest depth) the carried
fields are ``c(x)``, whatever they held before, and ``c(x)`` is the only
solution of ``c = J_c(x, c)``.  (The lemma needs "feeds" to hold wherever
one field is in fact computed from another; a relation that overstates,
as the walk's may, costs evaluations and nothing else.)

*Lemma 2 (the fixed points are the pass's own).*  ``(x, c)`` is a fixed
point of ``P`` -- of either sweep, the carried fields included -- if and
only if ``c = c(x)`` and ``x = G(x)``.

  "Only if": a state a Gauss-Seidel sweep leaves unchanged is one a Jacobi
  pass leaves unchanged (each member in turn finds the state it would have
  found in a Jacobi pass, and returns its own part of it).  So
  ``c = J_c(x, c)``, which by Lemma 1 is ``c = c(x)``, and then
  ``x = P_x(x, c) = G(x)``.

  "If": take ``x = G(x)`` and start the pass from ``s = (x, c(x))``.  A
  Jacobi pass returns ``x`` (that is ``x = G(x)``) and ``c(x)`` (Lemma 1).
  A Gauss-Seidel sweep visits its members in order; the first reads ``s``
  as it would in a Jacobi pass, so its carried fields come back as
  ``c(x)``, and its floating fields come back as ``x`` because the sweep's
  result is ``G(x) = x``.  The state the second member reads is therefore
  ``s`` again, and so on down the sweep.

*Consequences.*  (a) A residual of exactly zero is reported only at a fixed
point of the pass, non-floating fields included -- under either solver and
either sweep.  (b) All four combinations of solver and sweep have the same
set of fixed points, which is also that of the same graph with the fields
stored in floating point.  (c) The two solvers iterate the same map from the
same start with the same acceleration, as they do for an all-floating group.

What is not claimed
-------------------
* Under a non-zero tolerance ``converged=True`` says the *floating* fields
  move by no more than the tolerance over a pass that starts from
  ``(x*, c(x*))``.  A Gauss-Seidel sweep recomputes a carried field from
  floating fields the sweep has already moved, by up to that tolerance; a
  flag whose threshold lies within the tolerance of the iterate can come
  back different from ``c(x*)``.  Under Jacobi it cannot (Lemma 1).
* A group whose map has no fixed point (a flag that flips the floating
  fields back and forth) reaches its cap with ``converged=False``, as an
  all-floating group that oscillates does.
* ``max_iterations=1`` iterates nothing: its one staggered pass reads the
  carried fields as they are stored, one step behind, and nothing here
  applies to it.

What is refused
---------------
A group with ``max_iterations >= 2`` in which carried fields feed one
another in a closed loop: in the trace of its Jacobi pass, a non-floating
state field is an operand of the arithmetic that computes a second, that
one of a third, and so on back to the first (a loop of one is a field that
is an operand of its own value).  Two members that each compute a flag from
the flag the other hands them are the plain case.  No number of evaluations
yields "their value at ``x``" (Lemma 1 has no base): the fields would have
to be carried from pass to pass outside the residual, and a solve could
read ``converged=True`` while they still change.
``GraphManager.compile()`` raises a ``ValueError`` naming the fields
(:func:`nonfloat_loop_refusal`), ``validate()`` lists it as an error, and
the step raises it too if it is ever traced for such a group: compile and
the step ask the same question of the state each holds, so the step's
refusal is reached only for a state ``compile()`` did not see -- one a
write gave other dtypes after it (``set_node_state`` says to call
``validate()`` after such a write), or a pass that returns a field in
another dtype than the one it is stored in.  For each link of the loop the
message says whether it was established or assumed, and through which
primitive.

Not refused: a member that holds a step counter or a PRNG key and
exchanges boundary fluxes with another, both ways -- the trace shows the
counter is computed from the pre-step state and that no flux reads it, so it
is not carried at all, printed or not; and two members that hand each other
flags which each computes from floating inputs alone -- carried, each of
depth one.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from maddening.core._param_probes import _CALL_PRIMITIVES, _live_jaxpr_inputs

#: The structured primitives the walk follows into.  It asks
#: :func:`_unfollowed_overstates` about one of them only when it does not
#: recognise its layout.
_FOLLOWED = frozenset({"while", "scan", "cond"}) | _CALL_PRIMITIVES


class CarriedChain(NamedTuple):
    """The carried non-floating fields of a group and how they feed one another.

    Attributes
    ----------
    fields : tuple
        ``((node, field), ...)``, sorted.
    depth : int
        The length of the longest chain of carried fields that feed one
        another: how many evaluations settle them (Lemma 1 of the module
        docstring).  Zero for a group with no carried field, and where
        ``loop`` is not ``None``.
    loop : tuple or None
        ``None``, or carried fields that feed one another in a closed loop,
        in order: each feeds the next and the last feeds the first.
    assumed : str or None
        ``None`` where "feeds" was read from a trace of the group's pass,
        or for a group whose edges allow no carried field.  Otherwise why
        it could not be, in which case the fields and the loop are the
        bound the edges give (:func:`possible_chain`).
    assumed_links : tuple
        For a loop read from the trace, its links that are *assumed* (the
        module docstring): ``((feeder, fed, primitives), ...)``, each
        ``primitives`` the names of the equations the read passes through
        that the walk does not look inside.  Empty when every link is
        established, which a loop that has an established alternative is
        reported as.
    """

    fields: tuple
    depth: int
    loop: Optional[tuple]
    assumed: Optional[str] = None
    assumed_links: tuple = ()


def _non_floating(value) -> bool:
    """An integer, Boolean or PRNG-key leaf: what a group carries beside its floats.

    Not a complex one: the coupling solver carries none, and says so
    itself at the first step (``acceleration.float_image``).
    """
    return not jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact)


def _chain(fields, fed_by, assumed=None) -> CarriedChain:
    """The longest chain among *fields*, or a loop; ``fed_by[f]`` are the fields feeding ``f``."""
    fields = sorted(fields)
    depth: dict[tuple, int] = {}
    on_path: list[tuple] = []

    def visit(field):
        """Depth of *field*, or the loop found below it."""
        if field in depth:
            return None
        if field in on_path:
            return tuple(on_path[on_path.index(field):])
        on_path.append(field)
        deepest = 0
        for feeder in sorted(fed_by.get(field, ())):
            loop = visit(feeder)
            if loop is not None:
                return loop
            deepest = max(deepest, depth[feeder])
        on_path.pop()
        depth[field] = deepest + 1
        return None

    for field in fields:
        loop = visit(field)
        if loop is not None:
            # In feeding order (``visit`` walked from a field to its
            # feeders), starting at the loop's first field in sorted order.
            ring = list(reversed(loop))
            start = ring.index(min(ring))
            return CarriedChain(tuple(fields), 0, tuple(ring[start:] + ring[:start]), assumed)
    return CarriedChain(tuple(fields), max(depth.values(), default=0), None, assumed)


def possible_chain(plan, state, assumed=None) -> CarriedChain:
    """The non-floating fields the group *may* carry, as far as its edges say.

    An upper bound, decided without looking at a node's code
    (``InterfacePlan.iterate_reads``): a member is taken to use every
    field an edge hands it in every field it computes, and a boundary flux
    to use its producer's whole state and everything the producer is
    handed.  ``fields`` empty means the group carries none, whatever its
    nodes compute; ``loop`` ``None`` means it carries none in a closed
    loop.  Anything else is a question for :func:`carried_nonfloat_chain`.
    """
    return _chain(*_edge_relation(plan, state), assumed)


def _edge_relation(plan, state) -> tuple:
    """``(fields, fed_by)`` of :func:`possible_chain`: the edges' bound, before chaining."""
    reads = plan.iterate_reads()
    carried = {
        (nn, fld) for pairs in reads.values() for nn, fld in pairs
        if fld in state.get(nn, {}) and _non_floating(state[nn][fld])}
    return carried, {(nn, fld): carried & reads[nn] for nn, fld in carried}


def _field_of(path) -> tuple:
    """``(node, field)`` of a leaf of ``{node: {field: leaf}}``."""
    return path[0].key, path[1].key


def _unfollowed_overstates(eqn) -> bool:
    """Whether "each result uses every operand" may overstate for *eqn*.

    Asked by the walk only of an equation it does not follow into, which
    it takes to use every operand for each result.  That is exact for a
    primitive with one result or with one operand.  It may overstate for a
    ``custom_*`` call or a callback (the walk does not look inside), a
    primitive with several operands and several results, a structured
    primitive whose layout the walk did not recognise (:data:`_FOLLOWED`),
    and an equation with an effect, whose result may come from outside the
    program.  The reads through such an equation are the *assumed* ones of
    the module docstring.
    """
    name = eqn.primitive.name
    return (bool(eqn.effects) or name.startswith("custom_") or "callback" in name
            or name in _FOLLOWED
            or (len(eqn.invars) > 1 and len(eqn.outvars) > 1))


def traced_reads(evaluate: Callable, members: dict, *, assumed: Optional[dict] = None) -> dict:
    """``{output field: the input fields it reads}`` in one abstract trace of *evaluate*.

    *evaluate* maps ``{node: {field: array}}`` (the members' iterate) to a
    dict of the same layout (what one Jacobi pass computes from it);
    fields are ``(node, field)`` pairs.  ``jax.make_jaxpr`` traces it once
    -- nothing is compiled, nothing runs -- and each output is walked back
    to the inputs that can reach it (``_live_jaxpr_inputs``), following an
    equation only through its results: an equation with no result that
    reaches the output (a ``jax.debug.print``) makes nothing a read.

    *assumed*, if given, is filled with ``{(output, input): primitives}``
    for each read that is assumed rather than established (the module
    docstring): ``primitives`` names the equations it passes through that
    the walk does not look inside.  An output reached through such an
    equation with an effect (an ``io_callback`` that returns a value) is
    taken to read every input.  Raises whatever the trace raises.
    """
    closed, out_shape = jax.make_jaxpr(evaluate, return_shape=True)(members)
    in_fields = [_field_of(p) for p, _ in jax.tree_util.tree_flatten_with_path(members)[0]]
    out_fields = [_field_of(p) for p, _ in jax.tree_util.tree_flatten_with_path(out_shape)[0]]
    jaxpr = closed.jaxpr
    if len(in_fields) != len(jaxpr.invars) or len(out_fields) != len(jaxpr.outvars):
        raise ValueError("the pass's fields do not line up with its traced program")
    reads: dict[tuple, set] = {}
    established: dict[tuple, set] = {}
    through: dict[tuple, set] = {}
    for i, out_field in enumerate(out_fields):
        live_out = [j == i for j in range(len(out_fields))]
        met: dict[str, bool] = {}

        def note(eqn, met=met) -> bool:
            """Record an equation the read may overstate through; take it whole."""
            if _unfollowed_overstates(eqn):
                met[eqn.primitive.name] = met.get(eqn.primitive.name, False) or bool(eqn.effects)
            return False

        live = _live_jaxpr_inputs(jaxpr, live_out, effects=False, opaque=note)
        sure = live
        if met:
            sure = _live_jaxpr_inputs(jaxpr, live_out, effects=False,
                                      opaque=_unfollowed_overstates)
            if any(met.values()):
                live = [True] * len(in_fields)
            through.setdefault(out_field, set()).update(met)
        reads.setdefault(out_field, set()).update(
            in_field for in_field, keep in zip(in_fields, live) if keep)
        established.setdefault(out_field, set()).update(
            in_field for in_field, keep in zip(in_fields, sure) if keep)
    if assumed is not None:
        for out_field, fields in reads.items():
            for in_field in fields - established[out_field]:
                assumed[(out_field, in_field)] = tuple(sorted(through[out_field]))
    return reads


def carried_nonfloat_chain(plan, state, evaluate: Optional[Callable] = None) -> CarriedChain:
    """The group's carried non-floating fields, their longest chain, or a loop.

    *evaluate* is the group's Jacobi pass as a function of its members'
    iterate (:func:`traced_reads`).  A group whose edges allow no carried
    field (:func:`possible_chain`) is not traced.  Otherwise the carried
    fields are the non-floating fields of *state* that any output of the
    traced pass reads, and one feeds another when the second's output
    reads the first -- each only where the edges' bound allows it too, so
    that a loop returned here is a loop of that bound (the module
    docstring: ``compile()`` and the step then decide alike).  Of a loop,
    one whose links are all established is returned if there is one;
    otherwise ``assumed_links`` names the links that are not.  Where the
    trace raises, or *evaluate* is ``None``, the bound from the edges is
    returned with ``assumed`` saying why.
    """
    bound_fields, bound_fed_by = _edge_relation(plan, state)
    bound = _chain(bound_fields, bound_fed_by)
    if not bound.fields:
        return bound
    if evaluate is None:
        return bound._replace(assumed="no trace of the group's pass was asked for")
    members = {nn: state[nn] for nn in plan.order}
    how: dict = {}
    try:
        reads = traced_reads(evaluate, members, assumed=how)
    except Exception as exc:  # noqa: BLE001 - whatever a node's hooks raise when traced alone
        return bound._replace(assumed=(
            f"its pass could not be traced on its own ({type(exc).__name__}: "
            f"{str(exc)[:200]})"))
    nonfloat = {(nn, fld) for nn, fields in members.items() for fld, value in fields.items()
                if _non_floating(value)}
    carried = nonfloat & set().union(*reads.values()) & bound_fields
    fed_by = {field: carried & reads.get(field, set()) & bound_fed_by[field]
              for field in carried}
    chain = _chain(carried, fed_by)
    if chain.loop is None:
        return chain
    sure = _chain(carried, {field: {f for f in feeders if (field, f) not in how}
                            for field, feeders in fed_by.items()})
    if sure.loop is not None:
        return sure
    ring = chain.loop
    return chain._replace(assumed_links=tuple(
        (feeder, fed, how[(fed, feeder)])
        for feeder, fed in zip(ring, ring[1:] + ring[:1]) if (fed, feeder) in how))


def nonfloat_loop_refusal(group_nodes, chain: CarriedChain, state) -> Optional[str]:
    """Why a group whose carried fields close a loop cannot be iterated, or ``None``."""
    if chain.loop is None:
        return None

    def named(field):
        nn, fld = field
        return f"{nn}.{fld} ({jnp.asarray(state[nn][fld]).dtype})"

    ring = " -> ".join(named(f) for f in (*chain.loop, chain.loop[0]))
    if chain.assumed is None and not chain.assumed_links:
        how = (
            "This is read from one trace of the group's pass: each field named is "
            "an operand of the arithmetic that computes the next.")
    elif chain.assumed is None:
        shaky = {(feeder, fed): prims for feeder, fed, prims in chain.assumed_links}
        links = []
        for feeder, fed in zip(chain.loop, chain.loop[1:] + chain.loop[:1]):
            pair = f"{feeder[0]}.{feeder[1]} -> {fed[0]}.{fed[1]}"
            if (feeder, fed) in shaky:
                links.append(f"{pair} is ASSUMED, through {', '.join(shaky[feeder, fed])}")
            else:
                links.append(f"{pair} is established through operands")
        how = (
            "Each link as a trace of the group's pass gives it: " + "; ".join(links) + ". "
            "An assumed link passes through an operation the trace does not look "
            "inside (a custom_jvp or custom_vjp function, a callback, a primitive "
            "with several operands and several results, or one with an effect whose "
            "result is used), which is taken to use every operand for each of its "
            "results, so the loop may not be one; computing the field outside that "
            "operation lets the trace see what it reads.")
    else:
        how = (
            f"This could not be read from the group's pass, because {chain.assumed}; "
            "it is assumed from the edges instead, a member taken to use every "
            "field an edge hands it in every field it computes and a boundary flux "
            "its producer's whole state, so the loop may not be one.")
    return (
        f"Coupling group {sorted(group_nodes)}: non-floating fields are computed "
        f"from one another in a closed loop: {ring}. {how} The group's iteration "
        "acts on floating-point fields and reads its convergence from them; an "
        "integer, Boolean or key field that a pass reads from the iterate is "
        "evaluated at each iterate from the fields it is computed from, which needs "
        "the fields that are computed from one another to form chains that end. "
        "Around a closed loop they have no value at an iterate: they would be "
        "carried from pass to pass where the residual does not read them, and the "
        "group could report converged=True while they still change. Store one field "
        "of the loop in floating point (the residual then measures it), or compute "
        "it in a node outside the group, or set max_iterations=1 (one staggered "
        "pass, which iterates nothing)."
    )


def _refuse_nonfloat_loops(refusals) -> None:
    """Raise one ``ValueError`` for *refusals* (texts of :func:`nonfloat_loop_refusal`), if any."""
    refusals = [text for text in refusals if text]
    if refusals:
        raise ValueError("\n".join(refusals))
