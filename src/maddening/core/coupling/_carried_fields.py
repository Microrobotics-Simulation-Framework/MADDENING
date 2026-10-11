"""Integer, Boolean and key fields that a coupling group's own edges carry.

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
  that a pass reads *from the iterate* (:func:`carried_nonfloat_fields`).
  A member reads the iterate in three ways only: the source field of an
  internal edge; the geometry field a mapping on such an edge reads from
  its source; and, through an internal edge whose source is a boundary
  flux, the producer's whole state and whatever the producer itself reads
  (its flux is computed from both);
* ``o``, the remaining non-floating fields.  No pass reads them (a member
  integrates from its pre-step state, not from its own iterate), so they
  are outputs only: ``P`` is a function of ``(x, c)``.

The rule
--------
**Every pass a solve measures starts from carried fields evaluated at its
floating iterate.**  For an iterate ``x`` the carried fields are set to
``c(x)``, obtained by evaluating the members as a *Jacobi* pass does, every
member reading the same frozen state, and keeping the carried fields; this
is repeated ``L`` times, ``L`` the length of the longest chain of carried
fields that feed one another (:func:`carried_nonfloat_chain`).  Both solvers
(``"ift"`` and ``"fori"``) and both sweeps then iterate the one map

    ``G(x) = P_x(x, c(x))``

and measure the residual ``r(x) = ||G(x) - x||`` in the group's norm.  The
state a solve returns holds ``x*`` and the non-floating fields ``P`` computes
at ``(x*, c(x*))``.

Why it is right
---------------
Write ``J`` for the Jacobi pass and ``J_c(x, c)`` for the carried fields it
computes.  "Field ``f`` feeds field ``g``" means: ``g``'s member reads ``f``
from the iterate.  That is decided from the edges, without looking inside a
node's ``update``, so it can only overstate what feeds what.

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
solution of ``c = J_c(x, c)``.

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
Where carried fields feed one another in a closed loop, no number of
evaluations yields "their value at ``x``" (Lemma 1 has no base), the fields
would have to be carried from pass to pass outside the residual, and a
solve could read ``converged=True`` while they still change.
``GraphManager.compile()`` refuses such a group
(:func:`_refuse_nonfloat_loops`).  The decision is structural, so a loop is
also seen where a node's ``update`` does not in fact use what it reads.  The
case to know: two members that exchange boundary fluxes both ways, one of
them holding a step counter or a PRNG key.  A flux is taken to use its
producer's whole state, so the counter is read as feeding itself round the
two fluxes, and the group is refused although the counter is computed from
the pre-step state alone.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp


class CarriedChain(NamedTuple):
    """The carried non-floating fields of a group and how they feed one another.

    Attributes
    ----------
    fields : tuple
        ``((node, field), ...)``, sorted: :func:`carried_nonfloat_fields`.
    depth : int
        The length of the longest chain of carried fields that feed one
        another: how many evaluations settle them (Lemma 1 of the module
        docstring).  Zero for a group with no carried field, and where
        ``loop`` is not ``None``.
    loop : tuple or None
        ``None``, or carried fields that feed one another in a closed loop,
        in order: each feeds the next and the last feeds the first.
    """

    fields: tuple
    depth: int
    loop: Optional[tuple]


def _non_floating(value) -> bool:
    """An integer, Boolean or PRNG-key leaf: what a group carries beside its floats.

    Not a complex one: the coupling solver carries none, and says so
    itself at the first step (``acceleration.float_image``).
    """
    return not jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact)


def _iterate_reads(plan) -> dict:
    """``{member: the (node, field) pairs it reads from the iterate}``.

    What one Jacobi evaluation of the member can depend on, among the
    iterate's state fields: the source field of each internal edge into it
    and the geometry a mapping on that edge reads from its source; and, for
    an edge whose source is not a state field (a boundary flux), the
    producer's whole state together with everything the producer reads in
    turn, since the pass computes the flux from the producer's state and
    from the producer's own boundary inputs.
    """
    direct: dict[str, set] = {nn: set() for nn in plan.members}
    flux_from: dict[str, set] = {nn: set() for nn in plan.members}
    for rec in plan.internal:
        (src, fld), tgt = rec.source, rec.target[0]
        if rec.read_from_state:
            direct[tgt].add((src, fld))
        else:
            flux_from[tgt].add(src)
        if rec.anchor is not None and rec.anchor[0] == "source":
            direct[tgt].add((src, rec.anchor[1]))

    def through(nn, seen):
        out = set(direct[nn])
        for src in sorted(flux_from[nn]):
            out.update((src, fld) for fld in plan.member_fields.get(src, ()))
            if src not in seen:
                out |= through(src, seen | {src})
        return out

    return {nn: through(nn, frozenset((nn,))) for nn in plan.members}


def carried_nonfloat_fields(plan, state) -> tuple:
    """The non-floating state fields a pass of the group reads from the iterate.

    Sorted ``(node, field)`` pairs; non-floating as *state* holds them (an
    integer, Boolean or PRNG-key leaf).  See the module docstring.
    """
    return carried_nonfloat_chain(plan, state).fields


def carried_nonfloat_chain(plan, state) -> CarriedChain:
    """The group's carried non-floating fields, their longest chain, or a loop.

    Field ``f`` feeds field ``g`` when ``g``'s member reads ``f`` from the
    iterate (:func:`_iterate_reads`): every field a member computes may
    depend on everything the member reads, so the relation is decided
    between members and overstates at most.
    """
    reads = _iterate_reads(plan)
    carried = sorted({
        (nn, fld) for pairs in reads.values() for nn, fld in pairs
        if fld in state.get(nn, {}) and _non_floating(state[nn][fld])})
    carried_set = set(carried)
    fed_by = {(nn, fld): sorted(carried_set & reads[nn]) for nn, fld in carried}

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
        for feeder in fed_by[field]:
            loop = visit(feeder)
            if loop is not None:
                return loop
            deepest = max(deepest, depth[feeder])
        on_path.pop()
        depth[field] = deepest + 1
        return None

    for field in carried:
        loop = visit(field)
        if loop is not None:
            # In feeding order (``visit`` walked from a field to its
            # feeders), starting at the loop's first field in sorted order.
            ring = list(reversed(loop))
            start = ring.index(min(ring))
            return CarriedChain(tuple(carried), 0, tuple(ring[start:] + ring[:start]))
    return CarriedChain(tuple(carried), max(depth.values(), default=0), None)


def nonfloat_loop_refusal(group_nodes, chain: CarriedChain, state) -> Optional[str]:
    """Why a group whose carried fields close a loop cannot be iterated, or ``None``."""
    if chain.loop is None:
        return None

    def named(field):
        nn, fld = field
        return f"{nn}.{fld} ({jnp.asarray(state[nn][fld]).dtype})"

    ring = " -> ".join(named(f) for f in (*chain.loop, chain.loop[0]))
    return (
        f"Coupling group {sorted(group_nodes)}: non-floating fields feed one another "
        f"in a closed loop of the group's own edges: {ring}. The group's iteration "
        "acts on floating-point fields and reads its convergence from them; an "
        "integer, Boolean or key field that an edge of the group carries is "
        "evaluated at each iterate from the fields that feed it, which needs the "
        "fields that feed one another to form chains that end. Around a closed "
        "loop they have no value at an iterate: they would be carried from pass "
        "to pass where the residual does not read them, and the group could "
        "report converged=True while they still change. Store one field of the "
        "loop in floating point (the residual then measures it), or compute it "
        "in a node outside the group, or set max_iterations=1 (one staggered "
        "pass, which iterates nothing). The loop is read from the edges: a member "
        "is taken to use every field it reads in every field it computes, and a "
        "boundary flux to use its producer's whole state and everything the "
        "producer reads."
    )


def _refuse_nonfloat_loops(groups, plans, state) -> None:
    """Refuse, at ``compile()``, a group whose carried non-floating fields close a loop.

    One ``ValueError`` naming every such group.  *plans* is ``{group key:
    InterfacePlan}`` as ``compile()`` builds it.  A group with
    ``max_iterations=1`` iterates nothing and is let through.
    """
    refusals = []
    for group in groups:
        if group.max_iterations <= 1:
            continue
        plan = plans["+".join(sorted(group.nodes))]
        refusal = nonfloat_loop_refusal(
            group.nodes, carried_nonfloat_chain(plan, state), state)
        if refusal is not None:
            refusals.append(refusal)
    if refusals:
        raise ValueError("\n".join(refusals))
