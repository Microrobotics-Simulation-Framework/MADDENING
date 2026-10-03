"""A deterministic t-way covering-array generator (IPOG-style greedy), with constraints.

A covering array of strength ``t`` over a set of knobs is a list of
configurations in which every combination of values of every ``t`` knobs
appears in at least one row.  Coupling's defects have repeatedly lived in
*interactions* between settings that no single-knob test reaches -- the
deprecated ``"fori"`` loop with Aitken on a 16-bit group, ``"fori"`` with
IQN-IMVJ dropping ``jacobian_reuse``, the interface norm with a flux edge
-- so ``test_differential_coupling_interactions.py`` runs its oracles over a
strength-3 array of the coupling group's knobs rather than over a hand-picked
table.

The generator is IPOG (Lei et al., "IPOG: a general strategy for t-way
software testing", 2007), greedy and deterministic: it grows the array one
knob at a time, first horizontally (each existing row takes the value of the
new knob that covers the most still-uncovered ``t``-tuples) and then
vertically (each tuple still uncovered goes into the first row whose
unassigned slots can hold it, or into a new row).  Unassigned slots left at
the end take the first value the constraints allow.  Nothing iterates over
a ``set``, so the array does not depend on ``PYTHONHASHSEED``.

Constraints are a predicate on *partial* assignments.  A ``t``-tuple the
predicate refuses is not required (it cannot occur in a valid
configuration), and no row is ever built that the predicate refuses.  The
predicate must be *extendable*: any partial assignment it accepts must be
completable to a full one it accepts.  That holds for implications of the
form "knob ``a`` takes value ``v`` => knob ``b`` takes value ``w``" whenever
each antecedent knob has a value that triggers nothing, which is the only
form the coupling knob space uses (:func:`coverage` would report a required
tuple left uncovered if it did not hold).

Nothing here is a test and nothing imports JAX.
"""

from __future__ import annotations

import itertools
from typing import Callable, Optional, Sequence

#: A partial assignment: ``{knob: value}``.
Assignment = dict


def _key(pairs) -> tuple:
    """A ``t``-tuple as ``((knob, value), ...)``, in the caller's knob order."""
    return tuple(pairs)


def _valid_tuples(knobs: Sequence[str], domains: dict, t: int,
                  valid: Callable[[Assignment], bool],
                  must_include: Optional[str] = None) -> dict:
    """Every valid ``t``-tuple over *knobs*, as an insertion-ordered ``{key: None}``."""
    out: dict = {}
    for combo in itertools.combinations(knobs, t):
        if must_include is not None and must_include not in combo:
            continue
        for values in itertools.product(*(domains[k] for k in combo)):
            if valid(dict(zip(combo, values))):
                out[_key(zip(combo, values))] = None
    return out


def _row_keys(row: Assignment, knobs: Sequence[str], t: int) -> list:
    """Every ``t``-tuple *row* covers, over the knobs it assigns."""
    have = [k for k in knobs if k in row]
    return [_key((k, row[k]) for k in combo) for combo in itertools.combinations(have, t)]


def ipog(domains: dict, t: int, valid: Callable[[Assignment], bool] = lambda a: True,
         order: Optional[Sequence[str]] = None) -> list[Assignment]:
    """A strength-*t* covering array over *domains* that respects *valid*.

    Parameters
    ----------
    domains : dict
        ``{knob: (value, ...)}``.  Values are compared with ``==`` and must
        be hashable.
    t : int
        The strength: every valid combination of values of any *t* knobs
        appears in some row.
    valid : callable
        A predicate on partial assignments (see the module docstring).
    order : sequence of str, optional
        The order knobs are added in.  Default: by decreasing domain size,
        ties in *domains* order -- the order IPOG grows best in.

    Returns
    -------
    list of dict
        Full assignments (keys in *domains* order), in the order the
        generator built them.  The result depends only on the arguments.
    """
    knobs = list(order) if order is not None else sorted(
        domains, key=lambda k: (-len(domains[k]), list(domains).index(k)))
    if len(knobs) < t:
        raise ValueError(f"need at least t={t} knobs, got {len(knobs)}")
    first = knobs[:t]
    rows: list[Assignment] = [dict(zip(first, values))
                              for values in itertools.product(*(domains[k] for k in first))
                              if valid(dict(zip(first, values)))]
    for i in range(t, len(knobs)):
        new = knobs[i]
        seen = knobs[: i + 1]
        pending = _valid_tuples(seen, domains, t, valid, must_include=new)
        # Horizontal growth: each row takes the value covering the most tuples.
        for row in rows:
            assigned = [k for k in seen[:-1] if k in row]
            best, best_hits = None, -1
            for v in domains[new]:
                trial = {**row, new: v}
                if not valid(trial):
                    continue
                hits = 0
                for combo in itertools.combinations(assigned, t - 1):
                    pairs = sorted(((k, trial[k]) for k in (*combo, new)),
                                   key=lambda kv: seen.index(kv[0]))
                    hits += _key(pairs) in pending
                if hits > best_hits:
                    best, best_hits = v, hits
            if best_hits < 0:
                # No value of *new* fits this row as it stands; vertical
                # growth or the final fill completes it (the predicate is
                # extendable).
                continue
            row[new] = best
            for key in _row_keys(row, seen, t):
                pending.pop(key, None)
        # Vertical growth: every tuple still uncovered goes into the first row
        # that can hold it, else into a row of its own.
        for key in list(pending):
            if key not in pending:
                continue
            for row in rows:
                merged = dict(row)
                clash = False
                for k, v in key:
                    if k in merged and merged[k] != v:
                        clash = True
                        break
                    merged[k] = v
                if not clash and valid(merged):
                    row.update(key)
                    break
            else:
                row = dict(key)
                rows.append(row)
            for covered in _row_keys(row, seen, t):
                pending.pop(covered, None)
    full = []
    for row in rows:
        row = dict(row)
        for k in knobs:
            if k in row:
                continue
            for v in domains[k]:
                if valid({**row, k: v}):
                    row[k] = v
                    break
            else:  # pragma: no cover - an unextendable predicate
                raise ValueError(f"no value of {k!r} completes row {row}")
        full.append({k: row[k] for k in domains})
    return full


def coverage(rows: Sequence[Assignment], domains: dict, t: int,
             valid: Callable[[Assignment], bool] = lambda a: True) -> tuple[int, int, list]:
    """``(covered, required, missing)``: how many valid *t*-tuples *rows* cover.

    ``required`` counts the valid *t*-tuples over every knob of *domains*;
    ``missing`` lists the ones no row covers, as ``{knob: value}`` dicts.
    """
    knobs = list(domains)
    required = _valid_tuples(knobs, domains, t, valid)
    covered: dict = {}
    for row in rows:
        for key in _row_keys(row, knobs, t):
            covered[key] = None
    missing = [dict(key) for key in required if key not in covered]
    return len(required) - len(missing), len(required), missing


def one_way_slice(rows: Sequence[Assignment],
                  keep: Callable[[Assignment], bool] = lambda a: True) -> list[int]:
    """Indices of a small subset of *rows* in which every value of every knob appears.

    Greedy and deterministic: repeatedly take the eligible row (``keep``)
    covering the most still-unseen ``(knob, value)`` pairs, the earliest on
    a tie.  Values no eligible row holds are left out.  This is the
    per-push slice of a covering array: strength one on every push,
    strength *t* in the slow lane.
    """
    eligible = [i for i, row in enumerate(rows) if keep(row)]
    unseen: dict = {}
    for i in eligible:
        for pair in rows[i].items():
            unseen[pair] = None
    chosen: list[int] = []
    while unseen:
        best = max(eligible, key=lambda i: (sum(p in unseen for p in rows[i].items()), -i))
        gained = [p for p in rows[best].items() if p in unseen]
        if not gained:  # pragma: no cover - unseen is built from eligible rows
            break
        chosen.append(best)
        for p in gained:
            del unseen[p]
    return sorted(chosen)
