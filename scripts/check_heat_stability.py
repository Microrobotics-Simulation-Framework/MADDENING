#!/usr/bin/env python3
"""Every ``HeatNode(...)`` in the repository must be constructible.

Since 0.4.0 ``HeatNode.__init__`` refuses a configuration whose Fourier
number ``dt * alpha / dx^2`` exceeds its stencil's stability limit
(MADD-ANO-009).  That guard is correct and it is load-bearing -- the
explicit update diverges to NaN above the limit rather than losing
accuracy gracefully -- but it turns a formerly silent defect into a
``ValueError`` at construction, so a caller that has always been
unstable now fails loudly.

This gate finds those callers before CI does.  It exists because one
got through: ``tests/core/test_compile_cache.py`` builds a 257-cell
4th-order rod at ``dt=1e-4, alpha=0.1``, a Fourier number of 0.66,
inside a ``textwrap.dedent`` string passed to ``python -c``.  It had
been compiling a graph that would go to NaN on every step and passing
regardless, because it only ever measured compile *time*.  Two greps
missed it: one because ``HeatNode(`` inside a string literal is not
code to a regex scanning for calls, and a cruder regex sweep also
produced three false positives on expressions like
``thermal_diffusivity=1.0/n_cells**2``.

So this walks the AST, and:

* recurses into string literals that themselves contain source, after
  ``textwrap.dedent``, which is what the missed case needed;
* evaluates only genuine literals, which is what the false positives
  needed -- a computed argument is reported as unchecked, never as a
  pass;
* reads the limits from ``MAX_FOURIER_NUMBER`` and the parameter
  defaults from ``HeatNode.__init__``'s own signature, so the gate
  cannot drift from the code it checks.

Scope is the repository's *tracked* Python files (``git ls-files``),
so untracked scratch under ``benchmarks/results/`` is not scanned.

Every construction lands in exactly one of four counts, and only the
first is "verified":

* **verified** -- every argument that decides the Fourier number, or
  whether the constructor accepts the rod, was read as a literal (or is
  the constructor's default); the rod is within its stencil's limit; and
  ``HeatNode.__init__`` itself, called with those literals, builds it
  (:func:`constructor_probe`);
* **refused** -- the rod is past its limit, or the constructor raises on
  it (``n_cells=4, stencil_order=4``, a ``grid_points`` of the wrong
  length, a non-positive timestep or length, ``stencil_order=3``, an
  unknown keyword): the gate fails.  Until audit_040_p4_4 (L5) the gate
  mirrored only the Fourier test, so the first two were counted as
  verified and the rest as not evaluated;
* **refused as expected** -- the same, inside a ``with pytest.raises``
  block: a test asserting the refusal.  Counted apart; never verified;
* **not evaluated** -- something that decides the answer is not a
  literal: a computed argument, a ``**mapping`` held in a variable, a
  ``*args`` splat, a keyword given twice, or a computed ``grid_points``.
  These are counted and reported by reason (``--list-unevaluated`` prints
  each one), never folded into the verified count, and a scope in which
  *nothing* could be evaluated fails.

A rod given a literal non-uniform grid (``grid_points=[...]``) is judged
by the constructor's non-uniform criterion.  That is ``dt * alpha /
min(h_L * h_R)`` against ``MAX_FOURIER_NUMBER[2]``, with the spacing read
through the node's own ``_nonuniform_fourier_spacing``.  A grid that is not
strictly increasing is refused, as ``HeatNode.__init__`` refuses it.  Until
0.4.0 the constructor skipped non-uniform rods, and so did this gate.  A rod
at ten times its limit built without a word (audit_040_p4_2,
release-record, M3).  ``grid_points=None`` is the default spelled out, a
uniform rod, and is judged like one; it used to be skipped with every
other ``grid_points=`` (audit_040_p4_1, H6).

A literal ``**{"thermal_diffusivity": 1e3}`` or ``**dict(...)`` splat is
read like the keywords it spells.  It used to be dropped silently, so the
rod was judged on the constructor's *defaults* and counted as verified
while ``HeatNode.__init__`` refused it (audit_040_phase3_wave_d, H5).

A class defined in the same source as a subclass of ``HeatNode`` (or of
such a subclass, to any depth) is a ``HeatNode`` to this gate: a
construction through it runs ``HeatNode.__init__`` and its guard, so it
is judged the same way.  One whose chain defines its own ``__init__`` or
``__new__`` could map its arguments anywhere, and is reported as not
evaluated.  ``class Rod(HeatNode): pass; Rod(...)`` used to be invisible
(audit_040_phase3_confirm, release-record).

Limits, stated rather than implied:

* a subclass *imported from another module* is not recognised -- the gate
  reads one source at a time and does not resolve imports, so a rod built
  through such a class is not seen at all (it is neither verified nor
  counted as unevaluated);
* a class obtained at run time (``type(...)``, a factory, a registry
  lookup) is likewise invisible;
* a call is matched by name, so an unrelated class that happens to be
  called ``HeatNode`` -- or a local subclass name reused for something
  else -- is judged as a rod.  The failure that can cause is a spurious
  refusal, never a spurious pass.

The allowlist exempts named constructions, not files.  The one test file
on it may plant unstable rods only inside string literals (embedded
source it writes out as a fixture); a real construction in its code is
checked like any other.  The archived probes are exempt at the listed
lines only.  The whole-file exemption used to let a real unstable rod
appended to the test file pass.  What the gate cannot tell apart is a
string literal the test file writes out as a fixture and one it
*executes* (``python -c``, the way the rod this gate exists for hid): in
that one file, both are exempt, so a test there must not run a rod from
a string.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from maddening.nodes.heat import (  # noqa: E402
    MAX_FOURIER_NUMBER,
    HeatNode,
    _nonuniform_fourier_spacing,
)

def _positional_order() -> list:
    """``HeatNode.__init__``'s parameters, in order, so the gate cannot drift.

    This list used to stop at ``thermal_diffusivity``, five names in.
    ``stencil_order`` is the *seventh* parameter, so a rod that passes all
    seven arguments positionally had its order silently default to 2 and was
    judged against a limit of 0.5 instead of 0.3125 --
    the gate passed a construction ``__init__`` refuses.  Deriving the order
    from the signature is the same defence ``_defaults`` already uses for the
    values (audit_040_r2/gates, finding G4).
    """
    return [
        name for name, param in inspect.signature(HeatNode.__init__).parameters.items()
        if name != "self"
        and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    ]


#: Constructor parameters this gate needs, in positional order.
_POSITIONAL = _positional_order()

#: Files whose unstable constructions are deliberate, with the reason.
#:
#: Only one kind of file qualifies: this gate's own test, which has to
#: contain the defect in order to prove the gate catches it.  The entry
#: is kept honest by
#: ``TestHeatStabilityGate.test_the_allowlisted_file_still_plants_a_defect``
#: -- if the planted construction ever goes away, the exemption has to
#: go with it rather than sitting there quietly widening the scope.
#: Each entry is ``path: (reason, scope)``.  ``scope`` is
#: :data:`EMBEDDED_ONLY` -- only constructions inside string literals the
#: file writes out as source are exempt -- or a frozenset of the line
#: numbers of the exempt constructions.  Anything else in the file is
#: checked.  ``test_the_allowlisted_file_still_plants_a_defect`` holds each
#: scope to at least one unstable rod, and each listed line to one.
EMBEDDED_ONLY = "embedded source only"

_ALLOWED_UNSTABLE = {
    "tests/compliance/test_gate_scripts.py": (
        "plants unstable rods as fixtures, as source text it writes to a "
        "temporary directory, to prove this gate rejects them",
        EMBEDDED_ONLY),
    "benchmarks/results/audit_040_r2/numerics/repro/gate_probes/"
    "e_genuinely_unstable.py": (
        "archived audit probe: the positive control for this gate, a rod the "
        "guard must refuse.  Committed as round-2 evidence, so the gate now "
        "scans it",
        frozenset({2})),
    "benchmarks/results/audit_040_r2/numerics/repro/gate_probes/"
    "c_positional_stencil.py": (
        "archived audit probe: a 4th-order rod whose stencil_order is passed "
        "POSITIONALLY.  It was invisible while _POSITIONAL stopped at five "
        "names; widening it to the full signature made this probe the first "
        "thing the gate caught",
        frozenset({4})),
    "benchmarks/results/audit_040_r2/numerics/repro/gate_probes/"
    "d_attribute_call.py": (
        "archived audit probe: the same unstable rod spelled as the "
        "attribute heat.HeatNode.  It was invisible while the gate matched "
        "ast.Name only",
        frozenset({3})),
    "benchmarks/results/audit_040_r2/numerics/repro/gate_probes/"
    "a_nonpositive.py": (
        "archived audit probe: a rod with timestep=0.0, which the gate "
        "counted as verified until round 2 and then reported as merely not "
        "evaluated.  The constructor refuses it, and since the gate asks the "
        "constructor, so does the gate",
        frozenset({4})),
    "benchmarks/results/audit_040_r2/numerics/repro/gate_probes/"
    "b_unknown_order.py": (
        "archived audit probe: stencil_order=3, reported as not evaluated "
        "until the gate asked the constructor, which refuses it",
        frozenset({3})),
    "benchmarks/results/audit_040_final/params-io/repro/r11_fmi_desc.py": (
        "archived audit probe of the FMI description: grid_points=4 against "
        "the default 10 cells, a construction the constructor refuses on "
        "the grid's length; the probe records what the description did with "
        "it before that check",
        frozenset({48})),
}

#: How ``scan_source`` marks the origin of a construction found inside a
#: string literal.
_EMBEDDED = " [embedded source]"


def _is_allowed(origin: str, lineno: int) -> bool:
    """Is this unstable construction one the allowlist names?"""
    base, embedded = origin.split(_EMBEDDED, 1)[0], _EMBEDDED in origin
    entry = _ALLOWED_UNSTABLE.get(base)
    if entry is None:
        return False
    scope = entry[1]
    if scope == EMBEDDED_ONLY:
        return embedded
    return not embedded and lineno in scope

#: A ceiling on the named exemptions, enforced by :func:`main` before it
#: scans anything: an ``EMBEDDED_ONLY`` entry counts once, a line-pinned
#: entry once per line.  It was declared and never read, so fourteen
#: entries passed (audit_040_p4_4, L5).  One test file has to plant
#: defects to prove the gate catches them, and the archived audit probes
#: are the positive controls for the holes this gate has had; three of
#: those joined when the gate began asking the constructor, whose
#: refusals they plant.  Raising this is a reviewed decision, made beside
#: the entry that needs it -- an unstable rod in live code is fixed, not
#: allowlisted.
_MAX_ALLOWED_UNSTABLE = 7


def allowlist_ceiling_error():
    """Why the allowlist is over :data:`_MAX_ALLOWED_UNSTABLE`, or ``None``."""
    named = sum(1 if scope == EMBEDDED_ONLY else len(scope)
                for _reason, scope in _ALLOWED_UNSTABLE.values())
    if named <= _MAX_ALLOWED_UNSTABLE:
        return None
    return (f"_ALLOWED_UNSTABLE names {named} exempt construction(s) across "
            f"{len(_ALLOWED_UNSTABLE)} file(s), over the ceiling of "
            f"{_MAX_ALLOWED_UNSTABLE} (_MAX_ALLOWED_UNSTABLE).  Each one is a "
            f"rod this gate stops refusing: fix the rod rather than "
            f"allowlisting it, or raise the ceiling in a reviewed change "
            f"that says why")


def _defaults() -> dict:
    """``HeatNode.__init__``'s own defaults, so the gate cannot drift."""
    sig = inspect.signature(HeatNode.__init__)
    out = {}
    for key in ("n_cells", "length", "thermal_diffusivity", "stencil_order"):
        default = sig.parameters[key].default
        out[key] = default if isinstance(default, (int, float)) else None
    return out


def _number(node):
    """The literal number ``node`` evaluates to, or ``None``.

    ``None`` means "not a literal", never "zero": a computed argument is
    unchecked, and the caller must not read that as a pass.
    """
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _is_literal(node) -> bool:
    """Does ``node`` evaluate to a literal (``ast.literal_eval``)?"""
    try:
        ast.literal_eval(node)
    except Exception:
        return False
    return True


def _parse(src: str):
    """Parse ``src``, dedenting first if that is what it takes."""
    for candidate in (src, textwrap.dedent(src)):
        try:
            return ast.parse(candidate)
        except SyntaxError:
            continue
    return None


def _local_aliases(tree) -> set:
    """Every name in this source that is bound to ``HeatNode``.

    ``from maddening.nodes.heat import HeatNode as Rod`` binds a different
    ``ast.Name``, and matching only the literal identifier made the rod
    invisible.  An attribute-spelled call -- ``heat.HeatNode`` -- is handled
    separately: an ``ast.Attribute`` carries ``.attr``, not ``.id``.
    """
    names = {"HeatNode"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "HeatNode" and alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.Assign):
            # ``Rod = HeatNode`` -- the rebinding an import alias avoids.
            if isinstance(node.value, ast.Name) and node.value.id in names:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif (isinstance(node.value, ast.Attribute)
                  and node.value.attr == "HeatNode"):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
    return names


def _local_subclasses(tree, aliases) -> dict:
    """Classes defined in this source that are ``HeatNode`` subclasses.

    Maps each name to ``True`` when it (or a local class between it and
    ``HeatNode``) defines ``__init__`` or ``__new__``, so the constructor
    arguments may not reach ``HeatNode.__init__`` as written.  Iterated to
    a fixed point, so a subclass of a subclass is found in any order.
    """
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    found: dict = {}
    changed = True
    while changed:
        changed = False
        for cls in classes:
            if cls.name in found or cls.name in aliases:
                continue
            parents = []
            for base in cls.bases:
                name = getattr(base, "id", None) or getattr(base, "attr", None)
                if name in aliases or name == "HeatNode":
                    parents.append(False)
                elif name in found:
                    parents.append(found[name])
            if not parents:
                continue
            own = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and n.name in ("__init__", "__new__") for n in cls.body)
            found[cls.name] = own or any(parents)
            changed = True
    return found


def _is_heat_node_call(node, aliases, subclasses=None) -> bool:
    """Is this call constructing a ``HeatNode``, however it is spelled?

    Matching ``ast.Attribute`` can in principle pick up an unrelated
    ``something.HeatNode`` -- which is harmless here, because the worst it
    can do is report a construction as unchecked.
    """
    if not isinstance(node, ast.Call):
        return False
    if getattr(node.func, "attr", None) == "HeatNode":
        return True
    name = getattr(node.func, "id", None)
    return name in aliases or name in (subclasses or {})


def _literal_splat(value):
    """``{name: node}`` for a literal ``**{...}`` / ``**dict(...)``, else ``None``.

    ``None`` means "a mapping this gate cannot read", which the caller must
    treat as unevaluable -- never as "no extra arguments".
    """
    if isinstance(value, ast.Dict):
        out = {}
        for key, item in zip(value.keys, value.values):
            # ``key is None`` is a nested ``**other`` inside the literal.
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                return None
            if key.value in out:
                return None
            out[key.value] = item
        return out
    if (isinstance(value, ast.Call) and getattr(value.func, "id", None) == "dict"
            and not value.args and all(kw.arg for kw in value.keywords)):
        return {kw.arg: kw.value for kw in value.keywords}
    return None


def _call_arguments(node):
    """The constructor arguments by parameter name, or ``(None, reason)``.

    Returns ``(args, None)`` when every argument's *name* is known, and
    ``(None, reason)`` when a splat hides which parameter receives what.
    """
    for positional in node.args:
        if isinstance(positional, ast.Starred):
            return None, ("a *args splat hides which parameter receives each "
                          "positional argument")
    args = {}
    for i, positional in enumerate(node.args):
        if i < len(_POSITIONAL):
            args[_POSITIONAL[i]] = positional
    for kw in node.keywords:
        if kw.arg is not None:
            items = {kw.arg: kw.value}
        else:
            items = _literal_splat(kw.value)
            if items is None:
                return None, ("a **mapping the gate cannot read supplies "
                              "some of the arguments")
        for name, value in items.items():
            if name in args:
                return None, (f"{name!r} is given twice; the constructor "
                              f"would raise TypeError")
            args[name] = value
    return args, None


#: Parameters the gate passes through to the constructor probe only when
#: they are literal, and otherwise leaves at their defaults: none of them
#: decides whether ``HeatNode.__init__`` accepts the rod or what its Fourier
#: number is.  ``name`` is required, so a computed one is replaced.
_NOT_DECIDING = ("name", "initial_temperature", "geometry_source")

#: The name a probe construction gets when the call's own is computed.
_PROBE_NAME = "check_heat_stability_probe"

#: Literal rods larger than this are not built to ask the constructor: the
#: probe allocates the rod's grid.  None in the repository comes close.
_MAX_PROBE_CELLS = 1_000_000


def _signature_parameters():
    """``(every parameter, the required ones)`` of ``HeatNode.__init__``."""
    params = [(name, p) for name, p in
              inspect.signature(HeatNode.__init__).parameters.items()
              if name != "self"]
    return ([name for name, _p in params],
            [name for name, p in params if p.default is p.empty
             and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                            p.KEYWORD_ONLY)])


_PARAMETERS, _REQUIRED = _signature_parameters()


def _structural_refusal(node, args):
    """The ``TypeError`` the call raises before the constructor runs, or ``None``."""
    if len(node.args) > len(_POSITIONAL):
        return (f"{len(node.args)} positional arguments, but HeatNode.__init__ "
                f"takes at most {len(_POSITIONAL)}; the call raises TypeError")
    unknown = sorted(set(args) - set(_PARAMETERS))
    if unknown:
        return (f"HeatNode.__init__ has no parameter {unknown[0]!r}; the call "
                f"raises TypeError")
    missing = [name for name in _REQUIRED if name not in args]
    if missing:
        return (f"no {missing[0]!r} argument, which HeatNode.__init__ "
                f"requires; the call raises TypeError")
    return None


def constructor_probe(args):
    """Ask ``HeatNode.__init__`` itself: ``(outcome, detail)``.

    ``outcome`` is ``"refused"`` (``detail`` is what it raised),
    ``"accepted"``, or ``"unprobed"`` (``detail`` says why not: an argument
    that could decide the answer is computed, or the rod is too large to
    build).  Every argument except :data:`_NOT_DECIDING` must be a literal.

    This is what makes "verified" mean "the constructor builds it".  The
    gate used to mirror only the Fourier test, so ``n_cells=4,
    stencil_order=4`` and a ``grid_points`` of the wrong length -- both
    refused before the Fourier test is reached -- were counted as verified,
    and literal values the constructor refuses outright (a non-positive
    timestep or length, ``stencil_order=3``) were reported as merely not
    evaluated (audit_040_p4_4, L5).  Asking the constructor keeps the gate
    in step with every check it has, including ones added later.
    """
    import warnings

    kwargs = {}
    for name, value in args.items():
        if _is_literal(value):
            kwargs[name] = ast.literal_eval(value)
        elif name == "name":
            kwargs[name] = _PROBE_NAME
        elif name not in _NOT_DECIDING:
            return "unprobed", f"{name} is computed, not literal"
    n_cells = kwargs.get("n_cells", 0)
    points = kwargs.get("grid_points")
    size = len(points) if isinstance(points, (list, tuple)) else 0
    if (isinstance(n_cells, int) and n_cells > _MAX_PROBE_CELLS) or (
            size > _MAX_PROBE_CELLS):
        return "unprobed", (f"a rod of more than {_MAX_PROBE_CELLS} cells is "
                            f"not built to ask the constructor")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            HeatNode(**kwargs)
    except Exception as exc:  # noqa: BLE001 -- any raise is a refusal
        return "refused", f"{type(exc).__name__}: {exc}"
    return "accepted", None


def _expected_to_raise(tree) -> set:
    """``id``s of the calls inside a ``with pytest.raises(...)`` body.

    A construction a test expects to raise is the constructor's refusal
    under test (``tests/core/test_spatial_accuracy.py`` asserts
    ``stencil_order=3`` is refused), not a rod anybody builds.  It is
    counted apart, and is never verified.  ``self.assertRaises`` and a
    bare ``raises`` are read the same way.
    """
    ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        if not any(isinstance(item.context_expr, ast.Call)
                   and (getattr(item.context_expr.func, "attr", None)
                        or getattr(item.context_expr.func, "id", None))
                   in ("raises", "assertRaises", "assertRaisesRegex")
                   for item in node.items):
            continue
        for stmt in node.body:
            ids.update(id(sub) for sub in ast.walk(stmt)
                       if isinstance(sub, ast.Call))
    return ids


def _judge_grid(points, args, defaults):
    """``(outcome, detail)`` for a rod on a literal ``grid_points``.

    The non-uniform stencil is 2nd order whatever ``stencil_order`` says, and
    ``length`` and ``n_cells`` do not enter its Fourier number.  So only
    ``timestep``, ``thermal_diffusivity`` and the points are read here; the
    constructor probe judges the rest.  ``outcome`` is ``"unstable"``,
    ``"within"`` (the Fourier test passes), ``"undefined"`` (no Fourier
    number: the probe decides) or ``"unchecked"``.
    """
    try:
        coords = [float(v) for v in points]
    except (TypeError, ValueError):
        return "undefined", ("grid_points is a literal but not a sequence of "
                             "numbers")
    if len(coords) < 2:
        return "undefined", ("grid_points has fewer than two points, so no "
                             "spacing is defined")
    values = {}
    for key in ("timestep", "thermal_diffusivity"):
        values[key] = _number(args[key]) if key in args else defaults.get(key)
    if any(v is None for v in values.values()):
        return "unchecked", "a constructor argument is computed, not literal"
    dt, alpha = values["timestep"], values["thermal_diffusivity"]
    spacing = _nonuniform_fourier_spacing(coords)
    if spacing is None:
        return "unstable", (f"grid_points {coords!r} is not strictly "
                            f"increasing; HeatNode.__init__ raises ValueError")
    if dt <= 0 or alpha < 0:
        return "undefined", (f"a non-positive constructor argument "
                             f"(dt={dt!r}, alpha={alpha!r})")
    limit = MAX_FOURIER_NUMBER[2]
    fourier = dt * alpha / spacing
    if fourier > limit:
        return "unstable", (
            f"Fourier number {fourier:.4g} (dt*alpha/min(h_left*h_right) on "
            f"the non-uniform grid) exceeds the {limit:g} limit of the "
            f"2nd-order variable-spacing stencil (dt={dt!r}, "
            f"alpha={alpha!r}, min(h_left*h_right)={spacing:.6g}); "
            f"HeatNode.__init__ raises ValueError. "
            f"Use timestep <= {limit * spacing / alpha:.6g}, a wider finest "
            f"spacing, or a smaller thermal_diffusivity")
    return "within", None


def _judge_uniform(args, defaults):
    """``(outcome, detail)`` for a uniform rod; outcomes as :func:`_judge_grid`."""
    values = {}
    for key in ("timestep", "n_cells", "length",
                "thermal_diffusivity", "stencil_order"):
        values[key] = _number(args[key]) if key in args else defaults.get(key)
    if any(v is None for v in values.values()):
        return "unchecked", "a constructor argument is computed, not literal"
    dt, n_cells = values["timestep"], values["n_cells"]
    length, alpha = values["length"], values["thermal_diffusivity"]
    order = values["stencil_order"]
    # A zero diffusivity is a rod that does not conduct: the constructor
    # builds it, and its Fourier number is 0.  It used to be "not
    # evaluated" with the negative values.
    if min(dt, n_cells, length) <= 0 or alpha < 0:
        return "undefined", (f"a non-positive constructor argument "
                             f"(dt={dt!r}, n_cells={n_cells!r}, "
                             f"length={length!r}, alpha={alpha!r})")
    limit = MAX_FOURIER_NUMBER.get(order)
    if limit is None:
        return "undefined", (f"stencil_order={order!r} has no entry in "
                             f"MAX_FOURIER_NUMBER ({sorted(MAX_FOURIER_NUMBER)})")
    dx = length / n_cells
    fourier = dt * alpha / (dx * dx)
    if fourier > limit:
        return "unstable", (
            f"Fourier number {fourier:.4g} exceeds the {limit:g} limit of "
            f"the order-{order} stencil "
            f"(dt={dt!r}, alpha={alpha!r}, dx=length/n_cells={dx:.6g}); "
            f"HeatNode.__init__ raises ValueError. "
            f"Use timestep <= {limit * dx * dx / alpha:.6g}, more cells, "
            f"or a smaller thermal_diffusivity")
    return "within", None


def scan_source(src, origin, defaults, unstable, unchecked, seen,
                expected=None):
    """Walk one source string, recursing into embedded source.

    Each construction lands in exactly one list: ``seen`` (verified),
    ``unstable`` (refused: past its limit, or refused by the constructor --
    also recorded in ``seen``, which ``main`` reconciles with the
    allowlist), ``unchecked`` (not evaluated, with the reason), or
    ``expected`` (refused inside a ``pytest.raises`` block, as its test
    expects).
    """
    if expected is None:
        expected = []
    tree = _parse(src)
    if tree is None:
        if "HeatNode(" in src:
            unchecked.append((origin, 0, "unparseable source mentioning HeatNode"))
        return

    aliases = _local_aliases(tree)
    subclasses = _local_subclasses(tree, aliases)
    raising = _expected_to_raise(tree)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and "HeatNode" in node.value and "(" in node.value):
            scan_source(node.value, f"{origin}{_EMBEDDED}",
                        defaults, unstable, unchecked, seen, expected)
            continue
        if not _is_heat_node_call(node, aliases, subclasses):
            continue
        if subclasses.get(getattr(node.func, "id", None)):
            unchecked.append((
                origin, node.lineno,
                f"a HeatNode subclass ({node.func.id}) that defines its own "
                f"__init__ or __new__, so its arguments may not reach "
                f"HeatNode.__init__ as written"))
            continue

        def refuse(why, node=node):
            if id(node) in raising:
                expected.append((origin, node.lineno, why))
            else:
                # In ``seen`` too, as every refused rod is: ``main`` takes
                # an allowlisted one back out of it.
                seen.append((origin, node.lineno))
                unstable.append((origin, node.lineno, why))

        # A ``**`` splat used to be dropped (``if kw.arg``), so the rod was
        # judged on the defaults it overrode and counted as verified.
        args, why_not = _call_arguments(node)
        if args is None:
            unchecked.append((origin, node.lineno, why_not))
            continue
        structural = _structural_refusal(node, args)
        if structural:
            refuse(structural)
            continue

        # An explicit grid is a non-uniform rod, judged on the constructor's
        # non-uniform criterion -- but only a grid that is actually given.
        # Any ``grid_points=`` used to skip the rod, so ``grid_points=None``
        # (the default, spelled out) hid an unstable uniform rod that
        # ``HeatNode.__init__`` refuses (audit_040_p4_1, H6).
        grid = args.get("grid_points")
        if grid is not None and not (isinstance(grid, ast.Constant)
                                     and grid.value is None):
            if not _is_literal(grid):
                unchecked.append((
                    origin, node.lineno,
                    "grid_points is computed, so the grid the Fourier number "
                    "depends on cannot be read"))
                continue
            outcome, detail = _judge_grid(ast.literal_eval(grid), args, defaults)
        else:
            outcome, detail = _judge_uniform(args, defaults)

        # ``seen`` is appended *after* every way out of this block, because
        # it is the count the summary line calls "verified".  It used to be
        # appended before the non-positive and unknown-order exits, so the
        # headline counted rods nobody had evaluated: 132 reported against
        # 131 (audit_040_r2/gates, finding G3).
        if outcome == "unchecked":
            unchecked.append((origin, node.lineno, detail))
            continue
        if outcome == "unstable":
            refuse(detail)
            continue
        # "within" or "undefined": the constructor has the last word.
        verdict, why = constructor_probe(args)
        if verdict == "refused":
            refuse(f"HeatNode.__init__ refuses it: {why}")
        elif verdict == "unprobed":
            unchecked.append((origin, node.lineno,
                              f"{detail + '; ' if detail else ''}{why}, so "
                              f"whether the constructor accepts it was NOT "
                              f"checked"))
        elif outcome == "undefined":
            unchecked.append((origin, node.lineno,
                              f"{detail}; the constructor accepts it, but no "
                              f"Fourier number is defined, so this was NOT "
                              f"checked"))
        else:
            seen.append((origin, node.lineno))


def tracked_python_files(root: Path):
    """Tracked ``.py`` files, so untracked scratch is out of scope."""
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "*.py"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return [root / rel for rel in out]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "roots", nargs="*", default=None,
        help="directories to scan (default: the repository's tracked .py files)",
    )
    parser.add_argument(
        "--list-unevaluated", action="store_true",
        help="print every construction that could not be evaluated, with "
             "the reason",
    )
    args = parser.parse_args(argv)

    if args.roots:
        paths = [p for root in args.roots
                 for p in sorted(Path(root).rglob("*.py"))]
        scanned = ", ".join(args.roots)
    else:
        paths = tracked_python_files(REPO_ROOT)
        scanned = "the repository's tracked .py files"

    ceiling = allowlist_ceiling_error()
    if ceiling:
        print(f"FAIL: {ceiling}")
        return 1

    defaults = _defaults()
    unstable, unchecked, seen, expected = [], [], [], []
    for path in paths:
        try:
            src = path.read_text(errors="replace")
        except OSError:
            continue
        if "HeatNode" not in src:
            continue
        try:
            origin = str(path.relative_to(REPO_ROOT))
        except ValueError:
            origin = str(path)
        scan_source(src, origin, defaults, unstable, unchecked, seen, expected)

    # The allowlist names constructions, not files: everything else in an
    # allowlisted file was scanned above and is judged like any other rod.
    # An exempt rod is past its limit, so it is not "verified" either.
    allowed = [(o, n) for o, n, _why in unstable if _is_allowed(o, n)]
    unstable = [u for u in unstable if not _is_allowed(u[0], u[1])]
    # One ``seen`` entry per exempt rod: embedded snippets share an origin
    # and number their own lines, so (origin, line) is not unique.
    for key in allowed:
        seen.remove(key)

    if unstable:
        print(f"FAIL: {len(unstable)} HeatNode construction(s) the constructor "
              f"refuses (its stability guard or its other checks):")
        for origin, lineno, why in unstable:
            print(f"  {origin}:{lineno}: {why}")
        print("\nFix: give the construction a stable timestep and arguments "
              "the constructor accepts.  If the test does not care about the "
              "physics, it still cannot build a rod that diverges -- see "
              "MADD-ANO-009.  A test that asserts the refusal does it inside "
              "pytest.raises, which this gate reads.")
        return 1

    if not seen and not unchecked:
        print(
            f"FAIL: 0 HeatNode construction(s) found in {scanned}.\n"
            "A gate that verifies nothing cannot fail.  Either the scan "
            "scope is wrong or the call is spelled in a way this gate does "
            "not recognise; fix the scope rather than trusting the OK.",
            file=sys.stderr,
        )
        return 1

    if not seen:
        print(
            f"FAIL: {len(unchecked)} HeatNode construction(s) found in "
            f"{scanned}, and not one of them could be evaluated statically.\n"
            "A gate that verifies nothing cannot fail.  Fix the scope rather "
            "than trusting the OK.",
            file=sys.stderr,
        )
        for origin, lineno, why in unchecked[:20]:
            print(f"  {origin}:{lineno}: {why}", file=sys.stderr)
        return 1

    note = ""
    if unchecked:
        # Says what it means: these were not evaluated, for any of the
        # reasons recorded against them (a computed argument, a non-positive
        # one, an unparseable embedded snippet, or a stencil order with no
        # stability limit).  They are NOT part of the verified count.
        note = (f"; {len(unchecked)} further construction(s) could not be "
                f"evaluated statically and were NOT checked")
        by_reason = {}
        for _origin, _lineno, why in unchecked:
            key = _reason_key(why)
            by_reason[key] = by_reason.get(key, 0) + 1
        print("not evaluated, by reason:")
        for key, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4d}  {key}")
        if args.list_unevaluated:
            for origin, lineno, why in unchecked:
                print(f"  {origin}:{lineno}: {why}")
    exempt = (f"; {len(allowed)} deliberately unstable construction(s) "
              f"exempt by name (_ALLOWED_UNSTABLE)" if allowed else "")
    raised = (f"; {len(expected)} construction(s) refused inside "
              f"pytest.raises, as their test expects" if expected else "")
    print(f"OK: {len(seen)} HeatNode construction(s) verified: the "
          f"constructor builds each, within its stencil's stability "
          f"limit{note}{exempt}{raised}")
    return 0


def _reason_key(why: str) -> str:
    """The reason with its per-construction values dropped, for grouping."""
    return why.split(" (", 1)[0].split(";", 1)[0]


if __name__ == "__main__":
    sys.exit(main())
