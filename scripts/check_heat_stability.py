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

Every construction lands in exactly one of three counts, and only the
first is "verified":

* **verified** -- every argument that decides the Fourier number was read
  as a literal (or is the constructor's default), and the rod is within
  its stencil's limit;
* **refused** -- the same, and the rod is past the limit: the gate fails;
* **not evaluated** -- something decides the Fourier number that the gate
  cannot read: a computed argument, a ``**mapping`` held in a variable, a
  ``*args`` splat, a keyword given twice, a non-positive value, or a
  stencil order with no limit.  These are counted and reported by reason
  (``--list-unevaluated`` prints each one), never folded into the verified
  count, and a scope in which *nothing* could be evaluated fails.

A literal ``**{"thermal_diffusivity": 1e3}`` or ``**dict(...)`` splat is
read like the keywords it spells.  It used to be dropped silently, so the
rod was judged on the constructor's *defaults* and counted as verified
while ``HeatNode.__init__`` refused it (audit_040_phase3_wave_d, H5).
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

from maddening.nodes.heat import MAX_FOURIER_NUMBER, HeatNode  # noqa: E402

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
_ALLOWED_UNSTABLE = {
    "tests/compliance/test_gate_scripts.py":
        "plants unstable rods as fixtures to prove this gate rejects them",
    "benchmarks/results/audit_040_r2/numerics/repro/gate_probes/"
    "e_genuinely_unstable.py":
        "archived audit probe: the positive control for this gate, a rod the "
        "guard must refuse.  Committed as round-2 evidence, so the gate now "
        "scans it",
    "benchmarks/results/audit_040_r2/numerics/repro/gate_probes/"
    "c_positional_stencil.py":
        "archived audit probe: a 4th-order rod whose stencil_order is passed "
        "POSITIONALLY.  It was invisible while _POSITIONAL stopped at five "
        "names; widening it to the full signature made this probe the first "
        "thing the gate caught",
    "benchmarks/results/audit_040_r2/numerics/repro/gate_probes/"
    "d_attribute_call.py":
        "archived audit probe: the same unstable rod spelled as the "
        "attribute heat.HeatNode.  It was invisible while the gate matched "
        "ast.Name only",
}

#: A ceiling, not a target.  One test file has to plant defects to prove the
#: gate catches them, and the archived round-2 probes are the positive
#: controls for the three holes this gate has had.  A sixth entry means
#: unstable rods are being allowlisted rather than fixed.
_MAX_ALLOWED_UNSTABLE = 6


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


def _is_heat_node_call(node, aliases) -> bool:
    """Is this call constructing a ``HeatNode``, however it is spelled?

    Matching ``ast.Attribute`` can in principle pick up an unrelated
    ``something.HeatNode`` -- which is harmless here, because the worst it
    can do is report a construction as unchecked.
    """
    if not isinstance(node, ast.Call):
        return False
    if getattr(node.func, "attr", None) == "HeatNode":
        return True
    return getattr(node.func, "id", None) in aliases


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


def scan_source(src, origin, defaults, unstable, unchecked, seen):
    """Walk one source string, recursing into embedded source."""
    tree = _parse(src)
    if tree is None:
        if "HeatNode(" in src:
            unchecked.append((origin, 0, "unparseable source mentioning HeatNode"))
        return

    aliases = _local_aliases(tree)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and "HeatNode(" in node.value):
            scan_source(node.value, f"{origin} [embedded source]",
                        defaults, unstable, unchecked, seen)
            continue
        if not _is_heat_node_call(node, aliases):
            continue

        # A ``**`` splat used to be dropped (``if kw.arg``), so the rod was
        # judged on the defaults it overrode and counted as verified.
        args, why_not = _call_arguments(node)
        if args is None:
            unchecked.append((origin, node.lineno, why_not))
            continue

        # An explicit grid is not a uniform rod; the guard skips it too.
        if "grid_points" in args:
            continue

        values = {}
        for key in ("timestep", "n_cells", "length",
                    "thermal_diffusivity", "stencil_order"):
            if key in args:
                values[key] = _number(args[key])
            else:
                values[key] = defaults.get(key)
        if any(v is None for v in values.values()):
            unchecked.append((origin, node.lineno,
                              "a constructor argument is computed, not literal"))
            continue

        dt = values["timestep"]
        n_cells = values["n_cells"]
        length = values["length"]
        alpha = values["thermal_diffusivity"]
        order = values["stencil_order"]
        # ``seen`` is appended *after* every way out of this block, because
        # it is the count the summary line calls "verified".  It used to be
        # appended above, so the two ``continue``s below inflated the
        # headline: 132 reported against 131 actually evaluated
        # (audit_040_r2/gates, finding G3).
        if min(dt, n_cells, length, alpha) <= 0:
            unchecked.append((
                origin, node.lineno,
                f"a non-positive constructor argument (dt={dt!r}, "
                f"n_cells={n_cells!r}, length={length!r}, alpha={alpha!r}); "
                f"no Fourier number is defined, so this was NOT checked"
            ))
            continue
        limit = MAX_FOURIER_NUMBER.get(order)
        if limit is None:
            unchecked.append((
                origin, node.lineno,
                f"stencil_order={order!r} has no entry in MAX_FOURIER_NUMBER "
                f"({sorted(MAX_FOURIER_NUMBER)}), so this was NOT checked"
            ))
            continue
        seen.append((origin, node.lineno))

        dx = length / n_cells
        fourier = dt * alpha / (dx * dx)
        if fourier > limit:
            unstable.append((
                origin, node.lineno,
                f"Fourier number {fourier:.4g} exceeds the {limit:g} limit of "
                f"the order-{order} stencil "
                f"(dt={dt!r}, alpha={alpha!r}, dx=length/n_cells={dx:.6g}); "
                f"HeatNode.__init__ raises ValueError. "
                f"Use timestep <= {limit * dx * dx / alpha:.6g}, more cells, "
                f"or a smaller thermal_diffusivity"
            ))


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

    defaults = _defaults()
    unstable, unchecked, seen = [], [], []
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
        if origin in _ALLOWED_UNSTABLE:
            continue
        scan_source(src, origin, defaults, unstable, unchecked, seen)

    if unstable:
        print(f"FAIL: {len(unstable)} HeatNode construction(s) the stability "
              f"guard refuses:")
        for origin, lineno, why in unstable:
            print(f"  {origin}:{lineno}: {why}")
        print("\nFix: give the construction a stable timestep.  If the test "
              "does not care about the physics, it still cannot build a rod "
              "that diverges -- see MADD-ANO-009.")
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
    print(f"OK: {len(seen)} HeatNode construction(s) verified within their "
          f"stencil's stability limit{note}")
    return 0


def _reason_key(why: str) -> str:
    """The reason with its per-construction values dropped, for grouping."""
    return why.split(" (", 1)[0].split(";", 1)[0]


if __name__ == "__main__":
    sys.exit(main())
