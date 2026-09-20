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

#: Constructor parameters this gate needs, in positional order.
_POSITIONAL = ["name", "timestep", "n_cells", "length", "thermal_diffusivity"]

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
}


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


def scan_source(src, origin, defaults, unstable, unchecked, seen):
    """Walk one source string, recursing into embedded source."""
    tree = _parse(src)
    if tree is None:
        if "HeatNode(" in src:
            unchecked.append((origin, 0, "unparseable source mentioning HeatNode"))
        return

    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and "HeatNode(" in node.value):
            scan_source(node.value, f"{origin} [embedded source]",
                        defaults, unstable, unchecked, seen)
            continue
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "HeatNode"):
            continue

        args = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        for i, positional in enumerate(node.args):
            if i < len(_POSITIONAL):
                args.setdefault(_POSITIONAL[i], positional)

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
        seen.append((origin, node.lineno))

        dt = values["timestep"]
        n_cells = values["n_cells"]
        length = values["length"]
        alpha = values["thermal_diffusivity"]
        order = values["stencil_order"]
        if min(dt, n_cells, length, alpha) <= 0:
            continue
        limit = MAX_FOURIER_NUMBER.get(order)
        if limit is None:
            continue

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
        return 1

    note = ""
    if unchecked:
        note = (f"; {len(unchecked)} further construction(s) have computed "
                f"arguments and were NOT checked")
    print(f"OK: {len(seen)} HeatNode construction(s) verified within their "
          f"stencil's stability limit{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
