"""Reproducer: tests/property/test_coupling_error_bound.py::
test_the_bound_is_the_same_on_both_solvers fails whenever Hypothesis
draws solver="ift" together with a non-default linear_solver.

The strategy never draws an inert knob (strategies.py:721-724 gates
linear_solver on solver == "ift"), but the test then flips `solver` to
"fori" with dataclasses.replace, which makes the drawn
linear_solver="dense" inert -> UserWarning -> error under
pyproject's filterwarnings = ["error"].

  PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python repro_property_test_solver_flip.py
"""
import os, dataclasses, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from maddening.core.coupling import CouplingGroup

drawn = CouplingGroup(
    nodes=frozenset({"a", "b"}), solver="ift", linear_solver="dense",
)
print("drawn group  :", drawn.solver, drawn.linear_solver, "-> built quietly")

warnings.simplefilter("error")          # what pyproject.toml does in CI
for flipped in ("ift", "fori"):
    try:
        dataclasses.replace(drawn, solver=flipped)
        print(f"replace(solver={flipped!r}) -> ok")
    except UserWarning as w:
        print(f"replace(solver={flipped!r}) -> UserWarning: {' '.join(str(w).split())}")
