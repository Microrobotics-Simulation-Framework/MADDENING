"""Construction-time validation of ``CouplingGroup``'s Literal-typed fields.

The fields ``solver``, ``acceleration``, ``iteration_mode``,
``boundary_interpolation``, ``predictor`` and ``linear_solver`` are annotated
``typing.Literal[...]``, but Python treats that purely as a type-checker hint
at runtime — a typo like ``acceleration="aitkin"`` would silently set the
field to that string and the runtime dispatch (``if group.acceleration ==
"aitken": ...``) would simply fail to match, with the group quietly falling
back to the default branch.  ``CouplingGroup.__post_init__`` therefore
re-validates each Literal field against its declared options and raises
``ValueError`` on a mismatch.
"""

from __future__ import annotations

import pytest

from maddening.core.coupling.group import CouplingGroup


NODES = frozenset({"a", "b"})


# ---------------------------------------------------------------------------
# 1. Each valid option for each Literal field constructs successfully.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["fori", "ift"])
def test_solver_valid(value):
    g = CouplingGroup(nodes=NODES, solver=value)
    assert g.solver == value


@pytest.mark.parametrize(
    "value", ["none", "aitken", "fixed", "iqn-ils", "iqn-imvj"]
)
def test_acceleration_valid(value):
    g = CouplingGroup(nodes=NODES, acceleration=value)
    assert g.acceleration == value


@pytest.mark.parametrize("value", ["gauss-seidel", "jacobi"])
def test_iteration_mode_valid(value):
    g = CouplingGroup(nodes=NODES, iteration_mode=value)
    assert g.iteration_mode == value


@pytest.mark.parametrize("value", ["constant", "linear", "quadratic"])
def test_boundary_interpolation_valid(value):
    g = CouplingGroup(nodes=NODES, boundary_interpolation=value)
    assert g.boundary_interpolation == value


@pytest.mark.parametrize("value", ["none", "linear", "quadratic"])
def test_predictor_valid(value):
    g = CouplingGroup(nodes=NODES, predictor=value)
    assert g.predictor == value


@pytest.mark.parametrize("value", ["gmres", "dense"])
def test_linear_solver_valid(value):
    g = CouplingGroup(nodes=NODES, linear_solver=value)
    assert g.linear_solver == value


# ---------------------------------------------------------------------------
# 2. A typo in any Literal field raises ValueError that names the field,
#    the bad value, and the valid set.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field, bad",
    [
        ("solver", "for"),               # missing 'i'
        ("acceleration", "aitkin"),      # 'i' vs 'e'
        ("iteration_mode", "jacopi"),    # 'p' vs 'b'
        ("boundary_interpolation", "lineer"),
        ("predictor", "qaudratic"),      # transposed
        ("linear_solver", "gmrs"),       # missing 'e'
    ],
)
def test_typo_raises(field, bad):
    with pytest.raises(ValueError) as exc:
        CouplingGroup(nodes=NODES, **{field: bad})
    msg = str(exc.value)
    # Diagnostic message must identify field, bad value, and valid set.
    assert field in msg, msg
    assert repr(bad) in msg, msg
    assert "expected one of" in msg, msg


def test_typo_error_includes_valid_options():
    """The error lists every accepted option, so the fix is obvious."""
    with pytest.raises(ValueError) as exc:
        CouplingGroup(nodes=NODES, acceleration="aitkin")
    msg = str(exc.value)
    for valid in ("none", "aitken", "fixed", "iqn-ils", "iqn-imvj"):
        assert valid in msg, msg


def test_empty_string_rejected():
    """Empty string is not a member of any Literal — must raise."""
    with pytest.raises(ValueError, match="acceleration"):
        CouplingGroup(nodes=NODES, acceleration="")


# ---------------------------------------------------------------------------
# 3. Realistic experiment-style configurations still construct.
# ---------------------------------------------------------------------------

def test_default_construction_succeeds():
    """The default CouplingGroup (every field at its default) is valid."""
    g = CouplingGroup(nodes=NODES)
    # Each Literal field landed on its declared default.
    assert g.solver == "fori"
    assert g.acceleration == "none"
    assert g.iteration_mode == "gauss-seidel"
    assert g.boundary_interpolation == "linear"
    assert g.predictor == "none"
    assert g.linear_solver == "gmres"


def test_ift_imvj_gmres_config():
    """A realistic IFT + IMVJ + GMRES experiment config still constructs."""
    g = CouplingGroup(
        nodes=frozenset({"fluid", "solid"}),
        max_iterations=50,
        tolerance=1e-8,
        acceleration="iqn-imvj",
        jacobian_reuse=4,
        iteration_mode="gauss-seidel",
        solver="ift",
        linear_solver="gmres",
    )
    assert g.solver == "ift"
    assert g.acceleration == "iqn-imvj"
    assert g.linear_solver == "gmres"


def test_subcycling_quadratic_predictor_config():
    """Subcycling + quadratic boundary interpolation + predictor combo."""
    g = CouplingGroup(
        nodes=frozenset({"fast", "slow"}),
        subcycling=True,
        boundary_interpolation="quadratic",
        predictor="quadratic",
        waveform_iterations=2,
    )
    assert g.boundary_interpolation == "quadratic"
    assert g.predictor == "quadratic"
