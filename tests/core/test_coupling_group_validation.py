"""Construction-time validation of ``CouplingGroup``'s Literal-typed fields.

The fields ``solver``, ``acceleration``, ``iteration_mode``,
``boundary_interpolation``, ``predictor``, ``linear_solver`` and
``convergence_norm`` are annotated ``typing.Literal[...]``, but Python treats that purely as a type-checker hint
at runtime — a typo like ``acceleration="aitkin"`` would silently set the
field to that string and the runtime dispatch (``if group.acceleration ==
"aitken": ...``) would simply fail to match, with the group quietly falling
back to the default branch.  ``CouplingGroup.__post_init__`` therefore
re-validates each Literal field against its declared options and raises
``ValueError`` on a mismatch.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import pytest

from maddening.core.coupling.group import CouplingGroup, coupling_group_kwargs
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


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


@pytest.mark.parametrize("value", ["l2", "mixed", "interface"])
def test_convergence_norm_valid(value):
    g = CouplingGroup(nodes=NODES, convergence_norm=value)
    assert g.convergence_norm == value


@pytest.mark.parametrize("value", ["gauss-seidel", "jacobi"])
def test_iteration_mode_valid(value):
    g = CouplingGroup(nodes=NODES, iteration_mode=value)
    assert g.iteration_mode == value


@pytest.mark.parametrize("value", ["constant", "linear", "quadratic"])
def test_boundary_interpolation_valid(value):
    # With ``subcycling=True``, the setting that reads it: there is no
    # time to interpolate over without sub-steps, so a group that names
    # an interpolation and does not subcycle is a dead setting and warns.
    g = CouplingGroup(nodes=NODES, boundary_interpolation=value,
                      subcycling=True)
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
        ("convergence_norm", "mixxed"),  # doubled 'x'
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
    assert g.solver == "ift"
    assert g.acceleration == "none"
    assert g.iteration_mode == "gauss-seidel"
    assert g.convergence_norm == "l2"
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


# ---------------------------------------------------------------------------
# 4. ``accelerated_fields`` is checked for shape, not only for content.
# ---------------------------------------------------------------------------

def test_accelerated_fields_rejects_a_non_mapping():
    """A non-mapping is a ``ValueError``, not an ``AttributeError``.

    The validator reads the mapping's values, so a list used to escape
    as ``AttributeError: 'list' object has no attribute 'values'``
    raised from inside ``__post_init__`` -- the wrong exception type,
    naming an internal call rather than the setting.
    """
    with pytest.raises(ValueError, match="must be a mapping"):
        CouplingGroup(nodes=NODES, accelerated_fields=["a"])


def test_accelerated_fields_rejects_a_bare_string_value():
    """``{"a": "position"}`` names a field, not a sequence of fields.

    A string is iterable, so it survived construction and reached the
    traced coupling loop as a per-character field list
    (``names ['p', 'o', 's', ...]``) -- exactly the deep, confusing
    failure this validation exists to prevent.
    """
    with pytest.raises(ValueError, match="bare string"):
        CouplingGroup(nodes=NODES, accelerated_fields={"a": "position"})


def test_accelerated_fields_accepts_tuples_and_none():
    """The shape checks reject nothing that was valid before.

    Under an IQN acceleration, the only one that reads the mapping:
    naming fields for a group that solves no quasi-Newton problem is a
    dead setting and warns.
    """
    assert CouplingGroup(nodes=NODES).accelerated_fields is None
    g = CouplingGroup(
        nodes=NODES, acceleration="iqn-ils",
        accelerated_fields={"a": ("position",), "b": ()},
    )
    assert g.accelerated_fields == {"a": ("position",), "b": ()}


# ---------------------------------------------------------------------------
# 5. A tolerance knob the chosen norm never reads warns at construction.
# ---------------------------------------------------------------------------
#
# ``"l2"`` tests the global L2 state change against ``tolerance`` and is
# never handed ``atol`` / ``rtol``; ``"mixed"`` and ``"interface"`` fold
# ``atol`` / ``rtol`` into the residual and test it against a threshold
# hard-coded to 1.0, never reading ``tolerance``.  Nothing used to say so,
# so the unread knob turned silently: an investigation into a 2.14%
# discrepancy reported that tightening ``tolerance`` from 1e-4 to 1e-14 on
# an interface-norm group left every digit unchanged and concluded the
# solvers converge to different fixed points.  The control was inert.


def _warnings_from(**kwargs):
    """Every warning raised by constructing a group with ``kwargs``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CouplingGroup(nodes=NODES, **kwargs)
    return caught


@pytest.mark.parametrize("norm", ["mixed", "interface"])
def test_tolerance_set_under_a_norm_that_ignores_it_warns(norm):
    """``tolerance`` is dead under ``"mixed"`` / ``"interface"``."""
    with pytest.warns(UserWarning, match=r"CouplingGroup\.tolerance"):
        CouplingGroup(nodes=NODES, convergence_norm=norm, tolerance=1e-9)


@pytest.mark.parametrize("norm", ["mixed", "interface"])
def test_inert_tolerance_warning_names_the_live_knobs_and_the_norm(norm):
    """The message is actionable without opening the source.

    It has to say which norm is in force (the user may have set it far
    from the ``tolerance=`` line, or inherited it from a config) and
    which settings do work under that norm.
    """
    (w,) = _warnings_from(convergence_norm=norm, tolerance=1e-9)
    msg = str(w.message)
    assert "atol" in msg and "rtol" in msg, msg
    assert norm in msg, msg
    assert "1e-09" in msg, msg


def test_default_tolerance_under_an_ignoring_norm_is_silent():
    """Choosing a norm is not a mistake; only a dead setting is.

    A group that names ``"interface"`` and leaves ``tolerance`` alone has
    done nothing the user needs to hear about, and nagging it would train
    people to filter the warning that matters.
    """
    assert _warnings_from(convergence_norm="interface") == []
    assert _warnings_from(convergence_norm="mixed", atol=1e-10) == []


def test_tolerance_under_l2_is_silent():
    """Under ``"l2"`` the knob is live, so setting it is correct usage."""
    assert _warnings_from(convergence_norm="l2", tolerance=1e-9) == []
    assert _warnings_from(tolerance=1e-9) == []  # "l2" is the default


@pytest.mark.parametrize("kwargs", [{"rtol": 1e-9}, {"atol": 1e-10, "rtol": 1e-9}])
def test_rtol_set_under_l2_warns(kwargs):
    """The reverse footgun: ``"l2"`` never reads ``rtol``.

    ``coupling_residual_l2`` hard-codes the ratio's denominator to the
    field's bare magnitude and carries its threshold in ``tolerance``,
    so an ``rtol`` tightened under the default norm is as dead as a
    ``tolerance`` tightened under ``"interface"``.
    """
    with pytest.warns(UserWarning, match=r"CouplingGroup\.rtol"):
        CouplingGroup(nodes=NODES, **kwargs)


def test_atol_set_under_l2_is_silent_because_the_l2_norm_reads_it():
    """The warning that sent users away from the only live knob.

    ``coupling_residual_l2`` takes ``atol`` and drops every field at or
    below it out of the norm entirely, so under ``"l2"`` it decides
    which fields the residual is measuring at all.  Telling the caller
    it is ignored -- and, under ``filterwarnings = ["error"]``, making
    it unsettable -- left a group with a small unconverged field no way
    to be held to it.
    """
    assert _warnings_from(atol=1e-10) == []
    assert _warnings_from(convergence_norm="l2", atol=1e-3) == []


def test_inert_rtol_warning_names_tolerance_as_the_live_knob():
    (w,) = _warnings_from(rtol=1e-9)
    msg = str(w.message)
    assert "rtol" in msg, msg
    assert "tolerance" in msg, msg
    assert "l2" in msg, msg


def test_default_atol_rtol_under_l2_are_silent():
    """The overwhelmingly common group -- all defaults -- says nothing."""
    assert _warnings_from() == []
    assert _warnings_from(atol=0.0, rtol=1e-6) == []  # the declared defaults


@pytest.mark.parametrize("kwargs", [
    {"acceleration": "aitken"},
    {"relaxation": 0.5},
    {"jacobian_reuse": 2},
    {"accelerated_fields": {"a": ("x",)}},
    {"linear_solver": "dense"},
])
def test_the_acceleration_family_warns_at_a_cap_of_one(kwargs):
    """``max_iterations=1`` returns before any of them is reached.

    ``_run_coupling_inner`` takes one staggered pass and returns, ahead
    of the accelerator's construction and ahead of ``_run_ift_forward``
    -- so a cap of one makes the whole acceleration family and
    ``linear_solver`` dead whatever else the group says.  It is the one
    inert case that was decidable from the declared fields and was not
    being reported.
    """
    with pytest.warns(UserWarning, match="max_iterations=1"):
        CouplingGroup(nodes=NODES, max_iterations=1, **kwargs)


def test_a_cap_of_one_reports_its_dead_knobs_in_a_single_message():
    """One mistake, one message, even though two rules could speak.

    ``relaxation`` is gated on ``acceleration="fixed"`` as well as on
    the cap.  Here the acceleration is right and the cap is what kills
    it, so the acceleration-gated rule stands down: a message saying
    ``relaxation is ignored under acceleration='fixed'`` would be false
    twice over.
    """
    (w,) = _warnings_from(max_iterations=1, acceleration="fixed",
                          relaxation=0.5)
    msg = str(w.message)
    assert "acceleration='fixed'" in msg, msg
    assert "relaxation=0.5" in msg, msg
    assert "max_iterations=1" in msg, msg


def test_strict_convergence_survives_a_cap_of_one():
    """The single-pass branch checks it, so it is not in that family.

    ``max_iterations=1`` used to leave the seeded ``_meta`` zeros in
    place, which read as ``converged=True`` whatever the state; the
    branch now measures its own residual *and* honours
    ``strict_convergence`` on the ift path.  Warning that the flag is
    ignored there would send a user to turn off their only guard.
    """
    assert _warnings_from(max_iterations=1, strict_convergence=True) == []


def test_round_trip_through_to_dict_does_not_warn_twice():
    """Reloading a stored group is not a second chance to nag.

    ``to_dict`` writes every field and ``coupling_group_kwargs`` passes
    every field back by name, defaults included.  Because the check
    compares against the declared default rather than tracking what the
    caller passed, a group that was quiet when written stays quiet when
    it comes back.
    """
    g = CouplingGroup(nodes=NODES, convergence_norm="mixed", atol=1e-10)
    nodes, kwargs = coupling_group_kwargs(g.to_dict())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        back = CouplingGroup(nodes=frozenset(nodes), **kwargs)
    assert caught == []
    assert back == g


# ---------------------------------------------------------------------------
# 6. The rest of the family: every knob only some configurations read.
# ---------------------------------------------------------------------------
#
# ``tolerance`` was not special, only the one that got caught.  Six more
# settings are read under exactly one other setting and ignored outright
# under the rest, and each was as able to make a control look live while
# it did nothing.  The read sites (in ``maddening.core.graph_manager``):
#
# * ``relaxation``            -- ``acceleration="fixed"`` only;
# * ``jacobian_reuse``        -- ``acceleration="iqn-imvj"`` only;
# * ``accelerated_fields``    -- the two IQN accelerations only;
# * ``waveform_iterations``   -- ``subcycling=True`` only;
# * ``boundary_interpolation``-- ``subcycling=True`` only;
# * ``linear_solver``         -- ``solver="ift"`` only;
# * ``strict_convergence``    -- ``solver="ift"`` only.

#: ``(field, deliberate value, configuration that ignores it,
#:   configuration that reads it, the setting the message must name)``.
INERT_KNOBS = [
    pytest.param(
        "relaxation", 0.5, {"acceleration": "aitken"},
        {"acceleration": "fixed"}, "acceleration='fixed'", id="relaxation",
    ),
    pytest.param(
        "jacobian_reuse", 2, {"acceleration": "iqn-ils"},
        {"acceleration": "iqn-imvj"}, "acceleration='iqn-imvj'",
        id="jacobian_reuse",
    ),
    pytest.param(
        "accelerated_fields", {"a": ("position",)}, {"acceleration": "aitken"},
        {"acceleration": "iqn-ils"}, "acceleration='iqn-ils'",
        id="accelerated_fields",
    ),
    pytest.param(
        "waveform_iterations", 3, {"subcycling": False},
        {"subcycling": True}, "subcycling=True", id="waveform_iterations",
    ),
    pytest.param(
        "boundary_interpolation", "quadratic", {"subcycling": False},
        {"subcycling": True}, "subcycling=True", id="boundary_interpolation",
    ),
    pytest.param(
        "linear_solver", "dense", {"solver": "fori"},
        {"solver": "ift"}, "solver='ift'", id="linear_solver",
    ),
    pytest.param(
        "strict_convergence", True, {"solver": "fori"},
        {"solver": "ift"}, "solver='ift'", id="strict_convergence",
    ),
]


def _inert_warnings(**kwargs):
    """The inert-knob ``UserWarning``s raised by constructing a group.

    Filtered to ``UserWarning`` because ``solver="fori"`` -- half the
    "ignored" configurations below -- also raises its own
    ``DeprecationWarning``, which is about the solver, not the knob.
    """
    return [
        w for w in _warnings_from(**kwargs)
        if issubclass(w.category, UserWarning)
        and not issubclass(w.category, DeprecationWarning)
    ]


@pytest.mark.parametrize("field,value,dead,live,names", INERT_KNOBS)
def test_knob_set_under_a_configuration_that_ignores_it_warns(
    field, value, dead, live, names,
):
    """The warning fires on the deliberate setting, at construction."""
    with pytest.warns(UserWarning, match=rf"CouplingGroup\.{field}"):
        CouplingGroup(nodes=NODES, **dead, **{field: value})


@pytest.mark.parametrize("field,value,dead,live,names", INERT_KNOBS)
def test_inert_knob_warning_names_the_value_and_the_live_setting(
    field, value, dead, live, names,
):
    """Actionable without opening the source.

    The user may have set the knob far from the setting that killed it,
    or inherited either from a config, so the message has to carry both
    ends: the value that will not be used, and the setting that would
    use it.
    """
    (w,) = _inert_warnings(**dead, **{field: value})
    msg = str(w.message)
    assert f"{field}={value!r}" in msg, msg
    assert names in msg, msg


@pytest.mark.parametrize("field,value,dead,live,names", INERT_KNOBS)
def test_knob_left_at_its_default_is_silent(field, value, dead, live, names):
    """Choosing an acceleration, a solver or no subcycling is not a mistake.

    Only a dead *setting* is.  Nagging every group that does not use
    quasi-Newton about ``jacobian_reuse`` would train people to filter
    the warning that matters.
    """
    assert _inert_warnings(**dead) == []


@pytest.mark.parametrize("field,value,dead,live,names", INERT_KNOBS)
def test_knob_under_the_configuration_that_reads_it_is_silent(
    field, value, dead, live, names,
):
    """Under the setting that reads it, setting it is correct usage."""
    assert _inert_warnings(**live, **{field: value}) == []


def test_two_dead_knobs_produce_two_warnings():
    """One message per mistake, not one per group.

    ``solver="fori"`` kills ``linear_solver`` and ``strict_convergence``
    together, and a user who set both has two things to undo.
    """
    caught = _inert_warnings(
        solver="fori", linear_solver="dense", strict_convergence=True,
    )
    heads = sorted(str(w.message).split("=")[0] for w in caught)
    assert heads == [
        "CouplingGroup.linear_solver", "CouplingGroup.strict_convergence",
    ]


def test_every_compatible_live_knob_together_is_silent():
    """A group that uses everything it sets says nothing.

    Six of the seven can be live at once; ``relaxation`` cannot join
    them, because the acceleration that reads it is not one of the two
    that read ``jacobian_reuse`` and ``accelerated_fields``.  If this
    warns, a rule is gated on the wrong field.
    """
    assert _inert_warnings(
        acceleration="iqn-imvj", jacobian_reuse=2,
        accelerated_fields={"a": ("position",)},
        subcycling=True, waveform_iterations=3,
        boundary_interpolation="quadratic",
        solver="ift", linear_solver="dense", strict_convergence=True,
    ) == []


def test_full_round_trip_of_a_quiet_group_stays_quiet():
    """Every field written out and passed back by name must not warn.

    The round trip re-passes defaults explicitly, so a rule that tested
    "was this argument given?" rather than "is it the default?" would
    turn every reload into a warning storm.
    """
    g = CouplingGroup(
        nodes=NODES, acceleration="fixed", relaxation=0.5, subcycling=True,
        waveform_iterations=3,
    )
    nodes, kwargs = coupling_group_kwargs(g.to_dict())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        back = CouplingGroup(nodes=frozenset(nodes), **kwargs)
    assert caught == []
    assert back == g


# ---------------------------------------------------------------------------
# 7. The warning points at the line the user wrote.
# ---------------------------------------------------------------------------
#
# A ``UserWarning`` is a message to whoever can act on it, and the only
# line they can act on is their own.  ``CouplingGroup(...)`` is three
# frames below the warning, ``gm.add_coupling_group(...)`` four and
# ``gm.auto_couple()`` five, so a fixed ``stacklevel`` -- which is what
# these warnings had -- can be right for at most one of them.  It was
# right for the direct construction, which meant every warning reached
# through ``GraphManager`` was attributed to ``graph_manager.py``'s own
# ``CouplingGroup(...)`` line: a file the reader does not own, at a line
# that says nothing about which group or which knob.


def _line_of_next_statement() -> int:
    """The line number of the statement after the call to this function."""
    return inspect.currentframe().f_back.f_lineno + 1


def _cycle_of_two_springs() -> GraphManager:
    """Two spring nodes in a 2-cycle; no coupling group yet."""
    gm = GraphManager()
    for name, pos in (("spring_a", 0.0), ("spring_b", 2.0)):
        gm.add_node(SpringDamperNode(
            name=name, timestep=0.001, stiffness=50.0, damping=1.0,
            mass=1.0, rest_length=1.0, initial_position=pos,
        ))
    gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
    gm.add_edge("spring_b", "spring_a", "position", "anchor_position")
    return gm


def test_inert_knob_warning_points_at_the_direct_construction():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected = _line_of_next_statement()
        CouplingGroup(nodes=NODES, relaxation=0.5)
    (w,) = caught
    assert Path(w.filename) == Path(__file__)
    assert w.lineno == expected


def test_inert_knob_warning_points_at_the_add_coupling_group_call():
    """Not at ``graph_manager.py``, which is where it used to land."""
    gm = _cycle_of_two_springs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected = _line_of_next_statement()
        gm.add_coupling_group(["spring_a", "spring_b"], relaxation=0.5)
    (w,) = [x for x in caught if issubclass(x.category, UserWarning)]
    assert Path(w.filename) == Path(__file__), w.filename
    assert w.lineno == expected


def test_inert_knob_warning_points_at_the_auto_couple_call():
    """One frame deeper again: ``auto_couple`` calls ``add_coupling_group``."""
    gm = _cycle_of_two_springs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected = _line_of_next_statement()
        gm.auto_couple(waveform_iterations=3)
    (w,) = [x for x in caught if issubclass(x.category, UserWarning)]
    assert Path(w.filename) == Path(__file__), w.filename
    assert w.lineno == expected


def test_fori_deprecation_warning_points_at_the_users_call_too():
    """The same defect, in the warning next to it in ``__post_init__``.

    ``solver="fori"`` is a setting a user migrates off; a deprecation
    notice attributed to library source tells them nothing about which
    of their groups to change.
    """
    gm = _cycle_of_two_springs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected = _line_of_next_statement()
        gm.add_coupling_group(["spring_a", "spring_b"], solver="fori")
    (w,) = [x for x in caught if issubclass(x.category, DeprecationWarning)]
    assert Path(w.filename) == Path(__file__), w.filename
    assert w.lineno == expected
