"""A ``CouplingGroup`` warns for exactly the settings its configuration
never reads -- over generated configurations, not chosen ones.

``tests/core/test_coupling_group_validation.py`` states the seven knobs
one at a time: set ``relaxation`` under ``acceleration="aitken"`` and
you get a warning, set it under ``"fixed"`` and you do not.  Those are
the cases somebody thought of, and the whole reason this warning exists
is that *nobody thought of* ``tolerance`` under an interface norm until
a day had been spent concluding that two solvers converge to different
fixed points, on the evidence of a control knob that did nothing.

So the property is the total statement, over every combination of the
eighteen settings Hypothesis can reach:

    A ``CouplingGroup`` emits one warning for each setting that is both
    *deliberate* (different from the field's declared default) and
    *unread* (the rest of the configuration never looks at it), and no
    warning for anything else.

The "unread" half is written out here, in :data:`_READ_WHEN`, from the
read sites in ``maddening.core.graph_manager`` -- deliberately not
derived from ``_INERT_RULES``, the table the implementation walks.  A
property that re-derived its expectation from the table it is testing
would pass whatever the table said, including an empty one.  Two
independent statements of the same fact is the point: they are checked
against each other on every example, and against the dataclass itself
by :func:`test_every_field_is_classified`, which fails the moment a
nineteenth field is added without anyone deciding when it is read.
"""

from __future__ import annotations

import warnings
from dataclasses import fields

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.core.coupling.group import _FIELD_DEFAULTS, CouplingGroup
from tests.conftest import EXAMPLES_CHEAP


NODES = frozenset({"a", "b"})


# ---------------------------------------------------------------------------
# When each setting is read, from the read sites -- not from the validator.
# ---------------------------------------------------------------------------
#: ``field -> predicate(group)``, True when this group's configuration
#: actually reads the field.  Sources, all in
#: ``maddening.core.graph_manager``:
#:
#: * ``max_iterations <= 1`` returns from ``_run_coupling_inner`` after
#:   the single pass, before the accelerator is built and before
#:   ``_run_ift_forward`` runs -- which kills ``acceleration``,
#:   ``relaxation``, ``jacobian_reuse``, ``accelerated_fields`` and
#:   ``linear_solver`` whatever else is set.  ``strict_convergence``
#:   survives it: that branch checks it on the ift path;
#: * ``_compute_residual`` hands ``rtol`` to the mixed and interface
#:   residuals only, and ``conv_threshold_value`` is a literal ``1.0``
#:   for those two and ``float(group.tolerance)`` otherwise.  ``atol``
#:   is *not* in that set: all three residuals take it as the dead band
#:   that decides which fields enter the norm at all, and the cap-one
#:   branch measures a residual too, so it is read on every path;
#: * ``_accelerate`` (ift) and the ``elif group.acceleration == "fixed"``
#:   branch (fori) are the only readers of ``relaxation``;
#: * ``_iqn_warm_start`` returns before reaching ``jacobian_reuse``
#:   unless the acceleration is ``"iqn-imvj"``, and ``accel_fields`` is
#:   ``None`` unless it is one of the two IQN methods;
#: * ``n_waveform`` is ``1`` and the interpolation flags are never
#:   computed unless the group subcycles;
#: * ``linear_solver`` and ``strict_convergence`` are read inside
#:   ``_run_ift_forward``, which only ``solver="ift"`` calls.
#:
#: Everything else is read on every path, so it can never be inert.
_ALWAYS = (lambda g: True)
_MULTIPASS = (lambda g: g.max_iterations > 1)
_READ_WHEN = {
    "max_iterations": _ALWAYS,
    "tolerance": lambda g: g.convergence_norm == "l2",
    "convergence_norm": _ALWAYS,
    "atol": _ALWAYS,
    "rtol": lambda g: g.convergence_norm != "l2",
    "diagnostics": _ALWAYS,
    "acceleration": _MULTIPASS,
    "relaxation": lambda g: _MULTIPASS(g) and g.acceleration == "fixed",
    "iteration_mode": _ALWAYS,
    "accelerated_fields": lambda g: (
        _MULTIPASS(g) and g.acceleration in ("iqn-ils", "iqn-imvj")
    ),
    "subcycling": _ALWAYS,
    "boundary_interpolation": lambda g: g.subcycling,
    "jacobian_reuse": lambda g: _MULTIPASS(g) and g.acceleration == "iqn-imvj",
    "waveform_iterations": lambda g: g.subcycling,
    "predictor": _ALWAYS,
    "solver": _ALWAYS,
    "strict_convergence": lambda g: g.solver == "ift",
    "linear_solver": lambda g: _MULTIPASS(g) and g.solver == "ift",
}

#: ``(applies, fields)``: settings reported in one message instead of
#: one each, and the configurations where that happens.  ``atol`` and
#: ``rtol`` used to be the only pair and are not one any more -- the
#: 0.4.0 dead band made ``atol`` live under every norm.  A cap of one
#: kills the whole acceleration family and ``linear_solver`` at once,
#: and does it *ahead* of the settings that normally gate them, so
#: there they share a message and two warnings for one mistake is still
#: one too many.
_ONE_MESSAGE = (
    (lambda g: g.max_iterations <= 1,
     ("acceleration", "relaxation", "jacobian_reuse", "accelerated_fields",
      "linear_solver")),
)

#: Values drawn per field, the declared default first.  Each pool holds
#: the default and at least one deliberate alternative, so both halves
#: of the property are reachable for every field;
#: :func:`test_pools_start_at_the_declared_default` keeps that true.
_VALUES = {
    # ``1`` is not a smaller cap but a different branch: it returns
    # before the accelerator exists.  See ``_READ_WHEN``.
    "max_iterations": [10, 3, 1],
    "tolerance": [1e-6, 1e-9],
    "convergence_norm": ["l2", "mixed", "interface"],
    "atol": [0.0, 1e-10],
    "rtol": [1e-6, 1e-9],
    "diagnostics": [False, True],
    "acceleration": ["none", "aitken", "fixed", "iqn-ils", "iqn-imvj"],
    "relaxation": [1.0, 0.5],
    "iteration_mode": ["gauss-seidel", "jacobi"],
    "accelerated_fields": [None, {"a": ("position",)}],
    "subcycling": [False, True],
    "boundary_interpolation": ["linear", "constant", "quadratic"],
    "jacobian_reuse": [0, 2],
    "waveform_iterations": [1, 3],
    "predictor": ["none", "linear", "quadratic"],
    "solver": ["ift", "fori"],
    "strict_convergence": [False, True],
    "linear_solver": ["gmres", "dense"],
}

_configurations = st.fixed_dictionaries(
    {name: st.sampled_from(pool) for name, pool in _VALUES.items()}
)


def _expected_message_heads(group: CouplingGroup) -> set[str]:
    """``{"CouplingGroup.<field>", ...}`` this group must warn about.

    Every message opens with ``CouplingGroup.<field>=<value>``, naming
    the first field of the rule that produced it, so the heads identify
    the warnings exactly -- one per mistake, and no spurious one.
    """
    inert = [
        name for name in _READ_WHEN
        if getattr(group, name) != _FIELD_DEFAULTS[name]
        and not _READ_WHEN[name](group)
    ]
    heads, paired = [], set()
    for applies, shared in _ONE_MESSAGE:
        if not applies(group):
            continue
        named = [name for name in shared if name in inert]
        if named:
            heads.append(named[0])
        paired.update(shared)
    heads += [name for name in inert if name not in paired]
    return {f"CouplingGroup.{name}" for name in heads}


def _construct(config: dict) -> tuple[CouplingGroup, list]:
    """Build the group, returning it with the ``UserWarning``s it raised.

    ``solver="fori"`` also raises its own ``DeprecationWarning``, which
    is about the solver rather than about a dead knob, so it is not part
    of this property.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        group = CouplingGroup(nodes=NODES, **config)
    return group, [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and not issubclass(w.category, DeprecationWarning)
    ]


@settings(max_examples=EXAMPLES_CHEAP)
@given(config=_configurations)
def test_warns_exactly_for_the_settings_the_configuration_does_not_read(
    config,
):
    """The invariant: one warning per deliberate-and-unread setting.

    Both directions at once.  A missing warning is the original bug --
    a knob that turns without turning anything.  A spurious one is the
    bug that follows it, because a warning that fires on correct usage
    is a warning people learn to silence, and the next inert knob goes
    out with it.
    """
    group, caught = _construct(config)
    heads = {str(w.message).split("=")[0] for w in caught}
    assert heads == _expected_message_heads(group)


@settings(max_examples=EXAMPLES_CHEAP)
@given(config=_configurations)
def test_every_warning_carries_the_value_that_will_be_ignored(config):
    """A message without the value cannot be matched to the line that set it.

    The value is also the only part that says *how far* the setting is
    from the default -- the difference between a knob someone tightened
    by three orders of magnitude and one they barely touched.
    """
    group, caught = _construct(config)
    for w in caught:
        head = str(w.message).split("=")[0]
        name = head.removeprefix("CouplingGroup.")
        assert f"{name}={getattr(group, name)!r}" in str(w.message)


@settings(max_examples=EXAMPLES_CHEAP)
@given(config=_configurations)
def test_a_group_of_pure_defaults_never_warns(config):
    """Naming a configuration is not a mistake; a dead setting is.

    Only the fields that *select* a configuration are drawn here --
    everything they gate stays at its default -- so nothing is inert by
    construction, however exotic the combination.
    """
    selectors = ("convergence_norm", "acceleration", "subcycling", "solver",
                 "iteration_mode", "predictor", "diagnostics")
    _, caught = _construct({k: config[k] for k in selectors})
    assert caught == []


def test_every_field_is_classified():
    """A knob added without a rule fails here, not in production.

    ``_READ_WHEN`` has to name every field of the dataclass.  The one
    that got this warning written was inert for two releases because no
    list like this existed to leave it off.
    """
    declared = {f.name for f in fields(CouplingGroup)} - {"nodes"}
    assert set(_READ_WHEN) == declared


def test_pools_start_at_the_declared_default():
    """Each pool must hold the default and a deliberate alternative.

    Otherwise a field silently stops exercising one half of the
    property: all-default values can never warn, and a pool missing the
    default can never test the silent case.
    """
    for name, pool in _VALUES.items():
        assert pool[0] == _FIELD_DEFAULTS[name], name
        assert len(pool) > 1, name
    assert set(_VALUES) == set(_READ_WHEN)


@pytest.mark.parametrize("name", sorted(_READ_WHEN))
def test_the_validator_and_this_module_agree_on_when_a_field_is_read(name):
    """The two independent statements must classify every field alike.

    ``_INERT_RULES`` drives the warnings; ``_READ_WHEN`` drives this
    module's expectation.  They are written from the same read sites by
    different routes, so a disagreement means one of them is stale --
    which is the failure this whole feature exists to make loud.
    """
    from maddening.core.coupling.group import _INERT_RULES

    ruled = [r for r in _INERT_RULES if name in r.fields]
    if not ruled:
        # Unruled means "read on every path"; assert that, rather than
        # trusting that nobody forgot a rule.
        assert _READ_WHEN[name] is _ALWAYS, name
        return
    for config in (
        {}, {"convergence_norm": "mixed"}, {"acceleration": "fixed"},
        {"acceleration": "iqn-ils"}, {"acceleration": "iqn-imvj"},
        {"subcycling": True}, {"solver": "fori"},
        {"max_iterations": 1}, {"max_iterations": 1, "acceleration": "fixed"},
        {"max_iterations": 1, "solver": "fori"},
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = CouplingGroup(nodes=NODES, **config)
        # Two rules can govern one field -- the cap and the setting
        # that normally gates it -- and it is read only where both say
        # so.  The cap rule stands the other one down so the *message*
        # is single; the *fact* is still the conjunction.
        live = all(r.live(g) for r in ruled)
        assert live == _READ_WHEN[name](g), (name, config)
