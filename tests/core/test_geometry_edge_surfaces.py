"""What is around a geometry edge: the texts, the reports, a checkpoint.

A geometry-dependent mapping (experimental) is resolved by the step; the
tests of that are the differential ones.  These are of everything else
that meets such an edge or the report of a group that resolves one:

* the inspection texts name the edge's geometry (``format_graph``,
  ``to_mermaid``, ``to_dot``), and those of a graph without one do not
  mention the word;
* a single-rate group whose geometry edges are ``multilinear_grid``
  reports its bounds like any other group, and its printed report has no
  word about a geometry; a group the diagnostics do not read the geometry
  of (here a sub-cycled one) reports its solve and withholds every bound,
  with the string ``not_usable_reason``; every printer and serialiser of
  a report takes that entry and shows the reason instead of a caveat
  about a value that is not there;
* a checkpoint holds the geometry as the node state it is and nothing for
  the mapping, which has no weights; a restart is the uninterrupted run.

The graph is ``tests/core/geometry_surface_graphs.py``.
"""

from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest

from maddening.core.coupling import reason_codes
from maddening.core.simulation.profiler import profile_graph
from maddening.serialization import json_codec

from tests.core import geometry_surface_graphs as G

STEPS = 3


def _stepped(**kw):
    gm = G.graph(**kw)
    for _ in range(STEPS):
        gm.step()
    return gm


@pytest.fixture(scope="module")
def grouped():
    return _stepped(group=True)


@pytest.fixture(scope="module")
def withheld():
    """The group sub-cycled: the diagnostics do not read its geometry."""
    return _stepped(group=True, substeps=2)


@pytest.fixture(scope="module")
def plain():
    return _stepped()


@pytest.fixture(scope="module")
def grouped_twin():
    return _stepped(group=True, geometry=False)


def _flat(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Inspection texts
# ---------------------------------------------------------------------------


def test_the_graph_text_names_each_edge_s_geometry(grouped):
    text = _flat(grouped.format_graph())
    assert "MultilinearGridMapping 6->4" in text
    gather = text[text.index("grid.x -> markers.sampled"):text.index("markers.x -> grid.deposit")]
    scatter = text[text.index("markers.x -> grid.deposit"):text.index("Coupling groups")]
    assert "geometry target.pos" in gather and "geometry source.pos" not in gather
    assert "geometry source.pos" in scatter and "geometry target.pos" not in scatter
    buffer = io.StringIO()
    grouped.print_graph(file=buffer)
    assert "geometry target.pos" in _flat(buffer.getvalue())


@pytest.mark.parametrize("diagram", ["to_mermaid", "to_dot"])
def test_a_diagram_s_edge_label_names_the_geometry(grouped, diagram):
    lines = getattr(grouped, diagram)().splitlines()
    gather = [line for line in lines if "x→sampled" in line]
    scatter = [line for line in lines if "x→deposit" in line]
    assert len(gather) == 1 and "geometry target.pos" in gather[0], lines
    assert len(scatter) == 1 and "geometry source.pos" in scatter[0], lines


def test_the_texts_of_a_graph_without_a_geometry_edge_do_not_mention_one(grouped_twin):
    """The control: the word comes from the edge, not from the printer."""
    for text in (grouped_twin.format_graph(), grouped_twin.to_mermaid(), grouped_twin.to_dot(),
                 str(grouped_twin.coupling_report())):
        assert "geometry" not in text, text
    twin = grouped_twin.coupling_diagnostics()[G.GROUP]
    assert "geometry" not in twin.get("not_usable_reason", "")
    assert not any(code.startswith("geometry_")
                   for codes in twin["reason_codes"].values() for code in codes)


def test_an_uncompiled_graph_s_text_names_the_geometry_too():
    text = _flat(G.graph(compile=False).format_graph())
    assert "geometry target.pos" in text and "geometry source.pos" in text


# ---------------------------------------------------------------------------
# The report of a group that resolves a geometry-dependent mapping
# ---------------------------------------------------------------------------


def test_the_report_of_a_single_rate_grid_group_has_its_numbers_and_no_flag_with_the_rule_s_reason(
        grouped):
    """The diagnostics read this group's geometry: its numbers are any
    other group's, on every printer.  The group solves the markers'
    positions (the scatter is anchored at its source), so in 0.4.0 it has
    neither flag, and the one reason says so."""
    report = grouped.coupling_diagnostics()[G.GROUP]
    reason = report["not_usable_reason"]
    assert reason.startswith("the group solves position(s) ['markers.pos']"), reason
    assert G.SCATTER in reason and "fixed during the pass" in reason
    assert "reported as computed, uncertified" in reason
    assert report["spectral_usable"] is False and report["gradient_bound_usable"] is False
    # At its tolerance the solve stops on the float floor: the report says
    # so, as it does for the static twin, and prints that caveat.
    assert report["ratio_usable"] is True and report["precision_limited"] is True
    for bound in ("rho_spectral", "spectral_error_bound", "gradient_relative_error_bound",
                  "error_estimate"):
        assert math.isfinite(report[bound]), (bound, dict(report))
    slot = f"coupling_{G.GROUP}_geometry_gap"
    assert float(grouped._state["_meta"][slot]) < 0.025  # noqa: SLF001
    (row,) = list(grouped.coupling_report())
    assert row["spectral_error_bound"] == report["spectral_error_bound"]
    # Nothing is withheld: the printers show the numbers, the floor's
    # caveat, and the rule's reason beside the flag it explains.
    assert not any("no bound" in flag for flag in row["flags"])
    assert any(flag.startswith("precision_limited=True") for flag in row["flags"]), row["flags"]
    assert f"spectral_usable=False: {reason}" in row["flags"], row["flags"]
    buffer = io.StringIO()
    grouped.print_coupling_report(file=buffer)
    assert "solves position(s) ['markers.pos']" in buffer.getvalue()
    json.loads(json_codec.dumps(dict(report)))


def test_the_coupling_report_shows_the_reason_and_no_caveat_about_a_withheld_value(withheld):
    grouped = withheld
    report = grouped.coupling_diagnostics()[G.GROUP]
    reason = report["not_usable_reason"]
    assert isinstance(reason, str) and G.GATHER in reason and G.SCATTER in reason
    assert "in a sub-cycled group" in reason and "['markers']" in reason
    table = grouped.coupling_report()
    (row,) = list(table)
    assert row["iterations"] == report["iterations"] and row["converged"] is report["converged"]
    assert row["residual"] == report["residual"]
    assert math.isnan(row["error_estimate"]) and math.isnan(row["spectral_error_bound"])
    assert row["ratio_usable"] is False and row["spectral_usable"] is False
    flags = row["flags"]
    assert any(reason in flag for flag in flags), flags
    # ``ratio_usable=False`` here says "withheld", not "the criterion fell
    # back to the raw residual test": that caveat would be a false one.
    for caveat in ("ratio_usable=False", "precision_limited=True", "spectral_usable=False"):
        assert not any(flag.startswith(caveat) for flag in flags), flags


def test_the_twin_s_report_keeps_its_own_caveats(grouped_twin):
    """The control for the flags: an ordinary group's caveats still print."""
    (row,) = list(grouped_twin.coupling_report())
    assert not any("no bound or estimate reported" in flag for flag in row["flags"])
    report = grouped_twin.coupling_diagnostics()[G.GROUP]
    expected = {"ratio_usable=False": not report["ratio_usable"],
                "precision_limited=True": report["precision_limited"]}
    assert any(expected.values()), report      # the fixture can express the difference
    for caveat, due in expected.items():
        assert any(flag.startswith(caveat) for flag in row["flags"]) == bool(due), row["flags"]


def test_the_printed_coupling_report_prints_the_reason(withheld):
    grouped = withheld
    buffer = io.StringIO()
    grouped.print_coupling_report(file=buffer)
    text = _flat(buffer.getvalue())
    assert "no bound or estimate reported: the group resolves geometry-dependent" in text
    assert G.GATHER in text and "iterations, residual and converged are the solve's own" in text
    assert _flat(str(grouped.coupling_report())) == text
    for width in (40, 200):
        narrow = io.StringIO()
        grouped.print_coupling_report(file=narrow, width=width)
        assert "the solve's own" in _flat(narrow.getvalue())


def test_a_report_with_the_reason_is_json_and_comes_back_as_it_was(withheld):
    grouped = withheld
    report = dict(grouped.coupling_diagnostics()[G.GROUP])
    back = json_codec.loads(json_codec.dumps(report))
    assert back["not_usable_reason"] == report["not_usable_reason"]
    assert back.keys() == report.keys()
    for key, value in report.items():
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(back[key]), key
        else:
            assert back[key] == value, key
    rows = json_codec.loads(json_codec.dumps([dict(r) for r in grouped.coupling_report()]))
    assert any(report["not_usable_reason"] in flag for flag in rows[0]["flags"])
    # Strict JSON: nothing in the encoded text is a bare NaN.
    json.loads(json_codec.dumps(report), parse_constant=lambda token: pytest.fail(token))


def test_the_profiler_reads_the_pass_count_of_a_group_with_a_geometry_edge():
    gm = G.graph(group=True, diagnostics=False)
    report = profile_graph(gm, n_steps=2, n_warmup=1, counts=False)
    assert report.coupling_iters[G.GROUP] == gm.coupling_diagnostics()[G.GROUP]["iterations"] > 0
    # Nothing is withheld from this group; its flags are False for two
    # causes (no diagnostics; it solves positions), each named.
    quick = gm.coupling_diagnostics()[G.GROUP]
    assert "do not read a moving geometry" not in quick["not_usable_reason"]
    assert quick["reason_codes"]["spectral_usable"] == [
        reason_codes.DIAGNOSTICS_OFF, reason_codes.GEOMETRY_POSITIONS_SOLVED]
    slow = G.graph(group=True, diagnostics=False, substeps=2)
    report = profile_graph(slow, n_steps=2, n_warmup=1, counts=False)
    assert report.coupling_iters[G.GROUP] == slow.coupling_diagnostics()[G.GROUP]["iterations"] > 0
    withheld = slow.coupling_diagnostics()[G.GROUP]
    assert "do not read a moving geometry" in withheld["not_usable_reason"]
    assert withheld["reason_codes"]["spectral_usable"] == [reason_codes.GEOMETRY_SUBCYCLED]


def test_the_raw_meta_slots_are_still_the_step_s_own_while_the_report_withholds_them(withheld):
    """Documented, not promised (``coupling_diagnostics``): the internal
    ``_meta`` entry keeps what the step computed for such a group; only
    the report withholds it."""
    grouped = withheld
    slot = f"coupling_{G.GROUP}_amplification"
    assert slot in grouped._state["_meta"]  # noqa: SLF001
    assert math.isnan(grouped.coupling_diagnostics()[G.GROUP]["amplification"])
    assert "_meta" in type(grouped).coupling_diagnostics.__doc__


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group", [False, True], ids=["plain step", "group"])
def test_a_restart_from_a_checkpoint_is_the_uninterrupted_run(tmp_path, group):
    kw = dict(group=True, diagnostics=False) if group else {}
    through = _stepped(**kw)
    path = through.save_state(tmp_path / "at3.npz")
    for _ in range(STEPS):
        through.step()

    restarted = G.graph(**kw)
    restarted.load_state(path)
    assert np.array_equal(G.states(restarted)["markers"]["pos"],
                          np.load(path)["markers/pos"])
    for _ in range(STEPS):
        restarted.step()
    G.assert_same_states(through, restarted, "restart")
    # The geometry moved over the run, so a restart that had lost it differs.
    assert not np.array_equal(np.asarray(G.Markers("m", G.DT).initial_state()["pos"]),
                              G.states(restarted)["markers"]["pos"])


@pytest.mark.parametrize("which", ["plain", "grouped"])
def test_a_checkpoint_holds_the_geometry_as_state_and_nothing_for_the_mapping(
        tmp_path, request, which):
    gm = request.getfixturevalue(which)
    assert gm.params["mappings"] == {G.GATHER: {}, G.SCATTER: {}}
    with np.load(gm.save_state(tmp_path / "c.npz")) as archive:
        members = set(archive.files)
        assert archive["markers/pos"].shape == (G.N_MARKERS, 1)
        assert archive["markers/pos"].dtype == np.float32
    assert {"grid/x", "markers/x", "markers/pos"} <= members
    assert not [m for m in members if m.startswith("_params_mappings")], members


def test_a_checkpoint_of_another_geometry_shape_is_refused_by_the_existing_state_check(
        tmp_path, plain):
    """The geometry is checked as any state field is on load."""
    gm = G.graph()
    path = plain.save_state(tmp_path / "c.npz")
    with np.load(path) as archive:
        members = {name: archive[name] for name in archive.files}
    members["markers/pos"] = members["markers/pos"].reshape(G.N_MARKERS)
    np.savez(tmp_path / "flat.npz", **members)
    before = G.states(gm)
    with pytest.raises(ValueError, match="pos"):
        gm.load_state(tmp_path / "flat.npz")
    after = G.states(gm)
    assert all(np.array_equal(before[n][f], after[n][f]) for n in before for f in before[n])
