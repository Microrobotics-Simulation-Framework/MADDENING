"""Every cell of the coupling searches is the configuration its index was pinned to.

The coupling searches (``test_coupling_targeted_search.py``,
``test_coupling_nonlinear_search.py``, ``test_coupling_geometry_search.py``)
each compile a tuple of *cells*, and keep what they found as pinned cases
that name a cell by its INDEX in that tuple.  The comment beside a pin
says what the cell is ("the fan-out hub with products of two fields in
float64, Jacobi under IQN-ILS, cap 120"); nothing else did.  On 2026-10-07
a row appended to the linear search's ``KNOBS`` re-rotated the nonlinear
search's cells, which took a modulus of that table's length: 21 of its 43
cells became other configurations, four slow tests went red on the release
branch a day later, and two pinned cases went on passing on configurations
their comments no longer described.

This module holds, per push and with no compile, a digest of everything
each cell bakes into its compiled graph (:func:`fields_of`: the cell's own
fields, the group's configuration as it is handed to
``add_coupling_group``, and the structure itself), for every search whose
pins name cells by index.  A cell that is another configuration than the
one pinned at its index fails, and the message names the cell and the
pinned cases that live on it.

**Adding a cell:** append it after every cell the search has (never
insert, never re-rotate), run this file, and append the digests its
message prints to :data:`PINNED`.  **A cell that moved:** put it back.  A
digest in :data:`PINNED` is replaced only together with every pinned case
on that cell, each re-derived on the new configuration.  A search lays its
cells out over tuples frozen for the purpose (``ROTATED_KNOBS``), never
over the length of a table another module can grow.

**Adding a field to a cell's class:** give it a default at which every
cell there is stays the configuration it was, and name the field and that
value in :data:`ADDED_SINCE_PINNED`.  At that value the field is left out
of the digest, so the digests pinned before the field existed still hold
those cells, unreplaced; at any other value it is in the digest like every
other field.  What the field decides for the group is in the digest either
way, through the configuration ``add_coupling_group`` receives.

``python -m tests.property.test_coupling_search_cells_are_pinned`` prints
the digests of the tree it runs on.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from tests.property import test_coupling_geometry_search as geometry
from tests.property import test_coupling_geometry_search_under_the_interface_norm as geometry_interface
from tests.property import test_coupling_nonlinear_search as nonlinear
from tests.property import test_coupling_targeted_search as linear

#: ``{search: the digest of each cell, in the cells' order}``.
PINNED = {
    "linear": (
        "79e1519c52db", "ec8a81bcfac2", "c01831e78053", "c8fc4e6501c5", "c8c280ffc0ed", "4cc9822477ce",
        "772be4f59b3b", "7d6642ccf228", "424d56602d7e", "fcc156fa9f16", "be6cc18871e2", "f8ab3838acbb",
        "b61f6d03fa63", "f9e35c28ef31", "10a65dddd9db", "d03b064783d2", "03e170ab4875", "1db9dc8e14ed",
        "9a1fb27a3a8e", "bc2db33ae648", "56411f7ca08b", "4359bc66e013", "50eb20193ca6", "4089377e81e6",
        "0973c4e9bc1a", "3cf00749c897", "ba965965304c", "b0646ff5b42a", "9248fa5678c2", "919a5fc7d27a",
        "db83c7421933", "90a596ccf376", "5def49785db2", "7c8f5837422d", "37123c9f11a5", "2d950a585c87",
        "fccec7cc2c98", "28a6afca9ab1", "9d10e9563248", "34f68da1fb96", "1c56d5dc2374", "5c49ebb7e513",
        "94c32558bc50", "2261bc0e63a6", "b2dbb82dd779", "d06167f16664", "f9c8986538cd", "d0321f6d1d72",
        "b1de96789961", "ea837fc64f73", "ab4438fb7fe2", "d062936f4434", "ebb7aaed3083", "9f1dda5a5337",
        "1fa307caf2c6", "654645bc86e6", "37a9497746c6", "e80c4012cb4e", "278badab3aee", "f16ae0360b77",
        "db7ac38e3ef4", "394cea625493", "2ee9946c049d", "7b7aeab5539e", "5c7aa92af513", "dbb789fb77eb",
        "87347f46e326", "98f2146ff774", "4b4257c0461d", "b205436d464c", "c84d52b37b80", "075512273e05",
        "ab90d2182e8a", "b7ca03a71aad",
    ),
    "linear-side": (
        "ed3cee5fc5b2", "0f8a4c96a855", "969211b27d17", "4d706f8f48aa", "343bc0d2c522", "60cdb48ad7ae",
        "c71ff1772dac", "4011f9566b1a", "27ccf30bf7c4", "07479ec454e3", "97a364efb489", "116917005145",
        "941e1f79f1ac", "2c0ab58b2afe", "3da0d5e701c8", "4a372bb5eb58", "68dbcc76eb8b", "774753160b56",
        "714ea7a369c5", "57d571f2bf6a", "458e9490794c", "5e0b01831c18", "a2ef5f31e535", "f9d69b42ffee",
        "bde2fed156e2", "ac2efee847bf", "01c5b7b9608b", "c239d9bfabf1", "c314227b657d", "aa5433ed3c0b",
        "93b8dd9924b6", "b4b8227fb9ac", "8d50a4e13d18", "66587d9c84a8", "26e35aea65b0", "0843a3bffddc",
    ),
    # The linear search's cells of the returned-state score (the state a
    # step returns under the interface norm, beside its report): named by
    # index by the slow hunt's blocks and by the test of a reading that
    # crosses zero.
    "linear-returned": (
        "d0058576b2c7", "9c1ded0b6a7a", "3f106935d556", "3641518af7d3", "02fa758cd4e4", "b778825fc19b",
        "7c099daf432d", "d518ea3e027e", "035de05d9652", "fc9e73708288", "e1c03e076e36", "37143cc90043",
        "b3a110a20f64", "87f700c28fb2", "f0aac2e43a07", "131163ba2745", "072a96b29d4c", "09ba4c4462ba",
        "670185db8319", "dcd302b847cd", "654c8dc76eea",
    ),
    "nonlinear": (
        "8e49c418d2d1", "bf58985620af", "f3110b07f131", "8ca3833a19d5", "df129a7b075c", "d2c8f50b84b9",
        "b4ec903a5fe2", "8201b83f5d80", "17533a05273d", "4d707e5d0473", "81b531b65d4c", "28c46d048b12",
        "17d7946ccb9e", "9fa582176fa6", "ed695e509ac5", "f48481487129", "dae727509925", "a06008a133bb",
        "3acb3b998372", "2a448ee6d446", "267903dfd015", "d18380dd0e21", "c0427beb5538", "9bc66dbfd670",
        "2c54727e1bae", "f757229e4a09", "896e2ecfb035", "86bb0dcc210e", "33993cc27eaa", "f945d8fa4323",
        "8beff3697817", "b7a7c8a24ee6", "f7f1f5578a8e", "700519c011f8", "b4a7705690d4", "3dbe2c0e4863",
        "07c83370b5b1", "c23df1c1d6e8", "b886f8f64b96", "0aab610e2a1f", "f72615efce40", "2426cb647dfb",
        "4cc3be5b814f", "36c94cd61198", "ebe7082625d9", "94c462e9802d", "93a6e3dc56fc", "a5c93eb1810d",
        "64585479923a", "2600dc30271d", "6a480c5b3943", "ef332ebf0f69", "603fd00c5270",
    ),
    "geometry": (
        "5890ebd22583", "8aa7cc3c0029", "f98ee00d436f", "2862dc84fe5e", "e4854bef88e0", "39361471fd8f",
        "025307cc1482", "f742005a3de2", "50e2fe11e465", "4468116f5a63", "ff58b423de1d", "48c1d341dcd8",
        "b833271f00d7", "ce7c1389ec02", "6b48ee75950e", "7762e2bbbc77", "94a052de54e8", "253b848577be",
        "66f835ce1560", "c43fbca7726f", "97f568bed0d8",
    ),
    "geometry-gauss-seidel": (
        "a706422d1e57", "3397ffc920f5",
    ),
    "geometry-plane": (
        "e7c3bf2c372b", "d6593c68e37e", "b6b80d84bf42", "26904b3f0c1a", "582f6f14f89d", "6dae87b28fab",
        "3bed1f83b716", "fca27d3445eb",
    ),
    # (Pinned on the tree that has the cells' ``tolerance`` field: these
    # cells were added with it, so no ``ADDED_SINCE_PINNED`` rule is theirs.)
    "geometry-interface": (
        "a743c3cd107e", "88385459b770", "bfc700f42464", "336f2c57e467", "5d6de56c724b", "51734d73f904",
        "4d8aadb650e0", "f334dd682540", "e1482cf04624",
    ),
}

#: ``{a cell's class: {a field it gained after its search was pinned: the
#: value at which a cell is the configuration it was before}}``.  The
#: geometry cells gained ``tolerance`` with the plane cells: zero is "the
#: linear search's tolerance", which is what every earlier cell ran at.
ADDED_SINCE_PINNED = {geometry.Cell: {"tolerance": 0.0}}


def searches() -> dict:
    """``{search: its cells}``, read when asked (a test replaces a table)."""
    return {"linear": linear.CELLS, "linear-side": linear.SIDE_CELLS,
            "linear-returned": linear.RETURNED_CELLS,
            "nonlinear": nonlinear.CELLS, "geometry": geometry.CELLS,
            "geometry-gauss-seidel": geometry.GS_CELLS,
            "geometry-plane": geometry.PLANE_CELLS,
            "geometry-interface": geometry_interface.CELLS}


def plain(value):
    """*value* as JSON holds it: dataclasses by their fields, tuples as
    lists, numpy scalars as Python's.  Anything else is refused: a field
    this cannot read would be a field the digest does not see."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"a cell holds {value!r} ({type(value).__name__}), which the digest cannot read")


def fields_of(cell) -> dict:
    """Everything *cell* bakes into its compiled graph: its own fields, the
    group's configuration as ``add_coupling_group`` receives it (the row of
    ``KNOBS`` resolved, not its index), the structure, and where a cell has
    them the nonlinearity of each member and the lattice.  A field of
    :data:`ADDED_SINCE_PINNED` is left out at the value named there."""
    own = plain(cell)
    for name, as_before in ADDED_SINCE_PINNED.get(type(cell), {}).items():
        if type(own[name]) is type(as_before) and own[name] == as_before:
            del own[name]
    out = {"cell": own, "knobs": plain(cell.knobs)}
    topo = getattr(cell, "topo", None)
    if topo is not None:
        out["structure"] = plain(topo)
    if hasattr(cell, "kind_of"):
        out["kinds"] = [cell.kind_of(name) for name in topo.names]
    if hasattr(cell, "grid"):
        out["grid"] = plain(cell.grid)
    return out


def digest(cell) -> str:
    text = json.dumps(fields_of(cell), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def describe(cell) -> str:
    """*cell* in a line: its fields, and its configuration in words."""
    knobs = cell.knobs
    fields = " ".join(str(getattr(cell, f.name)) for f in dataclasses.fields(cell))
    return (f"{fields} [{knobs['acceleration']}, {knobs['iteration_mode']}, "
            f"{knobs['convergence_norm']} norm, cap {knobs['max_iterations']}]")


def _case_in(value, kinds):
    """The case a pin table's entry holds: the entry, its first member, or
    the first value of a ``pytest.param``."""
    if isinstance(value, kinds):
        return value
    if hasattr(value, "values") and hasattr(value, "marks"):        # pytest.param
        value = value.values
    if isinstance(value, tuple) and value and isinstance(value[0], kinds):
        return value[0]
    return None


#: The pin tables that index another tuple of cells than their module's ``CELLS``.
_OTHER_CELLS = {("geometry", "GS_RADIUS_SEEDS"): "geometry-gauss-seidel"}


def pins() -> dict:
    """``{search: {cell index: [the pinned cases that name it]}}``: every
    module-level case, and every case in a module-level table, of the three
    search modules."""
    out = {name: {} for name in PINNED}
    sources = (("linear", linear, {linear.Case: "linear", linear.SideCase: "linear-side",
                                   linear.ReturnedCase: "linear-returned"}),
               ("nonlinear", nonlinear, {nonlinear.Case: "nonlinear"}),
               # A plane draw is a case of the geometry search's own cells
               # unless its table is in ``_OTHER_CELLS``.
               ("geometry", geometry, {geometry.Case: "geometry",
                                       geometry.gc.PlaneCase: "geometry"}))
    for module_name, module, kinds in sources:
        for name, value in vars(module).items():
            entries = (value.items() if isinstance(value, dict) else [(None, value)])
            for key, entry in entries:
                case = _case_in(entry, tuple(kinds))
                if case is None:
                    continue
                search = _OTHER_CELLS.get((module_name, name), kinds[type(case)])
                label = name if key is None else f"{name}[{key!r}]"
                out[search].setdefault(case.cell, []).append(f"{module_name} {label}")
    return out


def problems(search: str, cells=None, *, pinned=None) -> list:
    """What is wrong with *search*'s cells against :data:`PINNED`, one line
    each: a cell that is another configuration, cells that are not pinned,
    cells that are gone."""
    cells = searches()[search] if cells is None else cells
    pinned = PINNED[search] if pinned is None else pinned
    named = pins()[search]

    def on(index) -> str:
        return f"; pinned cases on it: {', '.join(named[index])}" if index in named else ""

    out = []
    for i, (cell, was) in enumerate(zip(cells, pinned)):
        now = digest(cell)
        if now != was:
            out.append(f"cell {i} MOVED: it is now {describe(cell)} (digest {now}, pinned "
                       f"{was}){on(i)}")
    if len(cells) > len(pinned):
        new = ", ".join(repr(digest(c)) for c in cells[len(pinned):])
        out.append(f"cells {len(pinned)} to {len(cells) - 1} are not pinned: if they were "
                   f"appended on purpose, append to PINNED[{search!r}]: {new}")
    for i in range(len(cells), len(pinned)):
        out.append(f"cell {i} is GONE (pinned {pinned[i]}){on(i)}")
    return out


@pytest.mark.parametrize("search", sorted(PINNED))
def test_every_cell_is_the_configuration_its_index_was_pinned_to(search):
    found = problems(search)
    assert not found, (
        f"the cells of the {search} search are not the ones pinned (a pinned case names its "
        f"cell by index, and its comment describes the configuration it was found on; see "
        f"this module's docstring):\n  " + "\n  ".join(found))


def test_every_pinned_case_names_a_cell_that_is_pinned():
    """The scan that names the pins on a moved cell reads every table it
    should (so the message is not silently short), and no pin names an
    index past the cells."""
    found = pins()
    # (No pinned case names a returned-state cell yet: "linear-returned" has none to count.)
    assert set(found) == set(PINNED) and not found["linear-returned"], sorted(found)
    for search, least in (("linear", 15), ("linear-side", 2), ("nonlinear", 6), ("geometry", 2),
                          ("geometry-gauss-seidel", 2)):
        count = sum(len(v) for v in found[search].values())
        assert count >= least, (search, found[search])
        assert all(0 <= i < len(PINNED[search]) for i in found[search]), (search, found[search])
    assert "nonlinear UNRESOLVED['a-converged-hub']" in found["nonlinear"][26]
    assert "nonlinear _NEVER_RETURNS" in found["nonlinear"][30]
    assert any("FIXED['MADD-ANO-226" in label for label in found["linear"][53])
    assert any("KNOWN['MADD-ANO-212-the-floor']" in label for label in found["linear"][2])
    assert any("KNOWN_SIDE" in label for label in found["linear-side"][1])
    assert "geometry PINNED_PLANES" in found["geometry"][0]


def _rotated_over(rows: int) -> tuple:
    """The nonlinear search's cells as it would lay them out with a
    rotation over *rows* configurations (seven: the cells there are)."""
    out = []
    for s, name in enumerate(nonlinear.STRUCTURES):
        for q, kind in enumerate(nonlinear.KINDS):
            for t, dtype in enumerate(("float32", "float64")):
                out.append(nonlinear.Cell(name, dtype, (s + 3 * t + 2 * q) % rows,
                                          nonlinear.ROTATED_CAPS[(s + t + q) % 2], kind))
    first = nonlinear.CELLS[:1]
    return first + tuple(c for c in out if c not in first)


#: The cells the rotation over eight rows moved on 2026-10-07.
MOVED_BY_AN_EIGHTH_ROW = (6, 12, 16, 18, 22, 23, 24, 26, 28, 29, 30, 32, 33, 34, 35, 36, 38, 39,
                          40, 41, 42)


def test_the_guard_names_the_cells_a_rotation_over_a_longer_table_moves():
    """The fault this module exists for, seeded: the nonlinear search's
    cells rotated over eight rows where they were laid out over seven.
    The guard names the 21 cells that became other configurations, and
    the two pinned cases that were on them; the same rotation over seven
    is the cells there are."""
    rotated = len(_rotated_over(len(nonlinear.ROTATED_KNOBS)))
    pinned = PINNED["nonlinear"][:rotated]
    assert rotated == 43 and nonlinear.CELLS[:rotated] == _rotated_over(7)
    assert problems("nonlinear", _rotated_over(7), pinned=pinned) == []
    found = problems("nonlinear", _rotated_over(8), pinned=pinned)
    moved = tuple(int(line.split()[1]) for line in found if " MOVED: " in line)
    assert moved == MOVED_BY_AN_EIGHTH_ROW and len(found) == len(moved), found
    by_cell = {int(line.split()[1]): line for line in found}
    assert "UNRESOLVED['a-converged-hub']" in by_cell[26], by_cell[26]
    assert "_NEVER_RETURNS" in by_cell[30], by_cell[30]
    assert "none, jacobi, interface norm" in by_cell[26], by_cell[26]
    assert "aitken, gauss-seidel, interface norm, cap 5" in by_cell[41], by_cell[41]


def test_the_guard_tells_an_appended_cell_from_a_moved_one_and_a_missing_one():
    cells = searches()["nonlinear"]
    extra = dataclasses.replace(cells[0], cap=7)
    (line,) = problems("nonlinear", cells + (extra,))
    assert "not pinned" in line and repr(digest(extra)) in line and "MOVED" not in line
    gone = problems("nonlinear", cells[:-2])
    assert len(gone) == 2 and all(" is GONE " in g for g in gone), gone
    # An insertion at the head moves every cell after it.
    shifted = problems("nonlinear", (extra,) + cells)
    assert sum(" MOVED: " in s for s in shifted) == len(cells) and "not pinned" in shifted[-1]


def test_a_digest_reads_everything_a_cell_bakes_in(monkeypatch):
    """Each field of a cell, the content of its row of ``KNOBS`` (not the
    row's index) and the content of its structure change the digest."""
    cell = nonlinear.CELLS[0]
    was = digest(cell)
    others = [dataclasses.replace(cell, dtype="float64"), dataclasses.replace(cell, cap=120),
              dataclasses.replace(cell, knob=0), dataclasses.replace(cell, kind="product"),
              dataclasses.replace(cell, structure="hub"),
              dataclasses.replace(cell, mapping_kind="sparse-local")]
    digests = [digest(c) for c in others]
    assert was not in digests and len(set(digests)) == len(digests), digests
    for search, cells in searches().items():
        assert len({digest(c) for c in cells}) == len(set(cells)), search

    # The same index, another row behind it (a row inserted at the head).
    rows = linear.KNOBS
    monkeypatch.setattr(linear, "KNOBS", (dict(rows[0], acceleration="aitken"),) + rows)
    assert digest(cell) != was and digest(linear.CELLS[0]) != PINNED["linear"][0]
    monkeypatch.setattr(linear, "KNOBS", rows)
    assert digest(cell) == was
    # The same name, another structure behind it.
    topo = nonlinear.STRUCTURES["tri"]
    monkeypatch.setitem(nonlinear.STRUCTURES, "tri", dataclasses.replace(
        topo, nodes=(dataclasses.replace(topo.nodes[0], alpha=0.25),) + topo.nodes[1:]))
    assert digest(cell) != was
    # A geometry cell's lattice and a field no digest could read.
    cell = geometry.CELLS[0]
    assert digest(dataclasses.replace(cell, origin=40.0)) != digest(cell)
    # A field a cell's class gained after the pins: out of the digest at
    # the value that leaves a cell what it was, in it at any other (and in
    # the configuration the group receives either way).
    assert ADDED_SINCE_PINNED[type(cell)] == {"tolerance": 0.0} and cell.tolerance == 0.0
    assert "tolerance" not in fields_of(cell)["cell"]
    stopped_early = fields_of(dataclasses.replace(cell, tolerance=1e-3))
    assert stopped_early["cell"]["tolerance"] == 1e-3
    assert stopped_early["knobs"] != fields_of(cell)["knobs"]
    assert all(c.tolerance for c in geometry.PLANE_CELLS)
    with pytest.raises(TypeError, match="cannot read"):
        plain({"a": object()})


if __name__ == "__main__":
    for _search, _cells in searches().items():
        print(f'    "{_search}": (')
        _digests = [digest(c) for c in _cells]
        for _k in range(0, len(_digests), 6):
            print("        " + " ".join(f'"{d}",' for d in _digests[_k:_k + 6]))
        print("    ),")
