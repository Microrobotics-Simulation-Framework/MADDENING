"""The mapping-spec resolver is a security boundary, tested against
*generated* input rather than against the cases a human thought of.

``tests/core/test_mapping_spec_hardening.py`` states the refusals an
audit named one by one -- a symlink out of the base directory, a forged
``.npy`` header, a zip bomb, a non-finite inline point, a reference to a
node that is gone.  Every one of those is a shape somebody imagined.
This module states the *total* property instead, over dicts Hypothesis
reaches by mutating a config the serialiser itself wrote:

    For any dict reachable by mutating a valid serialised mapping spec,
    ``from_dict`` either rebuilds a graph that behaves like the saved
    one, or raises a clear error naming the edge at fault.  It never
    silently produces a different graph, never escapes the base
    directory, and never dies with an opaque error that names neither
    the edge nor the reason.

"Behaves like the saved one" is the trichotomy actually checked, which
is a little more precise than the sentence above because a mutation is
allowed to *ask* for a different mapping -- editing ``epsilon`` in a
config is a legitimate thing to do, and refusing it would be wrong:

1. **refused** -- a ``ValueError`` (``MappingRebuildError`` and the
   ``add_edge`` shape checks both are) whose message names one endpoint
   of the offending edge and says why.  Never a ``TypeError``, a
   ``KeyError``, an ``AttributeError``, a ``RecursionError`` or anything
   else the caller cannot act on;
2. **accepted, same recipe** -- the mutated dict parses to a
   ``MappingSpec`` equal to the original's, so the reloaded graph must
   step bit for bit like the graph that was saved.  A stepped
   trajectory is compared, not the dicts: a serialiser that drops a
   hyper-parameter produces equal dicts and a different operator;
3. **accepted, different recipe** -- the file asked for something else,
   and the loader must have built *that*: the rebuilt mapping's own
   ``MappingSpec`` agrees with the one the mutated dict names.  This is
   what "never silently produces a different graph" means once a user
   is allowed to edit the file.

The asset resolver gets its own properties, over path shapes rather
than dict shapes, with a canary file planted outside the base directory
that no resolution is ever allowed to return.
"""

from __future__ import annotations

import copy
import json
import os
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, note, settings
from hypothesis import strategies as st

from maddening.core.coupling.mapping import (
    matrix_mapping,
    nearest_neighbor_mapping,
    projection_1d_mapping,
    rbf_mapping,
)
from maddening.core.coupling.mapping_spec import (
    MappingRebuildError,
    MappingSpec,
    PointReferenceError,
    make_point_resolver,
    normalise_point_reference,
    point_array_digest,
    reference_for_array,
)
from maddening.core.graph_manager import GraphManager

from tests.conftest import EXAMPLES_CHEAP, EXAMPLES_COSTLY, EXAMPLES_STANDARD
from tests.property.invariants import assert_states_identical
from tests.property.strategies import graph_recipes

#: Rollout length, as in ``test_round_trips.py``: long enough for a
#: mapped interface to feed back into the source node, short enough that
#: XLA compilation dominates either way.
N_STEPS = 3

#: Content of the file planted *outside* every base directory used here.
#: No resolution of any generated path may ever return it.
CANARY = np.array([-98765.25, 12345.75], dtype=np.float64)


# ---------------------------------------------------------------------------
# A filesystem sandbox: what an asset path may and may not reach
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """``base/`` (the config directory) beside ``outside/`` (everything a
    config must not reach), with the link shapes a single ``os.path``
    check misses.

    Module-scoped on purpose: a ``@given`` test may not take a
    function-scoped fixture, and nothing here is mutated by a property.
    """
    root = tmp_path_factory.mktemp("mapping_assets")
    outside = root / "outside"
    outside.mkdir()
    np.save(outside / "secret.npy", CANARY)
    np.savez(outside / "secret.npz", pts=CANARY)

    base = root / "cfg"
    base.mkdir()
    (base / "sub").mkdir()
    (base / "sub" / "deep").mkdir()
    np.save(base / "own.npy", np.array([0.0, 1.0, 2.0]))
    np.save(base / "sub" / "inner.npy", np.array([3.0, 4.0]))
    np.savez(base / "own.npz", pts=np.array([5.0, 6.0]))

    # A three-hop symlink chain ending outside: resolving only the first
    # link, or comparing the *lexical* path, lets this through.
    (base / "hop1.npy").symlink_to(base / "hop2.npy")
    (base / "hop2.npy").symlink_to(base / "hop3.npy")
    (base / "hop3.npy").symlink_to(outside / "secret.npy")
    # A chain that leaves and comes back: legal, and it must still work.
    (base / "detour.npy").symlink_to(outside / "back.npy")
    (outside / "back.npy").symlink_to(base / "own.npy")
    # A directory symlink, and a link to a directory symlink.
    (base / "dirlink").symlink_to(outside, target_is_directory=True)
    (base / "dirlink2").symlink_to(base / "dirlink", target_is_directory=True)
    # A link that points at itself.
    (base / "loop.npy").symlink_to(base / "loop.npy")
    # A name that is a different byte string but the same Unicode text
    # after NFC normalisation.
    np.save(base / unicodedata.normalize("NFC", "café.npy"), np.array([7.0]))
    # Not a regular file.
    os.mkfifo(base / "fifo.npy")

    return {"root": root, "base": base, "outside": outside}


#: Path shapes worth their own name.  Hypothesis generates the rest
#: around them (see ``_asset_paths``); these are the ones whose *point*
#: is lost if a random generator happens not to produce them.
_NAMED_PATHS = (
    # what must work
    "own.npy", "./own.npy", "sub/inner.npy", "sub//inner.npy", "sub/./inner.npy",
    "own.npz", "detour.npy", unicodedata.normalize("NFC", "café.npy"),
    # traversal, at several depths and disguises
    "../outside/secret.npy", "../../outside/secret.npy", "sub/../../outside/secret.npy",
    "sub/deep/../../../outside/secret.npy", "..", "../", "....//outside/secret.npy",
    "..%2foutside%2fsecret.npy", "%2e%2e/outside/secret.npy",
    "sub/../own.npy",                       # lexically fine, still refused
    # absolute, and absolute-looking
    "/etc/hostname.npy", "//outside/secret.npy", "\\..\\outside\\secret.npy",
    "~/secret.npy", "~root/secret.npy",
    # links
    "hop1.npy", "hop3.npy", "dirlink/secret.npy", "dirlink2/secret.npy", "loop.npy",
    # bytes an operating system argues about
    "\x00.npy", "own\x00.npy", "own.npy\x00.npy", "a\nb.npy", "a\rb.npy", "a\tb.npy",
    # Unicode that normalises onto another path, or is invisibly
    # indistinguishable from one (NFD, fullwidth, zero-width, Cyrillic)
    unicodedata.normalize("NFD", "café.npy"),
    "\uff4fwn.npy", "own\u200b.npy", "\u043ewn.npy",
    # legal here, illegal on another filesystem
    "CON.npy", "own.npy ", " own.npy", "own.npy.", "own:1.npy", "own|1.npy",
    "own.npy/", "own.npy/.", "own.NPY",
    # not a file, empty, and far too long
    "fifo.npy", "sub", "", "." * 40 + ".npy", "x" * 4096 + ".npy",
)


@st.composite
def _asset_paths(draw) -> str:
    """Relative asset paths: the named shapes above, plus ones assembled
    from adversarial components at a drawn depth."""
    if draw(st.booleans()):
        return draw(st.sampled_from(_NAMED_PATHS))
    component = st.one_of(
        st.sampled_from(("..", ".", "", "sub", "deep", "outside", "own.npy", "~",
                         "\x00", " ", "...", "́", "%2e%2e")),
        st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=6),
    )
    parts = draw(st.lists(component, min_size=1, max_size=5))
    suffix = draw(st.sampled_from((".npy", ".npz", "", ".np", ".NPY", ".npy.npy")))
    return "/".join(parts) + suffix


def _resolves_under(base: Path, rel: str) -> bool:
    """Whether ``rel`` really names a regular file under ``base`` --
    computed independently of the code under test."""
    try:
        real = (base / rel).resolve(strict=True)
        return real.is_file() and real.is_relative_to(base.resolve())
    except (OSError, RuntimeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Valid mapping specs, and the mutations of them
# ---------------------------------------------------------------------------

_INLINE_1D = [0.0, 0.25, 0.5, 0.75, 1.0]
_KERNELS = ("gaussian", "multiquadric", "inverse_multiquadric", "thin_plate_spline")


@st.composite
def _valid_spec_dicts(draw) -> dict:
    """``describe()`` dicts of mappings the real factories built -- the
    exact shape ``GraphManager.to_dict`` writes into a config."""
    kind = draw(st.sampled_from(("rbf", "nearest_neighbor", "projection_1d", "matrix")))
    mode = draw(st.sampled_from(("consistent", "conservative")))
    src = np.asarray(_INLINE_1D[:draw(st.integers(3, 5))])
    tgt = np.asarray(_INLINE_1D[:draw(st.integers(3, 5))])
    if kind == "rbf":
        mapping = rbf_mapping(src, tgt, mode=mode,
                              kernel=draw(st.sampled_from(_KERNELS)),
                              epsilon=draw(st.sampled_from((0.5, 1.0, 4.0))),
                              polynomial=draw(st.booleans()),
                              ridge=draw(st.sampled_from((0.0, 1e-8))))
    elif kind == "nearest_neighbor":
        mapping = nearest_neighbor_mapping(src, tgt, mode=mode)
    elif kind == "projection_1d":
        mapping = projection_1d_mapping(src, tgt)
    else:
        # ``matrix`` is always an asset reference; ``from_dict`` never
        # touches the filesystem, so the file need not exist for the
        # spec-level properties that use this.
        H = np.eye(len(tgt), len(src), dtype=np.float64)
        mapping = matrix_mapping(H, kind=draw(st.sampled_from(("matrix", "supermesh"))),
                                 mode=mode, asset="H.npy")
    return mapping.describe()


#: Values that *look* like something the schema accepts but belong to a
#: different slot: a valid kind where a dtype goes, a reference form
#: where a hyper-parameter goes, and so on.  The mutation most likely to
#: slip past a type check is one whose type is right.
_LOOKALIKES = (
    "rbf", "nearest_neighbor", "projection_1d", "matrix", "supermesh",
    "gaussian", "thin_plate_spline", "consistent", "conservative",
    "float64", "float32", "int64", "complex128", "float128", "object", "U8",
    {"inline": _INLINE_1D, "dtype": "float64"}, {"node": "rod", "field": "grid_x"},
    {"asset": "own.npy"}, {"asset": "../outside/secret.npy"},
    {"asset": "own.npz", "key": "pts"}, _INLINE_1D, "0" * 64, "f" * 64,
    point_array_digest(np.asarray(_INLINE_1D)),
)

_WEIRD_TEXT = st.one_of(
    st.just(""),
    st.text(max_size=8),
    st.sampled_from(("..", "../..", "/etc/passwd", "\x00", "\n", "kind", "points",
                     "inline", "asset", "node", "sha256", "-0.0", "1e400", "NaN",
                     "x" * 4096)),
)

#: Junk of every JSON shape, nested a few levels deep.  ``max_leaves``
#: is small because the point is the *shape*, and a big structure only
#: makes the shrinker slower.
_JUNK = st.recursive(
    st.one_of(
        st.none(), st.booleans(), st.integers(-(2 ** 70), 2 ** 70),
        st.floats(allow_nan=True, allow_infinity=True),
        _WEIRD_TEXT, st.sampled_from(_LOOKALIKES),
    ),
    lambda child: st.one_of(st.lists(child, max_size=3),
                            st.dictionaries(_WEIRD_TEXT, child, max_size=3)),
    max_leaves=5,
)


def _paths(value: Any, prefix: tuple = ()) -> list[tuple]:
    """Every position inside a nested dict / list, as a key path."""
    out = [prefix]
    if isinstance(value, dict):
        for key, sub in value.items():
            out.extend(_paths(sub, prefix + (key,)))
    elif isinstance(value, list):
        for index, sub in enumerate(value):
            out.extend(_paths(sub, prefix + (index,)))
    return out


def _at(value: Any, path: tuple) -> Any:
    for step in path:
        value = value[step]
    return value


@st.composite
def _mutated(draw, spec: dict) -> tuple[dict, str]:
    """``spec`` with one mutation applied, and a line describing it.

    Every operation the brief names is here: drop a key, duplicate one
    under a name that differs only cosmetically, change a type in both
    directions, inject a nested structure, empty and very long strings,
    and swap a value for a valid-looking one of the wrong kind.
    """
    out = copy.deepcopy(spec)
    path = draw(st.sampled_from(_paths(out)))
    parent = _at(out, path[:-1]) if path else None
    key = path[-1] if path else None
    value = _at(out, path)
    ops = ["replace", "wrap", "retype"]
    if parent is not None:
        ops += ["drop", "lookalike"]
        if isinstance(parent, dict):
            ops += ["duplicate", "alias"]
    op = draw(st.sampled_from(ops))

    if op == "drop":
        del parent[key]
        return out, f"drop {path}"
    if op == "duplicate":
        # A dict cannot hold the same key twice, so duplicate it under a
        # name a reader might treat as the same one.
        twin = draw(st.sampled_from((f"{key} ", f"{key}".upper(), f"{key}_",
                                     f" {key}", f"{key}\u200b")))
        parent[twin] = copy.deepcopy(value)
        return out, f"duplicate {path} as {twin!r}"
    if op == "alias":
        siblings = [k for k in parent if k != key]
        if siblings:
            other = draw(st.sampled_from(siblings))
            parent[other] = copy.deepcopy(value)
            return out, f"alias {path} onto {other!r}"
        op = "replace"
    if op == "lookalike":
        new = draw(st.sampled_from(_LOOKALIKES))
    elif op == "wrap":
        # A deeply nested structure where a leaf belongs.
        new = copy.deepcopy(value)
        for _ in range(draw(st.integers(1, 40))):
            new = [new] if draw(st.booleans()) else {"value": new}
    elif op == "retype":
        # A string where a number is expected, and the reverse.
        if isinstance(value, bool):
            new = draw(st.sampled_from(("true", "True", 1, 0, "")))
        elif isinstance(value, (int, float)):
            new = draw(st.sampled_from((str(value), f" {value} ", [value], {"v": value})))
        elif isinstance(value, str):
            new = draw(st.sampled_from((len(value), float(len(value)), "", "x" * 4096,
                                        list(value[:3]), None)))
        else:
            new = draw(_JUNK)
    else:
        new = draw(_JUNK)

    if path:
        parent[key] = new
    else:
        out = new if isinstance(new, dict) else {"mutated": new}
    return out, f"{op} {path} -> {new!r}"[:200]


# ---------------------------------------------------------------------------
# Totality of the pieces: no opaque failure, ever
# ---------------------------------------------------------------------------

@settings(max_examples=EXAMPLES_CHEAP)
@given(value=_JUNK)
def test_normalise_point_reference_answers_or_raises_point_reference_error(value):
    """``normalise_point_reference`` is total on arbitrary JSON: it
    returns a canonical reference or says, as a ``PointReferenceError``,
    why it will not.  Anything else -- a ``TypeError`` out of numpy, a
    ``KeyError``, a ``RecursionError`` from a nested list -- reaches the
    caller as a stack trace about the wrong subject."""
    try:
        ref = normalise_point_reference(value, name="points")
    except PointReferenceError as exc:
        assert str(exc).startswith("points:"), f"error does not name the slot: {exc}"
        assert len(str(exc)) > len("points:") + 8, f"error gives no reason: {exc}"
        return
    assert isinstance(ref, dict) and len(ref) >= 1
    # Canonical: exactly one of the three forms, and normalising again
    # is a no-op (otherwise "canonical" means nothing).
    assert sum(k in ref for k in ("node", "asset", "inline")) == 1
    assert normalise_point_reference(ref, name="points") == ref


@settings(max_examples=EXAMPLES_CHEAP,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(data=st.data())
def test_a_mutated_spec_dict_parses_to_a_canonical_spec_or_a_value_error(data):
    """``MappingSpec.from_dict`` on a mutated recipe: a ``MappingSpec``
    whose own ``to_dict`` parses back to itself, or a ``ValueError``.

    The two halves matter equally.  A ``TypeError`` (``unhashable type:
    'list'`` for a ``kind`` that is a list, which is what this found)
    tells the caller nothing about the config; and a spec that parses
    but does not survive its own ``to_dict`` is a recipe the next writer
    silently changes."""
    spec_dict = data.draw(_valid_spec_dicts())
    mutated, what = data.draw(_mutated(spec_dict))
    note(f"original: {spec_dict}")
    note(f"mutation: {what}")
    try:
        spec = MappingSpec.from_dict(mutated)
    except ValueError as exc:
        assert str(exc).strip(), "refusal with an empty message"
        return
    assert isinstance(spec, MappingSpec)
    assert MappingSpec.from_dict(spec.to_dict()) == spec, (
        "a spec that does not survive its own to_dict: the next writer changes it"
    )


@st.composite
def _equivalent(draw, mapping: dict) -> tuple[dict, str]:
    """``mapping`` rewritten without changing what it means.

    Every rewrite here is a shape a hand-edited config, a different YAML
    dumper or an older writer really produces, and every one of them has
    to rebuild the same operator -- which is the other half of "never
    silently produces a different graph": a file that still means what
    it meant must not load as something else either.
    """
    out = copy.deepcopy(mapping)
    refs = [k for k, v in out["points"].items() if isinstance(v, dict)]
    ops = ["reorder"]
    if "shape" in out:
        ops.append("drop_shape")
    if any(isinstance(out["points"][k].get("sha256"), str) for k in refs):
        ops.append("drop_sha256")
    if any(isinstance(out.get(k), float) and float(out[k]).is_integer()
           for k in ("epsilon", "ridge")):
        ops.append("int_hyper")
    if any("inline" in out["points"][k] for k in refs):
        ops += ["dtype_alias", "bare_inline"]
    if any(set(out["points"][k]) == {"asset"} for k in refs):
        ops.append("bare_asset")
    op = draw(st.sampled_from(ops))

    if op == "reorder":
        keys = draw(st.permutations(list(out)))
        out = {k: out[k] for k in keys}
        out["points"] = {k: out["points"][k]
                         for k in draw(st.permutations(list(out["points"])))}
    elif op == "drop_shape":
        del out["shape"]
    elif op == "drop_sha256":
        for key in refs:
            out["points"][key].pop("sha256", None)
    elif op == "int_hyper":
        for key in ("epsilon", "ridge"):
            if isinstance(out.get(key), float) and float(out[key]).is_integer():
                out[key] = int(out[key])
    elif op == "dtype_alias":
        # ``np.dtype`` spells one dtype many ways; the canonical name is
        # what the spec must come back holding.
        aliases = {"float64": ("f8", "<f8", "double", "float"),
                   "float32": ("f4", "<f4", "single"),
                   "int64": ("i8", "<i8"), "int32": ("i4", "<i4")}
        for key in refs:
            name = out["points"][key].get("dtype")
            if name in aliases:
                out["points"][key]["dtype"] = draw(st.sampled_from(aliases[name]))
    elif op == "bare_inline":
        # A plain list is the documented shorthand for a float64 set.
        for key in refs:
            ref = out["points"][key]
            if "inline" in ref and ref.get("dtype", "float64") == "float64":
                out["points"][key] = ref["inline"]
    else:
        for key in refs:
            if set(out["points"][key]) == {"asset"}:
                out["points"][key] = out["points"][key]["asset"]
    return out, op


# ---------------------------------------------------------------------------
# The trichotomy, over a whole config
# ---------------------------------------------------------------------------

def _mapped_edges(config: dict) -> list[int]:
    return [i for i, e in enumerate(config["edges"]) if e.get("mapping") is not None]


def _endpoints(edge: dict) -> tuple[str, str]:
    return (f"{edge['source_node']}.{edge['source_field']}",
            f"{edge['target_node']}.{edge['target_field']}")


def _assert_names_the_edge(exc: BaseException, edge: dict) -> None:
    """A refusal has to be actionable: the type a caller can catch, the
    edge it happened on, and a reason."""
    assert isinstance(exc, ValueError), (
        f"{type(exc).__name__} is not a ValueError, so a loader cannot catch it "
        f"alongside every other bad-config failure: {exc}"
    )
    text = str(exc)
    source, target = _endpoints(edge)
    if isinstance(exc, MappingRebuildError):
        assert exc.edge == f"{source} -> {target}"
        assert exc.__cause__ is not None, "the original failure was not chained"
    assert source in text or target in text, (
        f"refusal names neither endpoint of the edge it happened on: {text}"
    )
    reason = text.replace(source, "").replace(target, "")
    assert len(reason.strip(" :->")) > 10, f"refusal gives no reason: {text}"


def _spec_agrees(named: MappingSpec, built: MappingSpec) -> bool:
    """Whether ``built`` is the recipe ``named`` asked for.

    Not plain equality, for two documented reasons: a factory fills in
    the hyper-parameters the file left out with its own defaults, and
    rebuilding *records* the content hash of every reference that did
    not carry one.  So ``built`` may say more than ``named``; it may not
    say anything *different*.
    """
    if named.kind != built.kind:
        return False
    for key, value in named.hyperparameters.items():
        if key not in built.hyperparameters or built.hyperparameters[key] != value:
            return False
    if set(named.points) != set(built.points):
        return False
    for key, ref in named.points.items():
        other = built.points[key]
        if ref is None or other is None:
            if ref is not other:
                return False
            continue
        if any(other.get(k) != v for k, v in ref.items()):
            return False
        if set(other) - set(ref) - {"sha256"}:
            return False
    return True


@settings(max_examples=EXAMPLES_COSTLY,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(data=st.data())
def test_a_mutated_mapping_is_refused_by_name_or_loads_exactly_what_it_says(
        data, sandbox):
    """The whole property, over a config the serialiser wrote and
    Hypothesis then edited: refused with the edge named, or loaded as
    the recipe the file holds -- and, when the mutation left the recipe
    alone, stepping identically to the graph that was saved.

    ``base_dir`` is the sandbox, never the working directory: a mutation
    is free to invent an ``{"asset": ...}`` reference, the default
    resolver would look for it in the repository checkout, and pointing
    it at a directory that really holds assets (and links out of itself)
    is what lets an invented reference get far enough to be interesting.
    """
    recipe = data.draw(graph_recipes(require_mapping=True, max_nodes=3,
                                     allow_coupling_groups=False))
    gm = recipe.build()
    config = gm.to_dict()
    mapped = _mapped_edges(config)
    assume(mapped)
    index = data.draw(st.sampled_from(mapped))
    edge = config["edges"][index]
    mutated_mapping, what = data.draw(_mutated(edge["mapping"]))
    note(f"edge {index}: {_endpoints(edge)}")
    note(f"mutation: {what}")

    mutated = copy.deepcopy(config)
    mutated["edges"][index]["mapping"] = mutated_mapping
    base = sandbox["base"]

    try:
        reloaded = GraphManager.from_dict(mutated, recipe.registry, base_dir=base)
    except Exception as exc:                              # noqa: BLE001 -- the point
        _assert_names_the_edge(exc, edge)
        return

    # Accepted.  Whatever it built has to be the recipe the file names.
    named = MappingSpec.from_dict(mutated_mapping)
    built = reloaded.edges[index].mapping.spec
    assert _spec_agrees(named, built), (
        f"the config names {named} but the loader built {built}: a silently "
        f"different graph"
    )
    assert MappingSpec.from_dict(built.to_dict()) == built, (
        "the rebuilt recipe does not survive being written out again"
    )
    # ... and nothing it resolved came from outside the config directory.
    resolve = reloaded.point_resolver(base)
    for ref in built.points.values():
        points = np.asarray(resolve(ref))
        assert not (points.shape == CANARY.shape and np.array_equal(points, CANARY)), (
            f"the mutated spec resolved {ref!r} to the canary outside {base}"
        )

    if named == MappingSpec.from_dict(edge["mapping"]):
        # The mutation was cosmetic (a dropped ``shape``, a re-spelled
        # hyper-parameter that normalises back).  Same recipe, same
        # graph -- and "same graph" is a trajectory, not a dict.
        reloaded.compile()
        assert_states_identical(gm.run_scan(N_STEPS), reloaded.run_scan(N_STEPS),
                                what="trajectory after a recipe-preserving mutation")


@settings(max_examples=EXAMPLES_COSTLY,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(data=st.data())
def test_a_spec_rewritten_without_changing_its_meaning_steps_the_same(data, sandbox):
    """The other half of the trichotomy's middle branch, reached on
    purpose rather than by luck.

    Blind mutation almost always changes (or breaks) the recipe, so the
    "same recipe, therefore the same trajectory" case above fires on a
    handful of examples at most.  These rewrites are the ones a hand
    edit, a different YAML dumper or an older writer really produces --
    reordered keys, a dropped advisory ``shape``, a dropped ``sha256``,
    an integer where a float was written, a dtype spelled ``f8``, the
    documented bare-list and bare-string shorthands -- and every one of
    them has to rebuild the same operator, compared as a trajectory.
    """
    recipe = data.draw(graph_recipes(require_mapping=True, max_nodes=3,
                                     allow_coupling_groups=False))
    gm = recipe.build()
    config = gm.to_dict()
    mapped = _mapped_edges(config)
    assume(mapped)
    index = data.draw(st.sampled_from(mapped))
    rewritten, how = data.draw(_equivalent(config["edges"][index]["mapping"]))
    note(f"rewrite: {how}")
    note(f"was: {config['edges'][index]['mapping']}")
    note(f"now: {rewritten}")

    mutated = copy.deepcopy(config)
    mutated["edges"][index]["mapping"] = rewritten
    base = sandbox["root"] / "empty"
    base.mkdir(exist_ok=True)

    reloaded = GraphManager.from_dict(mutated, recipe.registry, base_dir=base)
    reloaded.compile()
    assert_states_identical(gm.run_scan(N_STEPS), reloaded.run_scan(N_STEPS),
                            what=f"trajectory after an equivalent rewrite ({how})")


# ---------------------------------------------------------------------------
# The asset resolver: adversarial paths
# ---------------------------------------------------------------------------

@settings(max_examples=EXAMPLES_CHEAP)
@given(rel=_asset_paths())
def test_an_asset_path_reaches_a_file_under_the_base_directory_or_nothing(rel, sandbox):
    """For any relative path at all: the resolver returns the contents of
    a regular file that really lies under ``base_dir`` after every
    symlink is followed, or refuses with a ``PointReferenceError``.

    The canary planted in ``outside/`` is the direct statement of the
    security property -- if its contents ever come back, a config read a
    file it had no business reading, whatever the path looked like.
    """
    resolve = make_point_resolver(base_dir=sandbox["base"])
    try:
        array = resolve({"asset": rel})
    except PointReferenceError as exc:
        assert len(str(exc)) > 20, f"refusal gives no reason: {exc}"
        return
    array = np.asarray(array)
    assert not (array.shape == CANARY.shape and np.array_equal(array, CANARY)), (
        f"asset path {rel!r} read the canary from outside the base directory"
    )
    assert _resolves_under(sandbox["base"], rel), (
        f"asset path {rel!r} loaded an array from outside {sandbox['base']}"
    )
    assert array.dtype.kind in "biuf"


@settings(max_examples=EXAMPLES_STANDARD)
@given(rel=_asset_paths(), key=st.one_of(st.none(), _WEIRD_TEXT,
                                         st.sampled_from(("pts", "arr_0"))))
def test_an_npz_member_name_cannot_widen_what_an_asset_path_reaches(rel, key, sandbox):
    """The ``key`` of an ``.npz`` reference selects a member *inside* an
    archive; it is a second attacker-controlled string on the same code
    path, and it may not turn a refused path into an accepted one."""
    resolve = make_point_resolver(base_dir=sandbox["base"])
    ref = {"asset": rel} if key is None else {"asset": rel, "key": key}
    try:
        array = np.asarray(resolve(ref))
    except PointReferenceError:
        return
    assert not (array.shape == CANARY.shape and np.array_equal(array, CANARY))
    assert _resolves_under(sandbox["base"], rel)


@settings(max_examples=EXAMPLES_CHEAP)
@given(rel=_asset_paths())
def test_an_asset_reference_normalises_only_paths_that_stay_relative(rel):
    """Containment is decided twice -- lexically when the reference is
    normalised, and again against the resolved path when it is loaded.
    The lexical half must refuse an absolute path or a ``..`` component
    without touching the filesystem at all, so a config is refused the
    same way on a machine where the file happens not to exist."""
    path = Path(rel)
    try:
        ref = normalise_point_reference({"asset": rel}, name="points")
    except PointReferenceError:
        return
    assert not path.is_absolute() and ".." not in path.parts
    assert ref["asset"] == rel


def test_a_symlink_created_after_the_first_resolution_is_still_refused(tmp_path):
    """Each resolution re-walks the path: an answer is never cached, so a
    component that appears (or changes) between two loads is judged
    afresh.  The shape the brief asks about -- a path that resolves
    differently once a component is created concurrently -- is exactly
    this, and the second load must not inherit the first one's verdict.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    np.save(outside / "secret.npy", CANARY)
    base = tmp_path / "cfg"
    base.mkdir()
    resolve = make_point_resolver(base_dir=base)

    with pytest.raises(PointReferenceError, match="missing point asset"):
        resolve({"asset": "late.npy"})

    # ... the component appears, pointing outside.
    (base / "late.npy").symlink_to(outside / "secret.npy")
    with pytest.raises(PointReferenceError, match="outside the config directory"):
        resolve({"asset": "late.npy"})

    # ... and is replaced by an honest file of the same name.
    (base / "late.npy").unlink()
    np.save(base / "late.npy", np.array([1.0, 2.0]))
    np.testing.assert_array_equal(resolve({"asset": "late.npy"}), [1.0, 2.0])


def test_a_hard_link_is_read_because_no_path_check_can_tell_it_apart(tmp_path):
    """The one escape a path check cannot close, pinned so nobody reads
    the properties above as promising more than they do.

    A hard link *is* a directory entry under the base directory -- it has
    no target to resolve and no way to be distinguished from the original
    -- so the resolver reads it.  Making that an error would need the
    loader to reject every file with ``st_nlink > 1``, which would refuse
    ordinary deduplicated checkouts.  The threat it leaves open requires
    write access to the config directory, at which point the attacker can
    simply copy the file in; the containment check is about what a
    *config* can name, not about a hostile directory.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    np.save(outside / "secret.npy", CANARY)
    base = tmp_path / "cfg"
    base.mkdir()
    os.link(outside / "secret.npy", base / "hard.npy")

    loaded = make_point_resolver(base_dir=base)({"asset": "hard.npy"})
    np.testing.assert_array_equal(loaded, CANARY)


# ---------------------------------------------------------------------------
# Round-trip totality through an asset reference
# ---------------------------------------------------------------------------

@settings(max_examples=EXAMPLES_COSTLY,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(recipe=graph_recipes(require_mapping=True, max_nodes=3,
                            allow_coupling_groups=False))
def test_a_mapped_edge_rewritten_to_asset_references_reloads_the_same_trajectory(
        recipe, tmp_path_factory):
    """``tests/property/strategies.py`` draws node-field and inline point
    references; the third form, ``{"asset": "<file>.npy"}``, is the one
    the resolver's containment and size checks guard, and no property
    covered a whole graph through it.

    Every reference of every mapped edge is rewritten to an asset
    holding exactly the points that reference resolves to -- so the
    recipe is unchanged by construction -- and the reloaded graph has to
    step bit for bit like the saved one.
    """
    gm = recipe.build()
    config = gm.to_dict()
    mapped = _mapped_edges(config)
    assume(mapped)

    base = tmp_path_factory.mktemp("assets")
    resolve = gm.point_resolver()
    for count, index in enumerate(mapped):
        mapping = config["edges"][index]["mapping"]
        for slot, ref in list(mapping["points"].items()):
            if ref is None:
                continue
            points = np.asarray(resolve(ref))
            rel = f"e{count}_{slot}.npy"
            np.save(base / rel, points)
            mapping["points"][slot] = {"asset": rel,
                                       "sha256": point_array_digest(points)}

    reloaded = GraphManager.from_dict(config, recipe.registry, base_dir=base)
    reloaded.compile()
    assert_states_identical(gm.run_scan(N_STEPS), reloaded.run_scan(N_STEPS),
                            what="trajectory through asset references")


@settings(max_examples=EXAMPLES_STANDARD)
@given(data=st.data())
def test_an_asset_whose_bytes_changed_since_the_save_is_refused_not_rebuilt(
        data, tmp_path_factory):
    """The content hash is the only thing standing between "the file next
    to the config was edited" and "the operator quietly changed".  For
    any edit at all -- a different value, a different length, a different
    dtype -- the load must fail rather than rebuild."""
    original = np.asarray(data.draw(st.lists(st.integers(-20, 20), min_size=2,
                                             max_size=6)), dtype=np.float64)
    dtype = data.draw(st.sampled_from(("float64", "float32", "int64")))
    changed = np.asarray(data.draw(st.lists(st.integers(-20, 20), min_size=1,
                                            max_size=6)), dtype=dtype)
    assume(point_array_digest(changed) != point_array_digest(original))

    base = tmp_path_factory.mktemp("edited")
    np.save(base / "pts.npy", changed)
    ref = {"asset": "pts.npy", "sha256": point_array_digest(original)}
    spec = MappingSpec("nearest_neighbor", {"mode": "consistent"},
                       {"source_points": ref, "target_points": ref})
    with pytest.raises(PointReferenceError, match="differs from the points"):
        spec.build(make_point_resolver(base_dir=base))


# ---------------------------------------------------------------------------
# Which dtypes a point set may have
# ---------------------------------------------------------------------------
#
# The accepted set is not a list of names anybody has to keep in sync: it
# is whatever a reference can actually *do* with a dtype.  A reference is
# written as JSON (``{"inline": arr.tolist(), "dtype": ...}``) and it is
# identified by ``point_array_digest``, so a dtype is usable exactly when
# ``json.dumps(arr.tolist())`` round-trips it and equal arrays of it hash
# alike.  ``np.longdouble`` fails both and is refused; the property below
# states the equivalence rather than enumerating the survivors.

#: Dtypes to try, by ``dtype.str`` so a platform where ``longdouble`` *is*
#: ``float64`` simply contributes one dtype instead of two.  Byte-order
#: variants are in because a reference records ``str(dtype)`` verbatim.
_CANDIDATE_POINT_DTYPES = sorted({
    np.dtype(t).str for t in (
        np.bool_, np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64, np.longdouble,
        np.complex64, np.complex128, np.clongdouble,
    )
} | {"U8", "O", "M8[ns]", "m8[ns]", ">f8", "<f4", ">i4", ">u8"})

_POINT_VALUES = [[1, 2], [3, 4]]


def _equal_arrays_of(dtype: np.dtype) -> list[np.ndarray]:
    """Arrays of ``dtype`` that all compare equal, built by different routes.

    The routes matter: an extended-precision float leaves whatever was in
    the unused bytes of its slot behind, so two arrays that ``==`` each
    other can still differ byte for byte depending on how they were made.
    Routes a dtype cannot support are dropped rather than faked.
    """
    base = np.array(_POINT_VALUES, dtype=dtype)
    out = [base]
    for build in (
        lambda: np.array(_POINT_VALUES).astype(dtype),               # via int64
        lambda: np.array([[0, 0]] + _POINT_VALUES, dtype=dtype)[1:],  # a slice
        lambda: np.asarray(np.array(_POINT_VALUES, dtype=dtype).T.copy().T),
    ):
        try:
            candidate = build()
        except (TypeError, ValueError):
            continue
        if candidate.dtype == dtype and np.array_equal(candidate, base):
            out.append(np.ascontiguousarray(candidate))
    try:                                          # a deliberately dirty buffer
        poisoned = np.empty(base.shape, dtype=dtype)
        poisoned.view(np.uint8)[:] = 0xA5
        poisoned[:] = base
        out.append(poisoned)
    except (TypeError, ValueError):
        pass
    return out


def _digest_is_stable(dtype: np.dtype) -> bool:
    """Do equal arrays of ``dtype`` have identical canonical bytes?

    Deliberately not ``point_array_digest``: that function now refuses the
    dtypes this asks about, and the question here is about the bytes it
    would have hashed.
    """
    canonical = {(a.dtype.str, a.shape, a.tobytes(order="C"))
                 for a in _equal_arrays_of(dtype)}
    return len(canonical) == 1


def _survives_json(dtype: np.dtype) -> bool:
    """Does ``{"inline": arr.tolist(), "dtype": ...}`` survive JSON?"""
    arr = np.array(_POINT_VALUES, dtype=dtype)
    try:
        text = json.dumps({"inline": arr.tolist(), "dtype": str(dtype)})
    except (TypeError, ValueError):
        return False
    back = json.loads(text)
    try:
        restored = np.asarray(back["inline"], dtype=np.dtype(back["dtype"]))
    except (TypeError, ValueError):
        return False
    return restored.dtype == arr.dtype and bool(np.array_equal(restored, arr))


@settings(max_examples=EXAMPLES_CHEAP)
@given(dtype_str=st.sampled_from(_CANDIDATE_POINT_DTYPES))
def test_a_point_dtype_is_accepted_exactly_when_it_writes_and_hashes(dtype_str):
    """Accepted iff usable: the two entry points agree with each other, and
    with what JSON and the digest can actually do.

    ``normalise_point_reference`` (a hand-written config) and
    ``reference_for_array`` (a factory recording its own points) must reach
    the same verdict, an accepted dtype must genuinely round-trip and hash
    stably, and a *numeric* dtype may only be refused when it genuinely
    fails one of those two -- so nothing legitimate is caught by the
    narrowing, and nothing unusable slips through it.
    """
    dtype = np.dtype(dtype_str)
    arr = np.array(_POINT_VALUES, dtype=dtype)
    usable = _survives_json(dtype) and _digest_is_stable(dtype)
    note(f"{dtype_str}: kind={dtype.kind} itemsize={dtype.itemsize} "
         f"json={_survives_json(dtype)} stable={_digest_is_stable(dtype)}")

    try:
        ref = normalise_point_reference({"inline": arr.tolist(), "dtype": str(dtype)},
                                        name="source_points")
        accepted, refusal = True, None
    except PointReferenceError as exc:
        ref, accepted, refusal = None, False, exc

    # the factory entry point reaches the same verdict on the same array
    try:
        recorded = reference_for_array(arr, None, name="source_points")
    except PointReferenceError:
        recorded = None
    assert (recorded is not None) == accepted

    if accepted:
        # ...then everything downstream of acceptance has to work
        assert usable, f"{dtype_str} was accepted but cannot be written or hashed"
        assert json.loads(json.dumps(ref)) == ref
        assert point_array_digest(arr) == point_array_digest(arr.copy())
        assert ref == recorded
    else:
        assert dtype.kind not in "biuf" or not usable, (
            f"{dtype_str} writes and hashes cleanly but was refused: {refusal}")
        assert dtype.name in str(refusal) or str(dtype) in str(refusal), (
            f"a refusal must name the dtype it refused, got: {refusal}")


@pytest.mark.skipif(np.dtype(np.longdouble).itemsize <= 8,
                    reason="this platform's longdouble is float64, so there is no "
                           "extended-precision case to make")
def test_an_inline_extended_precision_point_set_is_refused_not_written_out():
    """The decided shape of the extended-precision corner.

    This was a strict xfail while the choice was open: an inline
    ``float128`` point set was accepted and then could not be written
    (``arr.tolist()`` yields ``np.longdouble`` objects that ``json.dumps``
    refuses) and could not be hashed stably.  The decision was to narrow
    the accepted dtypes and to say so out loud -- never to downcast to
    ``float64``, because a silent narrowing is a well-known source of
    numerical bugs and this library will not add to it.
    """
    points = np.array([1.0, 2.0, 3.0], dtype=np.longdouble)
    with pytest.raises(PointReferenceError) as excinfo:
        normalise_point_reference({"inline": points.tolist(), "dtype": "float128"},
                                  name="points")
    message = str(excinfo.value)
    assert "float128" in message and "extended-precision" in message
    assert "json.dumps" in message and "point_array_digest" in message
    # refused, not quietly downcast
    assert "float64" in message and "np.asarray(points, dtype=np.float64)" in message

    with pytest.raises(PointReferenceError, match="extended-precision"):
        reference_for_array(points, None, name="points")
