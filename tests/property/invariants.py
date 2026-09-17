"""Exact-equality helpers shared by the round-trip properties.

Every comparison here is *exact*: same pytree structure, same dtype,
same shape, same bits.  A round trip that changes a value by one ulp has
changed it, and a tolerance would hide precisely the dtype and
promotion bugs these properties exist to find.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def _leaf_report(path: str, expected, actual) -> str:
    e, a = np.asarray(expected), np.asarray(actual)
    return (f"{path}: expected {e.dtype}{e.shape} {e!r}, "
            f"got {a.dtype}{a.shape} {a!r}")


def assert_leaf_identical(expected, actual, path: str) -> None:
    """One array leaf: same dtype, same shape, same bits."""
    e, a = jnp.asarray(expected), jnp.asarray(actual)
    assert e.dtype == a.dtype, _leaf_report(path, e, a)
    assert e.shape == a.shape, _leaf_report(path, e, a)
    assert np.array_equal(np.asarray(e), np.asarray(a)), _leaf_report(path, e, a)


def assert_states_identical(expected: dict, actual: dict, *, what: str = "state") -> None:
    """Two ``{node: {field: array}}`` trees are the same tree."""
    assert set(expected) == set(actual), (
        f"{what}: node sets differ -- only in expected "
        f"{sorted(set(expected) - set(actual))}, only in actual "
        f"{sorted(set(actual) - set(expected))}"
    )
    for node in expected:
        assert set(expected[node]) == set(actual[node]), (
            f"{what}[{node!r}]: field sets differ -- "
            f"{sorted(expected[node])} vs {sorted(actual[node])}"
        )
        for field in expected[node]:
            assert_leaf_identical(expected[node][field], actual[node][field],
                                  f"{what}[{node!r}][{field!r}]")


def assert_params_identical(expected: dict, actual: dict, *, what: str = "params") -> None:
    """Two graph parameter pytrees agree in structure, dtype and value."""
    assert set(expected) == set(actual), (
        f"{what}: sections differ -- {sorted(expected)} vs {sorted(actual)}"
    )
    for section in expected:
        assert set(expected[section]) == set(actual[section]), (
            f"{what}[{section!r}]: owners differ -- only in expected "
            f"{sorted(set(expected[section]) - set(actual[section]))}, only in actual "
            f"{sorted(set(actual[section]) - set(expected[section]))}"
        )
        for owner in expected[section]:
            e, a = expected[section][owner], actual[section][owner]
            assert set(e) == set(a), (
                f"{what}[{section!r}][{owner!r}]: keys differ -- "
                f"{sorted(e)} vs {sorted(a)}"
            )
            for key in e:
                assert_leaf_identical(e[key], a[key],
                                      f"{what}[{section!r}][{owner!r}][{key!r}]")


def assert_param_specs_identical(expected: dict, actual: dict) -> None:
    """``GraphManager.param_specs()`` agrees, including the edge-key specs
    that only exist on mapped edges."""
    assert set(expected) == set(actual)
    for section in expected:
        assert set(expected[section]) == set(actual[section]), (
            f"param_specs[{section!r}]: owners differ -- only in expected "
            f"{sorted(set(expected[section]) - set(actual[section]))}, only in actual "
            f"{sorted(set(actual[section]) - set(expected[section]))}"
        )
        for owner in expected[section]:
            assert expected[section][owner] == actual[section][owner], (
                f"param_specs[{section!r}][{owner!r}] differs: "
                f"{expected[section][owner]} vs {actual[section][owner]}"
            )


def structure(gm) -> dict[str, Any]:
    """The graph's topology as plain data: node identities and timesteps,
    every ``EdgeSpec`` field (transform name, units, additive flag,
    mapping recipe, ordinal) and the external-input declarations.

    This is what a reload has to reproduce *besides* the numbers, and it
    is where a serialiser that silently renames or drops something shows
    up.
    """
    return {
        "nodes": [
            (name, type(gm.get_node(name)).__name__, float(gm.get_node(name).delta_t))
            for name in gm.node_names
        ],
        "edges": [e.to_dict() for e in gm.edges],
        "external_inputs": [
            (ei.target_node, ei.target_field, tuple(ei.shape))
            for ei in gm._external_inputs  # noqa: SLF001 -- no public accessor
        ],
    }


def assert_structure_identical(expected: dict, actual: dict) -> None:
    for key in ("nodes", "edges", "external_inputs"):
        assert expected[key] == actual[key], (
            f"structure[{key!r}] differs:\n  expected {expected[key]}\n  actual   {actual[key]}"
        )


#: ``params["mappings"]`` (and any other ``{owner: {key: array}}`` tree)
#: has the same shape as a state tree; the comparison is the same one.
assert_leaf_tree_identical = assert_states_identical
