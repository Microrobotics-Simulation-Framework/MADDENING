"""
Checkpoint / restore -- save and load simulation state to disk.

State is persisted as a NumPy ``.npz`` archive with flat keys of the
form ``node_name/field_name``.  Internal multi-rate metadata lives
under the ``_meta/`` prefix.  JAX arrays are converted to NumPy on
save and back to JAX on load.

v0.2 #8 additions: integrity manifest, ``save_state_with_manifest``
and ``load_state_with_manifest``.  The ``RESUME_FROM_URL`` transport
(``download_and_load_state``) lives in :mod:`maddening.cloud.resume`
since v0.4.0; the alias kept here is deprecated and removed in 1.0.
This module stays dependency-free (stdlib + NumPy + JAX only).
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import jax.numpy as jnp
import numpy as np

from maddening.core._exact_integers import lost_as_integer
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

if TYPE_CHECKING:
    from maddening.core.graph_manager import GraphManager

# Key used by GraphManager for internal multi-rate bookkeeping.
_META_KEY = "_meta"
_PARAMS_KEY = "_params"
_MAPPINGS_KEY = "_params_mappings"

# Schema version for the integrity manifest.
#
# Bump policy (settled v0.2.0)
# ----------------------------
# A release running schema version ``N`` can read checkpoints saved
# at version ``N`` or ``N-1``.  Reading any older version raises
# :class:`CheckpointVersionError` naming the intermediate release the
# user should load with first.
#
# *Bump-worthy* changes (require ``N`` → ``N+1``):
#   - a key is renamed, removed, or its semantics change
#   - a new MANDATORY key is added (one with no sensible default)
#
# *Non-bump* changes:
#   - a new optional key with a supplied default
#   - additions to the ``extra`` block (free-form caller dict)
#
# Migration helpers live in ``MIGRATIONS`` below, keyed by
# ``(from_version, to_version)``.  A helper becomes unreachable —
# and is removed — one release after its source version stops being
# loadable.  Concretely: the v1→v2 helper is removed in v3 (because
# v3 supports v2+v3 only).
#
# Enforce in code review:
#   - Every bump comes with a migration helper in MIGRATIONS.
#   - Every release that crosses a bump removes the now-unreachable
#     helpers from the previous bump.
CHECKPOINT_SCHEMA_VERSION = 1

# Migration helpers: (from_version, to_version) -> callable.  Each
# callable takes a manifest+data pair and returns the same shape
# upgraded to the target version.  Empty today; populated when the
# first bump (v1 → v2) lands.
MIGRATIONS: dict[tuple[int, int], Any] = {}


def save_state(graph_manager: "GraphManager", path: str | Path) -> Path:
    """Persist all node states (and ``_meta``) to an ``.npz`` file.

    Parameters
    ----------
    graph_manager : GraphManager
        The graph whose state should be saved.
    path : str or Path
        Destination file.  A ``.npz`` suffix is appended automatically by
        ``numpy.savez`` if not already present.

    Returns
    -------
    Path
        The resolved path of the written file (always ends in ``.npz``).
    """
    path = Path(path)

    arrays: dict[str, np.ndarray] = {}

    # Node states
    for node_name in graph_manager.node_names:
        node_state = graph_manager.get_node_state(node_name)
        for field_name, value in node_state.items():
            key = f"{node_name}/{field_name}"
            arrays[key] = np.asarray(value)

    # Internal _meta state (multi-rate step counter, etc.)
    # Access the raw internal state dict directly.
    raw_state = graph_manager._state  # noqa: SLF001
    if _META_KEY in raw_state:
        for field_name, value in raw_state[_META_KEY].items():
            key = f"{_META_KEY}/{field_name}"
            arrays[key] = np.asarray(value)

    # Differentiable graph parameters (node constants), so a calibrated
    # graph restores with the values it was calibrated to.
    for node_name, node_params in graph_manager.params.get("nodes", {}).items():
        for pname, value in node_params.items():
            arrays[f"{_PARAMS_KEY}/{node_name}/{pname}"] = np.asarray(value)
    # Interface-mapping weights are trainable leaves too (edge keys hold
    # no '/', so they nest under the same prefix scheme).
    for edge_key, weights in graph_manager.params.get("mappings", {}).items():
        for wname, value in weights.items():
            arrays[f"{_MAPPINGS_KEY}/{edge_key}/{wname}"] = np.asarray(value)

    # numpy types `savez` as `savez(file, *args, allow_pickle=True,
    # **kwds)`, so a checker matches every `**` value against
    # `allow_pickle: bool` as well as against `**kwds`.  Every member
    # name built above is prefixed, so none can collide with it.
    np.savez(path, **arrays)  # pyright: ignore[reportArgumentType]

    # numpy.savez appends .npz if not already present
    resolved = path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")
    return resolved


def load_state(graph_manager: "GraphManager", path: str | Path) -> None:
    """Restore node states (and ``_meta``) from an ``.npz`` file.

    Parameters
    ----------
    graph_manager : GraphManager
        The graph whose state will be overwritten.
    path : str or Path
        Source file.  If *path* has no ``.npz`` extension and the file
        does not exist, the function retries with ``.npz`` appended.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found.
    ValueError
        If the saved state does not match the current graph structure
        (different node names, field names, or a state field / params
        leaf whose shape differs from the live one), or holds a value the
        live leaf's dtype cannot hold: a finite value that would overflow
        to ``inf``, a non-zero one that would flush to ``0``, an integer
        that would wrap, text, or a boolean for a numeric leaf
        (:func:`_checked_cast`; a value that rounds to a subnormal loads,
        and a state value already ``inf`` or ``NaN`` loads as it was).
        A parameter leaf is not asked what ``PUT /graph/params`` asks --
        its ``ParamSpec`` bounds, finiteness, the node's constructor -- as a
        ``gm.params`` write is not: bounds are metadata to a graph, and a
        graph whose parameters Python moved outside them resumes its own
        checkpoint.  ``POST /checkpoint/load`` asks them.
        :class:`CheckpointFormatError`, a ``ValueError``, if the file is not
        an ``.npz`` archive of plain arrays.

    Notes
    -----
    The restore is atomic: every node name, field name, state shape and
    params-leaf shape is checked before anything is written, and a
    failure part-way through the write puts ``gm._state`` and
    ``gm.params`` back as they were.  A caller that catches the error
    (the cloud entry point does) is therefore looking at the graph it
    had before the attempt, never at a half-restored one.

    The one thing outside that guarantee is the ``compile()`` this
    function triggers on a dirty or never-compiled graph: it runs before
    any checkpoint data is read into the graph, so a failure there is
    the graph's own and leaves nothing of the checkpoint behind.
    """
    path = Path(path)
    if not path.exists() and path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # Every member's name, shape and dtype, from its header: no data is
    # read until the names and shapes have been checked against the live
    # graph, and a member the load does not use is never read
    # (_CheckpointArchive).
    archive = _CheckpointArchive(path)
    try:
        _load_from_archive(graph_manager, archive)
    finally:
        archive.close()


def _load_from_archive(graph_manager: "GraphManager", archive: "_CheckpointArchive") -> None:
    """:func:`load_state`'s checks and restore, on *archive*'s headers
    first and its data only for the members it restores."""

    def read(key: str, want: Any, what: str) -> np.ndarray:
        """Member *key*, cast to the live leaf's dtype; its shape is
        checked by the caller, its dtype here, both before it is read."""
        refusal = archive.numeric_refusal(key, what)
        if refusal is not None:
            raise ValueError(refusal)
        return _checked_cast(archive.read(key), want, what)

    # Separate meta keys from node keys (member names, not data).
    meta_keys: dict[str, str] = {}
    node_keys: dict[str, dict[str, str]] = {}
    param_keys: dict[str, dict[str, str]] = {}
    mapping_keys: dict[str, dict[str, str]] = {}

    for flat_key in archive:
        parts = flat_key.split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Unexpected key format in checkpoint: '{flat_key}' "
                f"(expected 'node_name/field_name')"
            )
        prefix, field = parts
        if prefix == _META_KEY:
            meta_keys[field] = flat_key
        elif prefix == _PARAMS_KEY:
            node_name, pname = field.split("/", 1)
            param_keys.setdefault(node_name, {})[pname] = flat_key
        elif prefix == _MAPPINGS_KEY:
            edge_key, wname = field.rsplit("/", 1)
            mapping_keys.setdefault(edge_key, {})[wname] = flat_key
        else:
            node_keys.setdefault(prefix, {})[field] = flat_key

    # A graph that has never compiled has no params pytree and no _meta;
    # compile first so load-then-run equals compile-then-load (otherwise
    # the compile triggered by the first step would re-seed _meta -- the
    # multirate step counter, predictor / IQN history -- and drop params).
    if (getattr(graph_manager, "_dirty", False)
            or getattr(graph_manager, "_compiled_step", None) is None):
        graph_manager.compile()

    # ---- Validate against current graph structure ----
    current_nodes = set(graph_manager.node_names)
    saved_nodes = set(node_keys.keys())

    if current_nodes != saved_nodes:
        missing = current_nodes - saved_nodes
        extra = saved_nodes - current_nodes
        parts = []
        if missing:
            parts.append(f"missing from checkpoint: {sorted(missing)}")
        if extra:
            parts.append(f"extra in checkpoint: {sorted(extra)}")
        raise ValueError(
            f"Checkpoint node mismatch. {'; '.join(parts)}"
        )

    for node_name in current_nodes:
        current_fields = set(graph_manager.get_node_state(node_name).keys())
        saved_fields = set(node_keys[node_name].keys())
        if current_fields != saved_fields:
            raise ValueError(
                f"Field mismatch for node '{node_name}': "
                f"current={sorted(current_fields)}, "
                f"saved={sorted(saved_fields)}"
            )

    # ---- Validate shapes, coerce dtypes, then apply ----
    # Every shape first, from the headers, so a member of another shape is
    # refused before any member is read.
    for node_name in current_nodes:
        live = graph_manager.get_node_state(node_name)
        for field, key in node_keys[node_name].items():
            want_shape = tuple(jnp.shape(live[field]))
            if archive.shape(key) != want_shape:
                raise ValueError(
                    f"Checkpoint field '{node_name}/{field}' has shape {archive.shape(key)}, "
                    f"graph has {want_shape}"
                )
    staged_states: dict[str, dict] = {}
    for node_name in current_nodes:
        live = graph_manager.get_node_state(node_name)
        new_state = {}
        for field, key in node_keys[node_name].items():
            want = jnp.asarray(live[field])
            new_state[field] = jnp.asarray(read(key, want.dtype, f"field '{node_name}/{field}'"))
        staged_states[node_name] = new_state

    # Stage the graph parameters the same way, so a leaf that does not
    # fit is found *before* any node state is applied.  Applying first
    # and validating afterwards left a failed resume married to the
    # checkpoint's states and the graph's fresh params -- silently, and
    # the cloud entry point logged it as a fresh start.  Unknown nodes
    # and keys are ignored (a node may have stopped accepting params);
    # a leaf whose shape differs from the live one is an error, like a
    # state field, because restoring it would run the graph wrong.
    def _stage_params(section: str, saved_tree: dict) -> list:
        """``[(leaf_dict, name, value)]`` to write; raises before any write."""
        current = graph_manager.params.get(section, {})
        writes: list[tuple[dict, str, Any]] = []
        for owner, saved in saved_tree.items():
            if owner not in current:
                continue
            for pname, key in saved.items():
                if pname not in current[owner]:
                    continue
                live = jnp.asarray(current[owner][pname])
                if archive.shape(key) != tuple(live.shape):
                    raise ValueError(
                        f"Checkpoint params {section}[{owner!r}][{pname!r}] has shape "
                        f"{archive.shape(key)}, graph has {tuple(live.shape)}"
                    )
                writes.append((current[owner], pname, jnp.asarray(read(
                    key, live.dtype, f"params {section}[{owner!r}][{pname!r}]"))))
        return writes

    staged_params = (
        _stage_params("nodes", param_keys)
        + _stage_params("mappings", mapping_keys)
    )

    # ---- Apply.  Everything above validated without mutating; the
    # rollback below is the net for whatever validation cannot see, so a
    # restore that fails leaves the graph exactly as it found it.
    undo = _state_and_params_snapshot(graph_manager)
    try:
        for node_name, new_state in staged_states.items():
            graph_manager.set_node_state(node_name, new_state)

        # Restore _meta if present in the checkpoint.
        #
        # Merged over the compiled graph's own ``_meta``, never
        # substituted for it.  The key set is the *graph's*: a checkpoint
        # written before ``predictor=`` was turned on carries fewer keys,
        # and replacing the dict dropped the ones the recompiled graph had
        # just seeded -- ``step()`` survived that (it rebuilds ``_meta``
        # each call) while ``run_scan`` died on a carry-structure
        # mismatch, the one place the execution paths disagreed.  A key
        # the checkpoint carries and this graph does not is stale (its
        # coupling group is gone) and is dropped; a key whose shape no
        # longer fits keeps the graph's freshly seeded value, since every
        # ``_meta`` entry but the step counter is a warm start that is
        # only ever an accelerator.
        raw_state = graph_manager._state  # noqa: SLF001
        if meta_keys:
            live_meta = raw_state.get(_META_KEY) or {}
            merged = dict(live_meta)
            dropped, reshaped = [], []
            for field, key in meta_keys.items():
                if field not in live_meta:
                    dropped.append(field)
                    continue
                want = jnp.asarray(live_meta[field])
                if archive.shape(key) != tuple(want.shape):
                    reshaped.append(field)
                    continue
                merged[field] = jnp.asarray(read(key, want.dtype, f"_meta '{field}'"))
            if merged:
                raw_state[_META_KEY] = merged
            if dropped or reshaped:
                detail = []
                if dropped:
                    detail.append(f"not present in this graph: {sorted(dropped)}")
                if reshaped:
                    detail.append(f"shape no longer fits: {sorted(reshaped)}")
                warnings.warn(
                    "checkpoint _meta entries ignored (" + "; ".join(detail)
                    + "); this graph's freshly compiled values are used instead",
                    RuntimeWarning,
                    stacklevel=2,
                )
        # A checkpoint without ``_meta`` (written by a graph that had none)
        # keeps the freshly compiled ``_meta`` of *this* graph: a multirate
        # step counter or coupling history seeded at zero is the right start,
        # whereas dropping the key made the next step raise KeyError.

        for leaves, pname, value in staged_params:
            leaves[pname] = value
    except BaseException:
        _restore_state_and_params(graph_manager, undo)
        raise


class CheckpointFormatError(ValueError):
    """The file is not a checkpoint: not an ``.npz`` archive of plain
    arrays, or one whose members cannot be read without unpickling."""


#: The npy header versions a checkpoint member may use (``np.savez``
#: writes 1.0, or 2.0 for a header over 64 KiB); 3.0 is for structured
#: dtypes with non-latin-1 field names, which no checkpoint holds.
_NPY_HEADER_READERS = {
    (1, 0): np.lib.format.read_array_header_1_0,
    (2, 0): np.lib.format.read_array_header_2_0,
}

#: Dtype kinds a checkpoint member may hold: numbers and booleans.  Text,
#: bytes, void and objects are refused from the header, before a byte of
#: their data is read (text was refused after it, by _checked_cast).
_NUMERIC_KINDS = "biufc"


class _CheckpointArchive:
    """The members of a checkpoint ``.npz``: every member's name, shape and
    dtype read from its ``.npy`` header -- nothing of its data -- and the
    data of a member read only when :meth:`read` asks for it.

    ``load_state`` used to decompress every member before it checked a
    single name or shape, so a small file cost as much memory as its
    members declared: a 2 MB archive holding an extra 2 GiB member of zeros
    filled 2 GiB before ``POST /checkpoint/load`` answered "node mismatch".
    Now the names and shapes are checked against the live graph first, and
    only a member whose shape is a live leaf's, with a numeric dtype, is
    read: what a load reads is bounded by the live graph's own sizes, at
    one numeric element of the archive's dtype per live element.
    """

    def __init__(self, path: Path) -> None:
        what = (f"{path} is not a checkpoint: an .npz archive of plain arrays, "
                "saved by GraphManager.save_state, is expected")
        self.path = path
        try:
            self._zip = zipfile.ZipFile(path)
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            raise CheckpointFormatError(what) from None
        self.headers: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        self._members: dict[str, str] = {}
        try:
            for info in self._zip.infolist():
                name = info.filename
                if not name.endswith(".npy") or info.is_dir():
                    raise CheckpointFormatError(
                        f"{path} is not a checkpoint: its member {name!r} is not an "
                        ".npy array")
                key = name[: -len(".npy")]
                if key in self.headers:
                    raise CheckpointFormatError(
                        f"{path} is not a checkpoint: it holds {key!r} twice")
                with self._zip.open(info) as member:
                    version = np.lib.format.read_magic(member)
                    reader = _NPY_HEADER_READERS.get(version)
                    if reader is None:
                        raise CheckpointFormatError(
                            f"{path} is not a checkpoint: member {key!r} has npy "
                            f"header version {version}")
                    shape, _fortran, dtype = reader(member)
                if dtype.hasobject:
                    raise CheckpointFormatError(
                        f"{path} is not a checkpoint: one of its members holds Python "
                        "objects or is damaged")
                self.headers[key] = (tuple(int(n) for n in shape), dtype)
                self._members[key] = name
        except CheckpointFormatError:
            self._zip.close()
            raise
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError):
            self._zip.close()
            raise CheckpointFormatError(
                f"{path} is not a checkpoint: one of its members holds Python objects "
                "or is damaged") from None

    def __iter__(self):
        return iter(self.headers)

    def shape(self, key: str) -> tuple[int, ...]:
        return self.headers[key][0]

    def dtype(self, key: str) -> np.dtype:
        return self.headers[key][1]

    def numeric_refusal(self, key: str, what: str) -> Optional[str]:
        """Why member *key* cannot be a number (its header's dtype), in
        :func:`_checked_cast`'s words, or ``None``."""
        dtype = self.dtype(key)
        if dtype.kind not in _NUMERIC_KINDS or dtype.fields is not None \
                or dtype.subdtype is not None:
            return f"Checkpoint {what} holds {dtype} data, not a number.  Nothing was loaded."
        return None

    def read(self, key: str) -> np.ndarray:
        """The data of member *key*, read now (``allow_pickle=False``)."""
        try:
            with self._zip.open(self._members[key]) as member:
                return np.lib.format.read_array(member, allow_pickle=False)
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            raise CheckpointFormatError(
                f"{self.path} is not a checkpoint: one of its members holds Python "
                "objects or is damaged") from None

    def close(self) -> None:
        self._zip.close()


def _read_checkpoint_archive(path: Path) -> dict[str, np.ndarray]:
    """Every member of the ``.npz`` at *path*, read with
    ``allow_pickle=False``, or :class:`CheckpointFormatError`.

    NumPy answers a file that is not an archive, or a member holding Python
    objects, with its own ``ValueError`` ("This file contains pickled
    (object) data ...") -- a message about how to load the file unsafely,
    which a caller passing ``ValueError`` on as a checkpoint mismatch used
    to show to whoever had named the file.  A damaged archive (``EOFError``,
    ``zipfile.BadZipFile``) is the same answer.  ``load_state`` reads its
    members through :class:`_CheckpointArchive` instead, which checks every
    member's header before it reads any data.
    """
    archive = _CheckpointArchive(path)
    try:
        return {key: archive.read(key) for key in archive}
    finally:
        archive.close()


def _checked_cast(arr: np.ndarray, dtype: Any, what: str) -> np.ndarray:
    """*arr* in the live leaf's *dtype*, refused (``ValueError``) when the
    cast would lose a value.

    A checkpoint written in a wider dtype -- a graph run under
    ``jax_enable_x64`` and loaded without it -- was cast with no check:
    a float64 ``1e39`` loaded into a float32 field as ``inf`` and ``1e-50``
    as ``0.0``, its sign lost, with nothing said.  The rule is the FMU's
    (:func:`maddening.fmi.sidecar._checked_value`): a finite value that
    overflows to ``inf``, or a non-zero one flushed to ``+-0``, is
    refused; one that rounds to a subnormal keeps its sign and magnitude
    and is kept; a value that is already ``inf`` or ``NaN`` is stored as
    it was (a diverged state, a ``NaN``-seeded diagnostics slot).  An
    integer that would wrap or truncate, and a non-finite value for an
    integer leaf, are refused too -- an integer of the other signedness
    included (``-1`` for an unsigned leaf, 4000000000 for an ``int32``
    one), which a cast there and back cannot see
    (:func:`maddening.core._exact_integers.lost_as_integer`) -- and so is a
    complex value with an imaginary part for a real leaf.
    """
    a = np.asarray(arr)
    target = np.dtype(dtype)
    # Text and booleans are not numbers: NumPy casts "1.5" to 1.5 and True
    # to 1.0 without a word, and every other surface (PUT /graph/params,
    # the FMU) refuses both.  A boolean for a boolean leaf is a boolean.
    if a.dtype.kind in "USO":
        raise ValueError(
            f"Checkpoint {what} holds {a.dtype} data, not a number.  Nothing was loaded.")
    if a.dtype.kind == "b" and target.kind != "b":
        raise ValueError(
            f"Checkpoint {what} holds a boolean, and this graph's leaf is {target}.  "
            "Nothing was loaded.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        cast = a.astype(target)
    if a.dtype == target:
        return cast
    if target.kind in "fc":
        finite = np.isfinite(a) if a.dtype.kind in "fc" else np.ones(a.shape, bool)
        lost = (finite & ~np.isfinite(cast)) | ((a != 0) & (cast == 0))
        if a.dtype.kind == "c" and target.kind == "f":
            lost = lost | (a.imag != 0)        # the cast drops an imaginary part
    elif target.kind in "iu":
        # By range and wholeness, never by a cast there and back: that is a
        # bijection between a signed and an unsigned type of one width, so
        # -1 for a uint64 leaf came back as -1 and loaded as
        # 18446744073709551615; and a comparison through float64 rounds
        # both sides above 2**53.
        lost = lost_as_integer(a, target)
    elif target.kind == "b":
        lost = (a != 0) & (a != 1)
    else:
        lost = np.zeros(a.shape, bool)
    if np.any(lost):
        index = tuple(int(i) for i in np.argwhere(lost)[0]) if a.ndim else ()
        raise ValueError(
            f"Checkpoint {what} holds {a[index].item()!r} ({a.dtype}), which this graph's "
            f"{target} cannot hold: it would load as {cast[index].item()!r}.  Nothing was "
            "loaded.")
    return cast


def _state_and_params_snapshot(graph_manager: "GraphManager") -> tuple[dict, dict]:
    """Copy of ``gm._state`` and ``gm.params`` deep enough to undo a restore.

    Two levels of dict for the state (node -> field -> array) and three
    for the params (section -> owner -> name -> array).  The leaves are
    JAX/NumPy arrays and are never mutated in place by a restore, so
    copying the dicts that hold them is enough to put everything back.
    """
    state = {
        name: (dict(fields) if isinstance(fields, dict) else fields)
        for name, fields in graph_manager._state.items()  # noqa: SLF001
    }
    params = {
        section: {
            owner: (dict(leaves) if isinstance(leaves, dict) else leaves)
            for owner, leaves in owners.items()
        }
        for section, owners in graph_manager.params.items()
        if isinstance(owners, dict)
    }
    return state, params


def _restore_state_and_params(
    graph_manager: "GraphManager", snapshot: tuple[dict, dict],
) -> None:
    """Put a snapshot from :func:`_state_and_params_snapshot` back.

    The container objects are refilled rather than replaced: callers
    (and the compiled step) hold references to ``gm.params`` and to the
    per-node leaf dicts, so swapping in new dicts would strand them.
    """
    state, params = snapshot
    raw_state = graph_manager._state  # noqa: SLF001
    raw_state.clear()
    raw_state.update(state)
    for section, owners in params.items():
        live_section = graph_manager.params.get(section)
        if not isinstance(live_section, dict):
            continue
        for owner, leaves in owners.items():
            live_leaves = live_section.get(owner)
            if isinstance(live_leaves, dict) and isinstance(leaves, dict):
                live_leaves.clear()
                live_leaves.update(leaves)
            else:
                live_section[owner] = leaves


# ---------------------------------------------------------------------------
# v0.2 #8: integrity manifest + URL-based resume
# ---------------------------------------------------------------------------


class CheckpointIntegrityError(ValueError):
    """Raised when a checkpoint fails its manifest integrity check.

    Use the subclasses for specific kinds of failure (version drift,
    SHA mismatch, malformed manifest).  Catch this class to handle
    any integrity failure uniformly.
    """

    pass


class CheckpointVersionError(CheckpointIntegrityError):
    """Schema version on disk is outside the readable window.

    The error message names the intermediate release the user should
    load the checkpoint with first.  Example::

        Checkpoint at /tmp/snap.npz is schema v1, but this release
        reads v3 and v4 only.  Load with maddening>=0.3,<0.4 first
        and re-save to upgrade to v2, then with the current release
        to upgrade to v4.
    """

    def __init__(
        self,
        path: "Path | str",
        file_version: int,
        readable_min: int,
        readable_max: int,
    ):
        self.path = path
        self.file_version = file_version
        self.readable_min = readable_min
        self.readable_max = readable_max
        if file_version < readable_min:
            hint_release = _hint_release_for_version(file_version + 1)
            msg = (
                f"Checkpoint at {path} is schema v{file_version}, but this "
                f"release reads v{readable_min}+ only.  Load it with "
                f"{hint_release} first and re-save to upgrade."
            )
        else:
            msg = (
                f"Checkpoint at {path} is schema v{file_version} — newer "
                f"than this release's readable range (v{readable_min}-v{readable_max}). "
                f"Upgrade the runtime to read this checkpoint."
            )
        super().__init__(msg)


def _hint_release_for_version(target_version: int) -> str:
    """Best-effort guess at which release tag introduced *target_version*.

    Today (v1 only) this returns a placeholder string.  When v2
    lands the table grows.  The hint is used in the error message
    above; if the table doesn't know the answer, we just say "the
    release that introduced schema vN".
    """
    _RELEASE_FOR_VERSION = {
        1: "maddening>=0.2",
        # 2: "maddening>=0.3", -- add when v2 lands
    }
    return _RELEASE_FOR_VERSION.get(
        target_version,
        f"the release that introduced schema v{target_version}",
    )


def compute_checkpoint_hash(path: str | Path) -> str:
    """SHA-256 of the .npz bytes — same value the manifest stores."""
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(npz_path: str | Path, *, extra: Optional[dict] = None,
                   manifest_path: str | Path | None = None) -> Path:
    """Write a sidecar ``<file>.manifest.json`` next to an ``.npz`` file.

    The manifest captures:
      * ``schema_version`` — bumps if the on-disk format changes
      * ``sha256`` — full hash of the .npz body
      * ``size_bytes``
      * ``extra`` — caller-supplied dict (commit hash, sim_time, etc.)

    *manifest_path* writes it elsewhere: the manifest of a checkpoint still
    under a temporary name, to be moved into place after its manifest.  The
    file is written under a temporary name and moved into place
    (``os.replace``), so a reader never sees half of one.

    Returns the manifest path.
    """
    npz_path = Path(npz_path)
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "sha256": compute_checkpoint_hash(npz_path),
        "size_bytes": npz_path.stat().st_size,
        "extra": dict(extra or {}),
    }
    if manifest_path is None:
        manifest_path = npz_path.with_suffix(npz_path.suffix + ".manifest.json")
    manifest_path = Path(manifest_path)
    partial = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.partial")
    try:
        partial.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(partial, manifest_path)
    finally:
        if partial.exists():
            partial.unlink()
    return manifest_path


def read_manifest(npz_path: str | Path) -> dict:
    """Read the sidecar manifest for *npz_path*.

    Raises ``FileNotFoundError`` if the manifest is missing.
    """
    npz_path = Path(npz_path)
    manifest_path = npz_path.with_suffix(npz_path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest beside checkpoint: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def verify_manifest(npz_path: str | Path, manifest: Optional[dict] = None) -> None:
    """Raise :class:`CheckpointIntegrityError` on mismatch.

    Checks both schema_version and SHA-256 hash.  Pass ``manifest`` to
    use an in-memory copy (e.g. a manifest downloaded separately from
    the .npz); otherwise the function reads the sidecar.

    Version policy: the current release reads schema version
    ``CHECKPOINT_SCHEMA_VERSION`` and ``CHECKPOINT_SCHEMA_VERSION - 1``
    (when applicable).  Older versions raise
    :class:`CheckpointVersionError` pointing at the intermediate
    release.  Future versions raise the same exception in the
    "newer than runtime" direction.
    """
    npz_path = Path(npz_path)
    if manifest is None:
        manifest = read_manifest(npz_path)

    sv = manifest.get("schema_version")
    if not isinstance(sv, int):
        raise CheckpointIntegrityError(
            f"Manifest at {npz_path} has no integer 'schema_version' "
            f"(found {sv!r}).",
        )
    # Readable window: current and previous version (when >1).
    readable_min = max(1, CHECKPOINT_SCHEMA_VERSION - 1)
    readable_max = CHECKPOINT_SCHEMA_VERSION
    if sv < readable_min or sv > readable_max:
        raise CheckpointVersionError(
            path=npz_path, file_version=sv,
            readable_min=readable_min, readable_max=readable_max,
        )

    expected = manifest.get("sha256")
    if not isinstance(expected, str):
        raise CheckpointIntegrityError(
            f"Manifest missing 'sha256' field for {npz_path}",
        )
    actual = compute_checkpoint_hash(npz_path)
    if actual != expected:
        raise CheckpointIntegrityError(
            f"SHA-256 mismatch for {npz_path}: "
            f"manifest says {expected[:12]}…, file hashes to {actual[:12]}…",
        )


def save_state_with_manifest(
    graph_manager: "GraphManager",
    path: str | Path,
    *,
    extra: Optional[dict] = None,
) -> tuple[Path, Path]:
    """:func:`save_state` + :func:`write_manifest`.

    Returns ``(npz_path, manifest_path)``.
    """
    npz_path = save_state(graph_manager, path)
    manifest_path = write_manifest(npz_path, extra=extra)
    return npz_path, manifest_path


def load_state_with_manifest(
    graph_manager: "GraphManager",
    path: str | Path,
    *,
    skip_integrity_check: bool = False,
) -> dict:
    """:func:`load_state` after verifying the sidecar manifest.

    Returns the manifest dict for caller use.
    """
    path = Path(path)
    if not path.exists() and path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    if not skip_integrity_check:
        verify_manifest(path)
    load_state(graph_manager, path)
    try:
        return read_manifest(path)
    except FileNotFoundError:
        return {}


@stability(StabilityLevel.DEPRECATED)
def download_and_load_state(
    graph_manager: "GraphManager",
    url: str,
    *,
    dest_dir: Optional[str | Path] = None,
    skip_integrity_check: bool = False,
) -> dict:
    """Deprecated alias for :func:`maddening.cloud.resume.download_and_load_state`.

    The resume-from-URL transport moved to the cloud package in v0.4.0;
    this alias forwards to it and emits a :class:`DeprecationWarning`.
    It is removed in 1.0.  The import of the cloud package is deferred
    to call time so this core module never imports ``maddening.cloud``
    (or ``fsspec``) at import time.
    """
    import warnings

    warnings.warn(
        "maddening.core.simulation.checkpoint.download_and_load_state moved to "
        "maddening.cloud.resume; the alias is removed in 1.0",
        DeprecationWarning,
        stacklevel=2,
    )
    from maddening.cloud.resume import download_and_load_state as _impl

    return _impl(
        graph_manager, url,
        dest_dir=dest_dir, skip_integrity_check=skip_integrity_check,
    )
