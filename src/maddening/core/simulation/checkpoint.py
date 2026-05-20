"""
Checkpoint / restore -- save and load simulation state to disk.

State is persisted as a NumPy ``.npz`` archive with flat keys of the
form ``node_name/field_name``.  Internal multi-rate metadata lives
under the ``_meta/`` prefix.  JAX arrays are converted to NumPy on
save and back to JAX on load.

v0.2 #8 additions: integrity manifest, ``save_state_with_manifest``,
``load_state_with_manifest``, and ``download_and_load_state`` for the
``RESUME_FROM_URL`` resume path.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from maddening.core.graph_manager import GraphManager

# Key used by GraphManager for internal multi-rate bookkeeping.
_META_KEY = "_meta"

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

    np.savez(path, **arrays)

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
        (different node names or field names).
    """
    path = Path(path)
    if not path.exists() and path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    data = np.load(path, allow_pickle=False)

    # Separate meta keys from node keys.
    meta_keys: dict[str, np.ndarray] = {}
    node_keys: dict[str, dict[str, np.ndarray]] = {}

    for flat_key in data.files:
        parts = flat_key.split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Unexpected key format in checkpoint: '{flat_key}' "
                f"(expected 'node_name/field_name')"
            )
        prefix, field = parts
        if prefix == _META_KEY:
            meta_keys[field] = data[flat_key]
        else:
            node_keys.setdefault(prefix, {})[field] = data[flat_key]

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

    # ---- Apply loaded state ----
    for node_name in current_nodes:
        new_state = {
            field: jnp.array(arr) for field, arr in node_keys[node_name].items()
        }
        graph_manager.set_node_state(node_name, new_state)

    # Restore _meta if present in the checkpoint.
    raw_state = graph_manager._state  # noqa: SLF001
    if meta_keys:
        raw_state[_META_KEY] = {
            field: jnp.array(arr) for field, arr in meta_keys.items()
        }
    else:
        # If checkpoint had no meta but graph currently has it, reset.
        raw_state.pop(_META_KEY, None)


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


def write_manifest(npz_path: str | Path, *, extra: Optional[dict] = None) -> Path:
    """Write a sidecar ``<file>.manifest.json`` next to an ``.npz`` file.

    The manifest captures:
      * ``schema_version`` — bumps if the on-disk format changes
      * ``sha256`` — full hash of the .npz body
      * ``size_bytes``
      * ``extra`` — caller-supplied dict (commit hash, sim_time, etc.)

    Returns the manifest path.
    """
    npz_path = Path(npz_path)
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "sha256": compute_checkpoint_hash(npz_path),
        "size_bytes": npz_path.stat().st_size,
        "extra": dict(extra or {}),
    }
    manifest_path = npz_path.with_suffix(npz_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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


def download_and_load_state(
    graph_manager: "GraphManager",
    url: str,
    *,
    dest_dir: Optional[str | Path] = None,
    skip_integrity_check: bool = False,
) -> dict:
    """Download a checkpoint + manifest from *url* and load it.

    Supported URL schemes:
      * ``file://`` — local file path
      * ``http://`` / ``https://`` — HTTP GET

    A manifest at ``<url>.manifest.json`` is downloaded alongside the
    .npz so the integrity check can run without a side channel.

    Returns the manifest dict.
    """
    import tempfile as _tempfile
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("file", "http", "https", ""):
        raise ValueError(
            f"Unsupported URL scheme {parsed.scheme!r}; "
            "expected file://, http://, or https://"
        )

    if dest_dir is None:
        # Per-call temp dir avoids cross-call leakage when multiple
        # downloads target the same filename.
        dest_dir = Path(_tempfile.mkdtemp(prefix="maddening_resume_"))
    else:
        dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Local filename = last URL path component.
    fname = Path(parsed.path).name or "checkpoint.npz"
    local_npz = dest_dir / fname
    local_manifest = local_npz.with_suffix(local_npz.suffix + ".manifest.json")

    _fetch(url, local_npz)
    # The manifest is optional in skip_integrity mode; otherwise required.
    try:
        _fetch(url + ".manifest.json", local_manifest)
    except Exception:
        if not skip_integrity_check:
            raise

    return load_state_with_manifest(
        graph_manager, local_npz,
        skip_integrity_check=skip_integrity_check,
    )


def _fetch(url: str, dest: Path) -> None:
    """Copy *url* contents into *dest*.

    Pure-stdlib so we don't pull in another HTTP dep.  Used by
    :func:`download_and_load_state`.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in ("file", ""):
        # file:///path/to/x or /path/to/x
        src = Path(parsed.path) if parsed.scheme else Path(url)
        if not src.exists():
            raise FileNotFoundError(f"file:// source not found: {src}")
        dest.write_bytes(src.read_bytes())
        return
    if parsed.scheme in ("http", "https"):
        with urllib.request.urlopen(url) as response:  # noqa: S310 — trusted
            dest.write_bytes(response.read())
        return
    raise ValueError(f"unsupported URL scheme: {parsed.scheme}")
