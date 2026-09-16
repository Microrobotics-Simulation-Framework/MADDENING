"""
Resume-from-URL transport for the cloud entry-point.

This module fetches a checkpoint (``.npz`` + sidecar ``.manifest.json``)
from a URL and hands it to
:func:`maddening.core.simulation.checkpoint.load_state_with_manifest`.
It lives in the cloud package rather than next to the checkpoint code
because URL transport is a *deployment* concern: which storage backends
exist, how credentials are found, and which optional packages (``fsspec``
and its ``s3fs`` / ``gcsfs`` / ``adlfs`` backends) are needed are all
questions about where the simulation runs, not about the checkpoint
format.  Keeping the transport here lets
:mod:`maddening.core.simulation.checkpoint` stay a dependency-free
local save/load/manifest module.

Supported URL schemes:

* ``file://`` (and bare paths) — local file copy.
* ``http://`` / ``https://`` — HTTP GET via the stdlib ``urllib``.
* ``s3://``, ``gs://`` / ``gcs://``, ``az://`` / ``abfs://`` / ``azure://``,
  ``memory://`` — via ``fsspec`` and the matching backend.

The historical import path
``maddening.core.simulation.checkpoint.download_and_load_state`` is a
deprecated alias that forwards here and is removed in 1.0.
"""

from __future__ import annotations

import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.simulation.checkpoint import load_state_with_manifest

if TYPE_CHECKING:
    from maddening.core.graph_manager import GraphManager


__all__ = ["download_and_load_state"]


@stability(StabilityLevel.EVOLVING)
def download_and_load_state(
    graph_manager: "GraphManager",
    url: str,
    *,
    dest_dir: Optional[str | Path] = None,
    skip_integrity_check: bool = False,
) -> dict:
    """Download a checkpoint + manifest from *url* and load it.

    Parameters
    ----------
    graph_manager : GraphManager
        Compiled graph whose state is restored from the checkpoint.
    url : str
        Location of the ``.npz`` checkpoint.  Supported schemes:

        * ``file://`` — local file path (a bare path is treated the same)
        * ``http://`` / ``https://`` — HTTP GET
        * ``s3://``, ``gs://`` / ``gcs://``, ``az://`` / ``abfs://`` /
          ``azure://`` (and any other ``fsspec`` protocol, e.g.
          ``memory://``) — via ``fsspec`` (C3, v0.4.0); install the
          matching backend (``s3fs``, ``gcsfs``, ``adlfs``).  Credentials
          come from the backend's usual environment / config.
    dest_dir : str or Path, optional
        Directory the files are downloaded into.  Defaults to a fresh
        per-call temporary directory so concurrent resumes that target
        the same filename do not collide.
    skip_integrity_check : bool, default False
        When True, a missing manifest is tolerated and the hash / schema
        verification is skipped.  Do not use in production.

    Returns
    -------
    dict
        The manifest dict (empty when no manifest was available and
        ``skip_integrity_check`` is set).

    Raises
    ------
    ValueError
        The URL scheme is not one of the supported schemes.
    FileNotFoundError
        A ``file://`` source, or its manifest, does not exist.
    ImportError
        The URL needs ``fsspec`` or a backend that is not installed.
    CheckpointIntegrityError
        The downloaded checkpoint does not match its manifest.

    Notes
    -----
    A manifest at ``<url>.manifest.json`` is downloaded alongside the
    .npz so the integrity check can run without a side channel.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("file", "http", "https", "") and not _is_fsspec_scheme(parsed.scheme):
        raise ValueError(
            f"Unsupported URL scheme {parsed.scheme!r}; expected file://, "
            "http://, https://, or an fsspec protocol (s3://, gs://, az://, ...)"
        )

    if dest_dir is None:
        # Per-call temp dir avoids cross-call leakage when multiple
        # downloads target the same filename.
        dest_dir = Path(tempfile.mkdtemp(prefix="maddening_resume_"))
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

    Pure-stdlib for ``file://`` and ``http(s)://`` so we don't pull in
    another HTTP dep; ``fsspec`` only for cloud-storage schemes.  Used by
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
    if _is_fsspec_scheme(parsed.scheme):
        fs, path = _fsspec_open(url)
        with fs.open(path, "rb") as f:
            dest.write_bytes(f.read())
        return
    raise ValueError(f"unsupported URL scheme: {parsed.scheme}")


_FSSPEC_SCHEMES = {"s3", "s3a", "gs", "gcs", "az", "abfs", "abfss", "azure", "adl", "memory"}
_FSSPEC_BACKENDS = {"s3": "s3fs", "s3a": "s3fs", "gs": "gcsfs", "gcs": "gcsfs",
                    "az": "adlfs", "abfs": "adlfs", "abfss": "adlfs", "azure": "adlfs",
                    "adl": "adlfs"}


def _is_fsspec_scheme(scheme: str) -> bool:
    return scheme in _FSSPEC_SCHEMES


def _fsspec_open(url: str):
    """``(filesystem, path)`` for an fsspec URL, with actionable errors."""
    scheme = urllib.parse.urlparse(url).scheme
    try:
        import fsspec  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            f"{scheme}:// checkpoint URLs need fsspec"
            + (f" and {_FSSPEC_BACKENDS[scheme]}" if scheme in _FSSPEC_BACKENDS else "")
            + f":  pip install fsspec {_FSSPEC_BACKENDS.get(scheme, '')}".rstrip()
        ) from e
    # "azure://" is not an fsspec protocol name; adlfs registers "az" / "abfs".
    if scheme == "azure":
        url = "az://" + url[len("azure://"):]
    try:
        fs, path = fsspec.core.url_to_fs(url)
    except (ImportError, ValueError) as e:
        raise ImportError(
            f"no fsspec backend for {scheme}://; install "
            f"{_FSSPEC_BACKENDS.get(scheme, 'the matching fsspec backend')}"
        ) from e
    return fs, path
