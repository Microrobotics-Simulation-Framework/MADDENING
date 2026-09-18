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

Supported URL schemes (a closed allow-list; anything else is a
``ValueError``):

* ``file://`` — local file copy; the URL path is percent-decoded
  (``%20`` is a space), as a URL requires.
* a bare POSIX path — local file copy of exactly that path, with **no**
  percent-decoding: ``%`` is an ordinary character in a POSIX filename,
  so ``/tmp/a%20b.npz`` is the file of that name, not ``/tmp/a b.npz``.
* ``http://`` / ``https://`` — HTTP GET via the stdlib ``urllib`` with a
  timeout.
* ``s3://``, ``s3a://``, ``gs://``, ``gcs://``, ``az://``, ``abfs://``,
  ``abfss://``, ``adl://``, ``azure://``, ``memory://`` — via ``fsspec``
  and the matching backend (``s3fs``, ``gcsfs``, ``adlfs``).  Other
  fsspec protocols (``ftp``, ``sftp``, ``hdfs``, ``oss``, ...) are *not*
  accepted.

Windows drive-letter paths (``C:\\...``, ``file:///C:/...``) are not
supported: the drive letter parses as a URL scheme.  The cloud entry
point runs in a Linux container.

The manifest is fetched from the same location with ``.manifest.json``
appended to the URL *path*; the query string and fragment (a presigned
URL's signature) are preserved, so ``https://h/snap.npz?X-Amz-Signature=…``
looks for ``https://h/snap.npz.manifest.json?X-Amz-Signature=…``.  A
presigned URL only authorises the one object it was signed for, so with
presigned storage pass a second presigned URL as ``manifest_url=``
(``RESUME_MANIFEST_URL`` in the entry point).

The historical import path
``maddening.core.simulation.checkpoint.download_and_load_state`` is a
deprecated alias that forwards here and is removed in 1.0.
"""

from __future__ import annotations

import shutil
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.simulation.checkpoint import load_state_with_manifest

if TYPE_CHECKING:
    from maddening.core.graph_manager import GraphManager


__all__ = ["download_and_load_state", "DEFAULT_TIMEOUT", "SUPPORTED_SCHEMES"]

#: Seconds an HTTP(S) fetch may block before :class:`TimeoutError`.
DEFAULT_TIMEOUT: float = 60.0

#: Bytes per chunk when streaming a download to disk.
_CHUNK_SIZE = 1 << 20

_FSSPEC_SCHEMES = frozenset(
    {"s3", "s3a", "gs", "gcs", "az", "abfs", "abfss", "azure", "adl", "memory"}
)
_FSSPEC_BACKENDS = {"s3": "s3fs", "s3a": "s3fs", "gs": "gcsfs", "gcs": "gcsfs",
                    "az": "adlfs", "abfs": "adlfs", "abfss": "adlfs", "azure": "adlfs",
                    "adl": "adlfs"}

#: The closed allow-list of URL schemes (``""`` is a bare local path).
SUPPORTED_SCHEMES: frozenset[str] = frozenset({"", "file", "http", "https"}) | _FSSPEC_SCHEMES


@stability(StabilityLevel.EVOLVING)
def download_and_load_state(
    graph_manager: "GraphManager",
    url: str,
    *,
    dest_dir: Optional[str | Path] = None,
    skip_integrity_check: bool = False,
    manifest_url: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Download a checkpoint + manifest from *url* and load it.

    Parameters
    ----------
    graph_manager : GraphManager
        Compiled graph whose state is restored from the checkpoint.
    url : str
        Location of the ``.npz`` checkpoint.  Exactly these schemes are
        accepted (closed allow-list):

        * ``file://`` — local file path; the URL path is percent-decoded
          (``%20`` is a space).  A bare POSIX path is accepted too and
          is used verbatim: ``%`` is a legal character in a filename, so
          decoding it would look for a file the caller did not name.
        * ``http://`` / ``https://`` — HTTP GET with *timeout*.
        * ``s3://``, ``s3a://``, ``gs://``, ``gcs://``, ``az://``,
          ``abfs://``, ``abfss://``, ``adl://``, ``azure://``,
          ``memory://`` — via ``fsspec`` (C3, v0.4.0); install the
          matching backend (``s3fs``, ``gcsfs``, ``adlfs``).  Credentials
          come from the backend's usual environment / config.  No other
          fsspec protocol is accepted.

        Windows drive-letter paths are not supported (the drive letter
        parses as a URL scheme).
    dest_dir : str or Path, optional
        Directory the files are downloaded into.  Defaults to a fresh
        per-call temporary directory so concurrent resumes that target
        the same filename do not collide; that temporary directory is
        removed again when the call returns or raises.  A caller-supplied
        directory is left in place with the downloaded files in it.
    skip_integrity_check : bool, default False
        When True, a missing manifest is tolerated and the hash / schema
        verification is skipped.  Do not use in production.
    manifest_url : str, optional
        Where to fetch the sidecar manifest from.  Defaults to *url*
        with ``.manifest.json`` appended to its path component (query
        string and fragment preserved).  Presigned URLs authorise one
        object only, so pass the manifest's own presigned URL here.
    timeout : float, default 60
        Seconds an HTTP(S) connection or read may block before the fetch
        fails with :class:`TimeoutError`.  Forwarded to the fsspec
        backend where one is known to accept a timeout (``s3fs``,
        ``gcsfs``); best effort otherwise.

    Returns
    -------
    dict
        The manifest dict (empty when no manifest was available and
        ``skip_integrity_check`` is set).

    Raises
    ------
    ValueError
        The URL is empty, its scheme is not in the allow-list, or a
        ``file://`` source is a directory.
    FileNotFoundError
        A ``file://`` source, or its manifest, does not exist.
    TimeoutError
        An HTTP(S) fetch exceeded *timeout*.
    ImportError
        The URL needs ``fsspec`` or a backend that is not installed.
    CheckpointIntegrityError
        The downloaded checkpoint does not match its manifest.
    """
    parsed = _parse_checkpoint_url(url)
    if manifest_url is None:
        manifest_url = _manifest_url_for(url)
    else:
        _parse_checkpoint_url(manifest_url)

    owns_dest = dest_dir is None
    if owns_dest:
        # Per-call temp dir avoids cross-call leakage when multiple
        # downloads target the same filename; removed in the finally.
        dest_dir = Path(tempfile.mkdtemp(prefix="maddening_resume_"))
    else:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Local filename = last path component, decoded for a real URL
        # and verbatim for a bare path -- the same rule _local_path
        # applies to the source, so the two halves of this function
        # cannot disagree about what the caller named.
        source_path = parsed.path if parsed.scheme else url
        if parsed.scheme:
            source_path = urllib.parse.unquote(source_path)
        fname = Path(source_path).name or "checkpoint.npz"
        local_npz = dest_dir / fname
        local_manifest = local_npz.with_suffix(local_npz.suffix + ".manifest.json")

        _fetch(url, local_npz, timeout=timeout)
        # The manifest is optional in skip_integrity mode; otherwise required.
        try:
            _fetch(manifest_url, local_manifest, timeout=timeout)
        except Exception:
            if not skip_integrity_check:
                raise

        return load_state_with_manifest(
            graph_manager, local_npz,
            skip_integrity_check=skip_integrity_check,
        )
    finally:
        if owns_dest:
            shutil.rmtree(dest_dir, ignore_errors=True)


def _parse_checkpoint_url(url: str) -> urllib.parse.SplitResult:
    """Validate *url* against the allow-list and return its split form."""
    if not isinstance(url, str):
        raise TypeError(f"checkpoint URL must be a str, got {type(url).__name__}")
    if not url.strip():
        raise ValueError("empty checkpoint URL")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in SUPPORTED_SCHEMES:
        raise ValueError(
            f"Unsupported URL scheme {parsed.scheme!r}; expected file://, "
            "http://, https://, or one of the fsspec schemes "
            + ", ".join(f"{s}://" for s in sorted(_FSSPEC_SCHEMES))
            + " (Windows drive-letter paths are not supported)"
        )
    return parsed


def _manifest_url_for(url: str) -> str:
    """The sidecar-manifest URL that pairs with checkpoint *url*.

    ``.manifest.json`` is appended to the *path* component; the query
    string and fragment stay where they are so a presigned URL's
    signature survives.  A bare local path has no query/fragment
    semantics and gets the suffix appended verbatim.
    """
    if urllib.parse.urlsplit(url).scheme == "":
        return url + ".manifest.json"
    cut = min((i for i in (url.find("?"), url.find("#")) if i >= 0), default=len(url))
    return url[:cut] + ".manifest.json" + url[cut:]


def _local_path(parsed: urllib.parse.SplitResult, url: str) -> Path:
    """Filesystem path for a ``file://`` URL or a bare path.

    A ``file://`` URL is percent-decoded, because that is what its
    encoding means.  A bare path is *not*: ``%`` is an ordinary
    character in a POSIX filename, and decoding it would send the caller
    to a file they did not name (and, for ``%2e%2e``, to a directory
    they did not name).  The caller who wants decoding has said so by
    writing a URL.
    """
    if parsed.scheme == "":
        return Path(url)
    if parsed.netloc not in ("", "localhost"):
        raise ValueError(
            f"file:// URL with a host is not supported: {url!r} "
            "(use file:///absolute/path; Windows drive letters are not supported)"
        )
    return Path(urllib.parse.unquote(parsed.path))


def _fetch(url: str, dest: Path, *, timeout: float = DEFAULT_TIMEOUT) -> None:
    """Copy *url* contents into *dest*, streaming in 1 MiB chunks.

    Pure-stdlib for ``file://`` and ``http(s)://`` so we don't pull in
    another HTTP dep; ``fsspec`` only for cloud-storage schemes.  Used by
    :func:`download_and_load_state`.
    """
    parsed = _parse_checkpoint_url(url)
    if parsed.scheme in ("file", ""):
        src = _local_path(parsed, url)
        if src.is_dir():
            raise ValueError(f"file:// source is a directory, not a checkpoint: {src}")
        if not src.is_file():
            raise FileNotFoundError(f"file:// source not found: {src}")
        shutil.copyfile(src, dest)
        return
    if parsed.scheme in ("http", "https"):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 — trusted
                with open(dest, "wb") as out:
                    shutil.copyfileobj(response, out, _CHUNK_SIZE)
        except (socket.timeout, TimeoutError) as e:
            raise TimeoutError(
                f"timed out after {timeout:g} s fetching {_redact(url)}"
            ) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                raise TimeoutError(
                    f"timed out after {timeout:g} s fetching {_redact(url)}"
                ) from e
            raise
        return
    # Allow-list already enforced by _parse_checkpoint_url: fsspec scheme.
    fs, path = _fsspec_open(url, timeout=timeout)
    with fs.open(path, "rb") as f, open(dest, "wb") as out:
        shutil.copyfileobj(f, out, _CHUNK_SIZE)


def _redact(url: str) -> str:
    """*url* with its query string and fragment replaced (for messages)."""
    parts = urllib.parse.urlsplit(url)
    if not parts.query and not parts.fragment:
        return url
    return urllib.parse.urlunsplit(
        parts._replace(query="<redacted>" if parts.query else "", fragment=""),
    )


def _is_fsspec_scheme(scheme: str) -> bool:
    return scheme in _FSSPEC_SCHEMES


def _fsspec_timeout_options(scheme: str, timeout: float) -> dict:
    """Backend constructor kwargs that carry *timeout* (best effort).

    Only the backends whose signatures are known get one; the others
    use their own defaults.  ``fsspec``'s base class accepts unknown
    ``storage_options`` silently, so this never has to raise.
    """
    if scheme in ("s3", "s3a"):
        return {"config_kwargs": {"connect_timeout": timeout, "read_timeout": timeout}}
    if scheme in ("gs", "gcs"):
        return {"timeout": timeout}
    return {}


def _fsspec_open(url: str, *, timeout: float = DEFAULT_TIMEOUT):
    """``(filesystem, path)`` for an fsspec URL, with actionable errors."""
    scheme = urllib.parse.urlsplit(url).scheme
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
    options = _fsspec_timeout_options(scheme, timeout)
    try:
        try:
            fs, path = fsspec.core.url_to_fs(url, **options)
        except TypeError:
            if not options:
                raise
            fs, path = fsspec.core.url_to_fs(url)  # backend rejects the timeout kwargs
    except (ImportError, ValueError) as e:
        raise ImportError(
            f"no fsspec backend for {scheme}://; install "
            f"{_FSSPEC_BACKENDS.get(scheme, 'the matching fsspec backend')}"
        ) from e
    return fs, path
