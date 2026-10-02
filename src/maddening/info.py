"""Version and environment report: :func:`show_versions` and ``python -m maddening info``.

Paste its output into a bug report.  It names what decides how a
simulation runs -- the MADDENING, Python, JAX, jaxlib, NumPy and SciPy
versions, the JAX backend and devices, whether 64-bit floats are on,
``XLA_FLAGS`` -- and which optional extras are installed.

It never prints a secret.  Environment variables are read from a fixed
allowlist (:data:`ENV_ALLOWLIST`) of JAX / XLA / MADDENING settings that
hold no credentials; nothing else in the environment is read, so an API
token, a cloud key or a password in it cannot reach the report.

It is also safe to run on a broken installation, which is when it is
most useful: nothing here imports JAX at module import time, every
probe is guarded and reported rather than raised, and extras are found
with :func:`importlib.util.find_spec`, which locates a package without
importing (running) it.  Asking for the devices initialises the JAX
backend; it compiles nothing.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import sys
from typing import Any, Callable, Optional, TextIO, TypeVar

_T = TypeVar("_T")

#: Environment variables the report may show, and only when set.  None of
#: them holds a credential.  Anything else in the environment -- an API
#: token (``MADDENING_API_TOKEN``), cloud keys, passwords -- is never read.
ENV_ALLOWLIST: tuple[str, ...] = (
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "CUDA_VISIBLE_DEVICES",
    "MADDENING_COMPILATION_CACHE_DIR",
    "MADDENING_IFT_DENSE_SOLVE",
    "MADDENING_ADAPTIVE_DIAGNOSTICS",
)

#: The feature extras of ``pyproject.toml`` and the top-level modules each
#: one installs.  An extra is "installed" when every module is found.
#: Bundles (``server``, ``client``, ``all``), the per-provider cloud
#: extras, the empty ``ift`` alias and the developer extras (``ci``,
#: ``sbom``) are not listed: they install nothing these do not.
#: ``tests/core/test_show_versions.py`` holds this table to pyproject.
EXTRAS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("api", ("fastapi", "uvicorn", "websockets")),
    ("cloud", ("sky",)),
    ("compression", ("zstandard",)),
    ("cuda12", ("jax_cuda12_plugin",)),
    ("gpu-viz", ("pygfx", "rendercanvas", "glfw", "skimage")),
    ("network", ("zmq",)),
    ("streaming", ("gi", "websockets")),
    ("surrogates", ("equinox", "optax")),
    ("terminal", ("rich",)),
    ("tpu", ("libtpu",)),
    ("usd", ("pxr",)),
    ("viz", ("matplotlib",)),
    ("viz3d", ("pyvista", "PIL")),
)

#: Distributions whose versions are reported, by name.
_DEPENDENCIES = ("jax", "jaxlib", "numpy", "scipy", "lineax")


def _tag_experimental(obj: _T) -> _T:
    """``@stability(StabilityLevel.EXPERIMENTAL)``, applied only if the
    compliance package imports: it pulls in JAX, and this module has to
    load on an installation whose JAX is broken."""
    try:
        from maddening.core.compliance.metadata import StabilityLevel  # noqa: PLC0415
        from maddening.core.compliance.stability import stability  # noqa: PLC0415
    except Exception:   # noqa: BLE001 - a broken JAX is what this module reports
        return obj
    return stability(StabilityLevel.EXPERIMENTAL)(obj)


def _guard(fn: Callable[[], Any]) -> Any:
    """Run a probe; a failure becomes a short string instead of a raise."""
    try:
        return fn()
    except Exception as exc:   # noqa: BLE001 - every probe is reported, never raised
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        return f"unavailable ({type(exc).__name__}{': ' + message[:200] if message else ''})"


def _dist_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _found(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _maddening() -> dict[str, Any]:
    import maddening  # noqa: PLC0415 - the package itself is light (lazy imports)
    location = os.path.dirname(os.path.abspath(maddening.__file__ or "")) or "unknown"
    return {"version": maddening.__version__, "location": location}


def _jax_runtime() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    def devices() -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], int] = {}
        for d in jax.devices():
            key = (str(d.platform), str(getattr(d, "device_kind", d.platform)))
            groups[key] = groups.get(key, 0) + 1
        return [{"platform": p, "kind": k, "count": n} for (p, k), n in sorted(groups.items())]

    return {
        "backend": _guard(jax.default_backend),
        "devices": _guard(devices),
        "x64_enabled": _guard(lambda: bool(jax.config.read("jax_enable_x64"))),
    }


def _extras() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    try:
        dists = importlib.metadata.packages_distributions()
    except Exception:   # noqa: BLE001
        dists = {}
    for extra, modules in EXTRAS:
        missing = [m for m in modules if not _found(m)]
        versions = {}
        for m in modules:
            if m in missing:
                continue
            for dist in sorted(set(dists.get(m, ()))):
                v = _dist_version(dist)
                if v:
                    versions[dist] = v
        out[extra] = {"installed": not missing, "missing": missing, "versions": versions}
    return out


def collect_versions() -> dict[str, Any]:
    """Everything :func:`show_versions` prints, as a dict.

    Keys: ``maddening`` (``version``, ``location``), ``python``,
    ``platform``, ``dependencies`` (version or ``None`` per
    distribution), ``jax`` (``backend``, ``devices``, ``x64_enabled``),
    ``environment`` (the set :data:`ENV_ALLOWLIST` variables only) and
    ``extras`` (per extra: ``installed``, ``missing`` modules,
    ``versions``).  A probe that fails reads ``"unavailable (...)"``.
    """
    return {
        "maddening": _guard(_maddening),
        "python": f"{platform.python_version()} ({platform.python_implementation()})",
        "platform": _guard(platform.platform),
        "machine": _guard(platform.machine),
        "dependencies": {name: _dist_version(name) for name in _DEPENDENCIES},
        "jax": _guard(_jax_runtime),
        "environment": {k: os.environ[k] for k in ENV_ALLOWLIST if k in os.environ},
        "extras": _guard(_extras),
    }


def _render(info: dict[str, Any]) -> str:
    lines = ["MADDENING version information", "=" * 29]

    def row(key: str, value: Any, indent: str = "") -> None:
        lines.append(f"{indent}{key:<16}{value}")

    mad = info["maddening"]
    if isinstance(mad, dict):
        row("maddening", f"{mad['version']}  ({mad['location']})")
    else:
        row("maddening", mad)
    row("python", info["python"])
    row("platform", info["platform"])
    row("machine", info["machine"])
    lines.append("")
    lines.append("dependencies")
    for name, version in info["dependencies"].items():
        row(name, version or "not installed", "  ")
    lines.append("")
    lines.append("JAX runtime")
    jax_info = info["jax"]
    if isinstance(jax_info, dict):
        row("backend", jax_info["backend"], "  ")
        devices = jax_info["devices"]
        if isinstance(devices, list):
            text = ", ".join(f"{d['count']} x {d['platform']} ({d['kind']})" for d in devices)
            row("devices", text or "none", "  ")
        else:
            row("devices", devices, "  ")
        row("x64 enabled", jax_info["x64_enabled"], "  ")
    else:
        row("jax", jax_info, "  ")
    lines.append("")
    lines.append("environment (allowlisted variables that are set)")
    if info["environment"]:
        for key, value in info["environment"].items():
            lines.append(f"  {key} = {value}")
    else:
        lines.append("  (none set)")
    lines.append("")
    lines.append("optional extras")
    extras = info["extras"]
    if isinstance(extras, dict):
        for extra, entry in extras.items():
            if entry["installed"]:
                versions = ", ".join(f"{k} {v}" for k, v in sorted(entry["versions"].items()))
                row(extra, "installed" + (f" ({versions})" if versions else ""), "  ")
            else:
                row(extra, "missing " + ", ".join(entry["missing"]), "  ")
    else:
        row("extras", extras, "  ")
    return "\n".join(lines) + "\n"


def show_versions(file: Optional[TextIO] = None, *, as_dict: bool = False) -> Optional[dict[str, Any]]:
    """Print MADDENING's version and environment report, for bug reports.

    Shows the MADDENING version and install location; the Python,
    platform, JAX, jaxlib, NumPy, SciPy and lineax versions; the JAX
    backend, its devices (grouped by kind) and whether x64 is enabled;
    the set environment variables among :data:`ENV_ALLOWLIST` (which
    includes ``XLA_FLAGS``); and which optional extras are installed.
    The same report is ``python -m maddening info``.

    No secret is printed: only allowlisted environment variables are
    read.  A probe that fails (a JAX that cannot initialise its backend,
    say) is reported as ``unavailable (...)`` rather than raised.
    Listing devices initialises the JAX backend but compiles nothing,
    and no optional extra is imported -- each is located with
    :func:`importlib.util.find_spec`.

    Parameters
    ----------
    file : text stream, optional
        Where to print (default ``sys.stdout``).  Ignored with
        ``as_dict=True``.
    as_dict : bool
        Return the report as a dict (see :func:`collect_versions`)
        instead of printing it.

    Returns
    -------
    dict or None
        The report with ``as_dict=True``; otherwise ``None``.
    """
    info = collect_versions()
    if as_dict:
        return info
    (sys.stdout if file is None else file).write(_render(info))
    return None


show_versions = _tag_experimental(show_versions)
collect_versions = _tag_experimental(collect_versions)

__all__ = ["ENV_ALLOWLIST", "EXTRAS", "collect_versions", "show_versions"]
