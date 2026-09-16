"""Build and package a MADDENING co-simulation FMU.

An FMU is a zip with ``modelDescription.xml``, a binary under
``binaries/<platform>/<modelIdentifier>.<ext>`` and a ``resources/``
directory.  The binary is the thin C wrapper in ``c/maddening_fmu.c``:
it holds no state and forwards every FMI call over TCP/JSON to a
:class:`~maddening.fmi.tcp_bridge.FmuTcpBridge` serving the JAX graph.
``resources/endpoint.txt`` tells the wrapper where that bridge listens.

Typical use::

    md = build_model_description(gm, model_name="Plant", model_identifier=MODEL_IDENTIFIER)
    binary = build_fmu_binary(out_dir)                  # needs a C compiler once
    write_fmu(md, "plant.fmu", binary=binary, endpoint="127.0.0.1:5555")

    bridge = FmuTcpBridge(sidecar, md, master_dt=gm_base_dt, port=5555).start()
    # ... the importer loads plant.fmu and drives it through the bridge ...
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.fmi.model_description import ModelDescription

MODEL_IDENTIFIER = "maddening_fmu"
_C_DIR = Path(__file__).resolve().parent / "c"
C_SOURCE = _C_DIR / "maddening_fmu.c"
FMI3_INCLUDE_DIR = _C_DIR / "include"


def fmi_platform_tuple() -> str:
    """FMI 3.0 platform directory name, e.g. ``x86_64-linux``."""
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    system = {"linux": "linux", "darwin": "darwin", "windows": "windows"}[
        platform.system().lower()
    ]
    return f"{arch}-{system}"


def _shared_lib_suffix() -> str:
    return {"windows": ".dll", "darwin": ".dylib"}.get(platform.system().lower(), ".so")


def find_c_compiler() -> Optional[str]:
    for cand in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cand and shutil.which(cand):
            return cand
    return None


@stability(StabilityLevel.EVOLVING)
def build_fmu_binary(
    out_dir: str | os.PathLike,
    *,
    cc: Optional[str] = None,
    extra_flags: tuple[str, ...] = (),
) -> Path:
    """Compile ``maddening_fmu.c`` into ``<out_dir>/<MODEL_IDENTIFIER>.so``.

    Only libc and the vendored FMI 3.0 headers are needed.  Raises
    ``RuntimeError`` when no C compiler is available.
    """
    cc = cc or find_c_compiler()
    if cc is None:
        raise RuntimeError(
            "no C compiler found (set $CC or install gcc/clang); the FMU "
            "binary cannot be built on this machine"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{MODEL_IDENTIFIER}{_shared_lib_suffix()}"
    cmd = [cc, "-shared", "-fPIC", "-O2", f"-I{FMI3_INCLUDE_DIR}", str(C_SOURCE),
           "-o", str(out), *extra_flags]
    if sys.platform.startswith("win"):
        cmd += ["-lws2_32"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"building the FMU binary failed:\n{' '.join(cmd)}\n{proc.stderr}")
    return out


@stability(StabilityLevel.EVOLVING)
def write_fmu(
    model_description: ModelDescription,
    path: str | os.PathLike,
    *,
    binary: Optional[str | os.PathLike] = None,
    endpoint: Optional[str] = None,
    platform_dir: Optional[str] = None,
) -> Path:
    """Write ``path`` (a ``.fmu`` zip).

    ``binary`` is the compiled wrapper from :func:`build_fmu_binary`
    (omit it for a description-only FMU); ``endpoint`` (``"host:port"``)
    is stored in ``resources/endpoint.txt`` for the wrapper to connect
    to.  The description must carry ``co_simulation_model_identifier``
    when a binary is packaged.
    """
    path = Path(path)
    md = model_description
    if binary is not None and not md.co_simulation_model_identifier:
        raise ValueError(
            "packaging a binary needs a <CoSimulation> element: build the "
            f"description with model_identifier={MODEL_IDENTIFIER!r}"
        )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("modelDescription.xml", md.to_xml())
        if binary is not None:
            binary = Path(binary)
            ident = md.co_simulation_model_identifier
            zf.write(binary, f"binaries/{platform_dir or fmi_platform_tuple()}/"
                             f"{ident}{binary.suffix}")
        if endpoint is not None:
            zf.writestr("resources/endpoint.txt", endpoint + "\n")
        else:
            zf.writestr("resources/README.txt",
                        "Set MADDENING_FMU_ENDPOINT=host:port or add endpoint.txt here.\n")
    return path


__all__ = [
    "C_SOURCE", "FMI3_INCLUDE_DIR", "MODEL_IDENTIFIER",
    "build_fmu_binary", "find_c_compiler", "fmi_platform_tuple", "write_fmu",
]
