"""
MADDENING USD integration -- codeless schemas and USD I/O.

**Import order matters**: this module registers MADDENING's codeless
USD schemas via ``Plug.Registry().RegisterPlugins``.  The USD
``SchemaRegistry`` is a process-level singleton that caches type
definitions on first access.  If a ``Usd.Stage`` has already been
created (or the registry queried) *before* this module is imported,
the schemas will be invisible to USD's type system.

Always ``import maddening.usd`` before any ``Usd.Stage`` operations.
"""

import pathlib

from pxr import Plug, Usd

_schema_dir = pathlib.Path(__file__).parent / "schema"
Plug.Registry().RegisterPlugins([_schema_dir.absolute().as_posix()])

# Verify registration succeeded
_registry = Usd.SchemaRegistry()
if not _registry.FindConcretePrimDefinition("MaddeningNode"):
    raise RuntimeError(
        "MADDENING USD schema registration failed. "
        "Import maddening.usd BEFORE any Usd.Stage operations."
    )

from maddening.usd.writer import USDWriter  # noqa: E402
from maddening.usd.serialization import (  # noqa: E402
    save_graph_to_usd,
    load_graph_from_usd,
    register_node_class,
)
from maddening.usd.geometry import (  # noqa: E402
    load_grid_from_usd,
    create_vessel_phantom,
)

__all__ = [
    "USDWriter",
    "save_graph_to_usd",
    "load_graph_from_usd",
    "register_node_class",
    "load_grid_from_usd",
    "create_vessel_phantom",
]
