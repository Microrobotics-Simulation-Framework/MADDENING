"""replace_node and helpers — node-replacement utilities for surrogates (v0.2 #2).

Was the leaf module ``maddening.surrogates.replace`` in v0.1; in v0.2
it becomes a subpackage so additional helpers (e.g. surrogate
preflight checks, batched replacement, sharding-aware variants) can
live alongside the core function without growing one monolithic file.

The public import path ``from maddening.surrogates.replace import
replace_node`` continues to work unchanged.
"""

from maddening.surrogates.replace._core import replace_node

__all__ = ["replace_node"]
