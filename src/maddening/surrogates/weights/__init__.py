"""Surrogate-model weight serialisation.

Promoted to a real subpackage in v0.3.0 (the v0.2.x lazy-re-export
shim is removed).  The ``checkpoint`` module now lives at
``maddening.surrogates.weights.checkpoint``; the v0.2.x
``maddening.surrogates.checkpoint`` import path was removed in v0.3.0.
"""

from maddening.surrogates.weights.checkpoint import (
    save_weights,
    load_weights,
    load_train_result,
)

__all__ = ["save_weights", "load_weights", "load_train_result"]
