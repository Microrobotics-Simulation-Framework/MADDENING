"""Weight serialisation for surrogate models (v0.2 #2).

Thin re-export of the existing
``maddening.surrogates.checkpoint`` module so callers can use the
restructured path:

    from maddening.surrogates.weights import save_weights, load_weights

The old import path continues to work for one minor version
(scheduled removal in v0.3).  Internal MADDENING code should migrate
to the new path; downstream packages (MIME, MICROROBOTICA) may move
on their own cadence.
"""

from maddening.surrogates.checkpoint import (
    save_weights,
    load_weights,
    load_train_result,
)

__all__ = ["save_weights", "load_weights", "load_train_result"]
