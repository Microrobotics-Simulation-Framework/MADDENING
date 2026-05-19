"""Training subpackage for surrogate models (v0.2 #2).

Thin re-export of the existing
``maddening.surrogates.{trainer,callbacks,physics_losses}`` modules
so callers can use the restructured path:

    from maddening.surrogates.training import (
        SurrogateTrainer, TrainResult,
        EarlyStopping, ModelCheckpoint, LRSchedule,
        residual_loss, energy_conservation_loss, momentum_conservation_loss,
        smoothness_loss, composite_loss,
    )

The original import paths continue to work for one minor version.

Imports are lazy because trainer/callbacks/physics_losses depend on
the optional ``equinox`` + ``optax`` packages.  Accessing any
attribute on this module triggers the corresponding source-module
import, mirroring the lazy pattern in
``maddening.surrogates.__init__``.
"""

from importlib import import_module
from typing import Any

# Map exported name → backing module.  Identical contract to
# ``maddening.surrogates.__getattr__`` so the package layout can be
# refactored further (physical file moves) without rewriting the
# call sites here.
_LAZY: dict[str, str] = {
    # Trainer & validation
    "SurrogateTrainer": "maddening.surrogates.trainer",
    "TrainResult": "maddening.surrogates.trainer",
    "mse_loss": "maddening.surrogates.trainer",
    # Callbacks
    "TrainingCallback": "maddening.surrogates.callbacks",
    "EarlyStopping": "maddening.surrogates.callbacks",
    "ModelCheckpoint": "maddening.surrogates.callbacks",
    "LRSchedule": "maddening.surrogates.callbacks",
    # Physics-informed losses
    "residual_loss": "maddening.surrogates.physics_losses",
    "energy_conservation_loss": "maddening.surrogates.physics_losses",
    "momentum_conservation_loss": "maddening.surrogates.physics_losses",
    "smoothness_loss": "maddening.surrogates.physics_losses",
    "composite_loss": "maddening.surrogates.physics_losses",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = import_module(_LAZY[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'maddening.surrogates.training' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_LAZY.keys())


__all__ = list(_LAZY.keys())
