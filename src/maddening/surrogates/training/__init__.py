"""Surrogate-model training subpackage.

Promoted to a real subpackage in v0.3.0 (the v0.2.x lazy-re-export
shim is removed).  ``trainer``, ``callbacks``, and ``physics_losses``
are now physical modules under this package.

Imports are still lazy because trainer / callbacks / physics_losses
depend on the optional ``equinox`` + ``optax`` packages — accessing
any attribute on this module triggers the corresponding source-module
import on demand.
"""

from importlib import import_module
from typing import Any

_LAZY: dict[str, str] = {
    # Trainer & validation
    "SurrogateTrainer": "maddening.surrogates.training.trainer",
    "TrainResult": "maddening.surrogates.training.trainer",
    "mse_loss": "maddening.surrogates.training.trainer",
    # Callbacks
    "TrainingCallback": "maddening.surrogates.training.callbacks",
    "EarlyStopping": "maddening.surrogates.training.callbacks",
    "ModelCheckpoint": "maddening.surrogates.training.callbacks",
    "LRSchedule": "maddening.surrogates.training.callbacks",
    # Physics-informed losses
    "residual_loss": "maddening.surrogates.training.physics_losses",
    "energy_conservation_loss": "maddening.surrogates.training.physics_losses",
    "momentum_conservation_loss": "maddening.surrogates.training.physics_losses",
    "smoothness_loss": "maddening.surrogates.training.physics_losses",
    "composite_loss": "maddening.surrogates.training.physics_losses",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = import_module(_LAZY[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(
        f"module 'maddening.surrogates.training' has no attribute {name!r}",
    )


def __dir__() -> list[str]:
    return sorted(_LAZY.keys())


__all__ = list(_LAZY.keys())
