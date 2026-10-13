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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # PEP 562: the names below are resolved lazily by the module
    # `__getattr__`, which a type checker cannot see through -- without
    # these re-exports it reports every one of them as missing from
    # `__all__`, and a downstream `from maddening.surrogates.training
    # import X` is an error.  Nothing here runs at import time.
    from maddening.surrogates.training.callbacks import (
        EarlyStopping,
        LRSchedule,
        ModelCheckpoint,
        TrainingCallback,
    )
    from maddening.surrogates.training.physics_losses import (
        composite_loss,
        energy_conservation_loss,
        momentum_conservation_loss,
        residual_loss,
        smoothness_loss,
    )
    from maddening.surrogates.training.trainer import (
        SurrogateTrainer,
        TrainResult,
        mse_loss,
    )

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


# Spelled out rather than `list(_LAZY.keys())`: a type checker cannot
# evaluate a computed `__all__`, so a downstream `from ... import X`
# against this package would be an error.  Kept in `_LAZY` order; the
# `tests/test_lazy_reexports.py` pins the two against each other.
__all__ = [
    "SurrogateTrainer",
    "TrainResult",
    "mse_loss",
    "TrainingCallback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRSchedule",
    "residual_loss",
    "energy_conservation_loss",
    "momentum_conservation_loss",
    "smoothness_loss",
    "composite_loss",
]
