"""
Modular Surrogate Model Framework for MADDENING.

Core abstractions (no extra dependencies beyond JAX):
    SurrogateArchitecture, SurrogateNode, SurrogateDataset,
    DatasetGenerator, replace_node, euler_integrator, rk4_integrator

Training & validation (requires equinox + optax):
    SurrogateTrainer, TrainResult, SurrogateValidator, ValidationReport

Callbacks:
    TrainingCallback, EarlyStopping, ModelCheckpoint, LRSchedule

Physics-informed losses:
    residual_loss, energy_conservation_loss, momentum_conservation_loss,
    smoothness_loss, composite_loss

Weight serialization:
    save_weights, load_weights, load_train_result

Built-in architectures (requires equinox):
    MLPDirect, MLPDerivative,
    DeepONetDirect, DeepONetDerivative,
    SDeepONetDirect, SDeepONetDerivative,
    FNODirect, FNODerivative
"""

from typing import TYPE_CHECKING, Any

from maddening.surrogates.architecture import SurrogateArchitecture
from maddening.surrogates.node import (
    SurrogateNode,
    euler_integrator,
    rk4_integrator,
)
from maddening.surrogates.dataset import SurrogateDataset, DatasetGenerator
from maddening.surrogates.replace import replace_node

if TYPE_CHECKING:
    # PEP 562: the names below are resolved lazily by the module
    # `__getattr__`, which a type checker cannot see through -- without
    # these re-exports it reports every one of them as missing from
    # `__all__`, and a downstream `from maddening.surrogates import X` is an error.
    # Nothing here runs at import time; the lazy table stays the only
    # runtime path.
    from maddening.surrogates.architectures.deeponet import (
        DeepONetDerivative,
        DeepONetDirect,
        SDeepONetDerivative,
        SDeepONetDirect,
    )
    from maddening.surrogates.architectures.fno import FNODerivative, FNODirect
    from maddening.surrogates.architectures.mlp import MLPDerivative, MLPDirect
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
    from maddening.surrogates.validator import (
        SurrogateValidator,
        ValidationReport,
    )
    from maddening.surrogates.weights.checkpoint import (
        load_train_result,
        load_weights,
        save_weights,
    )



def __getattr__(name: str) -> Any:
    """Lazy imports for components that need equinox/optax."""
    _lazy = {
        # Trainer & validation
        "SurrogateTrainer": "maddening.surrogates.training.trainer",
        "TrainResult": "maddening.surrogates.training.trainer",
        "mse_loss": "maddening.surrogates.training.trainer",
        "SurrogateValidator": "maddening.surrogates.validator",
        "ValidationReport": "maddening.surrogates.validator",
        # Callbacks
        "TrainingCallback": "maddening.surrogates.training.callbacks",
        "EarlyStopping": "maddening.surrogates.training.callbacks",
        "ModelCheckpoint": "maddening.surrogates.training.callbacks",
        "LRSchedule": "maddening.surrogates.training.callbacks",
        # Physics losses
        "residual_loss": "maddening.surrogates.training.physics_losses",
        "energy_conservation_loss":
            "maddening.surrogates.training.physics_losses",
        "momentum_conservation_loss":
            "maddening.surrogates.training.physics_losses",
        "smoothness_loss": "maddening.surrogates.training.physics_losses",
        "composite_loss": "maddening.surrogates.training.physics_losses",
        # Checkpoint
        "save_weights": "maddening.surrogates.weights.checkpoint",
        "load_weights": "maddening.surrogates.weights.checkpoint",
        "load_train_result": "maddening.surrogates.weights.checkpoint",
        # Architectures
        "MLPDirect": "maddening.surrogates.architectures.mlp",
        "MLPDerivative": "maddening.surrogates.architectures.mlp",
        "DeepONetDirect": "maddening.surrogates.architectures.deeponet",
        "DeepONetDerivative": "maddening.surrogates.architectures.deeponet",
        "SDeepONetDirect": "maddening.surrogates.architectures.deeponet",
        "SDeepONetDerivative": "maddening.surrogates.architectures.deeponet",
        "FNODirect": "maddening.surrogates.architectures.fno",
        "FNODerivative": "maddening.surrogates.architectures.fno",
    }
    if name in _lazy:
        import importlib
        try:
            mod = importlib.import_module(_lazy[name])
            return getattr(mod, name)
        except ImportError as exc:
            raise ImportError(
                f"'{name}' requires equinox and/or optax. "
                f"Install with:  pip install maddening[surrogates]"
            ) from exc
    raise AttributeError(f"module 'maddening.surrogates' has no attribute {name!r}")


__all__ = [
    "SurrogateArchitecture",
    "SurrogateNode",
    "SurrogateDataset",
    "DatasetGenerator",
    "SurrogateTrainer",
    "TrainResult",
    "SurrogateValidator",
    "ValidationReport",
    "TrainingCallback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRSchedule",
    "residual_loss",
    "energy_conservation_loss",
    "momentum_conservation_loss",
    "smoothness_loss",
    "composite_loss",
    "save_weights",
    "load_weights",
    "load_train_result",
    "MLPDirect",
    "MLPDerivative",
    "DeepONetDirect",
    "DeepONetDerivative",
    "SDeepONetDirect",
    "SDeepONetDerivative",
    "FNODirect",
    "FNODerivative",
    "replace_node",
    "euler_integrator",
    "rk4_integrator",
    "mse_loss",
]
