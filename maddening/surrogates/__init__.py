"""
Modular Surrogate Model Framework for MADDENING.

Core abstractions (no extra dependencies beyond JAX):
    SurrogateArchitecture, SurrogateNode, SurrogateDataset,
    DatasetGenerator, replace_node, euler_integrator, rk4_integrator

Training & validation (requires equinox + optax):
    SurrogateTrainer, TrainResult, SurrogateValidator, ValidationReport

Built-in architectures (requires equinox):
    MLPDirect, MLPDerivative
"""

from maddening.surrogates.architecture import SurrogateArchitecture
from maddening.surrogates.node import (
    SurrogateNode,
    euler_integrator,
    rk4_integrator,
)
from maddening.surrogates.dataset import SurrogateDataset, DatasetGenerator
from maddening.surrogates.replace import replace_node


def __getattr__(name: str):
    """Lazy imports for components that need equinox/optax."""
    _lazy = {
        "SurrogateTrainer": "maddening.surrogates.trainer",
        "TrainResult": "maddening.surrogates.trainer",
        "mse_loss": "maddening.surrogates.trainer",
        "SurrogateValidator": "maddening.surrogates.validator",
        "ValidationReport": "maddening.surrogates.validator",
        "MLPDirect": "maddening.surrogates.architectures.mlp",
        "MLPDerivative": "maddening.surrogates.architectures.mlp",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
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
    "MLPDirect",
    "MLPDerivative",
    "replace_node",
    "euler_integrator",
    "rk4_integrator",
    "mse_loss",
]
