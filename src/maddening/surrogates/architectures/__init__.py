"""Built-in surrogate architectures."""

from typing import TYPE_CHECKING, Any

from maddening.surrogates.architectures.mlp import MLPDirect, MLPDerivative

if TYPE_CHECKING:
    # PEP 562: the names below are resolved lazily by the module
    # `__getattr__`, which a type checker cannot see through -- without
    # these re-exports it reports every one of them as missing from
    # `__all__`, and a downstream `from maddening.surrogates.architectures import X` is an error.
    # Nothing here runs at import time; the lazy table stays the only
    # runtime path.
    from maddening.surrogates.architectures.deeponet import (
        DeepONetDerivative,
        DeepONetDirect,
        SDeepONetDerivative,
        SDeepONetDirect,
    )
    from maddening.surrogates.architectures.fno import FNODerivative, FNODirect



def __getattr__(name: str) -> Any:
    """Lazy imports for architectures that need equinox."""
    _lazy = {
        "DeepONetDirect": "maddening.surrogates.architectures.deeponet",
        "DeepONetDerivative": "maddening.surrogates.architectures.deeponet",
        "SDeepONetDirect": "maddening.surrogates.architectures.deeponet",
        "SDeepONetDerivative": "maddening.surrogates.architectures.deeponet",
        "FNODirect": "maddening.surrogates.architectures.fno",
        "FNODerivative": "maddening.surrogates.architectures.fno",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'maddening.surrogates.architectures' has no attribute {name!r}")


__all__ = [
    "MLPDirect",
    "MLPDerivative",
    "DeepONetDirect",
    "DeepONetDerivative",
    "SDeepONetDirect",
    "SDeepONetDerivative",
    "FNODirect",
    "FNODerivative",
]
