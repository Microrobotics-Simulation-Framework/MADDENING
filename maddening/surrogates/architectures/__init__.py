"""Built-in surrogate architectures."""

from maddening.surrogates.architectures.mlp import MLPDirect, MLPDerivative


def __getattr__(name: str):
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
