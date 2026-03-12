"""
Simulation provenance tracking (Section 9.2).

Opt-in provenance capture for reproducibility and regulatory traceability.

Pure Python — no JAX dependency for the dataclass itself.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SimulationProvenance:
    """Captures the provenance of a simulation run.

    Attributes
    ----------
    maddening_version : str
        MADDENING version used.
    python_version : str
        Python version.
    platform_info : str
        OS and architecture.
    jax_version : str
        JAX version (populated lazily).
    jax_backend : str
        JAX backend (cpu, gpu, tpu).
    timestamp : float
        Unix timestamp of provenance capture.
    graph_config : dict
        Serialized graph configuration.
    random_seed : int or None
        Random seed used, if any.
    custom : dict
        User-supplied metadata.
    """
    maddening_version: str = ""
    python_version: str = field(default_factory=lambda: sys.version)
    platform_info: str = field(default_factory=platform.platform)
    jax_version: str = ""
    jax_backend: str = ""
    timestamp: float = field(default_factory=time.time)
    graph_config: dict = field(default_factory=dict)
    random_seed: Optional[int] = None
    custom: dict = field(default_factory=dict)

    @classmethod
    def capture(cls, graph_config: Optional[dict] = None, **kwargs: Any) -> "SimulationProvenance":
        """Capture current environment provenance.

        Parameters
        ----------
        graph_config : dict, optional
            Serialized graph configuration (from ``GraphManager.to_dict()``).
        **kwargs
            Additional fields to set (e.g., ``random_seed=42``).
        """
        jax_version = ""
        jax_backend = ""
        try:
            import jax
            jax_version = jax.__version__
            jax_backend = str(jax.default_backend())
        except ImportError:
            pass

        maddening_version = ""
        try:
            from maddening import __version__
            maddening_version = __version__
        except (ImportError, AttributeError):
            pass

        return cls(
            maddening_version=maddening_version,
            jax_version=jax_version,
            jax_backend=jax_backend,
            graph_config=graph_config or {},
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        from dataclasses import asdict
        return asdict(self)
