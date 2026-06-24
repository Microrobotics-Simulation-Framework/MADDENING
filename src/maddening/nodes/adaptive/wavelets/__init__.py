"""Production Deslauriers-Dubuc adaptive-wavelet engine for WaveletAdaptiveNode.

Matrix-free DD wavelet transforms (:mod:`.transform`), Galerkin BCOO operators
with JAX-traceable variable coefficients (:mod:`.operator`), hybrid-Jacobi
preconditioning (:mod:`.precond`), and CDD active-set selection (:mod:`.cdd`).

These are the production rewrite of the derisking-spike numerics
(``spikes/wavelet_derisking/``): matrix-free, JIT-compilable, static-shape,
autodiff-correct.
"""

from __future__ import annotations

from maddening.nodes.adaptive.wavelets import cdd, operator, precond, transform

__all__ = ["transform", "operator", "precond", "cdd"]
