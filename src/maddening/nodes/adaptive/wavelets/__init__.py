"""The interpolating-wavelet engine behind :class:`~maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode`.

* :mod:`.transform` -- matrix-free Deslauriers-Dubuc lifting transforms
  (1-D, and the isotropic Mallat multiresolution in 2-D / 3-D), periodic.
* :mod:`.dirichlet` -- the boundary-adapted basis for homogeneous
  Dirichlet walls, built dense at construction.
* :mod:`.operator` -- the Galerkin operator ``Wn^T A_phys Wn`` and the
  two frozen-active-set solves (gathered dense block; masked full-size
  operator for ``ift_linear_solve``).
* :mod:`.precond` -- the diagonal (hybrid-Jacobi / Jacobi / Dahmen-Kunoth)
  scalings.
* :mod:`.cdd` -- Cohen-Dahmen-DeVore bulk-chasing selection of the
  active set, capped at a budget.

Every public function is ``@stability(EXPERIMENTAL)``: the engine is
importable, but the node is the surface the 0.4.0 freeze round covers.
"""

from __future__ import annotations

from maddening.nodes.adaptive.wavelets import cdd, dirichlet, operator, precond, transform

__all__ = ["cdd", "dirichlet", "operator", "precond", "transform"]
