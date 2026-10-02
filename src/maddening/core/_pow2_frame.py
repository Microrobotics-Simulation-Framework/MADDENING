"""Scale-free arithmetic by exact powers of two: the one place it is done.

A float computation that is meant to be *relative* -- a ratio, a relaxed
step, a Krylov solve with a relative tolerance -- still depends on the units
its operands are written in wherever an intermediate leaves the normal range.
XLA's CPU backend flushes a subnormal result to zero, so the difference of two
iterates of a field near ``1e-34`` (float32), the square of a residual change
near ``1e-20``, or one ulp of a JVP tangent below ``tiny / eps`` is read as
exactly zero, and an absolute constant beside them (a ``1e-8`` tolerance, a
``1e-12`` floor) is not negligible at small units the way it is at scale one.

Multiplying by a power of two is exact: ``x * 2**k`` has the same significand
as ``x`` and changes only the exponent, so a computation posed on operands
rescaled by ``2**k`` and scaled back rounds exactly as the bare one does,
**bit for bit**, wherever the bare one's intermediates were normal numbers --
and unlike the bare one it stays in the normal range at any magnitude.  That
is why every frame here is a power of two and not, say, ``1 / max|x|``: a
decimal factor would change the rounding of every result at ordinary
magnitudes, and the coupling runtime's bit-identity claims (a group at any
power-of-two scale reproduces its unscaled run, 125 dumped configurations
unchanged by the frames' introduction) rest on the factor being exact.

:func:`pow2_frame` computes the factor in three modes:

* ``"common"``: one factor ``p`` with ``max_i |a_i| * p`` in ``[0.5, 1)``
  over every array passed -- for a dot product, a norm, a least-squares or a
  Krylov solve on vectors that share units (Aitken's two residuals, IQN's
  secant matrix and residual, the IFT solve's right-hand side, an
  accelerator's state vector, the sharded solvers' right-hand side, Adam's
  first gradient);
* ``"entrywise"``: one factor per entry, ``max_j |a_j[i]| * k[i]`` in
  ``[0.5, 1)`` -- for a difference or a sum taken entry by entry, so a small
  field beside a large one in the same flat vector is framed by its own size
  (an accelerator's relaxed step, the coupling report's residual, curvature
  step and secant);
* ``"lift"``: the least power of two ``>= 1`` that brings ``max|a|`` up to
  ``tiny / eps`` (``2**-102`` in float32: the smallest magnitude whose
  one-ulp change is a normal number), and exactly ``1`` at every larger
  magnitude -- for the coupling report's JVP tangents (spectral rate and
  gradient bound), which are taken in state units (``v * max|field|``, a
  relative perturbation of order one) so that a node's derivative
  intermediates keep their natural size.  Below ``tiny / eps`` the small
  components of such a tangent flush and the products read a different
  Jacobian, so there the tangent is lifted and the product divided by the
  lift again, exactly (``J`` is linear).  Normalising to order one instead
  was tried and is wrong both ways: at ``|x| ~ 1e18`` an order-one tangent
  is a relative perturbation of ``1/|x|`` and a nonlinear node's
  ``d(1/u) = -du/u**2`` underflowed (a gradient bound 0.79x its control,
  usable), and at a tiny state ``1/u`` times an order-one tangent
  overflowed.  A node whose derivative intermediates are within ``2**24``
  of overflow at a near-subnormal state can still overflow under the lift,
  which reads as a non-finite report (``spectral_usable=False``), not a
  wrong number.

In every mode the exponent is clamped so the factor itself is a normal number
of the arrays' dtype, and a zero or non-finite input gets the factor ``1``,
which the callers' own guards then handle.  :func:`pow2_rescue` is the fixed
factor ``1 / finfo.tiny`` the convergence norm applies, under a ``where``,
to a field whose scale or one-ulp change is below the normal range
(``maddening.core.coupling.acceleration._scaled_change``).

**The expressions are kept verbatim.**  Each mode is the exact sequence of
operations its call sites evaluated before they shared this module (Aitken
and IQN's ``_pow2_normaliser``, the relaxed step's ``_pow2_entrywise``, the
report's ``_tangent_lift``), because the frames sit in compiled loop bodies
that XLA fuses with the node updates around them, and an equivalent rewrite
of a frame has moved a node update by an ulp before.  The consolidation was
proved bit-identical on the coupling bit-identity dump under every jaxlib CI
runs; ``tests/core/test_pow2_frame.py`` pins what each mode returns.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np

_MODES = ("common", "entrywise", "lift")


def pow2_frame(*arrays, mode: str = "common"):
    """The power-of-two factor that frames *arrays* (see the module docstring).

    Parameters
    ----------
    *arrays : array_like
        Floating arrays.  ``"common"`` and ``"entrywise"`` take any number,
        the latter of one shape (broadcastable); ``"lift"`` takes one.
    mode : {"common", "entrywise", "lift"}
        ``"common"``: a scalar ``p`` with ``max |a| * p`` in ``[0.5, 1)``
        over every array.  ``"entrywise"``: an array ``k`` with
        ``max_j |a_j[i]| * k[i]`` in ``[0.5, 1)``.  ``"lift"``: a scalar,
        the least power of two ``>= 1`` with ``max|a| * p >= tiny / eps``.

    Returns
    -------
    jnp.ndarray
        The factor, an exact power of two in the arrays' (promoted) dtype;
        ``1`` where the input is zero or non-finite.  For example, in
        float32: ``[3.0, -0.1]`` is framed by ``0.25`` (common);
        ``[3.0, 0.0, 1e-3]`` by ``[0.25, 1.0, 512.0]`` (entrywise); and
        ``[1e-3]`` by ``1.0`` (lift: it is far above ``tiny / eps``).
        ``tests/core/test_pow2_frame.py`` pins each mode.
    """
    if mode == "common":
        dtype = jnp.result_type(*arrays)
        biggest = functools.reduce(
            jnp.maximum, [jnp.max(jnp.abs(v)) for v in arrays])
        info = jnp.finfo(dtype)
        _, exponent = jnp.frexp(jnp.where(jnp.isfinite(biggest), biggest,
                                          jnp.zeros_like(biggest)))
        exponent = jnp.clip(exponent, 1 - int(info.maxexp), -int(info.minexp))
        return jnp.ldexp(jnp.ones((), dtype), -exponent)
    if mode == "entrywise":
        dtype = jnp.result_type(*arrays)
        info = jnp.finfo(dtype)
        biggest = functools.reduce(jnp.maximum, [jnp.abs(jnp.asarray(a, dtype)) for a in arrays])
        _, exponent = jnp.frexp(jnp.where(jnp.isfinite(biggest), biggest, jnp.zeros_like(biggest)))
        exponent = jnp.clip(exponent, 1 - int(info.maxexp), -int(info.minexp))
        return jnp.ldexp(jnp.ones_like(biggest), -exponent)
    if mode == "lift":
        if len(arrays) != 1:
            raise ValueError(f"pow2_frame(mode='lift') takes one array, got {len(arrays)}")
        (scale,) = arrays
        dtype = jnp.asarray(scale).dtype
        info = jnp.finfo(dtype)
        biggest = jnp.max(jnp.abs(scale))
        usable = jnp.logical_and(jnp.isfinite(biggest), biggest > 0)
        _, have = jnp.frexp(jnp.where(usable, biggest, jnp.ones_like(biggest)))
        _, want = np.frexp(float(info.tiny) / float(info.eps))
        return jnp.ldexp(jnp.ones((), dtype), jnp.maximum(int(want) - have, 0))
    raise ValueError(f"pow2_frame: mode must be one of {_MODES}, got {mode!r}")


def pow2_rescue(dtype):
    """``1 / finfo(dtype).tiny``: the fixed power of two that lifts the smallest
    normal number to one, as a scalar of *dtype*.

    The convergence norm multiplies a field's pair by it, under a ``where``,
    where the field's scale or a one-ulp change of it is below the normal
    range; ``float()`` of it is the same number as a Python float
    (``2**126`` for float32).
    """
    return 1.0 / jnp.finfo(dtype).tiny
