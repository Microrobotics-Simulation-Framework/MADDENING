"""The precision MADDENING traces its float32 matrix products at.

On NVIDIA GPUs, JAX's default ``jax_default_matmul_precision`` runs float32
matrix-matrix products at reduced precision (TF32-class passes).  Measured on
an RTX A2000, relative error against float64 at the default and at
``"highest"``: ``U.T @ JU`` with ``U`` of shape ``(1e5, 8)``, 3.0e-4 against
4.9e-7 (the coupling bounds' Krylov basis); ``H @ field`` for a 3-component
field, 2.5e-4 against 6.5e-7 (an interface mapping); ``w @ cells`` of shapes
``(2, 3) @ (3, 1e5)``, 4.0e-4 against 7.3e-8 (the heat node's ghost cells);
``J.T @ J`` with ``J`` of shape ``(2e4, 12)``, 9.0e-6 against 1.1e-7 (the
Fisher information).  Vector-vector and matrix-vector products and anything
in float64 are unaffected, and XLA's CPU backend computes every float32
product in full whatever is requested, so on CPU the setting changes nothing.

The setting is read when a program is **traced** (``lax.dot_general`` bakes
it into each product), so the library traces the body of every program it
builds under :func:`jax.default_matmul_precision`:

* :func:`pinned` wraps a function so that its body runs inside the context.
  ``GraphManager``'s step builders return pinned steps (see
  :func:`pins_the_built_step`), so the compiled step, every scan, sweep and
  adaptive program built around it, and a step traced inside a caller's own
  ``jax.jit`` or ``jax.grad`` all trace their products at the graph's
  :attr:`~maddening.core.graph_manager.GraphManager.matmul_precision`.
* :func:`custom_jvp_at_call_precision` covers the one place a context around
  a body does not reach: a ``jax.custom_jvp`` rule is traced lazily, when a
  derivative is taken -- by the caller's ``jax.grad`` or ``jax.jvp``, outside
  the body that called the function -- so the rule captures the precision in
  force at the call and traces under it.

``None`` means "inherit JAX's own setting": nothing is entered, and the
products take whatever ``jax_default_matmul_precision`` holds where they are
traced (the ``JAX_DEFAULT_MATMUL_PRECISION`` environment variable, or a
caller's own context).  It is the opt-out for reduced-precision tensor-core
speed, for instance in MLP or FNO surrogate nodes.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable, Iterator, Optional

import jax

#: What the library traces its programs at unless told otherwise.
DEFAULT: str = "highest"


def validated(value: Any) -> Optional[str]:
    """``value`` if it is ``None`` or a setting ``jax.default_matmul_precision``
    accepts (``"highest"``, ``"float32"``, ``"tensorfloat32"``, ``"bfloat16"``,
    ``"default"``, ``"high"``, or the name of a ``lax.DotAlgorithmPreset``),
    else ``ValueError``.

    JAX itself checks the value only when the context is entered -- at the next
    trace -- so a bad value is refused here, where it is set, by entering the
    context once.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"matmul_precision must be None or a string accepted by "
            f"jax.default_matmul_precision (for example 'highest'), got "
            f"{type(value).__name__} {value!r}")
    try:
        with jax.default_matmul_precision(value):
            pass
    except ValueError as exc:
        raise ValueError(
            f"matmul_precision={value!r} is not a setting "
            f"jax.default_matmul_precision accepts: {exc}") from None
    return value


def of(owner: Any) -> Optional[str]:
    """The ``matmul_precision`` of a graph (or anything carrying one): the
    library's :data:`DEFAULT` for an object that has none."""
    return getattr(owner, "matmul_precision", DEFAULT)


@contextlib.contextmanager
def traced_at(precision: Optional[str]) -> Iterator[None]:
    """Trace (and run eagerly) inside ``jax.default_matmul_precision(precision)``;
    ``None`` enters nothing."""
    if precision is None:
        yield
        return
    with jax.default_matmul_precision(precision):
        yield


def pinned(fn: Callable, precision: Optional[str]) -> Callable:
    """``fn`` with its body traced at ``precision``.

    The context is entered inside the function, so it is in force wherever
    the function is traced: under the library's ``jax.jit`` or
    ``jax.lax.scan``, and under a caller's own transformation of it.
    Attributes set on ``fn`` (a step builder's ``strict_messages``) are
    carried over.  ``None`` returns ``fn`` itself.
    """
    if precision is None:
        return fn

    @functools.wraps(fn)
    def at_precision(*args, **kwargs):
        with jax.default_matmul_precision(precision):
            return fn(*args, **kwargs)

    return at_precision


def pins_the_built_step(build: Callable) -> Callable:
    """Decorate a ``GraphManager`` step builder so that the step it returns is
    :func:`pinned` at the graph's ``matmul_precision``, read when the step is
    built.

    Every program the graph runs is built from such a step -- ``compile()``'s
    ``jax.jit``, the scans of ``run_scan``, ``run_scan_with_history``,
    ``run_sweep`` and ``run_adaptive_scan``, ``run_adaptive``'s stepper, the
    windows of :func:`maddening.sysid.windowed_loss` -- so this one decorator
    covers them all, including a step the caller traces under a
    transformation of their own.
    """
    @functools.wraps(build)
    def built(self, *args, **kwargs):
        return pinned(build(self, *args, **kwargs),
                      getattr(self, "_matmul_precision", DEFAULT))

    return built


def _in_force() -> Optional[str]:
    """The ``jax_default_matmul_precision`` in force here: a context's value
    inside one (a :func:`pinned` body), else the global setting."""
    return jax.config.jax_default_matmul_precision


def custom_jvp_at_call_precision(
    fun: Callable, rule: Callable, *, nondiff_argnums: tuple = (),
) -> Callable:
    """``jax.custom_jvp(fun, nondiff_argnums)`` with ``rule`` as its JVP rule,
    the rule traced at the matmul precision in force where the function is
    **called**.

    A custom JVP rule is traced lazily: when ``jax.grad`` or ``jax.jvp`` of a
    program reaches the call, which is after the body that made the call has
    returned, so a :func:`pinned` body does not reach its products.  The
    returned function reads ``jax_default_matmul_precision`` at each call --
    inside a pinned body, the body's setting -- and hands it to the rule as an
    extra non-differentiable argument, which enters it.  ``fun`` and ``rule``
    keep their own signatures (``rule(*nondiff, primals, tangents)``), and the
    returned function is called as ``fun`` is, positionally.
    """
    shifted = (0,) + tuple(int(i) + 1 for i in nondiff_argnums)

    def impl(precision, *args):
        del precision
        return fun(*args)

    def jvp(precision, *rest):
        with traced_at(precision):
            return rule(*rest)

    derivative = jax.custom_jvp(impl, nondiff_argnums=shifted)
    derivative.defjvp(jvp)

    @functools.wraps(fun)
    def call(*args):
        return derivative(_in_force(), *args)

    return call
