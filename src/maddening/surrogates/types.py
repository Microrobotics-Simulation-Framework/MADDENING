"""Named shapes for the dictionaries the surrogate stack passes around.

Every ``dict`` parameter in :mod:`maddening.surrogates` is one of a small
number of shapes, and until this module existed each site described its
shape in a comment.  Two kinds of name are collected here, and which one
a site gets is decided by one question: **are the keys known statically?**

`PEP 589 <https://peps.python.org/pep-0589/>`_ ``TypedDict``
    For the dictionaries whose key set is fixed by this package --
    :class:`TrainState` and :class:`TrainMetrics`.  A checker knows the
    keys, so it rejects a misspelling and a wrong value type.

Type aliases
    For the dictionaries keyed by *field name*: :data:`StateDict`,
    :data:`BatchedStateDict`, :data:`SpecDict`, :data:`FieldValues` and
    :data:`WeightOverrides`.  Which fields a node has is a property of
    the node the caller passes, not of the function being called, so
    there is no static key set for a ``TypedDict`` to describe; naming
    the mapping is the whole of the available win.

A ``TypedDict`` *is* a ``dict`` at runtime -- same class, same
``PyTreeDef``, invisible to ``jax.jit`` -- so nothing here changes what
any of these values are or how they trace.  This is documentation the
type checker can read.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias, TypedDict

import jax
from jax.typing import ArrayLike

#: An arbitrary JAX pytree (network weights, optimiser state).  Mirrors
#: ``maddening.surrogates.architecture.PyTree``.
PyTree: TypeAlias = Any

#: A node's state, or its boundary inputs, or a prediction of either:
#: ``{field_name: array}``.  The field names come from whichever node is
#: being surrogated, so they are not known until runtime.
StateDict: TypeAlias = Mapping[str, jax.Array]

#: A :data:`StateDict` that a function builds and returns (or mutates).
#: Invariant and mutable, so it is a distinct name from :data:`StateDict`,
#: which is only ever read.
MutableStateDict: TypeAlias = dict[str, jax.Array]

#: A :data:`StateDict` whose arrays each carry a leading batch axis, as
#: consumed by the ``vmap``-ed training steps: ``{field_name: (B, ...)}``.
BatchedStateDict: TypeAlias = Mapping[str, jax.Array]

#: A state or boundary *specification*: ``{field_name: shape_tuple}``.
SpecDict: TypeAlias = Mapping[str, tuple[int, ...]]

#: Initial field values for a node: ``{field_name: float | array}``.
#: Wider than :data:`StateDict` because a scalar literal is accepted.
FieldValues: TypeAlias = Mapping[str, ArrayLike]

#: ``state -> d(state)/dt``, the function a derivative-mode surrogate
#: hands to its integrator.
DerivFn: TypeAlias = Callable[[StateDict], MutableStateDict]

#: ``(state, deriv_fn, dt) -> new_state``: the integrator contract that
#: :func:`~maddening.surrogates.node.euler_integrator` and
#: :func:`~maddening.surrogates.node.rk4_integrator` implement and that
#: ``SurrogateNode(integrator=...)`` accepts.
Integrator: TypeAlias = Callable[
    [StateDict, DerivFn, float], MutableStateDict
]

#: Flat per-leaf weight overrides for :meth:`SurrogateNode.update`:
#: ``{"weights" + leaf_path: array}``.  The keys are derived from the
#: weight pytree's own structure at call time.
WeightOverrides: TypeAlias = Mapping[str, jax.Array]


class TrainState(TypedDict):
    """The mutable training state shared with :mod:`.training.callbacks`.

    ``SurrogateTrainer.train`` creates exactly one of these and passes
    the same object to every callback hook, so a callback can replace
    ``weights`` (``EarlyStopping`` restores the best epoch this way) and
    the trainer picks the replacement up.

    Examples
    --------
    >>> state: TrainState = {"weights": (None, None), "opt_state": None}
    >>> sorted(state)
    ['opt_state', 'weights']
    """

    #: ``(arrays, static)`` as returned by the architecture's
    #: ``init_params`` -- ``equinox.partition`` splits the module there.
    weights: tuple[PyTree, PyTree]
    #: The optimiser state threaded through ``optax``.
    opt_state: PyTree


class TrainMetrics(TypedDict):
    """One epoch's losses, as passed to the callback hooks.

    ``EarlyStopping`` and ``ModelCheckpoint`` take a ``monitor`` key from
    the caller, so they read this with ``.get()`` rather than by
    subscript; any other reader knows both keys are present.

    Examples
    --------
    >>> metrics: TrainMetrics = {"train_loss": 0.5, "val_loss": 0.6}
    >>> metrics["val_loss"] > metrics["train_loss"]
    True
    """

    train_loss: float
    val_loss: float


__all__ = [
    "BatchedStateDict",
    "DerivFn",
    "FieldValues",
    "Integrator",
    "MutableStateDict",
    "PyTree",
    "SpecDict",
    "StateDict",
    "TrainMetrics",
    "TrainState",
    "WeightOverrides",
]
