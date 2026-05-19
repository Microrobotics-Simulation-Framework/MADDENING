"""Reusable neural-network primitives (v0.2 #2).

A future-home for shared building blocks that surrogate architectures
compose: decoders, encoders, attention blocks, normalisation layers.
Currently a thin scaffolding package — concrete primitives will land
here as they are extracted from MIME or as new architectures need them.

Design contract
---------------
A primitive lives here when:

  * It has **no dependence on the SurrogateArchitecture ABC** — it is
    a pure function or a small ``equinox.Module`` that can be used
    standalone, inside any architecture, or in a non-surrogate
    context (e.g. a JAX research script).
  * It is **stateless** beyond its parameters — i.e. no closure over
    a configuration object that other code can't reconstruct.
  * Its **input/output contract** is shape- and dtype-explicit
    rather than dict-of-arrays.  Primitives compose; node-style
    dict I/O is for architectures.

External code should import from this package by name, e.g.::

    from maddening.surrogates.primitives import (
        cholesky_decoder,        # incoming from MIME
        random_fourier_features,
    )

Re-exports for backward compat will be added to the package
``__init__`` as primitives are added.
"""

# When primitives land, list them here.  Today the package is
# intentionally empty so the import path is reserved and tests can
# assert it exists.
__all__: list[str] = []
