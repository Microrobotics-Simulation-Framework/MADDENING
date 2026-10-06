"""The interface-mapping kind registry: which factory a ``MappingSpec`` names.

A :class:`~maddening.core.coupling.mapping_spec.MappingSpec` records how a
mapping was built -- a ``kind``, hyper-parameters and references to the
arrays it was built from -- and a config or a USD stage carries that
recipe.  The ``kind`` is looked up here.  The four built-in kinds (``rbf``,
``nearest_neighbor``, ``projection_1d``, ``matrix``) are entries of this
registry like any other; :func:`register_mapping` adds one, so that a
mapping from another library survives ``GraphManager.to_dict`` /
``from_dict`` and a USD round trip.

Usage
-----
Define and register a factory::

    import jax.numpy as jnp
    import numpy as np

    from maddening.core.coupling.mapping import StaticLinearMapping
    from maddening.core.coupling.mapping_registry import register_mapping
    from maddening.core.coupling.mapping_spec import MappingSpec, reference_for_array

    @register_mapping("inverse_distance",
                      arrays=("source_points", "target_points"),
                      hyperparameters={"power": float})
    def inverse_distance_mapping(source_points, target_points, *, power=2.0,
                                 source_points_ref=None, target_points_ref=None):
        src = np.asarray(source_points, dtype=np.float64).reshape(-1, 1)
        tgt = np.asarray(target_points, dtype=np.float64).reshape(-1, 1)
        w = 1.0 / (np.abs(tgt - src.T) ** power + 1e-12)
        H = jnp.asarray(w / w.sum(axis=1, keepdims=True), dtype=jnp.float32)
        spec = MappingSpec("inverse_distance", {"power": float(power)}, {
            "source_points": reference_for_array(
                source_points, source_points_ref, name="source_points"),
            "target_points": reference_for_array(
                target_points, target_points_ref, name="target_points"),
        })
        return StaticLinearMapping(H, kind="inverse_distance", spec=spec)

and use it on an edge like a built-in one
(``gm.add_edge(..., mapping=inverse_distance_mapping(xs, xt))``).

A file can only name a kind
---------------------------
A config, a USD stage and a checkpoint are untrusted input.  All one of
them can say about a mapping is the *name* of its kind, which is looked up
in this table and nowhere else: nothing here imports a module, follows a
dotted path or discovers entry points because of what a file contains.  A
kind exists only because the running program imported the code that
registered it.  A file naming a kind this process has not registered is
refused (:class:`~maddening.core.coupling.mapping_spec.MappingRebuildError`
through ``from_dict`` / ``load_graph_from_usd``) with the registered kinds
in the message; import the module that registers it, then load.

The limits on references -- the asset size cap, the inline limits, the
accepted dtypes, the containment of asset paths in the config directory --
are enforced by the reference resolver before a factory is called, so they
hold for every kind, registered or built in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, TypeVar

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.node import _signature_required_arguments, _signature_takes_keyword

_F = TypeVar("_F", bound=Callable[..., Any])

#: The types a hyper-parameter may be declared as.  ``float`` is a *real
#: number*: a finite ``int`` or ``float`` that is not a ``bool``, stored as
#: a ``float`` (JSON has no ``Infinity`` / ``NaN``).  ``int`` is an integer:
#: an ``int`` that is not a ``bool``, of a magnitude a float64 can hold
#: (the bound a real has), stored as an ``int``.
_HYPERPARAMETER_TYPES: tuple[type, ...] = (str, bool, int, float)

#: Keys of a serialised spec that are not hyper-parameters: ``kind``,
#: ``points`` and the ``shape`` a mapping's description adds.
_RESERVED_KEYS = ("kind", "points", "shape")

#: ``label`` is the user-facing name of a ``matrix`` mapping, and a
#: serialised spec whose ``label`` repeats its ``kind`` is read as one
#: (``MappingSpec.from_dict``), so no other kind may take it.
_MATRIX_LABEL = "label"


@dataclass(frozen=True)
class _MappingKind:
    """One registered kind.

    A small record on purpose: what a kind declares is expected to grow
    (whether it needs the moving-geometry argument of ``Mapping.apply``,
    for one), and a new field with a default changes no caller.
    """

    kind: str
    factory: Callable[..., Any]
    #: The factory's array arguments, in the order a spec lists them.
    arrays: tuple[str, ...]
    #: Hyper-parameter name -> ``str``, ``bool``, ``int`` or ``float`` (a real).
    hyperparameters: dict[str, type]
    #: Array argument -> the factory keyword that carries its reference.
    references: dict[str, str]
    #: Shipped with the library; can never be replaced or removed.
    builtin: bool = False

    def declaration(self) -> tuple:
        """What was declared, without the factory, for comparing two
        registrations of one kind."""
        return (self.arrays, tuple(self.hyperparameters.items()),
                tuple(self.references.items()))


_MAPPING_REGISTRY: dict[str, _MappingKind] = {}

_BUILTINS_LOADED = False


def _ensure_builtins() -> None:
    """Import the modules that register the kinds this library ships.

    The built-in factories live in :mod:`maddening.core.coupling.mapping`,
    which registers them as it is imported.  Everything that reads the
    table asks for that import first, so a spec can be validated by a
    program that imported only ``mapping_spec``, and a built-in name is
    always taken before :func:`register_mapping` can be offered it.

    The sparse kinds of :mod:`maddening.core.coupling.sparse_mapping` are
    loaded with them, for the same two reasons: a config that names one
    loads in a program that imported nothing but the graph, and their
    names are taken first.  They are registered through the public
    :func:`register_mapping`, not as built-ins, so everything a registered
    kind is held to holds for them.

    These are the only imports the registry performs, and their targets
    are fixed: nothing a file contains chooses one.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    import maddening.core.coupling.mapping  # noqa: F401, PLC0415
    import maddening.core.coupling.sparse_mapping  # noqa: F401, PLC0415

    _BUILTINS_LOADED = True


def _lookup(kind: Any) -> Optional[_MappingKind]:
    """The entry registered as *kind*, or ``None``.

    *kind* may come straight from a file: anything that is not a string
    (a list, a dict, a number) is simply not a registered kind, rather
    than a ``TypeError`` out of a dict lookup.
    """
    _ensure_builtins()
    if not isinstance(kind, str):
        return None
    return _MAPPING_REGISTRY.get(kind)


def _registered_kinds() -> list[str]:
    """Every registered kind, sorted, for the "choose from" of a refusal."""
    _ensure_builtins()
    return sorted(_MAPPING_REGISTRY)


def _qualified(factory: Callable[..., Any]) -> str:
    name = getattr(factory, "__qualname__", None) or repr(factory)
    module = getattr(factory, "__module__", None)
    return f"{module}.{name}" if module else name


def _names(kind: str, what: str, names: Any) -> tuple[str, ...]:
    """*names* as a tuple of distinct identifiers, or a ``ValueError``."""
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise ValueError(
            f"mapping kind {kind!r}: {what} must be a tuple or list of argument "
            f"names, got {names!r}"
        )
    out = tuple(names)
    for name in out:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(
                f"mapping kind {kind!r}: {what} must be Python identifiers (they are "
                f"keyword arguments of the factory), got {name!r}"
            )
    if len(set(out)) != len(out):
        raise ValueError(f"mapping kind {kind!r}: {what} repeats a name: {list(out)}")
    return out


def _check_kind_name(kind: Any) -> None:
    from maddening.serialization.json_codec import (  # noqa: PLC0415
        NON_FINITE_TOKENS,
    )

    if not isinstance(kind, str) or not kind:
        raise ValueError(f"a mapping kind must be a non-empty string, got {kind!r}")
    if kind in NON_FINITE_TOKENS:
        raise ValueError(
            f"mapping kind {kind!r} spells a non-finite JSON token, which the "
            f"serialisers reserve: a config could not carry it.  A different "
            f"spelling ({kind.lower()!r}, say) is fine."
        )


def _declare(kind: str, factory: Callable[..., Any], arrays: Any, hyperparameters: Any,
             references: Any, *, builtin: bool) -> _MappingKind:
    """Validate one declaration and return its entry (nothing is stored)."""
    array_names = _names(kind, "arrays", arrays)

    if not isinstance(hyperparameters, Mapping):
        raise ValueError(
            f"mapping kind {kind!r}: hyperparameters must be a dict of name -> type "
            f"(str, bool, int or float), got {hyperparameters!r}"
        )
    hyper_names = _names(kind, "hyperparameters", list(hyperparameters))
    hyper: dict[str, type] = {}
    for name in hyper_names:
        declared = hyperparameters[name]
        if not any(declared is t for t in _HYPERPARAMETER_TYPES):
            raise ValueError(
                f"mapping kind {kind!r}: hyper-parameter {name!r} is declared as "
                f"{declared!r}; choose from str, bool, int and float (float is a "
                f"real number: a finite int or float that is not a bool)"
            )
        hyper[name] = declared
    reserved = sorted(set(hyper) & {*_RESERVED_KEYS, *(() if builtin else (_MATRIX_LABEL,))})
    if reserved:
        raise ValueError(
            f"mapping kind {kind!r}: hyper-parameter name(s) {reserved} are reserved; "
            f"{list(_RESERVED_KEYS)} are keys of a serialised mapping and "
            f"{_MATRIX_LABEL!r} is the user label of the 'matrix' kind"
        )

    if references is None:
        refs = {name: f"{name}_ref" for name in array_names}
    else:
        if not isinstance(references, Mapping):
            raise ValueError(
                f"mapping kind {kind!r}: references must be a dict of array name -> "
                f"factory keyword, got {references!r}"
            )
        if set(references) != set(array_names):
            raise ValueError(
                f"mapping kind {kind!r}: references must name exactly the arrays "
                f"{list(array_names)}, got {sorted(references, key=repr)}"
            )
        keywords = _names(kind, "references", [references[name] for name in array_names])
        refs = dict(zip(array_names, keywords))

    used: dict[str, str] = {}
    for role, names in (("an array", array_names), ("a hyper-parameter", hyper_names),
                        ("a reference keyword", tuple(refs.values()))):
        for name in names:
            if name in used:
                raise ValueError(
                    f"mapping kind {kind!r}: {name!r} is declared as both "
                    f"{used[name]} and {role}; the factory takes each as a "
                    f"keyword argument, so the names must differ"
                )
            used[name] = role

    _check_signature(kind, factory, used)
    return _MappingKind(kind=kind, factory=factory, arrays=array_names,
                        hyperparameters=hyper, references=refs, builtin=builtin)


def _check_signature(kind: str, factory: Callable[..., Any], used: dict[str, str]) -> None:
    """Refuse a declaration the factory could not be called with.

    The rebuild calls ``factory(**arrays, **hyperparameters, **references)``.
    A declared name the factory does not take would only surface there, as
    a ``TypeError`` while loading someone's config -- one it does not have,
    and one it has where no keyword reaches it (an optional positional-only
    parameter, the name of ``*args``); a required argument nothing declares
    would too, and so would a required positional-only one.  A callable
    whose signature cannot be read is taken on trust.

    Both questions are asked of ``maddening.core.node``, the one module
    that reads a signature.  Whether the factory takes a keyword is the
    rule every optional keyword in the package is probed with
    (``_signature_takes_keyword``: named in the signature, or forwarded by
    ``**kwargs``), not one of this module's own.
    """
    required = _signature_required_arguments(factory)
    if required is None:
        return
    # Before the keyword question: a required positional-only argument is
    # also a name no keyword reaches, and this is the refusal that says
    # what is wrong with it.
    positional = sorted(name for name, by_keyword in required.items() if not by_keyword)
    if positional:
        raise ValueError(
            f"mapping kind {kind!r}: {_qualified(factory)} requires positional-only "
            f"argument(s) {positional}, which no serialised mapping could supply: "
            f"the rebuild calls factory(**arrays, **hyperparameters, **references)"
        )
    absent = sorted(name for name in used if not _signature_takes_keyword(factory, name))
    if absent:
        raise ValueError(
            f"mapping kind {kind!r}: {_qualified(factory)} takes no keyword "
            f"argument(s) {absent}, which the registration declares "
            f"({', '.join(f'{n}: {used[n]}' for n in absent)}); the rebuild "
            f"calls factory(**arrays, **hyperparameters, **references)"
        )
    undeclared = sorted(name for name in required if name not in used)
    if undeclared:
        raise ValueError(
            f"mapping kind {kind!r}: {_qualified(factory)} requires argument(s) "
            f"{undeclared} that the registration declares as neither an array, a "
            f"hyper-parameter nor a reference keyword, so no serialised mapping "
            f"could supply them"
        )


def _add(kind: Any, factory: Any, arrays: Any, hyperparameters: Any, references: Any,
         *, builtin: bool) -> None:
    """Register *factory* as *kind*, or raise; the one way into the table.

    The same factory with the same declaration is a no-op, so a module
    that registers at import can be imported (or reloaded into the same
    function object) twice.  A different factory is refused, and so is the
    same one under a different declaration: silently keeping the first
    would leave the second caller with arguments the rebuild never passes.
    """
    _check_kind_name(kind)
    if not callable(factory):
        raise TypeError(
            f"mapping kind {kind!r}: the factory must be callable, got {factory!r}"
        )
    existing = _MAPPING_REGISTRY.get(kind)
    if existing is not None and existing.factory is not factory:
        if existing.builtin:
            raise ValueError(
                f"Mapping kind {kind!r} is built in ({_qualified(existing.factory)}) "
                f"and cannot be replaced. Register {_qualified(factory)} under "
                f"another name."
            )
        raise ValueError(
            f"Mapping kind {kind!r} is already registered to "
            f"{_qualified(existing.factory)}. Cannot re-register to "
            f"{_qualified(factory)}."
        )
    entry = _declare(kind, factory, arrays, hyperparameters, references,
                     builtin=builtin or (existing is not None and existing.builtin))
    if existing is None:
        _MAPPING_REGISTRY[kind] = entry
    elif existing.declaration() != entry.declaration():
        raise ValueError(
            f"Mapping kind {kind!r} is already registered to "
            f"{_qualified(existing.factory)} with a different declaration (arrays "
            f"{list(existing.arrays)}, hyper-parameters "
            f"{ {n: t.__name__ for n, t in existing.hyperparameters.items()} }, "
            f"references {existing.references}); registering it again must repeat "
            f"that declaration."
        )


@stability(StabilityLevel.EXPERIMENTAL)
def register_mapping(
    kind: str,
    *,
    arrays: Sequence[str],
    hyperparameters: Mapping[str, type],
    references: Optional[Mapping[str, str]] = None,
) -> Callable[[_F], _F]:
    """Decorator that registers a mapping factory under a kind name.

    A registered kind is serialisable: an edge carrying one of its
    mappings is written by ``GraphManager.to_dict`` and the USD writer as
    its :class:`~maddening.core.coupling.mapping_spec.MappingSpec`, and
    ``from_dict`` / ``load_graph_from_usd`` rebuild it by calling the
    factory as ``factory(**arrays, **hyperparameters, **references)`` --
    the arrays resolved from the spec's references, the hyper-parameters
    the spec holds, and each reference under its keyword -- provided the
    loading program has imported the module that registers the kind.  A
    file can only *name* a kind; see the module docstring.

    Parameters
    ----------
    kind : str
        The name a serialised mapping carries.  Any non-empty string but
        ``"NaN"``, ``"Infinity"`` and ``"-Infinity"``, which a config
        reserves for non-finite numbers; the names of the built-in kinds
        are taken.
    arrays : sequence of str
        The factory's array arguments: the point sets or matrices a spec
        refers to by reference (node field, asset file or small inline
        list).  May be empty.
    hyperparameters : mapping of str to type
        Every other argument a spec may carry, with its type: ``str``,
        ``bool``, ``int`` or ``float``.  ``float`` is a real number -- a
        finite ``int`` or ``float`` that is not a ``bool`` -- and reaches
        the factory as a ``float``.  ``int`` is an integer (a count, a
        size): an ``int`` that is not a ``bool``, of a magnitude a float64
        can hold, and reaches the factory as an ``int``; a float is refused
        for it, whatever its value.  A spec's hyper-parameters are checked
        against these declarations before the factory is called.  The
        names ``kind``, ``points``, ``shape`` and ``label`` are reserved.
    references : mapping of str to str, optional
        For each array, the factory keyword that carries its reference.
        ``"<array>_ref"`` when omitted.

    Returns
    -------
    Callable
        A decorator that returns the factory unchanged.

    Raises
    ------
    ValueError
        If *kind* is not a non-empty string, spells a non-finite JSON
        token, names a built-in kind, or is already registered to a
        different factory or with a different declaration (registering the
        same factory again with the same declaration is a no-op); if a
        name is not an identifier, is reserved or is declared twice; if a
        hyper-parameter type is not one of the four; or if the factory's
        signature does not take every declared name as a keyword, or
        requires an argument that is not declared or is positional-only.
    TypeError
        If the decorated object is not callable.

    Notes
    -----
    What the factory must return -- checked when a spec is rebuilt, with a
    refusal that names the kind:

    * an object with the members of the
      :class:`~maddening.core.coupling.mapping.Mapping` protocol whose
      ``kind`` is *kind*;
    * carrying, as ``spec``, a ``MappingSpec`` of this kind that records
      the references and the hyper-parameters the factory was given (build
      each reference with
      :func:`~maddening.core.coupling.mapping_spec.reference_for_array`,
      passing the reference keyword through), so that what is saved is
      what was built;
    * whose ``params_pytree()`` meets the contract below.

    The mapping may be a
    :class:`~maddening.core.coupling.mapping.StaticLinearMapping` -- a
    dense matrix, whose one weight is ``H`` -- or a class of your own.

    **What ``params_pytree()`` may contain.**  The graph snapshots it into
    ``gm.params["mappings"][edge.key]``, and checkpoints, ``POST
    /checkpoint/load``, system identification, the FMU's state archive and
    ``to_dict``'s "live weights differ" warning all walk that entry as a
    flat table of arrays.  So, for any mapping that is not a
    ``StaticLinearMapping``, ``add_edge`` refuses a ``params_pytree()``
    that is not:

    * a plain ``dict`` -- empty for a mapping that has no weights;
    * keyed by Python identifiers (a key is a member name in a checkpoint
      archive and a key of a config's ``param_specs``);
    * holding, under each key, one concrete JAX array of a floating-point
      dtype (any shape), all finite -- not a nested container, not a
      NumPy array or a Python number (whose dtype would depend on
      ``jax_enable_x64`` at the moment it is read), not an integer or
      complex array (keep indices and other structure as attributes of
      the mapping, outside the parameter tree);
    * the same on every call: same keys, shapes, dtypes and values.  It is
      what ``reset_params()`` restores and what ``to_dict`` compares the
      live weights with.

    Nothing public removes a kind: a serialised graph may name it for as
    long as the process runs.

    Examples
    --------
    See the module docstring, and "Registering your own mapping kind" in
    the interface-mapping guide.
    """
    _ensure_builtins()
    # Checked now rather than at decoration, so a bad name is reported at
    # the line that wrote it.
    _check_kind_name(kind)

    def decorator(factory: _F) -> _F:
        _add(kind, factory, arrays, hyperparameters, references, builtin=False)
        return factory

    return decorator


def _register_builtin(
    kind: str,
    *,
    arrays: Sequence[str],
    hyperparameters: Mapping[str, type],
    references: Mapping[str, str],
) -> Callable[[_F], _F]:
    """:func:`register_mapping` for the kinds the library ships.

    The same table and the same validation; the entry is marked so that it
    cannot be replaced or removed, and ``label`` -- reserved for the
    ``matrix`` kind -- is allowed.
    """
    def decorator(factory: _F) -> _F:
        _add(kind, factory, arrays, hyperparameters, references, builtin=True)
        return factory

    return decorator


def _unregister(kind: str) -> None:
    """Remove a registered kind.  **Test support only.**

    Private, and there is no public counterpart: a serialised graph names
    its kinds, so removing one from a running program strands every config
    that carries it.  A test that registers a kind for its own duration
    removes it again through this.  A built-in kind is never removed.
    """
    entry = _MAPPING_REGISTRY.get(kind)
    if entry is None:
        raise KeyError(f"mapping kind {kind!r} is not registered")
    if entry.builtin:
        raise ValueError(f"mapping kind {kind!r} is built in and cannot be removed")
    del _MAPPING_REGISTRY[kind]


__all__ = ["register_mapping"]
