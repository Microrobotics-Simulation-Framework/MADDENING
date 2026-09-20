"""JSON encoding of the non-finite floats, for every MADDENING JSON surface.

RFC 8259 has no literal for ``NaN`` or the infinities.  ``json.dumps``
defaults to ``allow_nan=True`` and writes the bare tokens ``NaN``,
``Infinity`` and ``-Infinity`` anyway; ``json.loads`` reads them back, so
a document that never leaves Python looks fine, and every conforming
reader of the same document rejects it.  That was ``MADD-ANO-006``, and
it bit exactly when a document is most wanted: the way a non-finite value
reaches a config, a stage or the FMI wire is that a model diverged and
someone serialised it to find out why.

**The encoding.**  A non-finite float is written as the *quoted* form of
the same three tokens -- ``"NaN"``, ``"Infinity"``, ``"-Infinity"``.
Three reasons for quoted tokens over a tagged object such as
``{"__float__": "NaN"}``:

* A 0.3.x document and a 0.4.0 document then differ by two characters
  per value, so a config or a ``.usda`` attribute stays diffable and a
  human reading it still sees what the number was.
* A tagged object changes the *shape* of the document, not just the type
  of one leaf: a reader walking ``values`` as an array of numbers meets a
  nested object, and one walking node params meets a dict where a scalar
  was.  Quoted tokens keep every container the shape it had.
* The FMU's C wrapper parses wire values with ``strtod``, which is C99
  and *does* parse ``nan``, ``inf`` and ``infinity`` case-insensitively.
  The only thing between it and a quoted token is the quote character,
  so the C side needs two lines rather than a JSON parser.  A tagged
  object would need the parser.  (The ``MADD-ANO-006`` entry said a
  tagged form "would break the shipped C wrapper, which parses values
  with strtod"; the wrapper's limit is the quote, not ``strtod``.)

**Disambiguation.**  A blind decoder cannot tell the float ``nan`` from a
string that happens to read ``"NaN"``, so :func:`encode_non_finite`
refuses the collision at *write* time: a string leaf equal to one of the
three tokens is a :class:`ValueError` naming its path.  No document
MADDENING writes is ambiguous, which is what makes blind decoding sound.
The alternative -- decoding only at leaves the schema types as float --
is not available here: node params are typed by the node class, and
``to_dict`` has no schema for them.

**Encode once, at one boundary.**  :func:`encode_non_finite` is
deliberately *not* idempotent, and cannot be made so: running it twice
means the second walk meets a string leaf spelling a token, and the
whole point of the disambiguation above is that such a leaf is refused.
A function that could tell "a token I wrote" from "a data string that
spells one" is exactly the function this module does not have.  So a
tree is encoded once, and a document that has already been encoded --
``GraphManager.to_dict()``, ``MappingSpec.to_dict()``, the USD JSON
attributes -- is written with :func:`dumps_encoded` or plain
``json.dumps``, never with :func:`dumps`, which would encode it again
and refuse its own output.  :func:`decode_non_finite` *is* idempotent
(a float passes through), so the reading side composes freely.

**Reading is lenient, writing is strict.**  :func:`loads` accepts the
bare tokens a 0.3.x document carries as well as the quoted form, because
``json.loads`` already parses them into floats and
:func:`decode_non_finite` passes a float through untouched.  :func:`dumps`
passes ``allow_nan=False``, so a non-finite value the walk somehow missed
is a :class:`ValueError` at the call site instead of an invalid document
on disk -- the failure mode this module exists to remove, closed from the
other side as well.
"""

from __future__ import annotations

import json
import math
from typing import Any

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

#: The token for a quiet or signalling NaN.  One token: JSON has no
#: payload or sign for a NaN, and neither does any reader of one.
NAN_TOKEN = "NaN"

#: The token for positive infinity.
INF_TOKEN = "Infinity"

#: The token for negative infinity.
NEG_INF_TOKEN = "-Infinity"

#: Every string a decoder turns back into a float.  A *data* string equal
#: to one of these is refused by :func:`encode_non_finite`.
NON_FINITE_TOKENS = frozenset({NAN_TOKEN, INF_TOKEN, NEG_INF_TOKEN})

_DECODE = {NAN_TOKEN: math.nan, INF_TOKEN: math.inf, NEG_INF_TOKEN: -math.inf}

__all__ = [
    "NAN_TOKEN", "INF_TOKEN", "NEG_INF_TOKEN", "NON_FINITE_TOKENS",
    "encode_non_finite", "decode_non_finite", "dumps", "dumps_encoded", "loads",
]


def _token_for(value: float) -> str:
    if math.isnan(value):
        return NAN_TOKEN
    return INF_TOKEN if value > 0 else NEG_INF_TOKEN


def _is_float_leaf(obj: Any) -> bool:
    """A leaf ``json.dumps`` would write as a bare non-finite token.

    Exactly ``isinstance(obj, float)``, which is also the C encoder's own
    test -- ``numpy.float64`` subclasses ``float`` and is included,
    ``numpy.float32`` does not and raises ``TypeError`` from
    ``json.dumps`` today whether it is finite or not.  Deliberately no
    wider than that: a type ``json`` refuses must go on refusing, rather
    than acquiring a representation only when its value is non-finite.
    ``bool`` is an ``int``, never a ``float``.
    """
    return isinstance(obj, float)


@stability(StabilityLevel.EVOLVING)
def encode_non_finite(obj: Any, *, _path: str = "$") -> Any:
    """Replace every non-finite float in *obj* with its quoted token.

    Walks dicts, lists and tuples.  Keys are never touched: a key is a
    name, never a number, so nothing decodes one back.  Containers whose
    contents did not change are returned as they are rather than copied,
    so a finite document costs one walk and no allocation.

    **Not idempotent.**  ``encode_non_finite(encode_non_finite(x))``
    raises wherever ``x`` held a non-finite float, because the first
    walk left a string spelling a token where the second walk refuses
    one.  That is the disambiguation working, not a bug: nothing can
    distinguish the token this function wrote from a data string that
    spells the same three characters, which is why the collision is
    refused at all.  Encode once; write an already-encoded document
    with :func:`dumps_encoded`.

    Parameters
    ----------
    obj : Any
        A JSON-shaped tree: dicts, lists, tuples and scalars.

    Returns
    -------
    Any
        The same tree with ``float('nan')`` as ``"NaN"``, ``float('inf')``
        as ``"Infinity"`` and ``-float('inf')`` as ``"-Infinity"``.

    Raises
    ------
    ValueError
        If a *string* leaf equals one of the three tokens.  Encoding it
        unchanged would make the document ambiguous -- the decoder would
        read it back as a float -- so it is refused here, where the path
        to the offending value is still known.

    Examples
    --------
    >>> encode_non_finite({"bounds": [float("-inf"), 1.0]})
    {'bounds': ['-Infinity', 1.0]}
    """
    if isinstance(obj, str):
        if obj in NON_FINITE_TOKENS:
            raise ValueError(
                f"{_path}: the string {obj!r} cannot be written to JSON, "
                f"because it is how a non-finite float is encoded and would "
                f"read back as that float.  Store it as something else (a "
                f"different spelling, or a tagged value of your own).  If "
                f"this tree was already encoded -- GraphManager.to_dict() "
                f"and the MappingSpec / USD helpers return encoded "
                f"documents -- write it with dumps_encoded() or json.dumps(), "
                f"not dumps(): encoding twice refuses the encoder's own "
                f"output."
            )
        return obj
    if _is_float_leaf(obj):
        value = float(obj)
        return obj if math.isfinite(value) else _token_for(value)
    if isinstance(obj, dict):
        out = {k: encode_non_finite(v, _path=f"{_path}.{k}")
               for k, v in obj.items()}
        return obj if all(a is b for a, b in zip(out.values(), obj.values())) else out
    if isinstance(obj, (list, tuple)):
        out = [encode_non_finite(v, _path=f"{_path}[{i}]")
               for i, v in enumerate(obj)]
        if all(a is b for a, b in zip(out, obj)):
            return obj
        return tuple(out) if isinstance(obj, tuple) else out
    return obj


@stability(StabilityLevel.EVOLVING)
def decode_non_finite(obj: Any) -> Any:
    """Turn every quoted non-finite token in *obj* back into a float.

    The inverse of :func:`encode_non_finite` on anything that function
    wrote.  A float is passed through, so a document carrying the bare
    0.3.x tokens -- which ``json.loads`` has already parsed into floats --
    comes out of this function unchanged and correct.

    Parameters
    ----------
    obj : Any
        A JSON-shaped tree, typically straight from ``json.loads``.

    Returns
    -------
    Any
        The same tree with ``"NaN"``, ``"Infinity"`` and ``"-Infinity"``
        replaced by the floats they denote.  Dict *keys* are left alone.

    Examples
    --------
    >>> decode_non_finite({"bounds": ["-Infinity", 1.0]})["bounds"][0]
    -inf
    """
    if isinstance(obj, str):
        return _DECODE.get(obj, obj)
    if isinstance(obj, dict):
        out = {k: decode_non_finite(v) for k, v in obj.items()}
        return obj if all(a is b for a, b in zip(out.values(), obj.values())) else out
    if isinstance(obj, (list, tuple)):
        out = [decode_non_finite(v) for v in obj]
        if all(a is b for a, b in zip(out, obj)):
            return obj
        return tuple(out) if isinstance(obj, tuple) else out
    return obj


@stability(StabilityLevel.EVOLVING)
def dumps(obj: Any, **kwargs: Any) -> str:
    """``json.dumps`` of *obj* that is always valid JSON.

    Encodes the non-finite floats first, then dumps with
    ``allow_nan=False`` so that anything the walk did not reach raises
    here rather than leaving a bare token in the document.

    *obj* must be a **raw** tree -- one whose non-finite values are still
    floats.  A document that has already been through
    :func:`encode_non_finite` (``GraphManager.to_dict()``,
    ``MappingSpec.to_dict()``, the USD JSON attributes) goes to
    :func:`dumps_encoded` instead: encoding is not idempotent, so
    ``dumps`` would walk it a second time, meet the tokens the first
    walk wrote and refuse them as ambiguous strings.

    Parameters
    ----------
    obj : Any
        A JSON-shaped tree that has *not* been encoded yet.
    **kwargs
        Passed to ``json.dumps``.  ``allow_nan`` is fixed to ``False``.

    Returns
    -------
    str
        A document every conforming JSON reader accepts.

    Raises
    ------
    ValueError
        From :func:`encode_non_finite` for an ambiguous string leaf --
        including every token in an already-encoded document, which is
        what :func:`dumps_encoded` exists for -- or from ``json.dumps``
        for a non-finite float the walk did not reach (a value inside a
        ``numpy`` array, say, or a custom container).

    See Also
    --------
    dumps_encoded : the same output for an already-encoded document.
    """
    kwargs.pop("allow_nan", None)
    return json.dumps(encode_non_finite(obj), allow_nan=False, **kwargs)


@stability(StabilityLevel.EVOLVING)
def dumps_encoded(obj: Any, **kwargs: Any) -> str:
    """``json.dumps`` of a document :func:`encode_non_finite` already walked.

    The write boundary for everything that encodes as it builds:
    ``GraphManager.to_dict()``, ``MappingSpec.to_dict()`` and the USD
    JSON attributes all return documents whose non-finite values are
    already quoted tokens.  Handing one of those to :func:`dumps` is the
    natural thing to write and raises, because the encoding is not
    idempotent (see the module docstring); this function is that call,
    spelled so it composes.

    ``allow_nan=False`` is kept, so a non-finite float that never reached
    an encoder still fails here rather than becoming a bare token in the
    document -- the backstop is the half of :func:`dumps` that is about
    the *output*, and it applies just as much to input somebody else
    encoded.

    Parameters
    ----------
    obj : Any
        A JSON-shaped tree whose non-finite floats are already quoted
        tokens.
    **kwargs
        Passed to ``json.dumps``.  ``allow_nan`` is fixed to ``False``.

    Returns
    -------
    str
        A document every conforming JSON reader accepts.

    Raises
    ------
    ValueError
        From ``json.dumps`` if *obj* still holds a non-finite float --
        which means it was not encoded after all, and belongs in
        :func:`dumps`.

    Examples
    --------
    >>> dumps_encoded({"bounds": ["-Infinity", 1.0]})
    '{"bounds": ["-Infinity", 1.0]}'
    """
    kwargs.pop("allow_nan", None)
    return json.dumps(obj, allow_nan=False, **kwargs)


@stability(StabilityLevel.EVOLVING)
def loads(text: Any, **kwargs: Any) -> Any:
    """``json.loads`` of *text*, decoding the non-finite tokens.

    Both forms are accepted: the quoted tokens 0.4.0 writes and the bare
    ones a 0.3.x document carries, the latter through ``json.loads``'s own
    leniency.  Reading stays lenient on purpose -- strictness belongs on
    the writing side, where it stops an invalid document being made.

    Parameters
    ----------
    text : str or bytes
        A JSON document.
    **kwargs
        Passed to ``json.loads``.

    Returns
    -------
    Any
        The decoded tree, with non-finite floats as floats.
    """
    return decode_non_finite(json.loads(text, **kwargs))
