"""Read-only inspection of a :class:`~maddening.core.graph_manager.GraphManager`.

The logic behind ``GraphManager.format_graph`` / ``print_graph``,
``to_mermaid`` / ``to_dot``, ``print_graph_diagram``, ``state_summary``,
``params_table``, ``coupling_report`` and ``memory_estimate``.  The
methods on the graph are thin delegates to the functions here.

Every function in this module is **strictly read-only**.  None of them
writes the graph's state, ``_meta``, ``params``, node parameters, the
dirty or compiled flags, the schedule or any cache, and none of them
compiles or traces the step: they do not call ``compile()``, they do not
call any entry point that recompiles a dirty graph, and they do not call
``_recover_from_escaped_tracers`` (which would put a graph left holding
a transform's tracers back to its pre-transform state -- a write).  What
each one does instead on a graph that cannot be read as it stands:

* **never compiled** -- the structure and the state ``add_node``
  initialised are shown, marked "not compiled"; what only ``compile()``
  computes (the schedule, rate dividers, ``_meta``) is reported as
  unknown rather than computed.
* **modified since the last compile** -- what the last compile
  committed is shown, marked stale.
* **holding escaped tracers** (``jax.grad`` of a loss that calls
  ``run_scan``) -- shapes and dtypes are read from the tracers, which
  carry them; values (statistics, the coupling report) are reported as
  unavailable, with the remedy: the next stepping method, or any entry
  point that reads the state, puts the graph back.

Statistics are taken on the host with NumPy (``np.asarray`` transfers a
device array to the host; that is a copy, not a computation), so no
JAX primitive is dispatched and nothing is compiled.  The one exception
is :func:`coupling_report`, which reads
:meth:`GraphManager.coupling_diagnostics`: that method evaluates the
residual's float floor with eager ``jax.numpy`` operations, which JAX
compiles once per shape the first time.  It is not a graph trace -- the
step's trace counts and compile generation do not move -- and the
report never calls it on a graph holding tracers, where it would write.

Plain text is the default output and is deterministic: rows are sorted
where their order carries no meaning, numbers are formatted with a
fixed precision, and the layout depends on ``width`` (default
:data:`DEFAULT_WIDTH`), never on the terminal.  ``rich=True`` renders
the same content with the optional ``rich`` package
(``pip install maddening[terminal]``) and is only ever used when asked
for.
"""

from __future__ import annotations

import math
import re
import sys
import textwrap
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields as dc_fields
from typing import TYPE_CHECKING, Any, Optional, TextIO, overload

import jax
# ``jax.core`` is not re-exported by ``jax/__init__.py``; imported by name
# so ``jax.core.Tracer`` resolves for a type checker (as in graph_manager).
import jax.core
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

if TYPE_CHECKING:
    from maddening.core.graph_manager import GraphManager

#: Line width of the plain-text layouts when none is given.  Fixed rather
#: than read from the terminal so the output is the same everywhere.
DEFAULT_WIDTH = 100

_META_KEY = "_meta"
_FLAGS = "flags"
_RICH_HINT = ("rich=True needs the optional 'rich' package "
              "(pip install maddening[terminal]); the default plain text needs nothing")


# ----------------------------------------------------------------------
# Formatting primitives
# ----------------------------------------------------------------------

def _cell(value: Any) -> str:
    """Deterministic text for one value."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.6g}"
    if isinstance(value, tuple):
        return "(" + ", ".join(_cell(v) for v in value) + (",)" if len(value) == 1 else ")")
    return str(value)


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{n} B"  # pragma: no cover - the loop always returns


def _wrap(text: str, width: int, indent: str, subsequent: Optional[str] = None) -> list[str]:
    """Wrap without ever splitting a word, so a long name stays whole on
    its own line rather than being cut in two."""
    return textwrap.wrap(
        text, width=max(width, len(indent) + 20), initial_indent=indent,
        subsequent_indent=indent + "  " if subsequent is None else subsequent,
        break_long_words=False, break_on_hyphens=False,
    ) or [indent.rstrip()]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


# ----------------------------------------------------------------------
# InspectionTable
# ----------------------------------------------------------------------

@stability(StabilityLevel.EXPERIMENTAL)
class InspectionTable(Sequence):
    """Rows returned by the inspection methods of ``GraphManager``.

    ``state_summary``, ``params_table``, ``coupling_report`` and
    ``memory_estimate`` all return one, and each has a ``print_*``
    counterpart on the graph that prints it.  It is a sequence of
    ``dict`` rows (``table[0]["max"]``, ``list(table)``,
    ``pandas.DataFrame(list(table))``) plus what a reader needs to
    interpret them:

    * :attr:`columns` -- the keys shown in the text table, in order (a
      row may carry more keys, such as ``params_table``'s ``section``);
    * :attr:`notes` -- caveats and status ("not compiled", "the state
      holds tracers"), printed under the table;
    * :attr:`summary` -- table-wide values (``memory_estimate``'s
      totals).

    A row's ``"flags"`` entry, when the table has one, is a tuple of
    warnings about that row (a field holding NaN, a parameter outside
    its bounds, a coupling group that hit ``max_iterations``); the text
    layout prints them beneath the table, each prefixed ``!``.

    The table is a snapshot: building it read the graph and changed
    nothing, and editing a row changes only the row.

    ``str(table)`` is the plain-text layout at :data:`DEFAULT_WIDTH`;
    :meth:`to_text` takes a width, and :meth:`print` can render with
    ``rich``.
    """

    def __init__(
        self,
        title: str,
        columns: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        *,
        label_columns: Sequence[str] = (),
        notes: Sequence[str] = (),
        summary: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._title = str(title)
        self._columns = tuple(columns)
        self._rows = tuple(dict(r) for r in rows)
        self._label_columns = tuple(label_columns) or self._columns[:1]
        self._notes = tuple(notes)
        self._summary = dict(summary or {})

    # -- sequence protocol ----------------------------------------------

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[dict[str, Any], ...]: ...

    def __getitem__(self, index: int | slice) -> dict[str, Any] | tuple[dict[str, Any], ...]:
        return self._rows[index]

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)

    # -- metadata --------------------------------------------------------

    @property
    def title(self) -> str:
        """One line naming what the table holds."""
        return self._title

    @property
    def columns(self) -> tuple[str, ...]:
        """The row keys the text layout shows, in order (``"flags"``
        excluded: flags are printed beneath the table)."""
        return tuple(c for c in self._columns if c != _FLAGS)

    @property
    def notes(self) -> tuple[str, ...]:
        """Caveats and status lines that apply to the whole table."""
        return self._notes

    @property
    def summary(self) -> dict[str, Any]:
        """Table-wide values (a fresh dict)."""
        return dict(self._summary)

    # -- rendering -------------------------------------------------------

    def _label(self, row: Mapping[str, Any]) -> str:
        return ".".join(_cell(row.get(c)) for c in self._label_columns)

    def _grid(self, width: int):
        """``(cells, column widths, numeric flags, fits)`` of the grid layout."""
        cols = list(self.columns)
        cells = [[_cell(r.get(c)) for c in cols] for r in self._rows]
        widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
        numeric = [
            all(_is_number(r.get(c)) or r.get(c) is None for r in self._rows)
            and any(_is_number(r.get(c)) for r in self._rows)
            for c in cols
        ]
        fits = 2 + sum(widths) + 2 * (len(cols) - 1) <= width
        return cells, widths, numeric, fits

    def to_text(self, *, width: int = DEFAULT_WIDTH) -> str:
        """The plain-text layout.

        A column-aligned table when it fits in ``width`` characters, and
        otherwise one block per row (the row's label, then its values
        wrapped at ``width``), so a long name or a wide row never
        produces a ragged grid.  Deterministic: the same table gives the
        same text on every machine.
        """
        cols = list(self.columns)
        lines = [self._title]
        if not self._rows:
            lines.append("  (no rows)")
        else:
            cells, widths, numeric, fits = self._grid(width)
            if fits:
                def fmt(values: Sequence[str]) -> str:
                    parts = [v.rjust(w) if num else v.ljust(w)
                             for v, w, num in zip(values, widths, numeric)]
                    return ("  " + "  ".join(parts)).rstrip()
                lines.append(fmt(cols))
                lines.append(fmt(["-" * w for w in widths]))
                lines.extend(fmt(row) for row in cells)
                flagged = [(self._label(r), f) for r in self._rows
                           for f in (r.get(_FLAGS) or ())]
                if flagged:
                    lines.append("")
                    lines.append("flags:")
                    for label, flag in flagged:
                        lines.extend(_wrap(f"! {label}: {flag}", width, "  ", "      "))
            else:
                rest = [c for c in cols if c not in self._label_columns]
                for r in self._rows:
                    lines.append("  " + self._label(r))
                    # Each "column value" pair is one unbreakable token
                    # (a no-break space inside it), so a line break only
                    # ever falls between pairs.
                    detail = ", ".join(f"{c}\u00a0{_cell(r.get(c))}".replace(" ", "\u00a0")
                                       for c in rest)
                    if detail:
                        lines.extend(line.replace("\u00a0", " ")
                                     for line in _wrap(detail, width, "      "))
                    for flag in r.get(_FLAGS) or ():
                        lines.extend(_wrap(f"! {flag}", width, "      ", "          "))
        if self._summary:
            lines.append("")
            for key, value in self._summary.items():
                text = _cell(value)
                if key.endswith("bytes") and isinstance(value, int):
                    text = f"{value} ({_human_bytes(value)})"
                lines.append(f"{key}: {text}")
        if self._notes:
            lines.append("")
            lines.append("notes:")
            for note in self._notes:
                lines.extend(_wrap(f"- {note}", width, "  ", "    "))
        return "\n".join(lines) + "\n"

    def print(self, file: Optional[TextIO] = None, *, width: int = DEFAULT_WIDTH,
              rich: bool = False) -> None:
        """Print the table to ``file`` (default ``sys.stdout``).

        ``rich=True`` renders it with the optional ``rich`` package and
        raises ``ImportError`` naming the extra when that is missing;
        the default is the plain text of :meth:`to_text`.
        """
        out = sys.stdout if file is None else file
        if not rich:
            out.write(self.to_text(width=width))
            return
        console, rich_table, text_cls = _rich_parts()
        con = console(file=out, width=width, soft_wrap=False)
        cols = list(self.columns)
        if not self._rows:
            con.print(text_cls(self._title, style="bold"))
            con.print(text_cls("  (no rows)"))
        elif self._grid(width)[3]:
            # The grid fits: one table, flags listed beneath it.
            cells, _widths, numeric, _fits = self._grid(width)
            table = rich_table(title=self._title, title_justify="left")
            for c, num in zip(cols, numeric):
                table.add_column(c, justify="right" if num else "left")
            for row in cells:
                table.add_row(*row)
            con.print(table)
            for r in self._rows:
                for flag in r.get(_FLAGS) or ():
                    con.print(text_cls(f"! {self._label(r)}: {flag}", style="bold red"))
        else:
            # Too wide for a readable grid: one two-column table per row.
            con.print(text_cls(self._title, style="bold"))
            rest = [c for c in cols if c not in self._label_columns]
            for r in self._rows:
                table = rich_table(title=self._label(r), title_justify="left",
                                   show_header=False)
                table.add_column("column")
                table.add_column("value")
                for c in rest:
                    table.add_row(c, _cell(r.get(c)))
                con.print(table)
                for flag in r.get(_FLAGS) or ():
                    con.print(text_cls(f"! {flag}", style="bold red"))
        for key, value in self._summary.items():
            con.print(text_cls(f"{key}: {_cell(value)}"))
        for note in self._notes:
            con.print(text_cls(f"- {note}"))

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return f"InspectionTable({self._title!r}, {len(self._rows)} rows)"


def _rich_parts():
    try:
        from rich.console import Console  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(_RICH_HINT) from exc
    return Console, Table, Text


# ----------------------------------------------------------------------
# Reading the graph without touching it
# ----------------------------------------------------------------------

def _is_tracer(x: Any) -> bool:
    return isinstance(x, jax.core.Tracer)


def _holds_tracers(tree: Any) -> bool:
    return any(_is_tracer(leaf) for leaf in jax.tree_util.tree_leaves(tree))


@dataclass(frozen=True)
class _Status:
    """What can be read off the graph as it stands."""

    ever_compiled: bool
    stale: bool          # compiled once, modified since
    traced: bool         # the state holds JAX tracers

    @property
    def compiled(self) -> bool:
        return self.ever_compiled and not self.stale

    def label(self) -> str:
        if not self.ever_compiled:
            text = "not compiled"
        elif self.stale:
            text = "modified since the last compile (the next step recompiles)"
        else:
            text = "compiled"
        if self.traced:
            text += "; state holds JAX tracers"
        return text


def _status(gm: "GraphManager") -> _Status:
    ever = bool(getattr(gm, "_compile_generation", 0)) and gm._compiled_step is not None
    return _Status(
        ever_compiled=ever,
        stale=ever and bool(gm._dirty),
        traced=bool(gm._state_traced) or _holds_tracers(gm._state),
    )


_TRACED_NOTE = (
    "the graph's state holds JAX tracers left behind by a transform (jax.grad of "
    "a loss that calls run_scan, say), and a tracer has no value to read. Shapes "
    "and dtypes are the tracers' own. This method changes nothing: call a stepping "
    "method (or get_node_state, which puts the graph back to the state it had "
    "before the transform) and inspect again."
)
_UNCOMPILED_STATE_NOTE = (
    "graph not compiled: this is the state add_node() initialised from each "
    "node's initial_state(); compile() makes its leaves strongly typed and "
    "builds _meta"
)


def _field_name(path: Sequence[Any]) -> str:
    if not path:
        return "<leaf>"
    head, rest = path[0], path[1:]
    name = str(head.key) if isinstance(head, jax.tree_util.DictKey) else jax.tree_util.keystr((head,))
    return name + (jax.tree_util.keystr(tuple(rest)) if rest else "")


def _fields(node_state: Any) -> list[tuple[str, Any]]:
    """``[(field name, leaf)]`` of one node's state, sorted by name."""
    flat, _ = jax.tree_util.tree_flatten_with_path(node_state)
    return sorted(((_field_name(p), leaf) for p, leaf in flat), key=lambda t: t[0])


@dataclass(frozen=True)
class _Leaf:
    shape: tuple[int, ...]
    dtype: str
    itemsize: Optional[int]
    concrete: bool
    devices: int = 1
    shard_shape: Optional[tuple[int, ...]] = None
    axis_names: Optional[tuple[Optional[str], ...]] = None   # per array axis

    @property
    def size(self) -> int:
        return int(math.prod(self.shape))

    @property
    def nbytes(self) -> Optional[int]:
        return None if self.itemsize is None else self.size * self.itemsize

    @property
    def per_device_nbytes(self) -> Optional[int]:
        if self.itemsize is None:
            return None
        shape = self.shard_shape if self.shard_shape is not None else self.shape
        return int(math.prod(shape)) * self.itemsize

    @property
    def sharded(self) -> bool:
        return self.devices > 1 and self.shard_shape is not None and self.shard_shape != self.shape

    def sharding_text(self) -> str:
        if self.devices <= 1:
            return "-"
        if not self.sharded:
            return f"replicated x{self.devices}"
        if self.axis_names is not None:
            named = [f"{name}@{i}" for i, name in enumerate(self.axis_names) if name]
            if named:
                return ",".join(named)
        return f"sharded x{self.devices}"

    def type_text(self) -> str:
        dims = []
        for i, d in enumerate(self.shape):
            name = self.axis_names[i] if self.axis_names and i < len(self.axis_names) else None
            dims.append(f"{d}@{name}" if (name and self.sharded) else str(d))
        text = f"{self.dtype}[{','.join(dims)}]"
        if self.devices > 1 and not self.sharded:
            text += f" replicated x{self.devices}"
        return text


def _itemsize(dtype: Any) -> Optional[int]:
    try:
        return int(dtype.itemsize)
    except (AttributeError, TypeError):
        try:
            return int(np.dtype(dtype).itemsize)
        except TypeError:
            return None


def _axis_names(spec: Any, ndim: int) -> tuple[Optional[str], ...]:
    out: list[Optional[str]] = []
    for i in range(ndim):
        entry = spec[i] if i < len(spec) else None
        if entry is None:
            out.append(None)
        elif isinstance(entry, (tuple, list)):
            out.append("+".join(str(e) for e in entry) or None)
        else:
            out.append(str(entry))
    return tuple(out)


def _leaf(leaf: Any) -> _Leaf:
    """Shape, dtype and placement of one leaf, read without computing on it."""
    if _is_tracer(leaf):
        return _Leaf(tuple(int(d) for d in leaf.shape), str(leaf.dtype),
                     _itemsize(leaf.dtype), concrete=False)
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        shape = tuple(int(d) for d in leaf.shape)
        devices, shard_shape, names = 1, None, None
        try:
            sharding = getattr(leaf, "sharding", None)
        except Exception:   # noqa: BLE001 - an exotic leaf; placement unknown
            sharding = None
        if sharding is not None:
            try:
                devices = len(sharding.device_set)
                if devices > 1:
                    shard_shape = tuple(int(d) for d in sharding.shard_shape(shape))
                    spec = getattr(sharding, "spec", None)
                    if spec is not None:
                        names = _axis_names(spec, len(shape))
            except Exception:   # noqa: BLE001 - not a jax.Array sharding
                devices, shard_shape, names = 1, None, None
        return _Leaf(shape, str(leaf.dtype), _itemsize(leaf.dtype), concrete=True,
                     devices=devices, shard_shape=shard_shape, axis_names=names)
    # A Python scalar (an uncompiled graph's initial_state may hold one);
    # compile() turns it into an array of JAX's canonical dtype.
    arr = np.asarray(leaf)
    dtype = jax.dtypes.canonicalize_dtype(arr.dtype)
    return _Leaf(tuple(arr.shape), f"{np.dtype(dtype).name} (Python {type(leaf).__name__})",
                 int(np.dtype(dtype).itemsize), concrete=True)


def _stats(leaf: Any) -> Optional[dict[str, Any]]:
    """``min`` / ``max`` / ``mean`` over the finite entries and the NaN /
    inf counts, on the host.  ``None`` for a leaf with no values to read
    (a tracer, a PRNG key)."""
    if _is_tracer(leaf):
        return None
    dtype = getattr(leaf, "dtype", None)
    if dtype is not None and jnp.issubdtype(dtype, jax.dtypes.extended):
        return None
    arr = np.asarray(leaf)
    if dtype is not None and jnp.issubdtype(dtype, jnp.floating) and arr.dtype.kind != "f":
        arr = arr.astype(np.float32)      # bfloat16 / float8: widen to read
    kind = arr.dtype.kind
    if kind == "c":
        return {"min": None, "max": None, "mean": None,
                "nan": int(np.isnan(arr).sum()), "inf": int(np.isinf(arr).sum())}
    if kind == "b":
        arr = arr.astype(np.int64)
        kind = "i"
    if kind in "iu":
        if arr.size == 0:
            return {"min": None, "max": None, "mean": None, "nan": 0, "inf": 0}
        return {"min": int(arr.min()), "max": int(arr.max()),
                "mean": float(np.mean(arr, dtype=np.float64)), "nan": 0, "inf": 0}
    if kind == "f":
        nan = int(np.isnan(arr).sum())
        inf = int(np.isinf(arr).sum())
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {"min": None, "max": None, "mean": None, "nan": nan, "inf": inf}
        return {"min": float(finite.min()), "max": float(finite.max()),
                "mean": float(np.mean(finite, dtype=np.float64)), "nan": nan, "inf": inf}
    return None


def _wrapper_chain(node: Any) -> list[Any]:
    """The node, then every node it wraps (``physics_node`` or ``_inner``)."""
    chain = [node]
    seen = {id(node)}
    for _ in range(64):
        nxt = getattr(node, "physics_node", None)
        if nxt is None:
            nxt = getattr(node, "_inner", None)
        if nxt is None or id(nxt) in seen:
            break
        chain.append(nxt)
        seen.add(id(nxt))
        node = nxt
    return chain


def _type_text(node: Any) -> str:
    chain = _wrapper_chain(node)
    names = [type(n).__name__ for n in chain]
    text = names[-1]
    for outer in reversed(names[:-1]):
        text = f"{outer}({text})"
    return text


def _mesh_text(mesh: Any) -> str:
    try:
        shape = dict(mesh.shape)
        return "{" + ", ".join(f"{k}: {int(v)}" for k, v in shape.items()) + "}"
    except Exception:   # noqa: BLE001
        return type(mesh).__name__


def _sharding_text(node: Any) -> Optional[str]:
    """How the node is sharded: wrapper type, mesh axes, what it splits."""
    for n in _wrapper_chain(node):
        mesh = getattr(n, "_mesh", None)
        if mesh is None:
            continue
        parts = [f"{type(n).__name__} over mesh {_mesh_text(mesh)}"]
        axis_map = getattr(n, "_axis_map", None)
        shard_axes = getattr(n, "_shard_axes", None)
        if isinstance(axis_map, dict) and axis_map:
            parts.append("axis map " + ", ".join(
                f"{k} -> state axis {v}" for k, v in sorted(axis_map.items(), key=lambda kv: str(kv[0]))))
        elif shard_axes is not None:
            parts.append(f"state axis {_cell(tuple(shard_axes))}")
        elif getattr(n, "_layout", None) is not None:
            parts.append("unstructured partition layout")
        return "; ".join(parts)
    return None


def _flux_node(node: Any) -> bool:
    from maddening.core.node import SimulationNode  # noqa: PLC0415
    method = getattr(type(node), "compute_boundary_fluxes", None)
    return method is not None and method is not SimulationNode.compute_boundary_fluxes


def _edge_kind(gm: "GraphManager", edge: Any) -> str:
    state = gm._state.get(edge.source_node)
    if isinstance(state, dict) and edge.source_field in state:
        return "state"
    spec = gm._nodes.get(edge.source_node)
    if spec is not None and _flux_node(spec.node):
        return "flux"
    return "unknown source field"


def _callable_name(fn: Any) -> str:
    from maddening.core.transforms import get_transform_name  # noqa: PLC0415
    registered = get_transform_name(fn)
    if registered:
        return registered
    inner = getattr(fn, "func", None)       # functools.partial
    if inner is not None and callable(inner):
        return f"partial({_callable_name(inner)})"
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    return str(name) if name else type(fn).__name__


def _mapping_text(mapping: Any) -> str:
    try:
        text = f"{type(mapping).__name__} {int(mapping.n_source)}->{int(mapping.n_target)}"
    except Exception:   # noqa: BLE001
        return type(mapping).__name__
    try:
        # A sparse mapping: how many slots a row has and how many are used.
        slots, entries = getattr(mapping, "k", None), getattr(mapping, "nnz", None)
        if slots is not None and entries is not None:
            text += f" (k={int(slots)}, nnz={int(entries)})"
    except Exception:   # noqa: BLE001
        pass
    return text


def _geometry_text(edge: Any) -> str:
    """``geometry <anchor>.<field>``: the moving geometry the edge's mapping reads."""
    anchor, field = edge.geometry
    return f"geometry {anchor}.{field}"


def _group_key(group: Any) -> str:
    return "+".join(sorted(group.nodes))


def _subcycle_dividers(gm: "GraphManager", group: Any) -> dict[str, int]:
    from maddening.core.coupling._group_layout import _group_dividers  # noqa: PLC0415
    if not all(n in gm._nodes for n in group.nodes):
        return {}
    return dict(_group_dividers(group, gm._nodes) or {})


def _norm_text(group: Any) -> str:
    if group.convergence_norm == "l2":
        return f"norm l2 (tolerance {_cell(float(group.tolerance))})"
    return (f"norm {group.convergence_norm} (atol {_cell(float(group.atol))}, "
            f"rtol {_cell(float(group.rtol))})")


def _group_details(gm: "GraphManager", group: Any) -> list[str]:
    from maddening.core.coupling.group import _FIELD_DEFAULTS  # noqa: PLC0415
    lines = [
        f"members: {', '.join(sorted(group.nodes))}",
        f"solver {group.solver}, acceleration {group.acceleration}, "
        f"mode {group.iteration_mode}",
        f"{_norm_text(group)}, max_iterations {int(group.max_iterations)}",
        f"diagnostics {'on' if group.diagnostics else 'off'}, "
        f"strict {'on' if group.strict_convergence else 'off'}",
    ]
    shown = {"nodes", "solver", "acceleration", "iteration_mode", "convergence_norm",
             "tolerance", "atol", "rtol", "max_iterations", "diagnostics",
             "strict_convergence"}
    extra = []
    for f in dc_fields(group):
        if f.name in shown:
            continue
        value = getattr(group, f.name)
        if value != _FIELD_DEFAULTS.get(f.name, value):
            extra.append(f"{f.name}={value!r}" if not isinstance(value, dict)
                         else f"{f.name}={dict(sorted(value.items()))!r}")
    if extra:
        lines.append("other settings: " + ", ".join(sorted(extra)))
    dividers = _subcycle_dividers(gm, group)
    if dividers:
        lines.append("sub-cycling: " + ", ".join(
            f"{n} x{d}" for n, d in sorted(dividers.items())) + " evaluations per pass")
    return lines


# ----------------------------------------------------------------------
# print_graph / format_graph
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class _Section:
    title: str
    items: tuple[tuple[str, tuple[str, ...]], ...]
    empty: str = "(none)"


def _graph_sections(gm: "GraphManager") -> tuple[list[str], list[_Section]]:
    status = _status(gm)
    nodes = gm._nodes
    groups = list(gm._coupling_groups)
    header_parts = [
        f"{len(nodes)} node{'s' * (len(nodes) != 1)}",
        f"{len(gm._edges)} edge{'s' * (len(gm._edges) != 1)}",
        f"{len(groups)} coupling group{'s' * (len(groups) != 1)}",
        f"{len(gm._external_inputs)} external input{'s' * (len(gm._external_inputs) != 1)}",
    ]
    header = [f"GraphManager: {', '.join(header_parts)}", f"status: {status.label()}"]
    if nodes:
        rate = "multi-rate" if status.ever_compiled and gm._is_multirate else (
            "uniform rate" if status.ever_compiled else "rate dividers: not compiled")
        # ``gm.timestep`` is the step compile() schedules (a sub-cycling
        # group at its largest member timestep), the one rule for both.
        header.append(f"base timestep: {_cell(float(gm.timestep))} ({rate})")
    mesh = getattr(gm, "_multigpu_mesh", None)
    if mesh is not None:
        device_map = getattr(gm, "_multigpu_device_map", None) or {}
        header.append(f"multi-GPU: mesh {_mesh_text(mesh)}; device map "
                      + ", ".join(f"{k}: {v}" for k, v in sorted(device_map.items())))

    dividers = dict(gm._committed_rate_dividers) if status.ever_compiled else {}
    group_of = {n: g for g in groups for n in g.nodes}
    node_items = []
    for name, spec in nodes.items():
        details = []
        timing = f"timestep {_cell(float(spec.timestep))}"
        if name in dividers:
            timing += f", rate divider {dividers[name]}"
        elif status.ever_compiled:
            timing += ", rate divider: not compiled (added since the last compile)"
        else:
            timing += ", rate divider: not compiled"
        details.append(timing)
        group = group_of.get(name)
        if group is not None:
            sub = _subcycle_dividers(gm, group).get(name)
            text = f"coupling group {_group_key(group)}"
            if sub and sub > 1:
                text += f", sub-cycled x{sub} per coupling pass"
            details.append(text)
        fields = _fields(gm._state.get(name, {}))
        details.append("state: " + (", ".join(
            f"{fname} {_leaf(leaf).type_text()}" for fname, leaf in fields) or "(no fields)"))
        shard = _sharding_text(spec.node)
        if shard:
            details.append(f"sharded: {shard}")
        node_items.append((f"{name}  {_type_text(spec.node)}", tuple(details)))

    from maddening.core.coupling._interface_plan import internal_edges  # noqa: PLC0415

    internal = {id(e) for g in groups for e in internal_edges(gm._edges, g.nodes)}
    back = {id(e) for e in gm._back_edges} if status.ever_compiled else set()
    edge_items = []
    for edge in gm._edges:
        details = [_edge_kind(gm, edge)]
        if edge.transform is not None:
            details.append(f"transform {_callable_name(edge.transform)}")
        if edge.mapping is not None:
            details.append(f"mapping {_mapping_text(edge.mapping)} "
                           f"(params['mappings'][{edge.key!r}])")
        if getattr(edge, "geometry", None) is not None:
            details.append(_geometry_text(edge))
        if edge.additive:
            details.append("additive")
        if edge.source_units or edge.target_units:
            details.append(f"units {edge.source_units or '?'} -> {edge.target_units or '?'}")
        if id(edge) in internal:
            details.append("iterated inside its coupling group")
        elif id(edge) in back:
            details.append("back edge: reads the previous step's value")
        edge_items.append((f"{edge.source_node}.{edge.source_field} -> "
                           f"{edge.target_node}.{edge.target_field}", ("; ".join(details),)))

    group_items = [(_group_key(g), tuple(_group_details(gm, g)))
                   for g in sorted(groups, key=_group_key)]
    ext_items = [
        (f"{ei.target_node}.{ei.target_field}",
         (f"{jnp.dtype(ei.dtype).name}[{','.join(str(int(d)) for d in ei.shape)}]",))
        for ei in sorted(gm._external_inputs, key=lambda e: (e.target_node, e.target_field))
    ]

    order_items: list[tuple[str, tuple[str, ...]]] = []
    if status.ever_compiled:
        handled: set[str] = set()
        step = 0
        for name in gm._schedule:
            group = group_of.get(name)
            if group is not None:
                key = _group_key(group)
                if key in handled:
                    continue
                handled.add(key)
                members = [n for n in gm._schedule if n in group.nodes]
                step += 1
                fire = _firing(dividers.get(members[0], 1))
                order_items.append((f"{step}. [{key}] coupled block: {', '.join(members)}",
                                    (fire,)))
            else:
                step += 1
                order_items.append((f"{step}. {name}", (_firing(dividers.get(name, 1)),)))
    order_title = "Execution order" + (
        "" if status.compiled else
        " (not compiled)" if not status.ever_compiled else " (last compile; stale)")
    order_empty = ("unknown until compile()" if not status.ever_compiled else "(none)")

    sections = [
        _Section(f"Nodes ({len(node_items)})", tuple(node_items)),
        _Section(f"Edges ({len(edge_items)})", tuple(edge_items)),
        _Section(f"Coupling groups ({len(group_items)})", tuple(group_items)),
        _Section(f"External inputs ({len(ext_items)})", tuple(ext_items)),
        _Section(order_title, tuple(order_items), empty=order_empty),
    ]
    return header, sections


def _firing(divider: int) -> str:
    return "every base step" if divider <= 1 else f"every {divider} base steps"


@stability(StabilityLevel.EXPERIMENTAL)
def format_graph(gm: "GraphManager", *, width: int = DEFAULT_WIDTH) -> str:
    """The text :meth:`GraphManager.print_graph` prints.  See there."""
    header, sections = _graph_sections(gm)
    lines = [wrapped for line in header for wrapped in _wrap(line, width, "", "  ")]
    for section in sections:
        lines.append("")
        lines.append(section.title)
        if not section.items:
            lines.append(f"  {section.empty}")
        for label, details in section.items:
            lines.extend(_wrap(label, width, "  ", "    "))
            for detail in details:
                lines.extend(_wrap(detail, width, "      ", "        "))
    return "\n".join(lines) + "\n"


@stability(StabilityLevel.EXPERIMENTAL)
def print_graph(gm: "GraphManager", *, file: Optional[TextIO] = None,
                width: int = DEFAULT_WIDTH, rich: bool = False) -> None:
    """Print :func:`format_graph` (plain) or its ``rich`` rendering."""
    out = sys.stdout if file is None else file
    if not rich:
        out.write(format_graph(gm, width=width))
        return
    console, _table, text_cls = _rich_parts()
    from rich.tree import Tree  # noqa: PLC0415
    header, sections = _graph_sections(gm)
    con = console(file=out, width=width, soft_wrap=False)
    for line in header:
        con.print(text_cls(line))
    for section in sections:
        tree = Tree(text_cls(section.title, style="bold"))
        if not section.items:
            tree.add(text_cls(section.empty))
        for label, details in section.items:
            branch = tree.add(text_cls(label))
            for detail in details:
                branch.add(text_cls(detail))
        con.print(tree)


# ----------------------------------------------------------------------
# Mermaid / DOT
# ----------------------------------------------------------------------

_DIRECTIONS = ("LR", "RL", "TB", "BT")


def _check_direction(direction: str) -> str:
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    return direction


def _mermaid_text(text: str) -> str:
    """Make arbitrary text safe inside a quoted Mermaid label.

    Mermaid reads ``#name;`` / ``#123;`` as an entity code, so ``#`` goes
    first; quotes, angle brackets, ampersands and backticks would end the
    label, open HTML or start a Markdown string.
    """
    out = text.replace("#", "#35;")
    for ch, code in (('"', "#quot;"), ("<", "#lt;"), (">", "#gt;"),
                     ("&", "#38;"), ("`", "#96;")):
        out = out.replace(ch, code)
    return out.replace("\r\n", "<br/>").replace("\n", "<br/>").replace("\r", "<br/>")


def _dot_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\r\n", "\\n") \
        .replace("\n", "\\n").replace("\r", "\\n")


def _edge_label(edge: Any) -> str:
    label = f"{edge.source_field}→{edge.target_field}"
    extra = []
    if edge.transform is not None:
        extra.append(_callable_name(edge.transform))
    if edge.mapping is not None:
        extra.append(_mapping_text(edge.mapping))
    if getattr(edge, "geometry", None) is not None:
        extra.append(_geometry_text(edge))
    if edge.additive:
        extra.append("additive")
    return label + (f" ({', '.join(extra)})" if extra else "")


def _export_layout(gm: "GraphManager"):
    """Node ids (insertion order), group members, edges, external inputs."""
    ids = {name: f"n{i}" for i, name in enumerate(gm._nodes)}
    groups = sorted(gm._coupling_groups, key=_group_key)
    grouped = {n for g in groups for n in g.nodes if n in ids}
    return ids, groups, grouped


def _flowchart(gm: "GraphManager", direction: str, *, text: Callable[[str], str],
               line_break: str, subgraph: str) -> str:
    """The flowchart :func:`to_mermaid` writes, in a given label dialect.

    ``text`` makes a name safe inside a quoted label, ``line_break``
    separates a node's name from its type, and ``subgraph`` is a coupling
    group's opening line (``{id}`` and ``{title}`` filled in).  One
    builder serves both dialects, so the terminal diagram cannot draw a
    different graph from the Mermaid export.
    """
    ids, groups, grouped = _export_layout(gm)
    lines = [f"flowchart {_check_direction(direction)}"]

    def node_line(name: str, indent: str) -> str:
        spec = gm._nodes[name]
        label = text(name) + line_break + text(_type_text(spec.node))
        return f'{indent}{ids[name]}["{label}"]'

    for gi, group in enumerate(groups):
        title = text("coupling group " + _group_key(group))
        lines.append("    " + subgraph.format(id=f"g{gi}", title=title))
        for name in gm._nodes:
            if name in group.nodes:
                lines.append(node_line(name, "        "))
        lines.append("    end")
    for name in gm._nodes:
        if name not in grouped:
            lines.append(node_line(name, "    "))
    for edge in gm._edges:
        if edge.source_node not in ids or edge.target_node not in ids:
            continue
        arrow = "-.->" if _edge_kind(gm, edge) == "flux" else "-->"
        lines.append(f'    {ids[edge.source_node]} {arrow}|"{text(_edge_label(edge))}"| '
                     f"{ids[edge.target_node]}")
    externals = sorted(gm._external_inputs, key=lambda e: (e.target_node, e.target_field))
    for i, ei in enumerate(externals):
        if ei.target_node not in ids:
            continue
        lines.append(f'    x{i}[/"{text("external: " + ei.target_field)}"/]')
        lines.append(f"    x{i} -.-> {ids[ei.target_node]}")
    return "\n".join(lines) + "\n"


@stability(StabilityLevel.EXPERIMENTAL)
def to_mermaid(gm: "GraphManager", *, direction: str = "LR") -> str:
    """The graph as a Mermaid flowchart.  See :meth:`GraphManager.to_mermaid`."""
    return _flowchart(gm, direction, text=_mermaid_text, line_break="<br/>",
                      subgraph='subgraph {id}["{title}"]')


# ----------------------------------------------------------------------
# Terminal diagram (termaid)
# ----------------------------------------------------------------------

_TERMAID_HINT = ("print_graph_diagram needs the optional 'termaid' package: "
                 'pip install "maddening[terminal]"')
_DIAGRAM_RICH_HINT = ("a diagram theme is a colouring, which needs the optional 'rich' "
                      'package: pip install "maddening[terminal]"; theme=None draws '
                      "the diagram uncoloured without it")

#: The colour themes termaid 0.9 ships (``termaid.renderer.themes.THEMES``;
#: ``tests/core/test_inspection_diagram.py`` holds this tuple to it).
#: termaid itself falls back to ``default`` for a name it does not know,
#: so the name is checked here rather than silently ignored.
_DIAGRAM_THEMES: tuple[str, ...] = (
    "default", "terra", "neon", "mono", "amber", "phosphor",
    "gruvbox", "monokai", "dracula", "nord", "solarized",
)

# termaid parses Mermaid but decodes no entity codes (``#quot;`` would
# print as typed), so nothing is escaped.  A few sequences act even
# inside a quoted label; each becomes a look-alike instead.
_TERMAID_LOOKALIKES = (
    (re.compile(r'["`]'), "'"),          # a quote ends the label; "`...`" is Markdown
    (re.compile(r"%(?=%)"), "% "),       # %% starts a comment anywhere on the line
    (re.compile(r":(?=::)"), ": "),      # ::: starts a class suffix on a node
    (re.compile(r"\\(?=n)"), "\\ "),     # a literal \n breaks a node's label
    (re.compile(r"\r\n|\r|\n"), " "),    # a line break ends the statement
)


def _termaid_text(text: str) -> str:
    """Make arbitrary text safe inside a quoted label that termaid parses.

    Where :func:`_mermaid_text` escapes, this replaces each sequence
    termaid's parser acts on inside quotes with a look-alike
    (:data:`_TERMAID_LOOKALIKES`): a double quote or backtick becomes
    ``'``, ``%%`` becomes ``% %``, ``:::`` becomes ``: ::``, a literal
    backslash-n becomes ``\\ n`` and a line break a space.  The diagram
    is for reading; :func:`to_mermaid` and :func:`format_graph` carry
    names exactly.
    """
    out = text
    for pattern, replacement in _TERMAID_LOOKALIKES:
        out = pattern.sub(replacement, out)
    return out


def _termaid_mermaid(gm: "GraphManager", *, direction: str = "LR",
                     use_ascii: bool = False) -> str:
    """:func:`to_mermaid`'s flowchart in the dialect termaid draws cleanly.

    Two differences, both forced by termaid 0.9's parser: a coupling
    group opens as ``subgraph g0 ["title"]``, because termaid reads a
    title only when a space separates it from the id and otherwise prints
    ``g0["coupling group a+b"]`` verbatim; and a node's name and type are
    joined by ``" :: "``, because termaid does not understand ``<br/>``.
    ``use_ascii`` also writes the edge labels' ``→`` as ``->``, so an
    ASCII drawing does not need Unicode for its own arrows.
    """
    def text(raw: str) -> str:
        out = _termaid_text(raw)
        return out.replace("→", "->") if use_ascii else out

    return _flowchart(gm, direction, text=text, line_break=" :: ",
                      subgraph='subgraph {id} ["{title}"]')


def _termaid() -> Any:
    try:
        import termaid  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(_TERMAID_HINT) from exc
    return termaid


@stability(StabilityLevel.EXPERIMENTAL)
def print_graph_diagram(gm: "GraphManager", *, theme: Optional[str] = None,
                        direction: str = "LR", use_ascii: bool = False,
                        file: Optional[TextIO] = None) -> None:
    """Draw the graph as boxes and arrows.  See :meth:`GraphManager.print_graph_diagram`."""
    _check_direction(direction)
    if theme is not None and theme not in _DIAGRAM_THEMES:
        raise ValueError(f"theme must be None or one of {_DIAGRAM_THEMES}, got {theme!r}")
    termaid = _termaid()
    out = sys.stdout if file is None else file
    if not gm._nodes:
        out.write("(empty graph: no nodes to draw)\n")
        return
    source = _termaid_mermaid(gm, direction=direction, use_ascii=use_ascii)
    try:
        from rich.console import Console  # noqa: PLC0415
    except ImportError:
        if theme is not None:
            raise ImportError(_DIAGRAM_RICH_HINT) from None
        out.write(termaid.render(source, use_ascii=use_ascii) + "\n")
        return
    drawing = termaid.render_rich(source, use_ascii=use_ascii, theme=theme or "default")
    # soft_wrap: rich must neither wrap nor crop a line wider than its
    # console, or the boxes come apart.  Colour reaches only a terminal;
    # rich writes plain text to any other file.
    Console(file=out).print(drawing, soft_wrap=True)


@stability(StabilityLevel.EXPERIMENTAL)
def to_dot(gm: "GraphManager", *, rankdir: str = "LR") -> str:
    """The graph in Graphviz DOT.  See :meth:`GraphManager.to_dot`."""
    ids, groups, grouped = _export_layout(gm)
    lines = ["digraph maddening {", f"    rankdir={_check_direction(rankdir)};",
             "    node [shape=box];"]

    def node_line(name: str, indent: str) -> str:
        spec = gm._nodes[name]
        label = _dot_text(name) + "\\n" + _dot_text(_type_text(spec.node))
        return f'{indent}{ids[name]} [label="{label}"];'

    for gi, group in enumerate(groups):
        lines.append(f"    subgraph cluster_g{gi} {{")
        lines.append(f'        label="{_dot_text("coupling group " + _group_key(group))}";')
        lines.append("        style=rounded;")
        for name in gm._nodes:
            if name in group.nodes:
                lines.append(node_line(name, "        "))
        lines.append("    }")
    for name in gm._nodes:
        if name not in grouped:
            lines.append(node_line(name, "    "))
    for edge in gm._edges:
        if edge.source_node not in ids or edge.target_node not in ids:
            continue
        style = ", style=dashed" if _edge_kind(gm, edge) == "flux" else ""
        lines.append(f'    {ids[edge.source_node]} -> {ids[edge.target_node]} '
                     f'[label="{_dot_text(_edge_label(edge))}"{style}];')
    externals = sorted(gm._external_inputs, key=lambda e: (e.target_node, e.target_field))
    for i, ei in enumerate(externals):
        if ei.target_node not in ids:
            continue
        lines.append(f'    x{i} [label="{_dot_text("external: " + ei.target_field)}", '
                     f"shape=parallelogram];")
        lines.append(f"    x{i} -> {ids[ei.target_node]} [style=dashed];")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# state_summary
# ----------------------------------------------------------------------

_STATE_COLUMNS = ("node", "field", "shape", "dtype", "min", "max", "mean",
                  "nan", "inf", "bytes", _FLAGS)


@stability(StabilityLevel.EXPERIMENTAL)
def state_summary(gm: "GraphManager", *, include_meta: bool = False) -> InspectionTable:
    """Per-field statistics of the graph's state.  See
    :meth:`GraphManager.state_summary`."""
    status = _status(gm)
    rows = []
    names = [n for n in sorted(gm._state) if n != _META_KEY]
    if include_meta and _META_KEY in gm._state:
        names.append(_META_KEY)
    for name in names:
        for fname, leaf in _fields(gm._state[name]):
            info = _leaf(leaf)
            stats = _stats(leaf)
            flags = []
            if stats and (stats["nan"] or stats["inf"]):
                flags.append(f"non-finite values: {stats['nan']} NaN, {stats['inf']} inf")
            row = {"node": name, "field": fname, "shape": info.shape, "dtype": info.dtype,
                   "min": None, "max": None, "mean": None, "nan": None, "inf": None,
                   "bytes": info.nbytes, _FLAGS: tuple(flags)}
            if stats is not None:
                row.update(stats)
            rows.append(row)
    notes = ["min, max and mean are taken over the finite entries; nan and inf count the rest"]
    if status.traced:
        notes.append(_TRACED_NOTE)
    if not status.ever_compiled:
        notes.append(_UNCOMPILED_STATE_NOTE)
    elif status.stale:
        notes.append("graph modified since the last compile; this is the state it holds now")
    if include_meta and _META_KEY not in gm._state:
        notes.append("no _meta: the graph has none until compile() builds it, and only a "
                     "multi-rate or coupled graph needs one")
    n_nodes = len({r["node"] for r in rows})
    return InspectionTable(
        f"State summary: {len(rows)} field{'s' * (len(rows) != 1)} in {n_nodes} "
        f"node{'s' * (n_nodes != 1)}",
        _STATE_COLUMNS, rows, label_columns=("node", "field"), notes=notes)


# ----------------------------------------------------------------------
# params_table
# ----------------------------------------------------------------------

def _effective_params(gm: "GraphManager", status: _Status) -> tuple[dict, Optional[str]]:
    """``gm.params`` as it stands, or -- never compiled -- the snapshot
    ``compile()`` would take, built fresh and not stored."""
    if status.ever_compiled:
        return gm.params, None
    # Copied down to the leaf dicts: a node's ``params_pytree()`` may hand
    # back a dict it keeps, and the overlay below must not write into it.
    fresh = {section: {owner: dict(leaves) for owner, leaves in tree.items()}
             for section, tree in gm._snapshot_params().items()}
    for section in ("nodes", "mappings"):
        for owner, leaves in (gm.params.get(section) or {}).items():
            base = fresh.get(section, {}).get(owner)
            if not isinstance(base, dict) or not isinstance(leaves, dict):
                continue
            for key, value in leaves.items():
                if key in base and np.shape(value) == np.shape(base[key]):
                    base[key] = value
    return fresh, ("graph not compiled: gm.params is filled by compile(); these are the "
                   "values it would take (each node's params_pytree(), with any live "
                   "gm.params entry that fits written over it). Nothing was stored.")


def _bound_violation(spec: Any, leaf: Any) -> Optional[str]:
    """What :meth:`ParamSpec.check` would refuse, on the host; ``None`` when
    the value passes.  The comparisons are ``check``'s own
    (:func:`~maddening.core.params._bound_operands`): the value and the
    bound rounded to the dtype JAX compares them in, then compared exactly,
    and under a ``log`` / ``logit`` transform a distance from the bound
    that the step's arithmetic flushes to zero counts as on it."""
    from maddening.core.params import _bound_operands, _step_gap  # noqa: PLC0415

    arr = np.asarray(leaf)
    if arr.dtype.kind not in "biufc" and not jnp.issubdtype(arr.dtype, jnp.floating):
        return None
    if arr.dtype.kind == "c":
        if not np.all(np.isfinite(arr)):
            return "not finite"
        arr = np.real(arr)
    if jnp.issubdtype(arr.dtype, jnp.floating) and not np.all(np.isfinite(arr.astype(np.float64))):
        return "not finite"
    lo, hi = spec.bounds
    strict = spec.transform in ("log", "logit")
    if lo is None and spec.transform == "log":
        a, b, tiny = _bound_operands(arr, 0.0)
        if np.any(_step_gap(a, b, tiny) <= 0.0):
            return "at or below 0 (transform='log' without a lower bound is measured from 0)"
    if lo is not None:
        a, b, tiny = _bound_operands(arr, lo)
        if np.any(_step_gap(a, b, tiny) <= 0.0) if strict else np.any(a < b):
            return f"below the lower bound {_cell(float(lo))}"
    if hi is not None:
        a, b, tiny = _bound_operands(arr, hi)
        if np.any(_step_gap(b, a, tiny) <= 0.0) if strict else np.any(a > b):
            return f"above the upper bound {_cell(float(hi))}"
    return None


_PARAMS_COLUMNS = ("owner", "param", "value", "shape", "dtype", "trainable", "bounds",
                   "transform", "out_of_bounds", _FLAGS)


@stability(StabilityLevel.EXPERIMENTAL)
def params_table(gm: "GraphManager") -> InspectionTable:
    """One row per leaf of the graph's parameters.  See
    :meth:`GraphManager.params_table`."""
    from maddening.core.params import _resolve_specs, _spec_for  # noqa: PLC0415
    status = _status(gm)
    params, note = _effective_params(gm, status)
    specs = gm.param_specs()
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    notes = []
    if note:
        notes.append(note)
    elif status.stale:
        notes.append("graph modified since the last compile: these are the live gm.params; "
                     "the next compile() refreshes them from the nodes and keeps the live "
                     "values that still fit")
    try:
        per_leaf = _resolve_specs(params, specs)
    except ValueError as exc:
        per_leaf = [_spec_for(specs, path) for path, _ in flat]
        notes.append(f"the ParamSpec tree has an entry the resolver refuses, so "
                     f"check_params() raises ({exc}); specs shown by the lenient reading")
    rows = []
    for (path, leaf), spec in zip(flat, per_leaf):
        section = str(path[0].key) if path and isinstance(path[0], jax.tree_util.DictKey) else "?"
        owner = (str(path[1].key) if len(path) > 1 and isinstance(path[1], jax.tree_util.DictKey)
                 else "?")
        param = _field_name(path[2:]) if len(path) > 2 else "<leaf>"
        info = _leaf(leaf)
        flags = []
        value: Any = None
        out_of_bounds: Optional[bool] = None
        if _is_tracer(leaf):
            flags.append("traced: a tracer has no value to read")
        else:
            if info.size == 1 and getattr(leaf, "dtype", None) is not None and not \
                    jnp.issubdtype(leaf.dtype, jax.dtypes.extended):
                scalar = np.asarray(leaf).reshape(())
                value = bool(scalar) if scalar.dtype.kind == "b" else (
                    int(scalar) if scalar.dtype.kind in "iu" else float(scalar))
            elif info.size == 1 and not hasattr(leaf, "dtype"):
                value = leaf
            violation = _bound_violation(spec, leaf)
            out_of_bounds = violation is not None
            if violation:
                flags.append(f"out of bounds: {violation}")
        rows.append({
            "section": section, "owner": owner, "param": param, "value": value,
            "shape": info.shape, "dtype": info.dtype, "trainable": bool(spec.trainable),
            "bounds": (spec.bounds[0], spec.bounds[1]), "transform": spec.transform,
            "units": spec.units or None, "out_of_bounds": out_of_bounds, _FLAGS: tuple(flags),
        })
    rows.sort(key=lambda r: (r["section"] != "nodes", r["section"], r["owner"], r["param"]))
    baked = sorted(gm.nodes_without_params())
    if baked:
        notes.append("nodes whose update() takes no params keyword (constants baked into "
                     "the step, not in this table): " + ", ".join(baked))
    notes.append("value is the scalar for a one-element leaf; an array leaf shows its shape. "
                 "out_of_bounds applies ParamSpec.check()'s rule (non-finite values count)")
    return InspectionTable(
        f"Parameters: {len(rows)} lea{'f' if len(rows) == 1 else 'ves'}",
        _PARAMS_COLUMNS, rows, label_columns=("owner", "param"), notes=notes)


# ----------------------------------------------------------------------
# coupling_report
# ----------------------------------------------------------------------

_COUPLING_COLUMNS = ("group", "iterations", "total_iterations", "max_iterations", "converged",
                     "residual", "error_estimate", "amplification", "ratio_usable",
                     "precision_limited", "rho_spectral", "spectral_error_bound",
                     "spectral_usable", _FLAGS)
_REPORT_KEYS = ("iterations", "total_iterations", "converged", "residual", "error_estimate",
                "amplification", "ratio_usable", "precision_limited", "rho_spectral",
                "spectral_error_bound", "spectral_usable")


def _coupling_flags(group: Any, d: Mapping[str, Any], whole: tuple = ()) -> list[str]:
    flags = []
    cap = int(group.max_iterations)
    if int(d["iterations"]) >= cap:
        flags.append(f"hit max_iterations ({cap}): the solve stopped on its budget")
    if not d["converged"]:
        flags.append("converged=False: the returned state is still outside the threshold"
                     + ("; under solver='ift' the gradient through this step is unreliable"
                        if group.solver == "ift" else ""))
    reason = d.get("not_usable_reason")
    estimate = d.get("error_estimate")
    bound = d.get("spectral_error_bound")
    # Every number is there and only the spectral flags are withdrawn: a
    # group with a geometry-dependent mapping whose bound reaches a
    # lattice plane (experimental), or one at its float floor behind a
    # long row of a static mapping (MADD-ANO-251).  The caveats below
    # still apply.
    flags_only = bool(reason) and all(
        isinstance(v, float) and not math.isnan(v) for v in (estimate, bound))
    if flags_only:
        pass
    elif reason and isinstance(estimate, float) and not math.isnan(estimate):
        # Only what rests on the float floor is withheld (a checkpoint
        # saved after the state was written): the estimates are there,
        # and their caveats below still apply.
        flags.append(f"no bound reported: {reason}")
        if not d["ratio_usable"]:
            flags.append("ratio_usable=False: the contraction ratio was unusable, so the "
                         "criterion fell back to the raw residual test; converged reports that "
                         "test, not a distance estimate (MADD-ANO-005)")
        return flags
    elif reason:
        # The report withholds every bound, estimate and ``*_usable`` flag
        # of this group, so the caveats below (which read them) would be
        # statements about values that are not there.
        flags.append(f"no bound or estimate reported: {reason}")
        return flags
    if not d["ratio_usable"]:
        flags.append("ratio_usable=False: the contraction ratio was unusable, so the "
                     "criterion fell back to the raw residual test; converged reports that "
                     "test, not a distance estimate (MADD-ANO-005)")
    if d["precision_limited"]:
        flags.append("precision_limited=True: the residual is at its float floor, so residual "
                     "and error_estimate are rounding and converged can be True on a stalled "
                     "iterate; read spectral_error_bound (solver='ift', diagnostics=True)")
    rho = d.get("rho_spectral")
    if flags_only:
        flags.append(f"spectral_usable=False: {reason}")
    elif isinstance(rho, float) and not math.isnan(rho) and not d.get("spectral_usable"):
        flags.append("spectral_usable=False: the spectral bound is not settled or not finite")
    if whole and d.get("gradient_bound_usable"):
        named = ", ".join(f"{name} ({size} entries)" for name, size in whole)
        flags.append("gradient bound: constants larger than the entry-probe limit were probed "
                     "as a whole, along one fixed direction weighted by their magnitudes, so for "
                     f"them it bounds that directional derivative, not each entry's: {named}")
    return flags


@stability(StabilityLevel.EXPERIMENTAL)
def coupling_report(gm: "GraphManager") -> InspectionTable:
    """One row per coupling group from :meth:`GraphManager.coupling_diagnostics`.
    See :meth:`GraphManager.coupling_report`."""
    status = _status(gm)
    groups = sorted(gm._coupling_groups, key=_group_key)
    notes: list[str] = []
    diags: dict[str, Any] = {}
    if not groups:
        notes.append("the graph has no coupling groups, so there is nothing to report")
    elif status.traced:
        notes.append(_TRACED_NOTE.replace(
            "Shapes and dtypes are the tracers' own. ", "")
            + " (coupling_diagnostics() itself would put the graph back first, which is "
              "a write, so this report does not call it.)")
    elif not status.ever_compiled:
        notes.append("graph not compiled: no step has run, so there are no diagnostics yet")
    else:
        # Read-only on a graph that holds no tracers: its one write,
        # ``_recover_from_escaped_tracers``, returns at once (checked above).
        diags = dict(gm.coupling_diagnostics())
    committed = getattr(gm, "_committed_coupling_groups", {}) or {}
    rows = []
    for group in groups:
        key = _group_key(group)
        d = diags.get(key)
        # A report was judged under the group the compiled step ran, which
        # differs from the registered one if it was replaced since.
        ran = committed.get(key, group) if d is not None else group
        row: dict[str, Any] = {"group": key, "solver": ran.solver,
                               "max_iterations": int(ran.max_iterations)}
        row.update({k: None for k in _REPORT_KEYS})
        if d is not None:
            for k in _REPORT_KEYS:
                row[k] = d.get(k)
            row[_FLAGS] = tuple(_coupling_flags(
                ran, d, (getattr(gm, "_gradient_whole_probes", {}) or {}).get(key, ())))
        elif status.traced or not status.ever_compiled:
            row[_FLAGS] = ()
        elif group.solver == "fori" and not group.diagnostics:
            row[_FLAGS] = ("no report: solver='fori' records diagnostics only with "
                           "diagnostics=True",)
        elif key not in committed:
            row[_FLAGS] = ("no report: the group was added after the last compile",)
        elif any(nn not in gm._state for nn in group.nodes):
            row[_FLAGS] = ("no report: a member was removed since the last step",)
        else:
            row[_FLAGS] = ("no report yet: no step has run since compile() or reset_state()",)
        rows.append(row)
    if diags:
        notes.append("error_estimate is an estimate, not a bound: it can understate the "
                     "distance to the fixed point by large factors even with "
                     "ratio_usable=True; spectral_error_bound (solver='ift', "
                     "diagnostics=True) is the bound where spectral_usable is True, for a "
                     "linear map (asymptotic for a non-linear one), in the group's own norm "
                     "at the returned state (under 'interface', what each edge delivers: "
                     "its mapping, then its transform; or its source field where a static "
                     "mapping delivers more entries than the source holds). "
                     "See coupling_diagnostics()")
        if gm._is_multirate:
            notes.append("multi-rate graph: a group's entry is its most recent applied solve")
    if status.stale and groups:
        notes.append("graph modified since the last compile: the report is the last step's, "
                     "judged under the groups that step ran")
    return InspectionTable(
        f"Coupling report: {len(groups)} group{'s' * (len(groups) != 1)}",
        _COUPLING_COLUMNS, rows, label_columns=("group",), notes=notes)


# ----------------------------------------------------------------------
# memory_estimate
# ----------------------------------------------------------------------

_MEMORY_COLUMNS = ("node", "fields", "bytes", "per_device_bytes", "devices", "sharding")


@stability(StabilityLevel.EXPERIMENTAL)
def memory_estimate(gm: "GraphManager") -> InspectionTable:
    """State memory per node from shapes and dtypes.  See
    :meth:`GraphManager.memory_estimate`."""
    status = _status(gm)
    rows = []
    unknown = []
    names = [n for n in sorted(gm._state) if n != _META_KEY]
    if _META_KEY in gm._state:
        names.append(_META_KEY)
    for name in names:
        leaves = [_leaf(leaf) for _, leaf in _fields(gm._state[name])]
        sized = [lf for lf in leaves if lf.nbytes is not None]
        if len(sized) != len(leaves):
            unknown.append(name)
        shardings = sorted({lf.sharding_text() for lf in leaves} - {"-"})
        rows.append({
            "node": name, "fields": len(leaves),
            "bytes": sum(lf.nbytes or 0 for lf in sized),
            "per_device_bytes": sum(lf.per_device_nbytes or 0 for lf in sized),
            "devices": max((lf.devices for lf in leaves), default=1),
            "sharding": ", ".join(shardings) if shardings else None,
        })
    node_bytes = sum(r["bytes"] for r in rows if r["node"] != _META_KEY)
    meta_bytes = sum(r["bytes"] for r in rows if r["node"] == _META_KEY)
    summary = {
        "state_bytes": node_bytes,
        "meta_bytes": meta_bytes,
        "total_bytes": node_bytes + meta_bytes,
        "total_per_device_bytes": sum(r["per_device_bytes"] for r in rows),
    }
    notes = [
        "state memory only, computed from shapes and dtypes: XLA workspace, compiled "
        "programs, the copies a step makes, scan histories, params and external inputs "
        "are not included",
        "bytes is a node's global (logical) size; per_device_bytes is what one device holds "
        "of it (a sharded field's shard, a replicated or unsharded field in full); "
        "total_per_device_bytes adds those, the most one device holds if every node's "
        "share lands on it",
    ]
    if not status.ever_compiled:
        notes.append("graph not compiled: the state add_node() initialised; there is no "
                     "_meta yet, and compile() may widen Python scalars to arrays")
    if status.traced:
        notes.append("the state holds JAX tracers; their shapes and dtypes are what is "
                     "counted here")
    if unknown:
        notes.append("a field with no fixed item size (counted as 0) in: " + ", ".join(unknown))
    return InspectionTable(
        f"State memory estimate: {_human_bytes(summary['total_bytes'])} in "
        f"{len(rows)} entr{'y' if len(rows) == 1 else 'ies'}",
        _MEMORY_COLUMNS, rows, label_columns=("node",), notes=notes, summary=summary)


__all__ = [
    "DEFAULT_WIDTH",
    "InspectionTable",
    "coupling_report",
    "format_graph",
    "memory_estimate",
    "params_table",
    "print_graph",
    "state_summary",
    "to_dot",
    "to_mermaid",
]
