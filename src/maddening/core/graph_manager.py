"""
GraphManager -- central orchestrator for the MADDENING simulation graph.

Owns all node state, builds the execution schedule, and JIT-compiles the
full graph step into a single XLA computation via ``jax.jit``.

Supports multi-rate timesteps: each node declares its own ``delta_t``,
and the graph manager derives a *base timestep* (GCD of all node
timesteps).  The compiled step advances at the base rate; each node
updates only when its own sub-step counter fires.  For JAX traceability
the update is always computed but conditionally applied via
``jnp.where``.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp

from maddening.core.coupling import CouplingGroup
from maddening.core.edge import EdgeSpec
from maddening.core.metadata import StabilityLevel
from maddening.core.node import SimulationNode
from maddening.core.stability import stability
from maddening.core.schedule import (
    detect_cycles,
    find_strongly_connected_components,
    identify_back_edges,
    topological_sort,
)


# ------------------------------------------------------------------
# Internal bookkeeping structs
# ------------------------------------------------------------------

@dataclass
class _NodeSpec:
    """Everything the graph manager needs to know about a node."""
    node: SimulationNode          # the descriptor object
    update_fn: Callable           # node.update  (pure function)
    timestep: float


@dataclass(frozen=True)
class ExternalInputSpec:
    """Declares an external input that flows into a node's boundary_inputs.

    External inputs come from outside the graph (controllers, sensors,
    user commands) rather than from other nodes via edges.
    """
    target_node: str
    target_field: str
    shape: tuple
    dtype: Any = jnp.float32


# ------------------------------------------------------------------
# Observer event names
# ------------------------------------------------------------------
EVENT_NODE_ADDED = "node_added"
EVENT_NODE_REMOVED = "node_removed"
EVENT_EDGE_ADDED = "edge_added"
EVENT_EDGE_REMOVED = "edge_removed"
EVENT_COMPILED = "compiled"
EVENT_STEP = "step"


_EMPTY_EXTERNAL_INPUTS: dict[str, dict] = {}

# Key for internal multi-rate metadata in the full state dict.
_META_KEY = "_meta"


# ------------------------------------------------------------------
# Floating-point-tolerant GCD
# ------------------------------------------------------------------

def _float_gcd(a: float, b: float, tol: float = 1e-9) -> float:
    """GCD of two positive floats using Euclidean algorithm with tolerance."""
    if a < b:
        a, b = b, a
    while b > tol:
        a, b = b, a % b
    return a


def _multi_gcd(values: Sequence[float], tol: float = 1e-9) -> float:
    """GCD of multiple positive floats."""
    result = values[0]
    for v in values[1:]:
        result = _float_gcd(result, v, tol)
    return result


@stability(StabilityLevel.STABLE)
class GraphManager:
    """Build, validate, compile and run a simulation graph.

    Supports multi-rate scheduling: nodes may have different timesteps.
    The graph steps at the *base timestep* (GCD of all node timesteps).
    Each node updates only on the sub-steps that are multiples of its
    own rate divider.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, _NodeSpec] = {}
        self._edges: list[EdgeSpec] = []
        self._state: dict[str, dict] = {}
        self._schedule: list[str] = []
        self._compiled_step: Optional[Callable] = None
        self._dirty: bool = True
        self._observers: list[Callable] = []
        self._back_edges: list[EdgeSpec] = []
        self._external_inputs: list[ExternalInputSpec] = []
        self._is_multirate: bool = False
        self._rate_dividers: dict[str, int] = {}
        self._coupling_groups: list[CouplingGroup] = []

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, node: SimulationNode) -> None:
        """Register a node and initialise its state."""
        if node.name in self._nodes:
            raise ValueError(f"Node '{node.name}' already exists in the graph.")

        spec = _NodeSpec(
            node=node,
            update_fn=node.update,
            timestep=node.delta_t,
        )
        self._nodes[node.name] = spec
        self._state[node.name] = node.initial_state()
        self._dirty = True
        self._notify(EVENT_NODE_ADDED, node.name)

    def add_edge(
        self,
        source: str,
        target: str,
        source_field: str,
        target_field: str,
        transform: Optional[Callable] = None,
    ) -> None:
        """Add a data-dependency edge between two nodes."""
        edge = EdgeSpec(source, target, source_field, target_field, transform)
        self._edges.append(edge)
        self._dirty = True
        self._notify(EVENT_EDGE_ADDED, edge)

    def add_external_input(
        self,
        target_node: str,
        target_field: str,
        shape: tuple = (),
        dtype: Any = jnp.float32,
    ) -> None:
        """Declare an external input that will be injected each step.

        External inputs appear in the target node's ``boundary_inputs``
        dict alongside edge-delivered values.  They are supplied via the
        ``external_inputs`` argument to :meth:`step` or :meth:`run`.

        Parameters
        ----------
        target_node : str
            Name of the node that receives this input.
        target_field : str
            Key in the node's ``boundary_inputs`` dict.
        shape : tuple
            Array shape (default ``()`` for scalar).
        dtype
            JAX dtype (default ``jnp.float32``).
        """
        spec = ExternalInputSpec(target_node, target_field, shape, dtype)
        self._external_inputs.append(spec)
        self._dirty = True

    def remove_node(self, name: str) -> None:
        """Remove a node and all edges / external inputs that reference it."""
        if name not in self._nodes:
            raise KeyError(f"No node named '{name}'.")
        del self._nodes[name]
        del self._state[name]
        self._edges = [
            e for e in self._edges
            if e.source_node != name and e.target_node != name
        ]
        self._external_inputs = [
            e for e in self._external_inputs if e.target_node != name
        ]
        self._dirty = True
        self._notify(EVENT_NODE_REMOVED, name)

    def remove_edge(
        self,
        source: str,
        target: str,
        source_field: str,
        target_field: str,
    ) -> None:
        """Remove a specific edge."""
        edge = EdgeSpec(source, target, source_field, target_field)
        self._edges = [
            e for e in self._edges
            if not (
                e.source_node == edge.source_node
                and e.target_node == edge.target_node
                and e.source_field == edge.source_field
                and e.target_field == edge.target_field
            )
        ]
        self._dirty = True
        self._notify(EVENT_EDGE_REMOVED, edge)

    # ------------------------------------------------------------------
    # Coupling groups
    # ------------------------------------------------------------------

    def add_coupling_group(
        self,
        nodes: Sequence[str],
        max_iterations: int = 10,
        tolerance: float = 1e-6,
    ) -> CouplingGroup:
        """Register an iteratively-coupled group of nodes.

        Within each timestep, the nodes in the group are executed
        repeatedly (Gauss-Seidel) until convergence or *max_iterations*.
        All edges between nodes in the group use current-iteration
        values rather than staggered (previous-timestep) values.

        Parameters
        ----------
        nodes : sequence of str
            Node names forming the coupling group.  Must all exist in
            the graph and should form (part of) a cycle.
        max_iterations : int
            Maximum iterations per timestep.
        tolerance : float
            Convergence threshold (L2 norm of state change).

        Returns
        -------
        CouplingGroup
            The created coupling group descriptor.
        """
        for name in nodes:
            if name not in self._nodes:
                raise KeyError(f"No node named '{name}'.")
        # Check for overlap with existing coupling groups
        new_set = frozenset(nodes)
        for existing in self._coupling_groups:
            overlap = new_set & existing.nodes
            if overlap:
                raise ValueError(
                    f"Nodes {overlap} already belong to a coupling group."
                )
        group = CouplingGroup(
            nodes=new_set,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        self._coupling_groups.append(group)
        self._dirty = True
        return group

    def remove_coupling_group(self, nodes: Sequence[str]) -> None:
        """Remove a coupling group by its node set."""
        target = frozenset(nodes)
        self._coupling_groups = [
            g for g in self._coupling_groups if g.nodes != target
        ]
        self._dirty = True

    def auto_couple(
        self,
        max_iterations: int = 10,
        tolerance: float = 1e-6,
    ) -> list[CouplingGroup]:
        """Automatically create coupling groups from graph cycles.

        Uses Tarjan's algorithm to find strongly connected components
        and creates a coupling group for each SCC with more than one
        node.  Existing coupling groups are cleared first.

        Returns the list of created coupling groups.
        """
        self._coupling_groups.clear()
        sccs = find_strongly_connected_components(
            list(self._nodes.keys()), self._edges
        )
        groups = []
        for scc in sccs:
            g = self.add_coupling_group(scc, max_iterations, tolerance)
            groups.append(g)
        return groups

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Check graph integrity.  Returns a list of warning/error strings."""
        issues: list[str] = []
        node_names = set(self._nodes.keys())

        # Edge endpoint checks
        for e in self._edges:
            if e.source_node not in node_names:
                issues.append(f"ERROR: edge references non-existent source node '{e.source_node}'")
            if e.target_node not in node_names:
                issues.append(f"ERROR: edge references non-existent target node '{e.target_node}'")

            # Field existence
            if e.source_node in self._state:
                if e.source_field not in self._state[e.source_node]:
                    issues.append(
                        f"ERROR: source field '{e.source_field}' not in state of node '{e.source_node}'. "
                        f"Available: {list(self._state[e.source_node].keys())}"
                    )

        # External input endpoint checks
        for ei in self._external_inputs:
            if ei.target_node not in node_names:
                issues.append(
                    f"ERROR: external input references non-existent node '{ei.target_node}'"
                )

        # Disconnected-node warning (edges + external inputs count as connections)
        connected = set()
        for e in self._edges:
            connected.add(e.source_node)
            connected.add(e.target_node)
        for ei in self._external_inputs:
            connected.add(ei.target_node)
        for n in node_names:
            if n not in connected:
                issues.append(f"WARNING: node '{n}' is disconnected (no edges or external inputs)")

        # Multi-rate timestep informational message
        timesteps = {spec.timestep for spec in self._nodes.values()}
        if len(timesteps) > 1:
            base_dt = _multi_gcd(sorted(timesteps))
            dividers = {
                name: round(spec.timestep / base_dt)
                for name, spec in self._nodes.items()
            }
            issues.append(
                f"INFO: multi-rate scheduling enabled. "
                f"Base timestep: {base_dt}, rate dividers: {dividers}"
            )

        # Coupling group validation
        coupled_nodes: set[str] = set()
        for group in self._coupling_groups:
            for n in group.nodes:
                if n not in node_names:
                    issues.append(
                        f"ERROR: coupling group references non-existent node '{n}'"
                    )
            # Check uniform timestep within coupling group
            group_timesteps = {
                self._nodes[n].timestep
                for n in group.nodes
                if n in self._nodes
            }
            if len(group_timesteps) > 1:
                issues.append(
                    f"ERROR: coupling group {set(group.nodes)} has mixed "
                    f"timesteps {group_timesteps}. All nodes in a coupling "
                    f"group must share the same timestep."
                )
            coupled_nodes |= group.nodes

        # Cycle detection (only on edges with valid endpoints)
        valid_edges = [
            e for e in self._edges
            if e.source_node in node_names and e.target_node in node_names
        ]
        cycles = detect_cycles(list(self._nodes.keys()), valid_edges)
        for cyc in cycles:
            # Check if cycle is covered by a coupling group
            cyc_set = set(cyc)
            covered = any(cyc_set <= g.nodes for g in self._coupling_groups)
            if covered:
                issues.append(
                    f"INFO: cycle {' -> '.join(cyc)} handled by iterative "
                    f"coupling (Gauss-Seidel)."
                )
            else:
                issues.append(
                    f"WARNING: cycle detected: {' -> '.join(cyc)}. "
                    f"Back-edges will use previous-timestep values (staggering)."
                )

        return issues

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self) -> None:
        """Topologically sort the graph and JIT-compile the step function."""
        issues = self.validate()
        errors = [i for i in issues if i.startswith("ERROR")]
        if errors:
            raise RuntimeError(
                "Cannot compile graph with errors:\n" + "\n".join(errors)
            )
        for issue in issues:
            if issue.startswith("WARNING"):
                warnings.warn(issue, stacklevel=2)

        node_names = list(self._nodes.keys())
        self._schedule = topological_sort(node_names, self._edges)
        self._back_edges = identify_back_edges(self._schedule, self._edges)

        # Compute multi-rate info
        timesteps = sorted({spec.timestep for spec in self._nodes.values()})
        if len(timesteps) > 1:
            self._is_multirate = True
            base_dt = _multi_gcd(timesteps)
            self._rate_dividers = {
                name: round(spec.timestep / base_dt)
                for name, spec in self._nodes.items()
            }
            # Initialise the step counter in the state
            self._state[_META_KEY] = {
                "step_count": jnp.array(0, dtype=jnp.int32),
            }
        else:
            self._is_multirate = False
            self._rate_dividers = {name: 1 for name in self._nodes}
            # Remove meta if it existed from a previous compile
            self._state.pop(_META_KEY, None)

        step_fn = self._build_step_fn()
        self._compiled_step = jax.jit(step_fn)

        self._dirty = False
        self._notify(EVENT_COMPILED, self._schedule)

    def _build_step_fn(self) -> Callable:
        """Create a pure function ``(full_state, ext_inputs) -> full_state``.

        When multi-rate is active, the step function increments an
        internal step counter and conditionally applies each node's
        update based on whether ``step_count % rate_divider == 0``.
        The update is always *computed* (to keep the function
        JAX-traceable with static structure), but the result is applied
        only when the node should fire.

        When coupling groups are defined, nodes within each group are
        wrapped in a ``jax.lax.while_loop`` that iterates
        (Gauss-Seidel) until convergence or max_iterations.
        """
        schedule = list(self._schedule)
        nodes = dict(self._nodes)
        back_edge_set = set(self._back_edges)
        is_multirate = self._is_multirate
        rate_dividers = dict(self._rate_dividers)
        coupling_groups = list(self._coupling_groups)

        # Map node -> coupling group
        node_to_group: dict[str, CouplingGroup] = {}
        for group in coupling_groups:
            for name in group.nodes:
                node_to_group[name] = group

        # Build block schedule: list of (type, data) where type is
        # "node" (single node) or "coupled" (CouplingGroup, node_list)
        blocks: list[tuple] = []
        handled_groups: set[int] = set()
        for node_name in schedule:
            if node_name in node_to_group:
                group = node_to_group[node_name]
                gid = id(group)
                if gid not in handled_groups:
                    handled_groups.add(gid)
                    # Collect nodes in this group in schedule order
                    group_schedule = [n for n in schedule if n in group.nodes]
                    blocks.append(("coupled", group, group_schedule))
            else:
                blocks.append(("node", node_name))

        # Identify edges within each coupling group (these become
        # forward edges during iteration, not back-edges)
        coupled_internal_edges: set[EdgeSpec] = set()
        for group in coupling_groups:
            for edge in self._edges:
                if edge.source_node in group.nodes and edge.target_node in group.nodes:
                    coupled_internal_edges.add(edge)

        # Pre-index edges by target node -- O(E) setup, O(degree) per node
        edges_by_target: dict[str, list[EdgeSpec]] = defaultdict(list)
        for edge in self._edges:
            edges_by_target[edge.target_node].append(edge)

        # Pre-index external inputs by target node
        ext_by_target: dict[str, list[ExternalInputSpec]] = defaultdict(list)
        for ei in self._external_inputs:
            ext_by_target[ei.target_node].append(ei)

        # Capture which nodes have external inputs (for the fast path)
        has_external = set(ext_by_target.keys())

        has_coupling = bool(coupling_groups)

        def _resolve_and_update_node(
            node_name, new_state, full_state, external_inputs,
            force_forward_edges=None,
        ):
            """Resolve boundary inputs and update a single node.

            Parameters
            ----------
            force_forward_edges : set or None
                If provided, edges in this set are treated as forward
                (use new_state) even if they are in back_edge_set.
            """
            boundary_inputs: dict[str, Any] = {}

            for edge in edges_by_target[node_name]:
                # Determine source state: back-edges read from full_state
                # (previous timestep), forward edges from new_state.
                # Exception: edges within a coupling group are always
                # forward during Gauss-Seidel iteration.
                if edge in back_edge_set and (
                    force_forward_edges is None
                    or edge not in force_forward_edges
                ):
                    src_state = full_state
                else:
                    src_state = new_state
                value = src_state[edge.source_node][edge.source_field]
                if edge.transform is not None:
                    value = edge.transform(value)
                boundary_inputs[edge.target_field] = value

            if node_name in has_external:
                node_ext = external_inputs.get(node_name, {})
                for ei in ext_by_target[node_name]:
                    if ei.target_field in node_ext:
                        boundary_inputs[ei.target_field] = node_ext[ei.target_field]

            spec = nodes[node_name]
            return spec.update_fn(
                new_state[node_name], boundary_inputs, spec.timestep
            )

        def _run_coupled_block(group, group_schedule, new_state, full_state, external_inputs):
            """Execute a coupling group with Gauss-Seidel iteration.

            In Gauss-Seidel coupling, each iteration re-solves the SAME
            timestep with updated boundary conditions from the latest
            iteration.  Crucially, each node integrates from the
            *initial* state (beginning of timestep), NOT from the
            previous iteration's output.  Only the boundary conditions
            change between iterations.
            """
            max_iters = group.max_iterations
            tol = group.tolerance
            group_node_names = list(group_schedule)

            # Edges internal to this group are forced forward
            group_internal = set()
            for edge in self._edges:
                if edge.source_node in group.nodes and edge.target_node in group.nodes:
                    group_internal.add(edge)

            # Save the initial state for each node at the beginning of
            # the timestep -- this is what we always integrate FROM.
            initial_node_states = {nn: new_state[nn] for nn in group_node_names}

            def one_pass(latest_results):
                """Execute all nodes in the group once (Gauss-Seidel order).

                Each node integrates from its initial state but resolves
                boundary conditions from ``latest_results`` (which
                contains the most recent iteration outputs for other
                nodes in the group).
                """
                s = {k: v for k, v in latest_results.items()}
                for nn in group_node_names:
                    # Resolve boundary inputs from latest results
                    boundary_inputs: dict[str, Any] = {}
                    for edge in edges_by_target[nn]:
                        if edge in back_edge_set and (
                            edge not in group_internal
                        ):
                            src_state = full_state
                        else:
                            src_state = s
                        value = src_state[edge.source_node][edge.source_field]
                        if edge.transform is not None:
                            value = edge.transform(value)
                        boundary_inputs[edge.target_field] = value

                    if nn in has_external:
                        node_ext = external_inputs.get(nn, {})
                        for ei in ext_by_target[nn]:
                            if ei.target_field in node_ext:
                                boundary_inputs[ei.target_field] = node_ext[ei.target_field]

                    spec = nodes[nn]
                    # Integrate from INITIAL state (beginning of timestep),
                    # not from the previous iteration's output.
                    s[nn] = spec.update_fn(
                        initial_node_states[nn], boundary_inputs, spec.timestep
                    )
                return s

            # Run first iteration
            state_after_first = one_pass(new_state)

            def _compute_residual(s_new, s_old):
                total = jnp.array(0.0)
                for nn in group_node_names:
                    for field_name in s_new[nn]:
                        diff = s_new[nn][field_name] - s_old[nn][field_name]
                        total = total + jnp.sum(diff ** 2)
                return jnp.sqrt(total)

            if max_iters <= 1:
                # No iteration needed, just one pass
                result = {k: v for k, v in new_state.items()}
                for nn in group_node_names:
                    result[nn] = state_after_first[nn]
                return result

            # Use fori_loop (fixed iterations, differentiable) with
            # jnp.where for early convergence.  The update is always
            # computed (for JAX traceability) but applied only if the
            # solution has not yet converged.
            def body_fn(i, carry):
                s_cur, converged = carry
                s_new = one_pass(s_cur)
                residual = _compute_residual(s_new, s_cur)
                new_converged = converged | (residual <= tol)
                # If already converged, keep current state (no-op)
                s_merged = {}
                for k_s in s_cur:
                    if k_s in {nn for nn in group_node_names}:
                        s_merged[k_s] = jax.tree.map(
                            lambda n, o: jnp.where(new_converged, o, n),
                            s_new[k_s], s_cur[k_s],
                        )
                    else:
                        s_merged[k_s] = s_cur[k_s]
                return s_merged, new_converged

            init_carry = (state_after_first, jnp.array(False))
            final_state, _ = jax.lax.fori_loop(
                1, max_iters, body_fn, init_carry
            )

            # Merge coupled nodes back
            result = {k: v for k, v in new_state.items()}
            for nn in group_node_names:
                result[nn] = final_state[nn]
            return result

        if not is_multirate and not has_coupling:
            # ---- Uniform-rate, no coupling: fast path ----
            def graph_step(full_state, external_inputs):
                new_state = {k: v for k, v in full_state.items()}

                for node_name in schedule:
                    new_state[node_name] = _resolve_and_update_node(
                        node_name, new_state, full_state, external_inputs
                    )
                return new_state

            return graph_step

        if has_coupling and not is_multirate:
            # ---- Coupling groups, uniform rate ----
            def graph_step_coupled(full_state, external_inputs):
                new_state = {k: v for k, v in full_state.items()}

                for block in blocks:
                    if block[0] == "node":
                        node_name = block[1]
                        new_state[node_name] = _resolve_and_update_node(
                            node_name, new_state, full_state, external_inputs
                        )
                    else:
                        _, group, group_schedule = block
                        new_state = _run_coupled_block(
                            group, group_schedule, new_state,
                            full_state, external_inputs,
                        )

                return new_state

            return graph_step_coupled

        # ---- Multi-rate path (with or without coupling) ----
        def graph_step_multirate(full_state, external_inputs):
            step_count = full_state[_META_KEY]["step_count"]
            new_state = {k: v for k, v in full_state.items()}

            def _apply_multirate(node_name, updated, current_state):
                rd = rate_dividers[node_name]
                if rd == 1:
                    return updated
                should_run = (step_count % rd) == 0
                return jax.tree.map(
                    lambda new_val, old_val: jnp.where(should_run, new_val, old_val),
                    updated,
                    current_state[node_name],
                )

            if has_coupling:
                for block in blocks:
                    if block[0] == "node":
                        node_name = block[1]
                        updated = _resolve_and_update_node(
                            node_name, new_state, full_state, external_inputs
                        )
                        new_state[node_name] = _apply_multirate(
                            node_name, updated, new_state
                        )
                    else:
                        _, group, group_schedule = block
                        coupled_result = _run_coupled_block(
                            group, group_schedule, new_state,
                            full_state, external_inputs,
                        )
                        for nn in group_schedule:
                            new_state[nn] = _apply_multirate(
                                nn, coupled_result[nn], new_state
                            )
            else:
                for node_name in schedule:
                    updated = _resolve_and_update_node(
                        node_name, new_state, full_state, external_inputs
                    )
                    new_state[node_name] = _apply_multirate(
                        node_name, updated, new_state
                    )

            # Increment step counter
            new_state[_META_KEY] = {
                "step_count": step_count + 1,
            }
            return new_state

        return graph_step_multirate

    def _default_external_inputs(self) -> dict[str, dict]:
        """Build a zero-valued external_inputs dict matching declared specs."""
        if not self._external_inputs:
            return _EMPTY_EXTERNAL_INPUTS
        ext: dict[str, dict] = {}
        for ei in self._external_inputs:
            ext.setdefault(ei.target_node, {})[ei.target_field] = jnp.zeros(
                ei.shape, dtype=ei.dtype
            )
        return ext

    # ------------------------------------------------------------------
    # Internal helpers for _meta stripping
    # ------------------------------------------------------------------

    def _user_state(self, full_state: dict) -> dict:
        """Return state dict without the internal ``_meta`` key."""
        if _META_KEY not in full_state:
            return full_state
        return {k: v for k, v in full_state.items() if k != _META_KEY}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def step(self, external_inputs: Optional[dict[str, dict]] = None) -> dict[str, dict]:
        """Advance the simulation by one base timestep.

        Parameters
        ----------
        external_inputs : dict, optional
            Values injected from outside the graph, structured as
            ``{node_name: {field_name: value, ...}, ...}``.
            If ``None``, zeros are used for all declared external inputs.

        Returns the full state dict after the step (excluding internal
        metadata).
        """
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        self._state = self._compiled_step(self._state, external_inputs)
        user_state = self._user_state(self._state)
        self._notify(EVENT_STEP, user_state)
        return user_state

    def run(
        self,
        n_steps: int,
        callback: Optional[Callable] = None,
        external_inputs: Optional[dict[str, dict]] = None,
    ) -> None:
        """Run *n_steps* simulation steps (at the base timestep rate).

        Parameters
        ----------
        n_steps : int
            Number of base-rate steps to execute.
        callback : callable, optional
            Called after every step with ``(step_index, state_dict)``.
            The state dict excludes internal metadata.
        external_inputs : dict, optional
            Static external inputs applied every step.  For dynamic
            inputs that change each step, use :meth:`step` in a loop
            or use a ``CommandReceiver`` with ``RealtimeRunner``.
        """
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        for i in range(n_steps):
            self._state = self._compiled_step(self._state, external_inputs)
            user_state = self._user_state(self._state)
            self._notify(EVENT_STEP, user_state)
            if callback is not None:
                callback(i, user_state)

    def run_scan(
        self,
        n_steps: int,
        external_inputs: Optional[dict[str, dict]] = None,
    ) -> dict[str, dict]:
        """Run *n_steps* using ``jax.lax.scan`` for maximum performance.

        Unlike :meth:`run`, this method pushes the entire loop into XLA
        via ``jax.lax.scan``, eliminating Python-loop and JAX-dispatch
        overhead.  The full computation is JIT-compiled into a single
        XLA program.

        Trade-offs compared to :meth:`run` / :meth:`step`:

        * No per-step callback or observer notifications.
        * External inputs are **static** -- the same values are applied
          at every timestep.  For dynamic per-step inputs, use
          :meth:`step` in a loop or ``RealtimeRunner``.

        Parameters
        ----------
        n_steps : int
            Number of base-rate simulation steps to execute.
        external_inputs : dict, optional
            Static external inputs applied identically every step.
            If ``None``, zeros are used for all declared external inputs.

        Returns
        -------
        dict[str, dict]
            The final state of the graph after *n_steps* (excluding
            internal metadata).
        """
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        # Build the raw (unjitted) step function -- lax.scan will JIT
        # the entire scan body, so an inner jit would be redundant.
        step_fn = self._build_step_fn()

        # Close over the static external inputs so the scan body has
        # the correct signature: (carry, x) -> (carry, None)
        ext = external_inputs

        def scan_body(state, _unused):
            new_state = step_fn(state, ext)
            return new_state, None

        final_state, _ = jax.lax.scan(scan_body, self._state, None, length=n_steps)
        self._state = final_state
        return self._user_state(self._state)

    def run_scan_with_history(
        self,
        n_steps: int,
        external_inputs: Optional[dict[str, dict]] = None,
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """Run *n_steps* via ``jax.lax.scan``, returning all intermediate states.

        Like :meth:`run_scan` but also collects the state at every
        timestep into stacked arrays, which is useful for plotting and
        post-hoc analysis without Python-loop overhead.

        Parameters
        ----------
        n_steps : int
            Number of base-rate simulation steps to execute.
        external_inputs : dict, optional
            Static external inputs applied identically every step.
            If ``None``, zeros are used for all declared external inputs.

        Returns
        -------
        (final_state, history) : tuple
            *final_state* is the state dict after the last step
            (same as :meth:`run_scan` would return), excluding internal
            metadata.
            *history* has the same nested-dict structure as state (also
            excluding internal metadata), but each leaf is a JAX array
            with an extra leading axis of size *n_steps*.
            ``history["ball"]["position"]`` is a 1-D array of shape
            ``(n_steps,)`` (or ``(n_steps, *field_shape)`` for
            non-scalar fields) holding the value **after** each step.
        """
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        step_fn = self._build_step_fn()
        ext = external_inputs

        def scan_body(state, _unused):
            new_state = step_fn(state, ext)
            return new_state, new_state  # carry, output (stacked by scan)

        final_state, history = jax.lax.scan(scan_body, self._state, None, length=n_steps)
        self._state = final_state
        return self._user_state(final_state), self._user_state(history)

    # ------------------------------------------------------------------
    # Parameter sweeps via vmap
    # ------------------------------------------------------------------

    def run_sweep(
        self,
        n_steps: int,
        initial_states: dict[str, dict],
        external_inputs: Optional[dict[str, dict]] = None,
        return_history: bool = False,
    ):
        """Run a batch of simulations over different initial conditions.

        Uses ``jax.vmap`` over ``jax.lax.scan`` to execute all
        variations in parallel (vectorised on GPU/TPU).

        Parameters
        ----------
        n_steps : int
            Number of steps per simulation.
        initial_states : dict[str, dict]
            Batched initial states.  Each leaf array must have a
            leading batch dimension of the same size.  For example::

                {"ball": {"position": jnp.array([1.0, 2.0, 3.0]),
                          "velocity": jnp.zeros(3)}}

            runs 3 simulations with initial positions 1, 2, 3.
        external_inputs : dict, optional
            Static external inputs (not batched — same for all runs).
        return_history : bool
            If True, return ``(final_states, histories)`` where
            histories has shape ``(batch, n_steps, ...)``.
            If False (default), return only ``final_states``.

        Returns
        -------
        final_states : dict[str, dict]
            Batched final states (leading batch dimension on each leaf).
        histories : dict[str, dict], optional
            Only if ``return_history=True``.  Batched histories with
            shape ``(batch, n_steps, ...)`` on each leaf.
        """
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        step_fn = self._build_step_fn()
        ext = external_inputs

        # Build the scan function
        if return_history:
            def simulate(init_state):
                def scan_body(state, _unused):
                    new_state = step_fn(state, ext)
                    return new_state, new_state
                final, hist = jax.lax.scan(scan_body, init_state, None, length=n_steps)
                return self._user_state(final), self._user_state(hist)

            vmapped = jax.vmap(simulate)
            finals, histories = vmapped(initial_states)
            return finals, histories
        else:
            def simulate(init_state):
                def scan_body(state, _unused):
                    new_state = step_fn(state, ext)
                    return new_state, None
                final, _ = jax.lax.scan(scan_body, init_state, None, length=n_steps)
                return self._user_state(final)

            vmapped = jax.vmap(simulate)
            return vmapped(initial_states)

    # ------------------------------------------------------------------
    # Adaptive timestepping
    # ------------------------------------------------------------------

    def _build_dt_step_fn(self) -> Callable:
        """Build a step function parameterised by ``dt``.

        Returns a function ``(state, external_inputs, dt) -> new_state``
        where *dt* is a JAX scalar that overrides each node's compiled
        timestep.  Used by :meth:`run_adaptive`.
        """
        schedule = list(self._schedule)
        nodes_dict = dict(self._nodes)
        back_edge_set = set(self._back_edges)
        coupling_groups = list(self._coupling_groups)

        node_to_group: dict[str, CouplingGroup] = {}
        for group in coupling_groups:
            for name in group.nodes:
                node_to_group[name] = group

        blocks: list[tuple] = []
        handled_groups: set[int] = set()
        for node_name in schedule:
            if node_name in node_to_group:
                group = node_to_group[node_name]
                gid = id(group)
                if gid not in handled_groups:
                    handled_groups.add(gid)
                    group_schedule = [n for n in schedule if n in group.nodes]
                    blocks.append(("coupled", group, group_schedule))
            else:
                blocks.append(("node", node_name))

        edges_by_target: dict[str, list[EdgeSpec]] = defaultdict(list)
        for edge in self._edges:
            edges_by_target[edge.target_node].append(edge)

        ext_by_target: dict[str, list[ExternalInputSpec]] = defaultdict(list)
        for ei in self._external_inputs:
            ext_by_target[ei.target_node].append(ei)

        has_external = set(ext_by_target.keys())
        has_coupling = bool(coupling_groups)

        coupled_internal_edges: set[EdgeSpec] = set()
        for group in coupling_groups:
            for edge in self._edges:
                if edge.source_node in group.nodes and edge.target_node in group.nodes:
                    coupled_internal_edges.add(edge)

        def _resolve_and_update(node_name, new_state, full_state, ext, dt,
                                force_forward_edges=None):
            boundary_inputs: dict[str, Any] = {}
            for edge in edges_by_target[node_name]:
                if edge in back_edge_set and (
                    force_forward_edges is None
                    or edge not in force_forward_edges
                ):
                    src_state = full_state
                else:
                    src_state = new_state
                value = src_state[edge.source_node][edge.source_field]
                if edge.transform is not None:
                    value = edge.transform(value)
                boundary_inputs[edge.target_field] = value

            if node_name in has_external:
                node_ext = ext.get(node_name, {})
                for ei in ext_by_target[node_name]:
                    if ei.target_field in node_ext:
                        boundary_inputs[ei.target_field] = node_ext[ei.target_field]

            spec = nodes_dict[node_name]
            return spec.update_fn(new_state[node_name], boundary_inputs, dt)

        def dt_step_fn(state, external_inputs, dt):
            new_state = {k: v for k, v in state.items()}

            if has_coupling:
                for block in blocks:
                    if block[0] == "node":
                        nn = block[1]
                        new_state[nn] = _resolve_and_update(
                            nn, new_state, state, external_inputs, dt
                        )
                    else:
                        _, group, group_schedule = block
                        group_internal = set()
                        for edge in self._edges:
                            if edge.source_node in group.nodes and edge.target_node in group.nodes:
                                group_internal.add(edge)

                        # Save initial state for Gauss-Seidel
                        init_group = {nn2: new_state[nn2] for nn2 in group_schedule}

                        def one_pass(latest):
                            s2 = {k: v for k, v in latest.items()}
                            for nn2 in group_schedule:
                                # Resolve boundaries from latest results
                                bi: dict[str, Any] = {}
                                for edge in edges_by_target[nn2]:
                                    if edge in back_edge_set and edge not in group_internal:
                                        src = state
                                    else:
                                        src = s2
                                    val = src[edge.source_node][edge.source_field]
                                    if edge.transform is not None:
                                        val = edge.transform(val)
                                    bi[edge.target_field] = val
                                if nn2 in has_external:
                                    ne = external_inputs.get(nn2, {})
                                    for ei in ext_by_target[nn2]:
                                        if ei.target_field in ne:
                                            bi[ei.target_field] = ne[ei.target_field]
                                spec = nodes_dict[nn2]
                                # Integrate from INITIAL state
                                s2[nn2] = spec.update_fn(init_group[nn2], bi, dt)
                            return s2

                        result = one_pass(new_state)
                        if group.max_iterations > 1:
                            tol = group.tolerance
                            max_iters = group.max_iterations

                            def _residual(s_new, s_old):
                                total = jnp.array(0.0)
                                for nn2 in group_schedule:
                                    for f in s_new[nn2]:
                                        diff = s_new[nn2][f] - s_old[nn2][f]
                                        total += jnp.sum(diff ** 2)
                                return jnp.sqrt(total)

                            def body_fn_dt(i, carry):
                                s_cur, converged = carry
                                s_new2 = one_pass(s_cur)
                                residual = _residual(s_new2, s_cur)
                                new_converged = converged | (residual <= tol)
                                s_merged = {}
                                for ks in s_cur:
                                    if ks in set(group_schedule):
                                        s_merged[ks] = jax.tree.map(
                                            lambda n, o: jnp.where(new_converged, o, n),
                                            s_new2[ks], s_cur[ks],
                                        )
                                    else:
                                        s_merged[ks] = s_cur[ks]
                                return s_merged, new_converged

                            init = (result, jnp.array(False))
                            result, _ = jax.lax.fori_loop(
                                1, max_iters, body_fn_dt, init
                            )

                        for nn2 in group_schedule:
                            new_state[nn2] = result[nn2]
            else:
                for nn in schedule:
                    new_state[nn] = _resolve_and_update(
                        nn, new_state, state, external_inputs, dt
                    )

            return new_state

        return dt_step_fn

    def run_adaptive(
        self,
        t_end: float,
        dt_initial: float = 0.01,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        dt_min: float = 1e-8,
        dt_max: float = 0.1,
        external_inputs: Optional[dict[str, dict]] = None,
        callback: Optional[Callable] = None,
    ) -> tuple[dict[str, dict], dict]:
        """Run with adaptive timestepping until *t_end*.

        Uses Richardson extrapolation (step-doubling) for error
        estimation and a PI controller for step-size adjustment.
        Incompatible with multi-rate graphs.

        Parameters
        ----------
        t_end : float
            Target simulation end time.
        dt_initial : float
            Initial timestep guess.
        atol, rtol : float
            Absolute and relative error tolerances.
        dt_min, dt_max : float
            Timestep bounds.
        external_inputs : dict, optional
            Static external inputs applied every step.
        callback : callable, optional
            Called after every *accepted* step with
            ``(sim_time, dt_used, state_dict)``.

        Returns
        -------
        (final_state, info) : tuple
            *final_state* is the state dict (excluding metadata).
            *info* is a dict with ``n_steps``, ``n_rejected``,
            ``dt_history`` (list of used timesteps), and
            ``t_history`` (list of simulation times).
        """
        if self._is_multirate:
            raise RuntimeError(
                "Adaptive timestepping is incompatible with multi-rate "
                "graphs.  All nodes must share the same timestep."
            )
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        from maddening.core.adaptive import AdaptiveConfig, _tree_error_norm

        config = AdaptiveConfig(
            dt_initial=dt_initial,
            atol=atol,
            rtol=rtol,
            dt_min=dt_min,
            dt_max=dt_max,
        )

        dt_step_fn = self._build_dt_step_fn()
        # JIT-compile the dt-parameterised step
        dt_step_jit = jax.jit(dt_step_fn)

        t = 0.0
        dt = dt_initial
        n_steps = 0
        n_rejected = 0
        dt_history = []
        t_history = []
        state = self._state

        while t < t_end:
            # Clamp dt so we don't overshoot t_end
            dt = min(dt, t_end - t)
            dt = max(dt, dt_min)

            dt_jax = jnp.array(dt)

            # Full step
            state_full = dt_step_jit(state, external_inputs, dt_jax)
            # Two half-steps
            half_dt = dt_jax / 2.0
            state_half = dt_step_jit(state, external_inputs, half_dt)
            state_half = dt_step_jit(state_half, external_inputs, half_dt)

            # Error estimate
            user_full = self._user_state(state_full)
            user_half = self._user_state(state_half)
            error_norm = float(_tree_error_norm(
                user_half, user_full, config.atol, config.rtol
            ))

            if error_norm <= 1.0:
                # Accept step -- use the more accurate (half-step) result
                state = state_half
                t += dt
                n_steps += 1
                dt_history.append(dt)
                t_history.append(t)

                if callback is not None:
                    callback(t, dt, self._user_state(state))
                self._notify(EVENT_STEP, self._user_state(state))

                # Grow dt
                if error_norm > 0:
                    factor = config.safety * (1.0 / error_norm) ** (1.0 / (config.order + 1))
                else:
                    factor = config.max_factor
                factor = min(max(factor, config.min_factor), config.max_factor)
                dt = min(dt * factor, dt_max)
            else:
                # Reject step -- shrink dt and retry
                n_rejected += 1
                factor = config.safety * (1.0 / error_norm) ** (1.0 / (config.order + 1))
                factor = min(max(factor, config.min_factor), config.max_factor)
                dt = max(dt * factor, dt_min)

                if dt <= dt_min:
                    # Cannot shrink further; accept with warning
                    warnings.warn(
                        f"Adaptive stepper hit dt_min={dt_min} at t={t:.6g} "
                        f"(error={error_norm:.3e}). Accepting step.",
                        stacklevel=2,
                    )
                    state = state_half
                    t += dt_min
                    n_steps += 1
                    dt_history.append(dt_min)
                    t_history.append(t)
                    if callback is not None:
                        callback(t, dt_min, self._user_state(state))
                    self._notify(EVENT_STEP, self._user_state(state))

        self._state = state
        info = {
            "n_steps": n_steps,
            "n_rejected": n_rejected,
            "dt_history": dt_history,
            "t_history": t_history,
        }
        return self._user_state(self._state), info

    def run_adaptive_scan(
        self,
        t_end: float,
        max_steps: int = 10000,
        dt_initial: float = 0.01,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        dt_min: float = 1e-8,
        dt_max: float = 0.1,
        external_inputs: Optional[dict[str, dict]] = None,
    ) -> tuple[dict[str, dict], dict[str, dict], dict]:
        """Adaptive timestepping via ``jax.lax.scan`` (differentiable).

        Like :meth:`run_adaptive` but fully JIT-compiled and
        differentiable.  Uses a fixed *max_steps* allocation; steps
        past ``t_end`` are no-ops.

        Parameters
        ----------
        t_end : float
            Target end time.
        max_steps : int
            Maximum number of steps (scan length).  Steps after reaching
            ``t_end`` produce no-op outputs.
        dt_initial, atol, rtol, dt_min, dt_max : float
            Same as :meth:`run_adaptive`.
        external_inputs : dict, optional
            Static external inputs.

        Returns
        -------
        (final_state, history, info) : tuple
            *final_state*: state after last accepted step.
            *history*: stacked state at each step (shape ``(max_steps, ...)``).
            *info*: dict with ``n_steps`` (actual steps taken, as JAX array).
        """
        if self._is_multirate:
            raise RuntimeError(
                "Adaptive timestepping is incompatible with multi-rate graphs."
            )
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        from maddening.core.adaptive import AdaptiveConfig, _tree_error_norm

        config = AdaptiveConfig(
            dt_initial=dt_initial, atol=atol, rtol=rtol,
            dt_min=dt_min, dt_max=dt_max,
        )

        dt_step_fn = self._build_dt_step_fn()
        ext = external_inputs

        t_end_jax = jnp.array(t_end)
        dt_min_jax = jnp.array(dt_min)
        dt_max_jax = jnp.array(dt_max)

        def scan_body(carry, _unused):
            state, t, dt, n_accepted = carry

            # Clamp dt to not overshoot
            dt = jnp.minimum(dt, t_end_jax - t)
            dt = jnp.maximum(dt, dt_min_jax)

            # Check if we've already reached t_end
            done = t >= t_end_jax

            # Full step + two half-steps
            state_full = dt_step_fn(state, ext, dt)
            half_dt = dt / 2.0
            state_half = dt_step_fn(state, ext, half_dt)
            state_half = dt_step_fn(state_half, ext, half_dt)

            # Error estimate
            user_full = {k: v for k, v in state_full.items() if k != _META_KEY}
            user_half = {k: v for k, v in state_half.items() if k != _META_KEY}
            error_norm = _tree_error_norm(user_half, user_full, config.atol, config.rtol)

            accepted = (error_norm <= 1.0) | (dt <= dt_min_jax)

            # PI controller
            safe_error = jnp.maximum(error_norm, 1e-10)
            factor = config.safety * jnp.power(1.0 / safe_error, 1.0 / (config.order + 1))
            factor = jnp.clip(factor, config.min_factor, config.max_factor)
            dt_next = jnp.clip(dt * factor, dt_min_jax, dt_max_jax)

            # If done, keep state unchanged; if accepted, use half-step result
            new_state = jax.tree.map(
                lambda s, h: jnp.where(done, s, jnp.where(accepted, h, s)),
                state, state_half,
            )
            new_t = jnp.where(done, t, jnp.where(accepted, t + dt, t))
            new_dt = jnp.where(done, dt, dt_next)
            new_n = jnp.where(done, n_accepted, jnp.where(accepted, n_accepted + 1, n_accepted))

            # Output the state for history (will be no-op state if not accepted)
            output_state = self._user_state(new_state)

            return (new_state, new_t, new_dt, new_n), output_state

        init_carry = (
            self._state,
            jnp.array(0.0),
            jnp.array(dt_initial),
            jnp.array(0, dtype=jnp.int32),
        )

        (final_state, final_t, final_dt, n_accepted), history = jax.lax.scan(
            scan_body, init_carry, None, length=max_steps
        )

        self._state = final_state
        info = {"n_steps": n_accepted, "final_t": final_t, "final_dt": final_dt}
        return self._user_state(final_state), history, info

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_node_state(self, name: str) -> dict:
        if name not in self._nodes:
            raise KeyError(f"No node named '{name}'.")
        if name not in self._state:
            raise KeyError(f"No node named '{name}'.")
        return self._state[name]

    def set_node_state(self, name: str, state: dict) -> None:
        if name not in self._nodes:
            raise KeyError(f"No node named '{name}'.")
        self._state[name] = state

    # ------------------------------------------------------------------
    # Observer pattern
    # ------------------------------------------------------------------

    def add_observer(self, callback: Callable) -> None:
        """Register a callback.  Called as ``callback(event, data)``."""
        self._observers.append(callback)

    def _notify(self, event: str, data: Any = None) -> None:
        for cb in self._observers:
            cb(event, data)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the graph structure (not runtime state)."""
        return {
            "nodes": [spec.node.to_dict() for spec in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
            "external_inputs": [
                {
                    "target_node": ei.target_node,
                    "target_field": ei.target_field,
                    "shape": list(ei.shape),
                }
                for ei in self._external_inputs
            ],
        }

    @classmethod
    def from_dict(
        cls,
        config: dict,
        node_registry: dict[str, type],
    ) -> "GraphManager":
        """Reconstruct a GraphManager from a serialised config.

        *node_registry* maps node type names (e.g. ``"BallNode"``) to
        the corresponding class.
        """
        gm = cls()
        for nd in config["nodes"]:
            node_cls = node_registry[nd["type"]]
            node = node_cls(name=nd["name"], timestep=nd["timestep"], **nd.get("params", {}))
            gm.add_node(node)
        for ed in config["edges"]:
            gm.add_edge(
                source=ed["source_node"],
                target=ed["target_node"],
                source_field=ed["source_field"],
                target_field=ed["target_field"],
            )
        for ei in config.get("external_inputs", []):
            gm.add_external_input(
                target_node=ei["target_node"],
                target_field=ei["target_field"],
                shape=tuple(ei.get("shape", ())),
            )
        return gm

    # ------------------------------------------------------------------
    # Checkpoint / restore
    # ------------------------------------------------------------------

    def save_state(self, path) -> "Path":
        """Save all node states to an ``.npz`` file.

        See :func:`maddening.core.checkpoint.save_state` for details.
        """
        from maddening.core.checkpoint import save_state
        return save_state(self, path)

    def load_state(self, path) -> None:
        """Load node states from an ``.npz`` file.

        See :func:`maddening.core.checkpoint.load_state` for details.
        """
        from maddening.core.checkpoint import load_state
        load_state(self, path)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def timestep(self) -> float:
        """Return the base timestep (GCD of all node timesteps).

        For uniform-rate graphs this is the common timestep.  For
        multi-rate graphs this is the smallest step at which the
        compiled function advances.
        """
        timesteps = sorted({spec.timestep for spec in self._nodes.values()})
        if not timesteps:
            raise RuntimeError("No nodes registered.")
        if len(timesteps) == 1:
            return timesteps[0]
        return _multi_gcd(timesteps)

    @property
    def is_multirate(self) -> bool:
        """Whether the graph has nodes with different timesteps."""
        return self._is_multirate

    @property
    def rate_dividers(self) -> dict[str, int]:
        """Per-node rate divider (node_dt / base_dt, rounded).

        Only meaningful after :meth:`compile`.
        """
        return dict(self._rate_dividers)

    @property
    def base_timestep(self) -> float:
        """Alias for :attr:`timestep`."""
        return self.timestep

    @property
    def node_names(self) -> list[str]:
        return list(self._nodes.keys())

    @property
    def schedule(self) -> list[str]:
        return list(self._schedule)

    def __repr__(self) -> str:
        n = len(self._nodes)
        e = len(self._edges)
        ei = len(self._external_inputs)
        compiled = "compiled" if not self._dirty else "dirty"
        parts = [f"{n} nodes", f"{e} edges"]
        if ei:
            parts.append(f"{ei} external inputs")
        if self._is_multirate:
            parts.append("multi-rate")
        return f"GraphManager({', '.join(parts)}, {compiled})"
